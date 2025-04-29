#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度学习推荐模型
基于TensorFlow实现的用于混合推荐系统的深度学习组件
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout, BatchNormalization, Add, LeakyReLU, Multiply, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall

# 尝试导入tensorflow_addons，如果不可用则跳过
try:
    import tensorflow_addons as tfa
    tfa_available = True
except ImportError:
    tfa_available = False
    import warnings
    warnings.warn("tensorflow_addons库不可用，将使用基本版TensorFlow功能")

# 尝试导入混合精度训练功能
try:
    from tensorflow.keras.mixed_precision import global_policy, set_global_policy
    mixed_precision_available = True
except ImportError:
    mixed_precision_available = False
    import warnings
    warnings.warn("混合精度训练不可用，将使用默认精度")

import logging
import pickle
from sklearn.model_selection import KFold
import math

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepRecommender")

# 设置混合精度训练 - 禁用混合精度以避免数据类型问题
if mixed_precision_available:
    try:
        # 强制使用float32，不使用混合精度
        set_global_policy('float32')
        logger.info("使用标准精度(float32)训练")
    except Exception as e:
        logger.warning(f"设置精度策略时出错: {e}")


class AttentionLayer(tf.keras.layers.Layer):
    """增强型自注意力层，使用多头注意力机制提高模型表达能力"""
    
    def __init__(self, attention_dim=128, num_heads=8, dropout_rate=0.1, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # 为每个注意力头创建独立的权重矩阵 - 更高维度
        self.query_weights = [Dense(self.attention_dim) for _ in range(self.num_heads)]
        self.key_weights = [Dense(self.attention_dim) for _ in range(self.num_heads)]
        self.value_weights = [Dense(input_shape[-1]//self.num_heads) for _ in range(self.num_heads)]
        
        # 用于合并多头注意力的输出
        self.combine = Dense(input_shape[-1])
        self.dropout = Dropout(self.dropout_rate)
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 缩放因子
        self.scale = tf.sqrt(tf.cast(self.attention_dim // self.num_heads, tf.float32))
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # 确保输入为float32类型，避免混合精度问题
        inputs = tf.cast(inputs, tf.float32)
        
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        head_dim = self.attention_dim // self.num_heads
        
        # 生成查询、键、值
        queries = []
        keys = []
        values = []
        
        for i in range(self.num_heads):
            q = self.query_weights[i](inputs)  # [batch_size, seq_len, head_dim]
            k = self.key_weights[i](inputs)    # [batch_size, seq_len, head_dim]
            v = self.value_weights[i](inputs)  # [batch_size, seq_len, head_dim]
            
            # 确保float32类型
            q = tf.cast(q, tf.float32)
            k = tf.cast(k, tf.float32)
            v = tf.cast(v, tf.float32)
            
            queries.append(q)
            keys.append(k)
            values.append(v)
        
        # 多头注意力计算
        attention_outputs = []
        
        for i in range(self.num_heads):
            # 计算注意力分数 - 确保类型一致
            scores = tf.matmul(queries[i], keys[i], transpose_b=True)  # [batch_size, seq_len, seq_len]
            scores = tf.cast(scores, tf.float32)
            head_dim_float = tf.cast(head_dim, tf.float32)
            scores = scores / tf.sqrt(head_dim_float)
            
            # 应用softmax获取注意力权重
            attention_weights = tf.nn.softmax(scores, axis=-1)
            
            # 应用dropout (如果处于训练模式)
            if training:
                attention_weights = self.dropout(attention_weights, training=training)
            
            # 注意力加权求和
            attention_output = tf.matmul(attention_weights, values[i])  # [batch_size, seq_len, head_dim]
            attention_outputs.append(attention_output)
        
        # 拼接多头注意力的输出
        concat_attention = tf.concat(attention_outputs, axis=-1)  # [batch_size, seq_len, attention_dim]
        concat_attention = tf.cast(concat_attention, tf.float32)
        
        # 应用最终的线性变换
        output = self.combine(concat_attention)
        
        # 应用dropout (如果处于训练模式)
        if training:
            output = self.dropout(output, training=training)
            
        # 确保返回float32类型
        return tf.cast(output, tf.float32)


class GatingLayer(tf.keras.layers.Layer):
    """改进的门控机制层，使用Highway Network结构控制信息流动和特征重要性"""
    
    def __init__(self, units, activation="sigmoid", dropout_rate=0.1, **kwargs):
        super(GatingLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.gate = Dense(self.units, activation=self.activation)
        self.transform = Dense(self.units, activation='tanh')  # 非线性变换
        
        # 如果输入维度不匹配，预先创建投影层
        if input_dim != self.units:
            self.projection = Dense(self.units, use_bias=False)
        else:
            self.projection = None
            
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(self.dropout_rate)
        super(GatingLayer, self).build(input_shape)
    
    def call(self, inputs, training=None):
        gate_values = self.gate(inputs)
        transformed = self.transform(inputs)
        
        # Highway Network结构: T*H + (1-T)*x
        if self.projection is not None:
            # 使用预创建的投影层
            inputs_proj = self.projection(inputs)
        else:
            inputs_proj = inputs
            
        output = gate_values * transformed + (1 - gate_values) * inputs_proj
        
        # 训练中使用dropout
        if training:
            output = self.dropout(output, training=training)
            
        return self.layer_norm(output)


class CrossNetwork(tf.keras.layers.Layer):
    """交叉网络层 - 显式建模特征交叉"""
    
    def __init__(self, num_layers=3, **kwargs):
        super(CrossNetwork, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.layers_list = []
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # 为每一层创建权重和偏置
        for i in range(self.num_layers):
            self.layers_list.append({
                'w': self.add_weight(name=f'w_{i}',
                                    shape=(input_dim, 1),
                                    initializer='glorot_uniform',
                                    trainable=True),
                'b': self.add_weight(name=f'b_{i}',
                                    shape=(input_dim,),
                                    initializer='zeros',
                                    trainable=True)
            })
        super(CrossNetwork, self).build(input_shape)
    
    def call(self, inputs):
        x0 = inputs
        x = inputs
        
        for i in range(self.num_layers):
            # 计算交叉项 x_{i+1} = x_0 * (x_i^T w_i + b_i) + x_i
            xw = tf.matmul(tf.expand_dims(x, axis=2), tf.expand_dims(self.layers_list[i]['w'], axis=1))
            xw = tf.squeeze(xw, axis=2)
            x = x0 * (xw + self.layers_list[i]['b']) + x
            
        return x


class MMOELayer(tf.keras.layers.Layer):
    """多目标混合专家模型层 - 增强多任务学习能力"""
    
    def __init__(self, num_experts=4, expert_dim=64, num_tasks=2, **kwargs):
        super(MMOELayer, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.num_tasks = num_tasks
        
    def build(self, input_shape):
        self.input_dim = input_shape[-1]
        
        # 创建专家网络
        self.experts = [
            Dense(self.expert_dim, activation='relu', 
                  kernel_regularizer=l2(1e-4),
                  name=f'expert_{i}')
            for i in range(self.num_experts)
        ]
        
        # 为每个任务创建门控网络
        self.gates = [
            Dense(self.num_experts, activation='softmax', 
                  kernel_regularizer=l2(1e-4),
                  name=f'gate_{i}')
            for i in range(self.num_tasks)
        ]
        
        super(MMOELayer, self).build(input_shape)
    
    def call(self, inputs):
        # 获取专家输出
        expert_outputs = [expert(inputs) for expert in self.experts]
        expert_outputs = tf.stack(expert_outputs, axis=1)  # [batch, num_experts, expert_dim]
        
        # 计算每个任务的门控输出
        mmoe_outputs = []
        for i in range(self.num_tasks):
            gate_output = self.gates[i](inputs)  # [batch, num_experts]
            gate_output = tf.expand_dims(gate_output, axis=2)  # [batch, num_experts, 1]
            
            # 加权组合专家输出
            weighted_expert = expert_outputs * gate_output
            mmoe_outputs.append(tf.reduce_sum(weighted_expert, axis=1))  # [batch, expert_dim]
            
        return mmoe_outputs  # 返回每个任务的加权输出


class DeepRecommender:
    """深度学习推荐模型类"""
    
    def __init__(self, n_users, n_items, embedding_dim=128, item_features=None, use_mmoe=True, spotify_features=None):
        """
        初始化深度推荐模型，集成多种先进技术
        
        参数:
            n_users: 用户数量
            n_items: 物品数量
            embedding_dim: 嵌入维度
            item_features: 物品特征矩阵 (可选)
            use_mmoe: 是否使用多任务混合专家
            spotify_features: Spotify音频特征 (可选)
        """
        # 基础参数
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.item_features = item_features
        self.use_mmoe = use_mmoe
        self.spotify_features = spotify_features
        
        # 特征维度处理
        self.item_feature_dim = 0
        if item_features is not None:
            self.item_feature_dim = item_features.shape[1]
            logger.info(f"使用物品特征, 维度: {self.item_feature_dim}")
        
        # 完全禁用混合精度训练
        self.use_mixed_precision = False
        
        # 强制使用float32作为默认数据类型
        tf.keras.backend.set_floatx('float32')
        
        # 如果可用，显式设置TensorFlow全局策略
        try:
            import tensorflow as tf
            tf.keras.mixed_precision.set_global_policy('float32')
            logger.info("明确设置为float32精度训练")
        except Exception as e:
            logger.warning(f"设置全局精度时出错: {e}")
        
        # 构建模型
        self.model = self._build_model()
        
        # 用户和物品映射
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None
        
        logger.info(f"初始化增强型深度学习推荐模型: {n_users} 用户, {n_items} 物品, {embedding_dim} 嵌入维度")
        if item_features is not None:
            logger.info(f"使用物品特征: {self.item_feature_dim} 个特征")
        if spotify_features is not None:
            logger.info(f"加载Spotify特征: {len(spotify_features)} 首歌曲")
            
    def process_msd_data(self, user_song_data, song_features=None, spotify_data=None):
        """
        处理Million Song Dataset数据和Spotify API数据
        
        参数:
            user_song_data (pd.DataFrame): 用户-歌曲交互数据，包含user_id, song_id, play_count/rating列
            song_features (pd.DataFrame): 歌曲特征数据
            spotify_data (dict): Spotify API特征数据
            
        返回:
            处理后的用户索引、物品索引和评分数组
        """
        logger.info("处理MSD和Spotify数据...")
        
        # 创建用户ID和物品ID的映射
        unique_users = user_song_data['user_id'].unique()
        unique_items = user_song_data['song_id'].unique()
        
        self.user_map = {user: idx for idx, user in enumerate(unique_users)}
        self.item_map = {item: idx for idx, item in enumerate(unique_items)}
        self.reverse_user_map = {idx: user for user, idx in self.user_map.items()}
        self.reverse_item_map = {idx: item for item, idx in self.item_map.items()}
        
        # 转换为索引
        user_indices = np.array([self.user_map[u] for u in user_song_data['user_id']])
        item_indices = np.array([self.item_map[i] for i in user_song_data['song_id']])
        
        # 获取评分/播放次数
        if 'rating' in user_song_data.columns:
            ratings = np.array(user_song_data['rating'])
        elif 'play_count' in user_song_data.columns:
            # 将播放次数转换为评分 (对数变换)
            play_counts = np.array(user_song_data['play_count'])
            ratings = np.log1p(play_counts) / np.log1p(play_counts.max())
        else:
            raise ValueError("用户-歌曲数据必须包含'rating'或'play_count'列")
            
        # 处理歌曲特征
        if song_features is not None and len(song_features) > 0:
            feature_cols = [c for c in song_features.columns if c != 'song_id']
            unique_songs = list(self.item_map.keys())
            
            self.item_features = np.zeros((len(unique_songs), len(feature_cols)))
            
            # 填充特征
            for i, song_id in enumerate(unique_songs):
                if song_id in song_features['song_id'].values:
                    row = song_features[song_features['song_id'] == song_id]
                    self.item_features[i] = row[feature_cols].values[0]
            
            self.item_feature_dim = self.item_features.shape[1]
            logger.info(f"从MSD提取了 {self.item_feature_dim} 个歌曲特征")
        
        # 整合Spotify特征
        if spotify_data is not None and len(spotify_data) > 0:
            self.spotify_features = spotify_data
            logger.info(f"加载了 {len(spotify_data)} 个Spotify歌曲特征")
        
        logger.info(f"数据处理完成: {len(unique_users)} 用户, {len(unique_items)} 歌曲, {len(ratings)} 交互记录")
        return user_indices, item_indices, ratings
    
    def _build_residual_block(self, x, units, dropout_rate=0.3):
        """高级残差块，提高模型表达能力和梯度流动"""
        skip = x
        
        # 如果输入和输出维度不同，使用线性投影
        if int(x.shape[-1]) != units:
            skip = Dense(units, use_bias=False, kernel_initializer='he_normal')(x)
        
        # 主路径 - 使用更强的非线性变换
        x = Dense(units * 2, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = Dropout(dropout_rate)(x)
        
        x = Dense(units, kernel_initializer='he_normal')(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = LeakyReLU(alpha=0.1)(x)
        
        # 带有Squeeze-and-Excitation的注意力机制
        se = Dense(units // 16, activation='relu')(x)
        se = Dense(units, activation='sigmoid')(se)
        x = Multiply()([x, se])
        
        # 残差连接和正则化
        x = Dropout(dropout_rate)(x)
        output = Add()([x, skip])
        return tf.keras.layers.LayerNormalization(epsilon=1e-6)(output)
    
    def _build_model(self):
        """构建高级混合推荐系统，集成多种最先进推荐算法"""
        # 输入层
        user_input = Input(shape=(1,), name='user_input', dtype='int32')
        item_input = Input(shape=(1,), name='item_input', dtype='int32')
        
        # 确保所有输入转换为float32类型 - 使用Lambda层包装tf操作
        user_input_float = Lambda(lambda x: tf.cast(x, dtype='float32'), name='user_float')(user_input)
        item_input_float = Lambda(lambda x: tf.cast(x, dtype='float32'), name='item_float')(item_input)
        
        # 特征输入处理
        feature_inputs = []
        if self.item_features is not None:
            item_feature_input = Input(shape=(self.item_feature_dim,), name='item_feature_input')
            feature_inputs.append(item_feature_input)
        
        # Spotify特征处理
        if self.spotify_features is not None:
            spotify_feature_input = Input(shape=(self.spotify_features.shape[1],), name='spotify_feature_input')
            feature_inputs.append(spotify_feature_input)
        
        # 增强型嵌入层 - 使用正交初始化和更高维度
        user_embedding = Embedding(self.n_users, self.embedding_dim*2, 
                                  embeddings_initializer='glorot_normal', 
                                  embeddings_regularizer=l2(1e-5),
                                  name='user_embedding')(user_input_float)
        
        item_embedding = Embedding(self.n_items, self.embedding_dim*2, 
                                  embeddings_initializer='glorot_normal', 
                                  embeddings_regularizer=l2(1e-5),
                                  name='item_embedding')(item_input_float)
        
        # 展平嵌入向量
        user_vec = Flatten(name='flatten_users')(user_embedding)
        item_vec = Flatten(name='flatten_items')(item_embedding)
        
        # FM部分：一阶线性部分 (偏置项)
        user_bias = Embedding(self.n_users, 1, embeddings_initializer='zeros',
                            embeddings_regularizer=l2(1e-6))(user_input_float)
        item_bias = Embedding(self.n_items, 1, embeddings_initializer='zeros',
                            embeddings_regularizer=l2(1e-6))(item_input_float)
        user_bias = Flatten()(user_bias)
        item_bias = Flatten()(item_bias)
        
        # 全局偏置项
        # 修复：使用Lambda层包装tf.zeros_like操作
        zeros_tensor = Lambda(lambda x: tf.zeros_like(x))(user_input_float)
        global_bias = tf.keras.layers.GlobalAveragePooling1D()(
            Embedding(1, 1, embeddings_initializer='zeros')(zeros_tensor)
        )
        
        # 嵌入向量处理 - 使用高级门控机制
        user_gated = GatingLayer(self.embedding_dim, dropout_rate=0.2)(user_vec)
        item_gated = GatingLayer(self.embedding_dim, dropout_rate=0.2)(item_vec)
        
        # 特征处理路径
        features_list = []
        
        # 如果有物品特征，加入模型
        if self.item_features is not None:
            feature_input = feature_inputs[0]
            
            # 特征处理 - 多路径架构和残差连接
            # 路径1：深度特征抽取
            feature_dense1 = Dense(self.embedding_dim, activation=None)(feature_input)
            feature_bn1 = BatchNormalization()(feature_dense1)
            feature_act1 = LeakyReLU(alpha=0.1)(feature_bn1)
            feature_drop1 = Dropout(0.25)(feature_act1)
            
            # 路径2：通过注意力机制的特征重要性权重
            feature_dense2 = Dense(self.embedding_dim, activation=None)(feature_input)
            feature_bn2 = BatchNormalization()(feature_dense2)
            feature_act2 = LeakyReLU(alpha=0.1)(feature_bn2)
            
            # 路径3：非线性特征变换
            feature_dense3 = Dense(self.embedding_dim // 2, activation='relu')(feature_input)
            feature_dense3 = Dense(self.embedding_dim, activation=None)(feature_dense3)
            feature_bn3 = BatchNormalization()(feature_dense3)
            
            # 合并多路径
            feature_combined = Add()([feature_drop1, feature_act2, feature_bn3])
            feature_gate = GatingLayer(self.embedding_dim, dropout_rate=0.2)(feature_combined)
            features_list.append(feature_gate)
            
        # FM部分：二阶交互部分 (计算用户和物品嵌入的交互)
        # 实现因子分解机的二阶交互项
        user_fm = Dense(self.embedding_dim, activation='linear', 
                       kernel_regularizer=l2(1e-5))(user_gated)
        item_fm = Dense(self.embedding_dim, activation='linear', 
                       kernel_regularizer=l2(1e-5))(item_gated)
        
        # 因子分解机交互
        fm_interaction = Multiply()([user_fm, item_fm])  # 元素级乘法
        fm_interaction = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True))(fm_interaction)
        
        # 神经协同过滤部分 (MLP)
        # 合并用户和物品表示
        if features_list:
            concat_features = Concatenate()(features_list)
            concat = Concatenate()([user_gated, item_gated, concat_features])
            inputs = [user_input_float, item_input_float] + feature_inputs
        else:
            concat = Concatenate()([user_gated, item_gated])
            inputs = [user_input_float, item_input_float]
        
        # 交叉网络 - 显式建模特征交叉
        cross_output = CrossNetwork(num_layers=4)(concat)
        
        # 多头注意力层，增强特征交互
        attention = AttentionLayer(attention_dim=256, num_heads=8, dropout_rate=0.2)(concat)
        
        # 深度网络部分 - 多层残差网络
        deep_layers = [512, 256, 128, 64]  # 更深的网络结构
        x = attention
        for units in deep_layers:
            x = self._build_residual_block(x, units, dropout_rate=0.3)
        
        # 如果启用MMOE，添加多目标混合专家层
        if self.use_mmoe:
            mmoe_outputs = MMOELayer(num_experts=4, expert_dim=64, num_tasks=2)(concat)
            main_output = mmoe_outputs[0]  # 主任务输出
            aux_output = mmoe_outputs[1]  # 辅助任务输出
            
            # 添加辅助输出层，辅助训练
            aux_pred = Dense(32, activation='relu')(aux_output)
            aux_pred = Dense(1, activation='sigmoid', name='aux_output')(aux_pred)
        
        # 合并深度网络、交叉网络和FM部分
        deep_output = Dense(1, activation=None)(x)
        cross_output = Dense(1, activation=None)(cross_output)
        
        # 将所有部分组合在一起
        combined_output = Add()([
            deep_output,          # 深度网络部分
            cross_output,         # 特征交叉网络部分
            fm_interaction,       # FM二阶交互部分
            user_bias,            # 用户偏置
            item_bias,            # 物品偏置
            global_bias           # 全局偏置
        ])
        
        # 最终输出
        main_output = Lambda(lambda x: tf.sigmoid(x), name='main_output')(combined_output)
        
        # 构建模型
        if self.use_mmoe:
            model = Model(inputs=inputs, outputs=[main_output, aux_pred])
            logger.info("创建了高级混合推荐模型，整合多目标学习、多头注意力、交叉网络和增强型FM交互")
        else:
            model = Model(inputs=inputs, outputs=main_output)
            logger.info("创建了高级混合推荐模型，整合多头注意力、交叉网络和增强型FM交互")
        
        return model
    
    def compile_model(self, learning_rate=0.001):
        """编译模型，使用先进的优化器和损失函数配置"""
        # 定义加权二元交叉熵损失，增强对正样本的关注
        def weighted_binary_crossentropy(y_true, y_pred):
            # 增加正样本权重，解决样本不平衡问题
            weights = (y_true * 4.0) + 1.0  # 正样本权重是4:1
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weighted_bce = weights * bce
            return tf.reduce_mean(weighted_bce)
        
        # 使用带warmup的学习率调度
        lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=0.1
        )
        
        # 优化器配置
        optimizer_config = {
            'learning_rate': lr_schedule,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'epsilon': 1e-7,
            'amsgrad': True,
            'clipnorm': 1.0,
            'clipvalue': 0.5
        }
        
        # 使用AdamW优化器，结合权重衰减减轻过拟合（如果可用）
        if tfa_available:
            try:
                optimizer = tfa.optimizers.AdamW(
                    weight_decay=1e-4,
                    **optimizer_config
                )
                logger.info("使用AdamW优化器，带权重衰减")
            except Exception as e:
                optimizer = Adam(**optimizer_config)
                logger.warning(f"AdamW初始化失败，使用标准Adam优化器: {e}")
        else:
            optimizer = Adam(**optimizer_config)
            logger.info("使用标准Adam优化器 (tensorflow_addons不可用)")
        
        # 添加评估指标
        metrics = [
            'binary_accuracy',
            tf.keras.metrics.AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            tf.keras.metrics.RootMeanSquaredError(name='rmse')
        ]
        
        # MMOE模型有两个输出，使用不同的损失权重
        if self.use_mmoe:
            self.model.compile(
                optimizer=optimizer,
                loss={
                    'main_output': weighted_binary_crossentropy,
                    'aux_output': 'binary_crossentropy'
                },
                loss_weights={
                    'main_output': 1.0,
                    'aux_output': 0.3  # 辅助任务权重较小
                },
                metrics={
                    'main_output': metrics,
                    'aux_output': ['binary_accuracy']
                }
            )
        else:
            self.model.compile(
                optimizer=optimizer,
                loss=weighted_binary_crossentropy,
                metrics=metrics
            )
        
        self.is_compiled = True
        logger.info("模型编译完成，使用加权二元交叉熵损失和AdamW优化器")
    
    def fit(self, user_indices, item_indices, ratings, epochs=30, batch_size=256, validation_split=0.1):
        """
        训练模型，采用先进的训练策略和数据增强
        
        参数:
            user_indices (np.ndarray): 用户索引数组
            item_indices (np.ndarray): 物品索引数组
            ratings (np.ndarray): 评分数组
            epochs (int): 训练轮数
            batch_size (int): 批次大小
            validation_split (float): 验证集比例
        
        返回:
            训练历史
        """
        if not self.is_compiled:
            self.compile_model()
        
        logger.info(f"开始训练高级混合推荐模型，数据集大小: {len(ratings)} 条记录")
        logger.info(f"训练参数: epochs={epochs}, batch_size={batch_size}")
        
        # 数据归一化和预处理
        ratings_normalized = ratings / np.max(ratings) if np.max(ratings) > 0 else ratings
        
        # 数据增强: 对高评分数据进行过采样
        if np.max(ratings_normalized) > 0.5:
            # 找出高评分样本（正样本）
            high_rating_indices = np.where(ratings_normalized > 0.8)[0]
            
            if len(high_rating_indices) > 0:
                # 过采样比例: 使高评分样本增加20%
                oversample_count = int(len(high_rating_indices) * 0.2)
                
                # 随机选择要过采样的高评分索引
                oversample_indices = np.random.choice(high_rating_indices, oversample_count, replace=True)
                
                # 添加到原始数据中
                user_indices = np.concatenate([user_indices, user_indices[oversample_indices]])
                item_indices = np.concatenate([item_indices, item_indices[oversample_indices]])
                ratings_normalized = np.concatenate([ratings_normalized, ratings_normalized[oversample_indices]])
                
                logger.info(f"数据增强: 过采样 {oversample_count} 个高评分样本，增强训练效果")
        
        # 创建先进的学习率调度器
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
            cooldown=2
        )
        
        # 早停策略，恢复最佳权重以提高泛化能力
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1
        )
        
        # 添加模型检查点回调
        model_checkpoint = ModelCheckpoint(
            filepath='temp/best_model',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        # 基本回调列表
        callbacks = [lr_scheduler, early_stopping, model_checkpoint]
        
        # 尝试添加TensorBoard回调
        try:
            # 确保日志目录存在
            if not os.path.exists('./logs'):
                os.makedirs('./logs')
                
            tensorboard = tf.keras.callbacks.TensorBoard(
                log_dir='./logs',
                histogram_freq=1,
                update_freq='epoch'
            )
            callbacks.append(tensorboard)
            logger.info("已添加TensorBoard回调")
        except Exception as e:
            logger.warning(f"无法添加TensorBoard回调: {e}")
        
        # 为MMOE模型准备辅助任务标签
        if self.use_mmoe:
            # 为辅助任务创建标签 - 这里简单地使用原始标签的变形
            # 实际应用中，这应该是一个相关但不同的任务
            aux_labels = np.clip(ratings_normalized + np.random.normal(0, 0.1, ratings_normalized.shape), 0, 1)
            
            # 准备Y标签
            train_labels = {
                'main_output': ratings_normalized,
                'aux_output': aux_labels
            }
        else:
            train_labels = ratings_normalized
        
        # 准备输入数据
        if self.item_features is not None:
            # 获取每个物品的特征
            features = np.zeros((len(item_indices), self.item_feature_dim))
            for i, item_idx in enumerate(item_indices):
                if 0 <= item_idx < len(self.item_features):
                    features[i] = self.item_features[item_idx]
            
            # 特征标准化
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0) + 1e-8
            features = (features - feature_mean) / feature_std
            
            # 训练模型
            history = self.model.fit(
                [user_indices, item_indices, features],
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
        else:
            # 不使用物品特征进行训练
            history = self.model.fit(
                [user_indices, item_indices],
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
        
        logger.info("高级混合推荐模型训练完成")
        return history
    
    def predict(self, user_id, item_ids):
        """
        预测用户对物品的评分 - 增强型预测支持批处理和单条预测
        
        参数:
            user_id: 用户ID或用户索引数组
            item_ids: 物品ID列表或物品索引数组
        
        返回:
            评分预测结果
        """
        # 检查输入是否为批处理格式 [user_indices, item_indices]
        if isinstance(user_id, list) and len(user_id) == 2 and isinstance(user_id[0], np.ndarray) and isinstance(user_id[1], np.ndarray):
            user_indices = user_id[0]
            item_indices = user_id[1]
            
            # 进行预测
            if self.item_features is not None:
                # 获取物品特征
                features = np.zeros((len(item_indices), self.item_feature_dim))
                for i, item_idx in enumerate(item_indices):
                    if 0 <= item_idx < len(self.item_features):
                        features[i] = self.item_features[item_idx]
                
                # 特征标准化
                feature_mean = np.mean(features, axis=0)
                feature_std = np.std(features, axis=0) + 1e-8
                features = (features - feature_mean) / feature_std
                
                # 预测
                if self.use_mmoe:
                    # MMOE模型返回主输出和辅助输出，我们只需要主输出
                    return self.model.predict([user_indices, item_indices, features], verbose=0)[0]
                else:
                    return self.model.predict([user_indices, item_indices, features], verbose=0)
            else:
                # 不使用物品特征
                if self.use_mmoe:
                    return self.model.predict([user_indices, item_indices], verbose=0)[0]
                else:
                    return self.model.predict([user_indices, item_indices], verbose=0)
        
        # 原始单用户预测逻辑
        if not self.user_map or not self.item_map:
            logger.error("模型未提供用户或物品映射，无法预测")
            return None
        
        # 检查用户ID是否在映射中，如果不在则使用冷启动策略
        if user_id not in self.user_map:
            logger.warning(f"用户ID {user_id} 不在训练集中，使用冷启动策略")
            # 使用平均用户嵌入作为冷启动策略
            user_embedding_layer = self.model.get_layer('user_embedding')
            user_embeddings = user_embedding_layer.get_weights()[0]
            # 使用最活跃的前10个用户的平均嵌入作为冷启动表示
            popular_users = list(self.user_map.values())[:10]
            avg_embedding = np.mean(user_embeddings[popular_users], axis=0)
            
            # 找到最接近平均嵌入的用户作为代理
            distances = np.sum((user_embeddings - avg_embedding) ** 2, axis=1)
            user_idx = np.argmin(distances)
            logger.info(f"为新用户找到最佳代理用户索引: {user_idx}")
        else:
            user_idx = self.user_map[user_id]
        
        # 检查物品是否在映射中，只保留有映射的
        valid_items = []
        valid_indices = []
        
        for item_id in item_ids:
            if item_id in self.item_map:
                valid_items.append(item_id)
                valid_indices.append(self.item_map[item_id])
        
        if not valid_items:
            logger.warning("没有有效的物品ID可以预测")
            return {}
        
        # 准备模型输入
        n_items = len(valid_indices)
        user_indices = np.full(n_items, user_idx)
        item_indices = np.array(valid_indices)
        
        # 进行预测
        if self.item_features is not None:
            # 获取物品特征
            features = np.zeros((n_items, self.item_feature_dim))
            for i, item_idx in enumerate(item_indices):
                if item_idx < len(self.item_features):
                    features[i] = self.item_features[item_idx]
            
            # 特征标准化
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0) + 1e-8
            features = (features - feature_mean) / feature_std
            
            # 预测评分
            if self.use_mmoe:
                predictions = self.model.predict([user_indices, item_indices, features], verbose=0)[0]
            else:
                predictions = self.model.predict([user_indices, item_indices, features], verbose=0)
        else:
            # 不使用物品特征
            if self.use_mmoe:
                predictions = self.model.predict([user_indices, item_indices], verbose=0)[0]
            else:
                predictions = self.model.predict([user_indices, item_indices], verbose=0)
        
        # 将预测结果整理为字典
        result = {}
        for i, item_id in enumerate(valid_items):
            result[item_id] = float(predictions[i][0])
        
        return result
    
    def predict_batch(self, inputs, verbose=0):
        """
        批量预测评分（用于内部调用，适配原始TensorFlow API）
        
        参数:
            inputs: 一个列表 [user_indices, item_indices]，包含用户和物品的索引数组
            verbose: 冗余输出级别（兼容TensorFlow API）
        
        返回:
            评分预测结果数组
        """
        user_indices, item_indices = inputs
        
        # 准备输入数据
        if self.item_features is not None:
            # 获取物品特征
            features = np.zeros((len(item_indices), self.item_feature_dim))
            for i, item_idx in enumerate(item_indices):
                if 0 <= item_idx < len(self.item_features):
                    features[i] = self.item_features[item_idx]
            
            # 特征标准化
            feature_mean = np.mean(features, axis=0)
            feature_std = np.std(features, axis=0) + 1e-8
            features = (features - feature_mean) / feature_std
            
            # 预测
            if self.use_mmoe:
                return self.model.predict([user_indices, item_indices, features], verbose=verbose)[0]
            else:
                return self.model.predict([user_indices, item_indices, features], verbose=verbose)
        else:
            # 不使用物品特征
            if self.use_mmoe:
                return self.model.predict([user_indices, item_indices], verbose=verbose)[0]
            else:
                return self.model.predict([user_indices, item_indices], verbose=verbose)
    
    def get_user_embedding(self, user_id):
        """获取用户嵌入向量"""
        if not self.user_map:
            logger.error("模型未提供用户映射，无法获取嵌入向量")
            return None
            
        if user_id not in self.user_map:
            logger.warning(f"用户ID {user_id} 不在训练集中")
            return None
            
        user_idx = self.user_map[user_id]
        user_embedding_layer = self.model.get_layer('user_embedding')
        return user_embedding_layer.get_weights()[0][user_idx]
    
    def get_item_embedding(self, item_id):
        """获取物品嵌入向量"""
        if not self.item_map:
            logger.error("模型未提供物品映射，无法获取嵌入向量")
            return None
            
        if item_id not in self.item_map:
            logger.warning(f"物品ID {item_id} 不在训练集中")
            return None
            
        item_idx = self.item_map[item_id]
        item_embedding_layer = self.model.get_layer('item_embedding')
        return item_embedding_layer.get_weights()[0][item_idx]
    
    def save(self, model_path):
        """保存模型"""
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        # 保存模型架构和权重
        self.model.save(os.path.join(model_path, 'deep_model.h5'))
        
        # 保存映射和其他信息
        metadata = {
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'item_feature_dim': self.item_feature_dim,
            'user_map': self.user_map,
            'item_map': self.item_map,
            'reverse_user_map': self.reverse_user_map,
            'reverse_item_map': self.reverse_item_map,
        }
        
        # 保存物品特征（如果有）
        if self.item_features is not None:
            np.save(os.path.join(model_path, 'item_features.npy'), self.item_features)
            
        # 保存元数据
        with open(os.path.join(model_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
            
        logger.info(f"深度学习模型已保存到 {model_path}")
        
    @classmethod
    def load(cls, model_path):
        """加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
            
        # 加载元数据
        with open(os.path.join(model_path, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            
        # 检查是否有物品特征
        item_features = None
        item_features_path = os.path.join(model_path, 'item_features.npy')
        if os.path.exists(item_features_path):
            item_features = np.load(item_features_path)
            
        # 创建模型实例
        instance = cls(
            n_users=metadata['n_users'],
            n_items=metadata['n_items'],
            embedding_dim=metadata['embedding_dim'],
            item_features=item_features
        )
        
        # 加载模型架构和权重
        model_file = os.path.join(model_path, 'deep_model.h5')
        if os.path.exists(model_file):
            # 注册自定义层，确保模型可以正确加载
            custom_objects = {
                'AttentionLayer': AttentionLayer,
                'GatingLayer': GatingLayer
            }
            instance.model = load_model(model_file, custom_objects=custom_objects)
            instance.is_compiled = True
            
        # 恢复映射和其他信息
        instance.user_map = metadata['user_map']
        instance.item_map = metadata['item_map']
        instance.reverse_user_map = metadata['reverse_user_map']
        instance.reverse_item_map = metadata['reverse_item_map']
        
        logger.info(f"深度学习模型已从 {model_path} 加载")
        return instance
    
    def k_fold_cross_validation(self, user_indices, item_indices, ratings, k=5, epochs=20, batch_size=256):
        """
        执行K折交叉验证，评估模型性能
        
        参数:
            user_indices (np.ndarray): 用户索引数组
            item_indices (np.ndarray): 物品索引数组
            ratings (np.ndarray): 评分数组
            k (int): 折数
            epochs (int): 每折训练的轮数
            batch_size (int): 批次大小
            
        返回:
            评估指标的平均值和标准差
        """
        logger.info(f"开始 {k} 折交叉验证")
        
        # 标准化评分
        ratings_normalized = ratings / np.max(ratings) if np.max(ratings) > 0 else ratings
        
        # 初始化KFold
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        
        # 评估指标
        metrics = {
            'accuracy': [],
            'auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 迭代每一折
        fold = 1
        for train_idx, test_idx in kf.split(ratings_normalized):
            logger.info(f"训练折 {fold}/{k}")
            
            # 获取训练和测试数据
            train_users = user_indices[train_idx]
            train_items = item_indices[train_idx]
            train_ratings = ratings_normalized[train_idx]
            
            test_users = user_indices[test_idx]
            test_items = item_indices[test_idx]
            test_ratings = ratings_normalized[test_idx]
            
            # 重置模型权重
            self.model = self._build_model()
            self.compile_model()
            
            # 训练模型
            if self.item_features is not None:
                # 获取特征
                train_features = np.zeros((len(train_items), self.item_feature_dim))
                for i, item_idx in enumerate(train_items):
                    if 0 <= item_idx < len(self.item_features):
                        train_features[i] = self.item_features[item_idx]
                
                test_features = np.zeros((len(test_items), self.item_feature_dim))
                for i, item_idx in enumerate(test_items):
                    if 0 <= item_idx < len(self.item_features):
                        test_features[i] = self.item_features[item_idx]
                
                # 标准化特征
                feature_mean = np.mean(train_features, axis=0)
                feature_std = np.std(train_features, axis=0) + 1e-8
                
                train_features = (train_features - feature_mean) / feature_std
                test_features = (test_features - feature_mean) / feature_std
                
                # 准备辅助任务标签
                if self.use_mmoe:
                    train_aux_labels = np.clip(train_ratings + np.random.normal(0, 0.1, train_ratings.shape), 0, 1)
                    train_labels = {
                        'main_output': train_ratings,
                        'aux_output': train_aux_labels
                    }
                else:
                    train_labels = train_ratings
                
                # 训练
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                self.model.fit(
                    [train_users, train_items, train_features],
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # 预测测试集
                if self.use_mmoe:
                    test_preds = self.model.predict([test_users, test_items, test_features])[0]
                else:
                    test_preds = self.model.predict([test_users, test_items, test_features])
            else:
                # 不使用物品特征
                if self.use_mmoe:
                    train_aux_labels = np.clip(train_ratings + np.random.normal(0, 0.1, train_ratings.shape), 0, 1)
                    train_labels = {
                        'main_output': train_ratings,
                        'aux_output': train_aux_labels
                    }
                else:
                    train_labels = train_ratings
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
                
                self.model.fit(
                    [train_users, train_items],
                    train_labels,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=0.1,
                    callbacks=[early_stopping],
                    verbose=1
                )
                
                # 预测测试集
                if self.use_mmoe:
                    test_preds = self.model.predict([test_users, test_items])[0]
                else:
                    test_preds = self.model.predict([test_users, test_items])
            
            # 计算评估指标
            test_preds = test_preds.reshape(-1)
            binary_preds = (test_preds > 0.5).astype(int)
            binary_true = (test_ratings > 0.5).astype(int)
            
            # 计算准确率
            accuracy = np.mean(binary_preds == binary_true)
            metrics['accuracy'].append(accuracy)
            
            # 计算AUC
            from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
            try:
                auc = roc_auc_score(binary_true, test_preds)
                metrics['auc'].append(auc)
            except:
                logger.warning("无法计算AUC")
            
            # 计算精确率
            precision = precision_score(binary_true, binary_preds, zero_division=0)
            metrics['precision'].append(precision)
            
            # 计算召回率
            recall = recall_score(binary_true, binary_preds, zero_division=0)
            metrics['recall'].append(recall)
            
            # 计算F1分数
            f1 = f1_score(binary_true, binary_preds, zero_division=0)
            metrics['f1'].append(f1)
            
            logger.info(f"折 {fold} - 准确率: {accuracy:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
            fold += 1
        
        # 计算平均值和标准差
        results = {}
        for metric, values in metrics.items():
            if values:  # 确保列表非空
                mean_value = np.mean(values)
                std_value = np.std(values)
                results[metric] = {
                    'mean': float(mean_value),
                    'std': float(std_value)
                }
                logger.info(f"{metric.capitalize()}: {mean_value:.4f} ± {std_value:.4f}")
        
        return results 