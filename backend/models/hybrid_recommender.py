#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合推荐系统
结合协同过滤、基于内容和上下文感知推荐的混合系统
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import logging
import scipy.sparse as sp

# 配置日志器
logger = logging.getLogger("hybrid_recommender")

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    tf_available = True
except ImportError:
    tf_available = False

class HybridRecommender:
    """混合推荐系统，结合协同过滤、基于内容和上下文感知推荐"""
    
    def __init__(self):
        # 初始化组件
        self.cf_model = None
        self.content_model = None
        self.context_model = None
        self.mood_model = None  # 新增: 情绪感知模型
        self.deep_model = None
        
        # 初始化权重
        self.cf_weight = 0.4     # 降低协同过滤权重
        self.content_weight = 0.25
        self.context_weight = 0.15
        self.mood_weight = 0.2   # 新增: 情绪权重
        self.deep_weight = 0.0  # 深度学习模型权重初始为0
        
        # 保存ID映射
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.song_to_idx = {}
        self.idx_to_song = {}
        
        # 训练标志
        self.is_trained = False
        self.has_deep_model = False
        self.has_mood_model = False  # 新增: 情绪模型标志
        
        # 存储歌曲信息
        self.songs_df = None
        
    def train(self, interactions, audio_features, songs, user_features=None, train_deep_model=False):
        """训练混合推荐系统"""
        print(f"开始训练混合推荐系统 (Deep Learning: {'启用' if train_deep_model else '禁用'})")
        
        if interactions.empty or songs.empty:
            raise ValueError("交互数据或歌曲数据为空，无法训练模型")
            
        # 保存歌曲信息用于推荐展示
        self.songs_df = songs.copy()
            
        # 确保ID列是字符串类型
        if 'user_id' in interactions.columns and interactions['user_id'].dtype != 'object':
            interactions['user_id'] = interactions['user_id'].astype(str)
        if 'song_id' in interactions.columns and interactions['song_id'].dtype != 'object':
            interactions['song_id'] = interactions['song_id'].astype(str)
        if 'song_id' in songs.columns and songs['song_id'].dtype != 'object':
            songs['song_id'] = songs['song_id'].astype(str)
        if audio_features is not None and 'song_id' in audio_features.columns and audio_features['song_id'].dtype != 'object':
            audio_features['song_id'] = audio_features['song_id'].astype(str)
            
        # 转换评分为数值类型（以防是字符串）
        if 'rating' in interactions.columns:
            interactions['rating'] = pd.to_numeric(interactions['rating'], errors='coerce')
            interactions = interactions.dropna(subset=['rating'])
            if interactions.empty:
                raise ValueError("转换评分后交互数据为空")
                
        # 创建映射
        self.user_id_map, self.song_id_map = self._create_mappings(interactions, songs)
        
        # 训练协同过滤模型
        self._train_cf_model(interactions)
        
        # 训练基于内容的模型
        self._train_content_model(audio_features, songs)
        
        # 训练上下文感知模型
        self._train_context_model(interactions, songs)
        
        # 训练情绪感知模型
        if audio_features is not None:
            self._train_mood_model(audio_features, songs)
        
        # 如果启用，训练深度学习模型
        if train_deep_model and tf:
            try:
                self._train_deep_model(interactions, songs)
                self.deep_weight = 0.2  # 启用深度学习模型并分配权重
                # 重新调整其他权重
                self.cf_weight = 0.35
                self.content_weight = 0.2
                self.context_weight = 0.1
                self.mood_weight = 0.15
            except Exception as e:
                print(f"深度学习模型训练失败: {str(e)}，将使用简化模型")
        
        # 根据用户特征定制权重
        if user_features is not None:
            self._customize_weights(user_features)
            
        self.is_trained = True
        print("混合推荐系统训练完成")
        
    def _create_mappings(self, interactions, songs):
        """创建用户ID和歌曲ID的映射"""
        # 确保song_id在两个数据框中都是字符串类型
        if 'song_id' in songs.columns and songs['song_id'].dtype != 'object':
            songs['song_id'] = songs['song_id'].astype(str)
            
        if 'song_id' in interactions.columns and interactions['song_id'].dtype != 'object':
            interactions['song_id'] = interactions['song_id'].astype(str)
            
        if 'user_id' in interactions.columns and interactions['user_id'].dtype != 'object':
            interactions['user_id'] = interactions['user_id'].astype(str)
            
        # 获取唯一的用户ID和歌曲ID
        unique_user_ids = interactions['user_id'].unique()
        unique_song_ids = pd.concat([songs['song_id'], interactions['song_id']]).unique()
        
        # 创建ID到索引的映射
        user_id_map = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        song_id_map = {song_id: idx for idx, song_id in enumerate(unique_song_ids)}
        
        # 创建索引到ID的映射
        self.id_to_user = {idx: user_id for user_id, idx in user_id_map.items()}
        self.id_to_song = {idx: song_id for song_id, idx in song_id_map.items()}
        
        # 添加对象级别的映射
        self.user_to_idx = user_id_map
        self.idx_to_user = self.id_to_user  # 确保一致性
        self.song_to_idx = song_id_map
        self.idx_to_song = self.id_to_song  # 确保一致性
        
        # 打印映射中的前几个用户ID作为调试信息
        user_samples = list(user_id_map.keys())[:5] if len(user_id_map) > 5 else list(user_id_map.keys())
        logger.info(f"用户ID映射示例: {user_samples}")
        
        print(f"创建ID映射完成. 用户数: {len(user_id_map)}, 歌曲数: {len(song_id_map)}")
        
        return user_id_map, song_id_map
        
    def _train_cf_model(self, interactions):
        """训练协同过滤模型 - 优化版本"""
        print("训练协同过滤模型(优化版)...")
        
        # 创建用户-项目矩阵
        user_indices = interactions['user_id'].map(self.user_id_map).values
        item_indices = interactions['song_id'].map(self.song_id_map).values
        
        # 如果评分列不存在，默认为1
        if 'rating' in interactions.columns:
            # 应用sigmoid函数压缩评分范围，提高区分度
            ratings = interactions['rating'].values
            # 归一化评分到[0, 1]，提高模型敏感性
            max_rating = np.max(ratings)
            if max_rating > 0:
                ratings = ratings / max_rating
        else:
            ratings = np.ones(len(interactions))
        
        # 获取交互数据中的唯一用户和歌曲
        unique_users = np.unique(user_indices)
        unique_items = np.unique(item_indices)
        
        # 创建映射到压缩索引的字典
        compact_user_map = {original: compact for compact, original in enumerate(unique_users)}
        compact_item_map = {original: compact for compact, original in enumerate(unique_items)}
        
        # 创建压缩的索引
        compact_user_indices = np.array([compact_user_map[ui] for ui in user_indices])
        compact_item_indices = np.array([compact_item_map[ii] for ii in item_indices])
        
        # 添加基于评分频率的权重调整
        item_counts = np.bincount(compact_item_indices)
        item_weights = np.log1p(item_counts) / np.max(np.log1p(item_counts))
        weighted_ratings = ratings * np.array([item_weights[i] for i in compact_item_indices])
        
        # 创建压缩的稀疏矩阵
        compact_cf_matrix = sp.csr_matrix((weighted_ratings, (compact_user_indices, compact_item_indices)), 
                                        shape=(len(unique_users), len(unique_items)))
        
        # 应用隐式反馈增强: 填充稀疏矩阵
        # 对于零评分的项目，添加一个小的默认值
        alpha = 0.1  # 隐式反馈权重
        rows, cols = compact_cf_matrix.nonzero()
        for i in range(len(unique_users)):
            if i not in rows:
                # 用户没有任何评分，添加全局流行度评分
                for j in range(len(unique_items)):
                    compact_cf_matrix[i, j] = alpha * item_weights[j]
        
        # 保存完整矩阵的维度信息，但实际使用压缩矩阵
        self.full_matrix_shape = (len(self.user_id_map), len(self.song_id_map))
        self.cf_matrix = compact_cf_matrix
        self.compact_user_map = compact_user_map
        self.compact_item_map = compact_item_map
        self.reverse_compact_user_map = {v: k for k, v in compact_user_map.items()}
        self.reverse_compact_item_map = {v: k for k, v in compact_item_map.items()}
        
        # 添加完整的物品ID列表，用于处理维度匹配问题
        self.all_item_ids = set(range(len(self.song_id_map)))
        self.compact_items_set = set(unique_items)
        
        # L2归一化CF矩阵的每一行，提高相似度计算准确性
        for i in range(len(unique_users)):
            row_norm = np.linalg.norm(compact_cf_matrix[i].toarray())
            if row_norm > 0:
                compact_cf_matrix[i] = compact_cf_matrix[i] / row_norm
        
        print(f"协同过滤矩阵构建完成，压缩形状: {self.cf_matrix.shape}，全矩阵形状: {self.full_matrix_shape}")
        print(f"节省内存: {100*(1 - (len(unique_users)*len(unique_items))/(len(self.user_id_map)*len(self.song_id_map))):.2f}%")
    
    def _train_content_model(self, audio_features, songs):
        """训练基于内容的模型 - 优化版本"""
        print("训练基于内容的模型(优化版)...")
        
        # 将歌曲特征和音频特征合并
        content_features = songs.merge(audio_features, on='song_id', how='inner')
        
        # 扩展特征集，包含更多音频特征
        all_numeric_cols = [col for col in content_features.columns 
                          if content_features[col].dtype in [np.float64, np.int64] 
                          and col not in ['song_id', 'year', 'artist_id', 'album_id']]
        
        # 基础音频特征
        base_features = ['duration', 'tempo', 'loudness', 'key', 'mode', 'energy_ratio', 'tempo_norm']
        
        # 加入情感特征
        emotion_features = [col for col in all_numeric_cols if 'valence' in col or 'arousal' in col or 'mood' in col]
        
        # 加入声学特征
        acoustic_features = [col for col in all_numeric_cols if 'acoustic' in col or 'timbre' in col]
        
        # 所有可能的特征
        potential_features = base_features + emotion_features + acoustic_features
        
        # 确保所有需要的列都存在
        valid_cols = [col for col in potential_features if col in content_features.columns]
        
        if not valid_cols:
            print("警告: 没有找到有效的音频特征列，使用随机特征")
            # 创建随机特征，但使用更合理的分布
            content_features['random_feature1'] = np.random.normal(0.5, 0.15, len(content_features))
            content_features['random_feature2'] = np.random.normal(0.5, 0.15, len(content_features))
            valid_cols = ['random_feature1', 'random_feature2']
        
        # 提取数值特征
        numeric_features = content_features[valid_cols].values
        
        # 处理缺失值
        numeric_features = np.nan_to_num(numeric_features, nan=0.0)
        
        # 应用非线性变换增强特征表达
        # 1. 对数变换处理偏态分布特征
        skewed_features = ['duration', 'loudness']
        for col in [c for c in skewed_features if c in valid_cols]:
            col_idx = valid_cols.index(col)
            # 确保值为正
            min_val = np.min(numeric_features[:, col_idx])
            if min_val < 0:
                numeric_features[:, col_idx] = numeric_features[:, col_idx] - min_val + 1e-6
            # 应用对数变换
            numeric_features[:, col_idx] = np.log1p(numeric_features[:, col_idx])
        
        # 2. 添加交叉特征
        if 'energy_ratio' in valid_cols and 'tempo' in valid_cols:
            energy_idx = valid_cols.index('energy_ratio')
            tempo_idx = valid_cols.index('tempo')
            # 创建能量与速度的交叉特征
            energy_tempo = numeric_features[:, energy_idx] * numeric_features[:, tempo_idx]
            # 添加到特征矩阵
            numeric_features = np.column_stack((numeric_features, energy_tempo))
            valid_cols.append('energy_tempo_cross')
        
        # 标准化特征 - 使用稳健的标准化
        mean = np.median(numeric_features, axis=0)  # 使用中位数代替均值，抵抗异常值
        # 使用四分位距代替标准差，抵抗异常值
        q75 = np.percentile(numeric_features, 75, axis=0)
        q25 = np.percentile(numeric_features, 25, axis=0)
        std = (q75 - q25) / 1.349  # 标准差的稳健估计
        std = np.where(std < 1e-8, 1.0, std)  # 避免除以零
        
        normalized_features = (numeric_features - mean) / std
        
        # 特征权重 - 对不同特征赋予权重
        feature_weights = np.ones(normalized_features.shape[1])
        # 增加情感和情绪特征的权重
        for i, col in enumerate(valid_cols):
            if 'valence' in col or 'energy' in col or 'mood' in col:
                feature_weights[i] = 1.5
            elif 'tempo' in col or 'acoustic' in col:
                feature_weights[i] = 1.2
        
        # 应用权重
        weighted_features = normalized_features * feature_weights
        
        # 创建歌曲到特征的映射
        song_features = {}
        for i, row in content_features.iterrows():
            song_id = row['song_id']
            song_idx = self.song_id_map.get(song_id)
            if song_idx is not None:
                song_features[song_idx] = weighted_features[i]
        
        # 生成默认特征值（全零向量）
        default_features = np.zeros(len(valid_cols) + ('energy_tempo_cross' in valid_cols))
        
        # 确保所有歌曲都有特征
        for song_idx in range(len(self.song_id_map)):
            if song_idx not in song_features:
                song_features[song_idx] = default_features
        
        # PCA降维，保留95%方差
        try:
            from sklearn.decomposition import PCA
            if weighted_features.shape[0] > 1 and weighted_features.shape[1] > 1:
                # 收集所有特征向量
                all_features = np.array(list(song_features.values()))
                # 计算合适的组件数量
                n_components = min(all_features.shape[1], all_features.shape[0] // 10)
                if n_components > 1:
                    pca = PCA(n_components=n_components)
                    transformed_features = pca.fit_transform(all_features)
                    # 更新特征映射
                    for i, song_idx in enumerate(song_features.keys()):
                        song_features[song_idx] = transformed_features[i]
                    print(f"应用PCA降维，将特征从{all_features.shape[1]}维减少到{transformed_features.shape[1]}维")
                    valid_cols = [f"pc_{i}" for i in range(transformed_features.shape[1])]
        except (ImportError, Exception) as e:
            print(f"PCA降维失败: {e}，使用原始特征")
        
        self.content_model = {
            'features': song_features,
            'mean': mean,
            'std': std,
            'columns': valid_cols,
            'weights': feature_weights.tolist()
        }
        
        # 打印完整统计信息
        print(f"基于内容的模型训练完成，使用特征: {valid_cols}")
        print(f"内容模型包含 {len(song_features)} 首歌曲的特征 (应有 {len(self.song_id_map)} 首)")
        
        # 验证内容模型包含所有歌曲
        if len(song_features) != len(self.song_id_map):
            logger.warning(f"内容特征数量 ({len(song_features)}) 与歌曲映射大小 ({len(self.song_id_map)}) 不匹配")
    
    def _train_context_model(self, interactions, songs):
        """训练上下文感知模型（例如流行度）"""
        print("训练上下文感知模型...")
        
        # 计算歌曲流行度（基于互动次数或评分）
        song_popularity = interactions.groupby('song_id')['rating'].agg(['count', 'mean']).reset_index()
        song_popularity.columns = ['song_id', 'play_count', 'avg_rating']
        
        # 归一化流行度分数
        max_count = song_popularity['play_count'].max()
        min_popularity = 0.001  # 设置最小流行度，避免完全为0
        song_popularity['popularity'] = song_popularity['play_count'] / max_count
        
        # 创建歌曲索引到流行度的映射
        self.context_model = {}
        
        # 确保所有歌曲都有流行度分数，遍历所有可能的歌曲索引
        for song_idx in range(len(self.song_id_map)):
            song_id = self.id_to_song.get(song_idx)
            if song_id is not None:
                # 查找该歌曲的流行度
                song_data = song_popularity[song_popularity['song_id'] == song_id]
                if not song_data.empty:
                    # 对于有交互的歌曲，使用计算出的流行度
                    self.context_model[song_idx] = song_data['popularity'].iloc[0]
                else:
                    # 对于没有交互的歌曲，分配最低流行度
                    self.context_model[song_idx] = min_popularity
        
        # 确认上下文模型包含所有歌曲
        missing_songs = set(range(len(self.song_id_map))) - set(self.context_model.keys())
        if missing_songs:
            for song_idx in missing_songs:
                self.context_model[song_idx] = min_popularity
            
        print(f"上下文感知模型训练完成，共 {len(self.context_model)} 首歌曲 (应有 {len(self.song_id_map)} 首)")
        
        # 验证上下文模型大小
        if len(self.context_model) != len(self.song_id_map):
            logger.warning(f"上下文模型大小 ({len(self.context_model)}) 与歌曲映射大小 ({len(self.song_id_map)}) 不匹配")
        
        return

    def _train_deep_model(self, interactions, songs, max_samples=100000):
        """训练深度学习模型"""
        print("训练深度学习模型...")
        
        if not tf:
            print("未安装TensorFlow，无法训练深度学习模型")
            return
        
        # 准备训练数据
        train_data = interactions.copy()
        
        # 如果数据太多，随机抽样
        if len(train_data) > max_samples:
            train_data = train_data.sample(max_samples, random_state=42)
        
        # 转换用户和物品ID为索引
        user_indices = [self.user_to_idx.get(uid) for uid in train_data['user_id']]
        item_indices = [self.song_to_idx.get(sid) for sid in train_data['song_id']]
        
        # 移除无效的索引
        valid_indices = [(i, ui, ii) for i, (ui, ii) in enumerate(zip(user_indices, item_indices)) 
                        if ui is not None and ii is not None]
        
        if not valid_indices:
            print("没有有效的训练数据，跳过深度学习模型训练")
            return
        
        indices, user_indices, item_indices = zip(*valid_indices)
        ratings = train_data.iloc[list(indices)]['rating'].values
        
        # 归一化评分到 [0, 1] 区间
        ratings = (ratings - ratings.min()) / (ratings.max() - ratings.min() + 1e-8)
        
        # 模型参数
        n_users = len(self.user_to_idx)
        n_items = len(self.song_to_idx)
        n_factors = 50  # 嵌入维度
        
        # 构建模型
        user_input = Input(shape=(1,), name='user_input')
        item_input = Input(shape=(1,), name='item_input')
        
        # 嵌入层
        user_embedding = Embedding(n_users, n_factors, embeddings_regularizer=l2(1e-6), name='user_embedding')(user_input)
        item_embedding = Embedding(n_items, n_factors, embeddings_regularizer=l2(1e-6), name='item_embedding')(item_input)
        
        # 展平嵌入
        user_flat = Flatten()(user_embedding)
        item_flat = Flatten()(item_embedding)
        
        # 连接嵌入
        concat = Concatenate()([user_flat, item_flat])
        
        # MLP层
        dense1 = Dense(128, activation='relu')(concat)
        dropout1 = Dropout(0.2)(dense1)
        dense2 = Dense(64, activation='relu')(dropout1)
        dropout2 = Dropout(0.2)(dense2)
        dense3 = Dense(32, activation='relu')(dropout2)
        
        # 输出层
        output = Dense(1, activation='sigmoid')(dense3)
        
        # 编译模型
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # 训练模型
        print(f"开始训练深度学习模型，样本数: {len(user_indices)}")
        history = model.fit(
            [np.array(user_indices), np.array(item_indices)],
            ratings,
            epochs=10,
            batch_size=256,
            verbose=2,
            validation_split=0.1
        )
        
        # 保存模型
        self.deep_model = model
        self.has_deep_model = True
        print("深度学习模型训练完成")
        
    def _train_mood_model(self, audio_features, songs):
        """训练基于情绪的模型
        
        情绪模型将音乐特征映射到情绪类别，使系统可以基于用户当前情绪推荐相应的音乐
        """
        print("训练情绪感知模型...")
        
        # 定义情绪类别及其对应的音乐特征规则
        mood_categories = {
            'happy': {'valence': (0.6, 1.0), 'energy': (0.6, 1.0), 'tempo': (100, 200)},
            'sad': {'valence': (0.0, 0.4), 'energy': (0.0, 0.4), 'tempo': (50, 90)},
            'calm': {'valence': (0.3, 0.7), 'energy': (0.0, 0.4), 'tempo': (60, 100)},
            'energetic': {'valence': (0.5, 1.0), 'energy': (0.7, 1.0), 'tempo': (110, 200)},
            'angry': {'valence': (0.0, 0.5), 'energy': (0.7, 1.0), 'tempo': (100, 200)},
            'nostalgic': {'valence': (0.4, 0.7), 'energy': (0.3, 0.6), 'tempo': (70, 120)},
            'romantic': {'valence': (0.5, 0.8), 'energy': (0.3, 0.6), 'tempo': (60, 110)},
            'excited': {'valence': (0.7, 1.0), 'energy': (0.7, 1.0), 'tempo': (120, 200)}
        }
        
        # 将歌曲特征和音频特征合并
        mood_features = songs.merge(audio_features, on='song_id', how='inner')
        
        # 检查必要的特征列是否存在
        required_features = ['energy_ratio', 'tempo', 'valence']
        found_features = [col for col in required_features if col in mood_features.columns]
        
        missing_features = set(required_features) - set(found_features)
        if missing_features:
            print(f"警告: 缺少情绪分析所需特征: {missing_features}")
            # 添加默认列
            for feature in missing_features:
                if feature == 'valence' and 'valence' not in mood_features.columns:
                    # 从loudness和energy推测valence
                    if 'loudness' in mood_features.columns and 'energy_ratio' in mood_features.columns:
                        mood_features['valence'] = (mood_features['energy_ratio'] - 
                                                   (mood_features['loudness'].abs() / 60)) / 2
                    else:
                        mood_features['valence'] = np.random.uniform(0.4, 0.6, len(mood_features))
                
                elif feature == 'energy_ratio' and 'energy_ratio' not in mood_features.columns:
                    if 'loudness' in mood_features.columns:
                        # 从loudness估计energy
                        mood_features['energy_ratio'] = 1 - (mood_features['loudness'].abs() / 60)
                    else:
                        mood_features['energy_ratio'] = np.random.uniform(0.3, 0.7, len(mood_features))
                
                elif feature == 'tempo' and 'tempo' not in mood_features.columns:
                    mood_features['tempo'] = np.random.uniform(80, 130, len(mood_features))
        
        # 标准化特征范围
        if 'valence' in mood_features.columns:
            mood_features['valence'] = mood_features['valence'].clip(0, 1)
        if 'energy_ratio' in mood_features.columns:
            mood_features['energy_ratio'] = mood_features['energy_ratio'].clip(0, 1)
        if 'tempo' in mood_features.columns:
            mood_features['tempo'] = mood_features['tempo'].clip(40, 220)
        
        # 为每首歌曲计算情绪相似度得分
        mood_scores = {}
        for i, row in mood_features.iterrows():
            song_id = row['song_id']
            song_idx = self.song_id_map.get(song_id)
            
            if song_idx is not None:
                # 获取特征值，如果缺失则使用默认值
                valence = row.get('valence', 0.5)
                energy = row.get('energy_ratio', 0.5)
                tempo = row.get('tempo', 100)
                
                # 计算每个情绪类别的匹配度
                mood_match = {}
                for mood, features in mood_categories.items():
                    val_match = max(0, 1 - abs(valence - (features['valence'][0] + features['valence'][1])/2) * 2)
                    energy_match = max(0, 1 - abs(energy - (features['energy'][0] + features['energy'][1])/2) * 2)
                    tempo_factor = min(1.0, max(0, (tempo - features['tempo'][0]) / (features['tempo'][1] - features['tempo'][0])))
                    
                    # 加权平均匹配度
                    mood_match[mood] = 0.4 * val_match + 0.4 * energy_match + 0.2 * tempo_factor
                
                mood_scores[song_idx] = mood_match
        
        # 为缺失的歌曲生成默认情绪评分
        default_mood_scores = {mood: 0.5 for mood in mood_categories.keys()}
        for song_idx in range(len(self.song_id_map)):
            if song_idx not in mood_scores:
                mood_scores[song_idx] = default_mood_scores.copy()
        
        # 保存情绪模型
        self.mood_model = {
            'scores': mood_scores,
            'categories': list(mood_categories.keys())
        }
        
        self.has_mood_model = True
        print(f"情绪感知模型训练完成，包含 {len(mood_scores)} 首歌曲")
        
        # 输出部分歌曲的情绪分布示例
        sample_size = min(5, len(mood_scores))
        sample_indices = list(mood_scores.keys())[:sample_size]
        for idx in sample_indices:
            song_id = self.id_to_song.get(idx)
            print(f"歌曲 {idx} (ID: {song_id}): {mood_scores[idx]}")
        
        return

    def _customize_weights(self, user_features):
        """根据用户活跃度和偏好定制权重 - 优化版"""
        print("根据用户特征定制权重(优化版)...")
        
        try:
            # 确保用户特征包含必要的列
            required_cols = ['user_id', 'total_plays']
            if not all(col in user_features.columns for col in required_cols):
                print(f"用户特征中缺少必要的列 {required_cols}，使用默认权重")
                return
                
            # 计算用户活跃度阈值 - 使用分位数分析
            if len(user_features) > 1:
                q25_plays = user_features['total_plays'].quantile(0.25)
                median_plays = user_features['total_plays'].quantile(0.5)
                q75_plays = user_features['total_plays'].quantile(0.75)
                q90_plays = user_features['total_plays'].quantile(0.9)
            else:
                # 如果只有一个用户，设置默认阈值
                q25_plays = 5
                median_plays = 10
                q75_plays = 20
                q90_plays = 50
            
            # 为每个用户创建定制权重
            self.user_weights = {}
            
            # 获取可用的模型
            available_models = []
            if hasattr(self, 'cf_matrix') and self.cf_matrix is not None:
                available_models.append('cf')
            if hasattr(self, 'content_model') and self.content_model is not None:
                available_models.append('content')
            if hasattr(self, 'context_model') and self.context_model:
                available_models.append('context')
            if hasattr(self, 'mood_model') and self.mood_model:
                available_models.append('mood')
            if self.has_deep_model:
                available_models.append('deep')
            
            for _, user in user_features.iterrows():
                user_id = user['user_id']
                total_plays = user['total_plays']
                
                # 基于用户行为模式的自适应权重
                if total_plays < q25_plays:
                    # 新用户/不活跃用户：更多依赖内容和上下文
                    weights = {
                        'cf': 0.2,
                        'content': 0.35,
                        'context': 0.25,
                        'mood': 0.2,
                        'deep': 0.0
                    }
                elif total_plays < median_plays:
                    # 轻度用户：平衡但偏向内容
                    weights = {
                        'cf': 0.3,
                        'content': 0.3,
                        'context': 0.2,
                        'mood': 0.2,
                        'deep': 0.0
                    }
                elif total_plays < q75_plays:
                    # 中等活跃用户：平衡权重
                    weights = {
                        'cf': 0.4,
                        'content': 0.25,
                        'context': 0.15,
                        'mood': 0.2,
                        'deep': 0.0
                    }
                elif total_plays < q90_plays:
                    # 活跃用户：偏向协同过滤
                    weights = {
                        'cf': 0.5,
                        'content': 0.2,
                        'context': 0.1,
                        'mood': 0.2,
                        'deep': 0.0
                    }
                else:
                    # 高度活跃用户：更依赖协同过滤
                    weights = {
                        'cf': 0.6,
                        'content': 0.15,
                        'context': 0.05,
                        'mood': 0.2,
                        'deep': 0.0
                    }
                
                # 检查用户特定偏好
                if 'prefers_novelty' in user_features.columns and not pd.isna(user.get('prefers_novelty')):
                    # 如果用户喜欢新颖性，增加内容权重
                    if user['prefers_novelty'] > 0.5:
                        weights['content'] = min(0.7, weights['content'] * 1.5)
                        weights['context'] = max(0.05, weights['context'] * 0.8)
                        weights['cf'] = max(0.1, weights['cf'] * 0.8)
                
                # 如果有其他用户偏好，进一步调整
                if 'prefers_popular' in user_features.columns and not pd.isna(user.get('prefers_popular')):
                    # 如果用户偏好流行歌曲，增加上下文权重
                    if user['prefers_popular'] > 0.5:
                        weights['context'] = min(0.5, weights['context'] * 1.5)
                        weights['content'] = max(0.1, weights['content'] * 0.8)

                # 根据情绪偏好调整
                if 'mood_sensitivity' in user_features.columns and not pd.isna(user.get('mood_sensitivity')):
                    # 如果用户对情绪敏感，增加情绪权重
                    mood_sensitivity = user['mood_sensitivity']
                    weights['mood'] = min(0.4, weights['mood'] * (1 + mood_sensitivity))
                
                # 如果有深度学习模型，调整权重
                if self.has_deep_model:
                    # 将部分权重分配给深度学习模型
                    deep_weight = 0.25
                    scale_factor = 1.0 - deep_weight
                    for key in weights:
                        if key != 'deep':
                            weights[key] = weights[key] * scale_factor
                    weights['deep'] = deep_weight
                
                # 移除不可用的模型并重新归一化
                available_weights = {k: weights[k] for k in available_models}
                total_weight = sum(available_weights.values())
                if total_weight > 0:
                    for k in available_weights:
                        available_weights[k] /= total_weight
                
                self.user_weights[user_id] = available_weights
            
            print(f"为 {len(self.user_weights)} 个用户定制了权重")
            # 输出样例权重
            if self.user_weights:
                sample_user = next(iter(self.user_weights))
                print(f"样例用户 {sample_user} 权重: {self.user_weights[sample_user]}")
        except Exception as e:
            print(f"定制权重时出错: {str(e)}，使用默认权重")
                
    def recommend(self, user_id, n=10, exclude_items=None, include_cold_start=True, current_mood=None, diversity_level=0.5):
        """
        生成用户推荐
        
        参数:
            user_id: 用户ID
            n: 推荐数量
            exclude_items: 排除列表
            include_cold_start: 是否对新用户使用冷启动推荐
            current_mood: 当前情绪 ('happy', 'sad', 'energetic', 'calm', 'focused' 等)
            diversity_level: 多样性水平 (0-1)，0为最相似，1为最多样
            
        返回:
            推荐列表
        """
        if not self.is_trained:
            logger.error("模型未经训练，无法生成推荐")
            return []
            
        # 确保用户ID为字符串类型
        user_id = str(user_id)
            
        # 对于新用户使用冷启动推荐
        if user_id not in self.user_to_idx:
            if include_cold_start:
                logger.info(f"用户 {user_id} 不在用户映射中，使用冷启动推荐")
                return self._cold_start_recommend(user_id, n, exclude_items, current_mood)
            else:
                logger.warning(f"用户 {user_id} 不在训练集中，且禁用了冷启动推荐")
                return []
                
        # 获取内部用户ID
        user_internal_id = self.user_to_idx[user_id]
        
        if exclude_items is None:
            exclude_items = []
        
        # 获取用户权重
        weights = self._get_user_weights(user_id)
        
        # 根据当前情绪调整权重
        if current_mood is not None and 'mood' in weights:
            # 如果提供了当前情绪，增强情绪模型的权重
            mood_boost = 0.2
            scale_factor = 1.0 - mood_boost
            adjusted_weights = {}
            for model, weight in weights.items():
                if model == 'mood':
                    adjusted_weights[model] = weight + mood_boost
                else:
                    adjusted_weights[model] = weight * scale_factor
            weights = adjusted_weights
        
        print(f"用户 {user_id} 的模型权重: {weights}")
        
        # 计算每个模型需要产生的推荐数量 (根据权重)
        total_weight = sum(weights.values())
        model_counts = {}
        remaining = n
        
        # 根据多样性级别调整权重分布
        # 高多样性意味着更均匀的分布，低多样性意味着更集中在高权重模型
        for model, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
            if diversity_level > 0.8:
                # 高多样性：更均匀分布
                model_counts[model] = max(1, int(n / len(weights)))
            elif diversity_level < 0.2:
                # 低多样性：集中在高权重模型
                adjusted_weight = weight ** 0.5  # 降低权重差异
                model_counts[model] = max(1, int(n * (adjusted_weight / total_weight)))
            else:
                # 中等多样性：按权重比例分配
                model_counts[model] = max(1, int(n * (weight / total_weight)))
            
            remaining -= model_counts[model]
        
        # 分配剩余的推荐数量给权重最高的模型
        if remaining > 0:
            top_model = max(weights.items(), key=lambda x: x[1])[0]
            model_counts[top_model] += remaining
        
        # 获取各个模型的推荐
        model_recommendations = {}
        
        # 协同过滤推荐
        if 'cf' in weights and weights['cf'] > 0 and model_counts['cf'] > 0:
            if hasattr(self, 'cf_matrix') and self.cf_matrix is not None and user_internal_id is not None:
                try:
                    cf_recs = self._cf_recommend(user_internal_id, model_counts['cf'] * 3, exclude_items)
                    model_recommendations['cf'] = cf_recs
                except Exception as e:
                    print(f"协同过滤推荐错误: {str(e)}")
        
        # 内容模型推荐
        if 'content' in weights and weights['content'] > 0 and model_counts['content'] > 0:
            if hasattr(self, 'content_model') and self.content_model is not None and user_internal_id is not None:
                try:
                    content_recs = self._content_recommend(user_internal_id, model_counts['content'] * 3, exclude_items)
                    model_recommendations['content'] = content_recs
                except Exception as e:
                    print(f"内容推荐错误: {str(e)}")
        
        # 上下文模型推荐
        if 'context' in weights and weights['context'] > 0 and model_counts['context'] > 0:
            if hasattr(self, 'context_model') and self.context_model and user_internal_id is not None:
                try:
                    context_recs = self._context_recommend(user_internal_id, model_counts['context'] * 3, exclude_items)
                    model_recommendations['context'] = context_recs
                except Exception as e:
                    print(f"上下文推荐错误: {str(e)}")
        
        # 情绪模型推荐
        if 'mood' in weights and weights['mood'] > 0 and model_counts['mood'] > 0:
            if hasattr(self, 'mood_model') and self.mood_model and user_internal_id is not None:
                try:
                    mood_recs = self._mood_recommend(user_internal_id, model_counts['mood'] * 3, exclude_items, current_mood)
                    model_recommendations['mood'] = mood_recs
                except Exception as e:
                    print(f"情绪推荐错误: {str(e)}")
        
        # 深度学习模型推荐
        if 'deep' in weights and weights['deep'] > 0 and model_counts['deep'] > 0:
            if self.has_deep_model and user_internal_id is not None:
                try:
                    deep_recs = self._deep_recommend(user_internal_id, model_counts['deep'] * 3, exclude_items)
                    model_recommendations['deep'] = deep_recs
                except Exception as e:
                    print(f"深度学习推荐错误: {str(e)}")
        
        # 组合推荐结果
        final_recs = []
        
        # 创建一个已添加的歌曲集合以避免重复
        added_items = set(exclude_items)
        
        # 第一轮：按照每个模型的分配数量获取推荐
        for model, count in model_counts.items():
            if model in model_recommendations:
                model_recs = model_recommendations[model]
                for item_id, score in model_recs:
                    if item_id not in added_items and len(final_recs) < n:
                        final_recs.append((item_id, score, model))
                        added_items.add(item_id)
                    if len(final_recs) >= n:
                        break
        
        # 第二轮：如果还需要更多推荐，使用权重最高的模型的剩余推荐
        if len(final_recs) < n:
            # 按权重排序模型
            sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            for model, _ in sorted_models:
                if model in model_recommendations:
                    model_recs = model_recommendations[model]
                    for item_id, score in model_recs:
                        if item_id not in added_items and len(final_recs) < n:
                            final_recs.append((item_id, score, model))
                            added_items.add(item_id)
                        if len(final_recs) >= n:
                            break
                if len(final_recs) >= n:
                    break
        
        # 第三轮：如果仍然不足，使用冷启动推荐
        if len(final_recs) < n and include_cold_start:
            remaining = n - len(final_recs)
            cold_start_recs = self._cold_start_recommend(user_id, remaining, list(added_items), current_mood)
            for item_id, score in cold_start_recs:
                if item_id not in added_items:
                    final_recs.append((item_id, score, 'cold_start'))
                    added_items.add(item_id)
        
        # 多样性增强：基于多样性级别可能重新排序推荐
        if diversity_level > 0.5 and len(final_recs) > 3:
            # 高多样性策略：在最终结果中轮流包含各个模型的推荐
            model_specific_recs = {}
            for item_id, score, model in final_recs:
                if model not in model_specific_recs:
                    model_specific_recs[model] = []
                model_specific_recs[model].append((item_id, score))
            
            # 重新排序以增加多样性
            diversified_recs = []
            remaining_models = list(model_specific_recs.keys())
            
            while remaining_models and len(diversified_recs) < n:
                for model in list(remaining_models):
                    if not model_specific_recs[model]:
                        remaining_models.remove(model)
                        continue
                    
                    item_id, score = model_specific_recs[model].pop(0)
                    diversified_recs.append((item_id, score, model))
                    
                    if len(diversified_recs) >= n:
                        break
            
            final_recs = diversified_recs
        
        # 最终处理：移除模型标签，只保留歌曲ID和分数
        result = [(item_id, score) for item_id, score, _ in final_recs]
        
        print(f"为用户 {user_id} 生成了 {len(result)} 条推荐")
        return result

    def _mood_recommend(self, user_internal_id, n=10, exclude_items=None, current_mood=None):
        """基于情绪的推荐方法
        
        Args:
            user_internal_id: 内部用户ID
            n: 推荐数量
            exclude_items: 排除的歌曲ID列表
            current_mood: 用户当前情绪 (可选)
        
        Returns:
            推荐歌曲列表 [(歌曲ID, 分数), ...]
        """
        if exclude_items is None:
            exclude_items = []
        
        if not hasattr(self, 'mood_model') or not self.mood_model:
            return []
        
        try:
            # 获取用户历史听歌情绪分布
            user_mood_profile = self.mood_model.get('user_profiles', {}).get(user_internal_id, {})
            
            # 使用当前情绪或历史情绪分布
            target_mood = current_mood
            if target_mood is None and user_mood_profile:
                # 如果没有提供当前情绪，使用用户历史情绪分布中最主要的情绪
                target_mood = max(user_mood_profile.items(), key=lambda x: x[1])[0]
            
            if not target_mood:
                return []
            
            # 获取符合目标情绪的歌曲
            mood_songs = self.mood_model.get('mood_songs', {}).get(target_mood, [])
            if not mood_songs:
                return []
            
            # 排除已经排除的歌曲
            filtered_songs = [(song_id, score) for song_id, score in mood_songs if song_id not in exclude_items]
            
            # 按分数排序并返回前n个
            return sorted(filtered_songs, key=lambda x: x[1], reverse=True)[:n]
        except Exception as e:
            print(f"情绪推荐出错: {str(e)}")
            return []
    
    def update_weights(self, user_id, cf_weight=None, content_weight=None, context_weight=None, deep_weight=None):
        """更新用户的权重"""
        if not hasattr(self, 'user_weights'):
            self.user_weights = {}
            
        if user_id not in self.user_weights:
            self.user_weights[user_id] = {
                'cf': self.cf_weight,
                'content': self.content_weight,
                'context': self.context_weight,
                'deep': self.deep_weight
            }
            
        weights = self.user_weights[user_id]
        
        if cf_weight is not None:
            weights['cf'] = cf_weight
        if content_weight is not None:
            weights['content'] = content_weight
        if context_weight is not None:
            weights['context'] = context_weight
        if deep_weight is not None:
            weights['deep'] = deep_weight
            
        # 确保权重总和为1
        total = sum(weights.values())
        if total > 0:
            for key in weights:
                weights[key] /= total
                
        print(f"用户 {user_id} 的权重已更新: {weights}")
        
    def save_model(self, path):
        """保存模型到磁盘"""
        logger.info(f"保存混合推荐模型到: {path}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 收集需要保存的数据
        model_data = {
            'cf_model': self.cf_model,
            'content_model': self.content_model,
            'context_model': self.context_model,
            'mood_model': self.mood_model,
            
            # 保存ID映射
            'user_id_map': self.user_id_map if hasattr(self, 'user_id_map') else {},
            'song_id_map': self.song_id_map if hasattr(self, 'song_id_map') else {},
            'id_to_user': self.id_to_user if hasattr(self, 'id_to_user') else {},
            'id_to_song': self.id_to_song if hasattr(self, 'id_to_song') else {},
            
            # 保存所有可能的映射变量
            'user_to_idx': self.user_to_idx if hasattr(self, 'user_to_idx') else {},
            'idx_to_user': self.idx_to_user if hasattr(self, 'idx_to_user') else {},
            'song_to_idx': self.song_to_idx if hasattr(self, 'song_to_idx') else {},
            'idx_to_song': self.idx_to_song if hasattr(self, 'idx_to_song') else {},
            
            # 权重
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'context_weight': self.context_weight,
            'mood_weight': self.mood_weight,
            'deep_weight': self.deep_weight,
            
            # 状态标志
            'is_trained': self.is_trained,
            'has_mood_model': self.has_mood_model,
            'has_deep_model': self.has_deep_model,
            
            # 歌曲数据
            'songs_df': self.songs_df
        }
        
        # 保存深度学习模型路径而不是模型本身
        model_data['deep_model_path'] = None
        if self.deep_model and self.has_deep_model:
            deep_model_dir = os.path.join(os.path.dirname(path), 'deep_model')
            os.makedirs(deep_model_dir, exist_ok=True)
            model_data['deep_model_path'] = deep_model_dir
            try:
                self.deep_model.save(deep_model_dir)
                logger.info(f"深度学习模型已保存到 {deep_model_dir}")
            except Exception as e:
                logger.error(f"保存深度学习模型失败: {e}")
        
        # 打印用户映射大小
        user_map_size = len(model_data['user_to_idx']) if 'user_to_idx' in model_data else 0
        logger.info(f"正在保存混合模型 (用户映射大小: {user_map_size})")
        
        # 保存模型数据
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
            
        logger.info(f"混合推荐模型已保存到 {path}")

    def load_model(self, path):
        """从磁盘加载模型"""
        logger.info(f"加载混合推荐模型: {path}")
        
        if not os.path.exists(path):
            logger.error(f"模型文件不存在: {path}")
            return False
            
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
                
            # 加载模型组件
            self.cf_model = model_data.get('cf_model')
            self.content_model = model_data.get('content_model')
            self.context_model = model_data.get('context_model')
            self.mood_model = model_data.get('mood_model')
            
            # 加载ID映射
            self.user_id_map = model_data.get('user_id_map', {})
            self.song_id_map = model_data.get('song_id_map', {})
            self.id_to_user = model_data.get('id_to_user', {})
            self.id_to_song = model_data.get('id_to_song', {})
            
            # 加载所有可能的映射变量
            self.user_to_idx = model_data.get('user_to_idx', self.user_id_map)
            self.idx_to_user = model_data.get('idx_to_user', self.id_to_user)
            self.song_to_idx = model_data.get('song_to_idx', self.song_id_map)
            self.idx_to_song = model_data.get('idx_to_song', self.id_to_song)
            
            # 加载权重
            self.cf_weight = model_data.get('cf_weight', 0.4)
            self.content_weight = model_data.get('content_weight', 0.25)
            self.context_weight = model_data.get('context_weight', 0.15)
            self.mood_weight = model_data.get('mood_weight', 0.2)
            self.deep_weight = model_data.get('deep_weight', 0.0)
            
            # 加载状态标志
            self.is_trained = model_data.get('is_trained', False)
            self.has_mood_model = model_data.get('has_mood_model', False)
            self.has_deep_model = model_data.get('has_deep_model', False)
            
            # 加载歌曲数据
            self.songs_df = model_data.get('songs_df')
            
            # 加载深度学习模型(如果有)
            deep_model_path = model_data.get('deep_model_path')
            if deep_model_path and os.path.exists(deep_model_path) and tf_available:
                try:
                    self.deep_model = load_model(deep_model_path)
                    logger.info(f"已加载深度学习模型: {deep_model_path}")
                except Exception as e:
                    logger.error(f"加载深度学习模型失败: {e}")
                    self.has_deep_model = False
                    
            # 打印映射大小
            user_map_size = len(self.user_to_idx) if hasattr(self, 'user_to_idx') else 0
            song_map_size = len(self.song_to_idx) if hasattr(self, 'song_to_idx') else 0
            logger.info(f"加载了用户映射 ({user_map_size} 个用户) 和歌曲映射 ({song_map_size} 首歌曲)")
            
            # 打印映射中的前几个用户ID作为调试信息
            if hasattr(self, 'user_to_idx') and self.user_to_idx:
                user_samples = list(self.user_to_idx.keys())[:5] if len(self.user_to_idx) > 5 else list(self.user_to_idx.keys())
                logger.info(f"用户ID映射示例: {user_samples}")
                
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            return False
    
    def _get_user_weights(self, user_id):
        """获取用户的模型权重"""
        # 如果不存在用户自定义权重，使用默认权重
        if not hasattr(self, 'user_weights') or user_id not in self.user_weights:
            default_weights = {
                'cf': self.cf_weight,
                'content': self.content_weight,
                'context': self.context_weight,
                'mood': self.mood_weight,
                'deep': self.deep_weight if self.has_deep_model else 0.0
            }
            return default_weights
        
        return self.user_weights[user_id]
    
    def _cf_recommend(self, user_internal_id, n=10, exclude_items=None):
        """基于协同过滤的推荐方法
        
        Args:
            user_internal_id: 内部用户ID
            n: 推荐数量
            exclude_items: 排除的歌曲ID列表
        
        Returns:
            推荐歌曲列表 [(歌曲ID, 分数), ...]
        """
        if exclude_items is None:
            exclude_items = []
            
        try:
            # 将用户ID转换为压缩索引
            if hasattr(self, 'compact_user_map') and user_internal_id in self.compact_user_map:
                compact_user_idx = self.compact_user_map[user_internal_id]
            else:
                print(f"用户ID {user_internal_id} 不在压缩映射中")
                return []
            
            # 获取用户评分向量
            if compact_user_idx >= self.cf_matrix.shape[0]:
                print(f"用户压缩索引 {compact_user_idx} 超出了矩阵维度 {self.cf_matrix.shape}")
                return []
                
            user_vector = self.cf_matrix[compact_user_idx].toarray().flatten()
            
            # 计算所有物品的预测评分
            scores = []
            for compact_item_idx in range(self.cf_matrix.shape[1]):
                original_item_idx = self.reverse_compact_item_map.get(compact_item_idx)
                if original_item_idx is not None and original_item_idx not in exclude_items:
                    score = user_vector[compact_item_idx]
                    scores.append((original_item_idx, float(score)))
            
            # 如果得分为空，返回空列表
            if not scores:
                return []
                
            # 按照评分排序，并返回前n个
            return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        except Exception as e:
            print(f"协同过滤推荐出错: {str(e)}")
            return []
    
    def _content_recommend(self, user_internal_id, n=10, exclude_items=None):
        """基于内容的推荐方法
        
        根据用户历史交互的歌曲内容特征，推荐相似的其他歌曲
        
        Args:
            user_internal_id: 内部用户ID
            n: 推荐数量
            exclude_items: 排除的歌曲ID列表
        
        Returns:
            推荐歌曲列表 [(歌曲ID, 分数), ...]
        """
        if exclude_items is None:
            exclude_items = []
        
        if not hasattr(self, 'content_model') or not self.content_model:
            return []
        
        try:
            # 获取用户历史交互的歌曲
            user_items = self._get_user_items(user_internal_id)
            if not user_items:
                return []
            
            # 构建用户内容偏好向量 (用户喜欢的歌曲特征的平均值)
            user_profile = np.zeros(len(next(iter(self.content_model['features'].values()))))
            item_count = 0
            
            for item_id, rating in user_items:
                if item_id in self.content_model['features']:
                    user_profile += self.content_model['features'][item_id] * rating
                    item_count += rating
            
            if item_count > 0:
                user_profile /= item_count
            
            # 计算所有歌曲与用户偏好的相似度
            scores = []
            for item_id, features in self.content_model['features'].items():
                if item_id not in exclude_items and item_id not in [i for i, _ in user_items]:
                    # 余弦相似度计算
                    similarity = np.dot(user_profile, features) / (np.linalg.norm(user_profile) * np.linalg.norm(features) + 1e-8)
                    scores.append((item_id, float(similarity)))
            
            # 按照相似度排序，取前n个
            return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        except Exception as e:
            print(f"内容推荐出错: {str(e)}")
            return []
    
    def _context_recommend(self, user_internal_id, n=10, exclude_items=None):
        """基于上下文的推荐方法
        
        推荐高流行度的歌曲，同时考虑用户历史交互
        
        Args:
            user_internal_id: 内部用户ID
            n: 推荐数量
            exclude_items: 排除的歌曲ID列表
        
        Returns:
            推荐歌曲列表 [(歌曲ID, 分数), ...]
        """
        if exclude_items is None:
            exclude_items = []
        
        if not hasattr(self, 'context_model') or not self.context_model:
            return []
        
        try:
            # 获取用户历史交互的歌曲
            user_items = self._get_user_items(user_internal_id)
            user_item_ids = [i for i, _ in user_items]
            
            # 收集所有歌曲的流行度分数
            scores = []
            for item_id, popularity in self.context_model.items():
                if item_id not in exclude_items and item_id not in user_item_ids:
                    scores.append((item_id, float(popularity)))
            
            # 按照流行度排序，取前n个
            return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        except Exception as e:
            print(f"上下文推荐出错: {str(e)}")
            return []
    
    def _deep_recommend(self, user_internal_id, n=10, exclude_items=None):
        """基于深度学习模型的推荐方法
        
        使用训练好的深度学习模型预测用户对歌曲的评分
        
        Args:
            user_internal_id: 内部用户ID
            n: 推荐数量
            exclude_items: 排除的歌曲ID列表
        
        Returns:
            推荐歌曲列表 [(歌曲ID, 分数), ...]
        """
        if exclude_items is None:
            exclude_items = []
        
        if not self.has_deep_model or self.deep_model is None:
            return []
        
        try:
            # 获取用户历史交互的歌曲
            user_items = self._get_user_items(user_internal_id)
            user_item_ids = [i for i, _ in user_items]
            
            # 为所有候选歌曲计算预测评分
            scores = []
            candidate_items = [i for i in range(len(self.song_id_map)) if i not in exclude_items and i not in user_item_ids]
            
            # 批量预测以提高效率
            batch_size = 128
            for i in range(0, len(candidate_items), batch_size):
                batch_items = candidate_items[i:i+batch_size]
                user_batch = np.full(len(batch_items), user_internal_id)
                pred = self.deep_model.predict([user_batch, np.array(batch_items)], verbose=0)
                
                for j, item_id in enumerate(batch_items):
                    scores.append((item_id, float(pred[j][0])))
            
            # 按照预测评分排序，取前n个
            return sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        except Exception as e:
            print(f"深度学习推荐出错: {str(e)}")
            return []
    
    def _get_user_items(self, user_internal_id):
        """获取用户历史交互的歌曲
        
        Args:
            user_internal_id: 内部用户ID
        
        Returns:
            用户交互过的歌曲列表 [(歌曲ID, 评分), ...]
        """
        try:
            if hasattr(self, 'compact_user_map') and user_internal_id in self.compact_user_map:
                compact_user_idx = self.compact_user_map[user_internal_id]
                user_vector = self.cf_matrix[compact_user_idx].toarray().flatten()
                
                # 获取用户评分不为0的物品
                items = []
                for compact_item_idx, rating in enumerate(user_vector):
                    if rating > 0:
                        original_item_idx = self.reverse_compact_item_map.get(compact_item_idx)
                        if original_item_idx is not None:
                            items.append((original_item_idx, float(rating)))
                
                return items
            else:
                return []
        except Exception as e:
            print(f"获取用户物品时出错: {str(e)}")
            return []
    
    def _cold_start_recommend(self, user_id, n=10, exclude_items=None, current_mood=None, user_preferences=None):
        """为新用户或没有历史记录的用户生成冷启动推荐"""
        if exclude_items is None:
            exclude_items = set()
        else:
            exclude_items = set(exclude_items)
            
        songs_df = self.songs_df.copy()
        if songs_df is None or songs_df.empty:
            logger.warning("没有歌曲数据，无法生成冷启动推荐")
            return []
            
        recommendations = []
        
        # 方案1：基于当前情绪的推荐（如果有情绪模型和当前情绪）
        if self.has_mood_model and current_mood and self.mood_model:
            logger.info(f"为用户 {user_id} 执行基于情绪的冷启动推荐, 当前情绪: {current_mood}")
            
            # 确保处理复合情绪
            if isinstance(current_mood, str):
                moods = [current_mood]
            elif isinstance(current_mood, (list, tuple)):
                moods = current_mood
            else:
                moods = ['happy']  # 默认情绪
                
            # 对每个情绪找到匹配的歌曲
            mood_matches = []
            for mood in moods:
                if mood in self.mood_model.keys():
                    # 按情绪分数排序的歌曲
                    sorted_songs = sorted(self.mood_model[mood].items(), key=lambda x: x[1], reverse=True)
                    # 只保留未被排除的歌曲
                    valid_songs = [(song_id, score) for song_id, score in sorted_songs 
                                   if song_id not in exclude_items]
                    mood_matches.extend(valid_songs[:min(n, len(valid_songs))])
            
            # 按情绪分数排序并去重
            seen_songs = set()
            for song_id, score in sorted(mood_matches, key=lambda x: x[1], reverse=True):
                if song_id not in seen_songs and len(recommendations) < n:
                    recommendations.append({
                        'song_id': song_id, 
                        'score': float(score),
                        'source': 'mood'
                    })
                    seen_songs.add(song_id)
                    
            # 如果根据情绪找到了足够的推荐
            if len(recommendations) >= n:
                logger.info(f"基于情绪生成了 {len(recommendations)} 条推荐")
                return recommendations[:n]
        
        # 方案2：基于用户偏好的内容推荐
        if user_preferences and self.content_model is not None:
            logger.info(f"为用户 {user_id} 执行基于偏好的冷启动推荐")
            
            # 用户偏好可以包括喜欢的流派、艺术家、时代等
            prefer_genres = user_preferences.get('genres', [])
            prefer_artists = user_preferences.get('artists', [])
            prefer_years = user_preferences.get('years', [])
            prefer_features = user_preferences.get('audio_features', {})
            
            # 匹配歌曲
            genre_matches = []
            artist_matches = []
            year_matches = []
            feature_matches = []
            
            # 基于流派匹配
            if prefer_genres and 'genre' in songs_df.columns:
                for genre in prefer_genres:
                    matching_songs = songs_df[songs_df['genre'].str.contains(genre, case=False, na=False)]
                    for _, song in matching_songs.iterrows():
                        if song['song_id'] not in exclude_items:
                            genre_matches.append((song['song_id'], 0.9))  # 高权重
            
            # 基于艺术家匹配
            if prefer_artists and 'artist_name' in songs_df.columns:
                for artist in prefer_artists:
                    matching_songs = songs_df[songs_df['artist_name'].str.contains(artist, case=False, na=False)]
                    for _, song in matching_songs.iterrows():
                        if song['song_id'] not in exclude_items:
                            artist_matches.append((song['song_id'], 0.85))  # 稍低权重
            
            # 基于年代匹配
            if prefer_years and 'year' in songs_df.columns:
                for year_range in prefer_years:
                    start_year, end_year = year_range[0], year_range[1]
                    matching_songs = songs_df[(songs_df['year'] >= start_year) & (songs_df['year'] <= end_year)]
                    for _, song in matching_songs.iterrows():
                        if song['song_id'] not in exclude_items:
                            year_matches.append((song['song_id'], 0.7))  # 较低权重
            
            # 基于音频特征匹配
            if prefer_features and self.content_model is not None:
                # 创建目标特征向量
                target_vector = np.zeros(len(next(iter(self.content_model.values()))))
                feature_count = 0
                
                # 从用户偏好中构建目标向量
                for feature, value in prefer_features.items():
                    if feature in self.content_model:
                        target_vector += self.content_model[feature] * value
                        feature_count += 1
                
                if feature_count > 0:
                    # 归一化
                    target_vector /= feature_count
                    
                    # 计算所有歌曲与目标向量的相似度
                    for song_id in self.song_to_idx.keys():
                        if song_id not in exclude_items and song_id in self.content_model:
                            similarity = cosine_similarity(
                                [target_vector], 
                                [self.content_model[song_id]]
                            )[0][0]
                            feature_matches.append((song_id, similarity))
            
            # 合并所有匹配，按权重排序
            all_matches = []
            all_matches.extend([(id, score * 0.9) for id, score in genre_matches])    # 流派权重
            all_matches.extend([(id, score * 0.85) for id, score in artist_matches])  # 艺术家权重
            all_matches.extend([(id, score * 0.7) for id, score in year_matches])     # 年代权重
            all_matches.extend([(id, score * 0.95) for id, score in feature_matches]) # 特征权重最高
            
            # 去重并排序
            song_scores = {}
            for song_id, score in all_matches:
                if song_id in song_scores:
                    song_scores[song_id] = max(song_scores[song_id], score)  # 保留最高分
                else:
                    song_scores[song_id] = score
            
            # 转换为推荐格式
            preference_recs = [
                {'song_id': song_id, 'score': float(score), 'source': 'preference'}
                for song_id, score in sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
            ]
            
            # 添加到推荐列表
            for rec in preference_recs:
                if len(recommendations) < n and rec['song_id'] not in [r['song_id'] for r in recommendations]:
                    recommendations.append(rec)
            
            # 如果基于偏好找到了足够的推荐
            if len(recommendations) >= n:
                logger.info(f"基于用户偏好生成了 {len(recommendations)} 条推荐")
                return recommendations[:n]
        
        # 方案3：基于流行度和多样性的推荐
        logger.info(f"为用户 {user_id} 执行基于流行度的冷启动推荐...")
        
        # 计算每首歌曲的流行度分数
        if 'popularity' in songs_df.columns:
            # 如果有明确的流行度列
            popularity_scores = dict(zip(songs_df['song_id'], songs_df['popularity']))
        else:
            # 基于交互次数计算流行度
            popularity_scores = {}
            if hasattr(self, 'cf_matrix') and self.cf_matrix is not None:
                # 计算每首歌曲的总交互次数
                song_interaction_counts = self.cf_matrix.sum(axis=0).A1
                for idx, count in enumerate(song_interaction_counts):
                    if idx in self.id_to_song:
                        song_id = self.id_to_song[idx]
                        popularity_scores[song_id] = float(count)
        
        # 流行度排名
        popular_songs = sorted(
            [(song_id, score) for song_id, score in popularity_scores.items() 
             if song_id not in exclude_items and song_id not in [r['song_id'] for r in recommendations]],
            key=lambda x: x[1], 
            reverse=True
        )
        
        # 增加多样性: 除了最流行的歌曲，也包括中等流行度和小众歌曲
        top_n_popular = int(n * 0.6)  # 60%是最流行的
        mid_n_popular = int(n * 0.3)   # 30%是中等流行的
        low_n_popular = n - top_n_popular - mid_n_popular  # 10%是小众歌曲
        
        # 添加最流行的歌曲
        for song_id, score in popular_songs[:min(top_n_popular, len(popular_songs))]:
            if len(recommendations) < n:
                recommendations.append({
                    'song_id': song_id, 
                    'score': float(score / max(popularity_scores.values()) if popularity_scores else 0.5),
                    'source': 'popular_top'
                })
        
        # 添加中等流行的歌曲
        middle_idx = len(popular_songs) // 2
        middle_range = popular_songs[middle_idx:min(middle_idx + mid_n_popular*2, len(popular_songs))]
        # 随机选择中等流行的歌曲
        import random
        random.shuffle(middle_range)
        for song_id, score in middle_range[:mid_n_popular]:
            if len(recommendations) < n and song_id not in [r['song_id'] for r in recommendations]:
                recommendations.append({
                    'song_id': song_id, 
                    'score': float(score / max(popularity_scores.values()) if popularity_scores else 0.3),
                    'source': 'popular_mid'
                })
        
        # 添加小众歌曲
        if low_n_popular > 0 and len(popular_songs) > top_n_popular + mid_n_popular:
            low_range = popular_songs[-(len(popular_songs)//4):]  # 取最后四分之一作为小众歌曲
            random.shuffle(low_range)
            for song_id, score in low_range[:low_n_popular]:
                if len(recommendations) < n and song_id not in [r['song_id'] for r in recommendations]:
                    recommendations.append({
                        'song_id': song_id, 
                        'score': float(score / max(popularity_scores.values()) if popularity_scores else 0.1),
                        'source': 'popular_low'
                    })
        
        # 如果还不够，填充最流行的歌曲
        remaining_popular = [
            song for song in popular_songs 
            if song[0] not in [r['song_id'] for r in recommendations]
        ]
        for song_id, score in remaining_popular:
            if len(recommendations) < n:
                recommendations.append({
                    'song_id': song_id, 
                    'score': float(score / max(popularity_scores.values()) if popularity_scores else 0.5),
                    'source': 'popular_fill'
                })
                
        logger.info(f"基于流行度生成了 {len(recommendations)} 条推荐")
        return recommendations[:n]

    def _get_item_features(self, item_id):
        """获取歌曲特征
        
        Args:
            item_id: 歌曲ID
        
        Returns:
            歌曲特征向量
        """
        if hasattr(self, 'content_model') and self.content_model and 'features' in self.content_model:
            return self.content_model['features'].get(item_id, None)
        return None 