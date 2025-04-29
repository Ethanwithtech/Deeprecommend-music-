#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高性能混合推荐系统模型
实现高性能的多模态混合推荐系统，目标准确率NDCG@10≥0.8
"""

import os
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Input, Embedding, Flatten, Dropout
from tensorflow.keras.models import Model
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score
import time

# 配置日志
logger = logging.getLogger(__name__)

class HybridRecommender:
    """
    高性能混合推荐系统
    
    结合深度学习和梯度提升树的混合推荐架构，支持多模态特征融合
    """
    
    def __init__(self, user_num, item_num, feature_dim=64, 
                 embedding_dim=64, dropout_rate=0.3, learning_rate=0.001):
        """
        初始化混合推荐系统
        
        参数:
            user_num: 用户数量
            item_num: 歌曲数量
            feature_dim: 特征维度
            embedding_dim: 嵌入层维度
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        self.user_num = user_num
        self.item_num = item_num
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # 多模态输入
        self.user_input = Input(shape=(1,), name='user_input')
        self.item_input = Input(shape=(1,), name='item_input')
        self.feature_input = Input(shape=(feature_dim,), name='feature_input')
        
        # 嵌入层（协同过滤部分）
        self.user_embed = Embedding(user_num, embedding_dim, 
                                    embeddings_regularizer=tf.keras.regularizers.l2(1e-5))(self.user_input)
        self.item_embed = Embedding(item_num, embedding_dim, 
                                   embeddings_regularizer=tf.keras.regularizers.l2(1e-5))(self.item_input)
        
        self.user_vec = Flatten()(self.user_embed)
        self.item_vec = Flatten()(self.item_embed)
        
        # 创建深度模型和GBDT模型
        self._build_deep_model()
        self._build_gbdt_model()
        
        # 混合权重
        self.deep_weight = 0.6
        self.gbdt_weight = 0.4
        
        # 特征工程组件
        self.scaler = StandardScaler()
        
        # 记录训练历史
        self.history = {}
        
    def _build_deep_model(self):
        """构建深度学习模型"""
        # 特征融合层
        concat = Concatenate()([self.user_vec, self.item_vec, self.feature_input])
        
        # 深度交叉网络
        x = Dense(256, activation='swish')(concat)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(128, activation='swish')(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='swish')(x)
        
        # 输出层
        output = Dense(1, activation='sigmoid', name='output')(x)
        
        # 创建模型
        self.deep_model = Model(
            inputs=[self.user_input, self.item_input, self.feature_input], 
            outputs=output
        )
        
        # 编译模型
        self.deep_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("深度学习模型构建完成")
        
    def _build_gbdt_model(self):
        """构建梯度提升树模型"""
        self.gbdt = lgb.LGBMRanker(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.1,
            max_depth=8,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            min_split_gain=0.01,
            reg_alpha=0.01,
            reg_lambda=0.01,
            n_jobs=-1,
            importance_type='gain'
        )
        
        logger.info("GBDT模型构建完成")
        
    def preprocess_features(self, X, y=None, is_training=False):
        """
        预处理特征数据
        
        参数:
            X: 特征数据
            y: 标签数据（可选）
            is_training: 是否为训练模式
            
        返回:
            预处理后的特征和标签
        """
        # 标准化音频特征
        if is_training:
            X_audio = self.scaler.fit_transform(X['audio_features'])
        else:
            X_audio = self.scaler.transform(X['audio_features'])
            
        # 创建高级特征
        X_advanced = self._create_advanced_features(X)
        
        # 合并所有特征
        X_processed = {
            'user_input': X['user_idx'].values,
            'item_input': X['item_idx'].values,
            'feature_input': np.hstack([X_audio, X_advanced])
        }
        
        if y is not None:
            # 处理标签
            y_processed = y.values
            return X_processed, y_processed
        
        return X_processed
        
    def _create_advanced_features(self, X):
        """
        创建高级音乐特征
        
        参数:
            X: 特征数据
            
        返回:
            高级特征矩阵
        """
        df = X.copy()
        advanced_features = []
        
        # 如果有相关列，创建高级特征
        if 'beats' in df.columns and 'duration' in df.columns:
            # 节奏复杂度
            df['rhythm_complexity'] = df['beats'] / (df['duration'] + 1e-6)
            advanced_features.append(df['rhythm_complexity'].values.reshape(-1, 1))
        
        if 'energy' in df.columns and 'valence' in df.columns:
            # 情感能量比
            df['energy_ratio'] = df['energy'] * df['valence']
            advanced_features.append(df['energy_ratio'].values.reshape(-1, 1))
        
        if 'release_year' in df.columns:
            # 时间衰减因子（偏好近期行为）
            df['time_decay'] = np.exp(-0.1 * (2023 - df['release_year']))
            advanced_features.append(df['time_decay'].values.reshape(-1, 1))
            
        # 如果有tempo和loudness
        if 'tempo' in df.columns:
            # 节奏活力
            df['tempo_norm'] = df['tempo'] / 200.0  # 归一化
            advanced_features.append(df['tempo_norm'].values.reshape(-1, 1))
            
        if 'loudness' in df.columns:
            # 音量强度（归一化）
            df['loudness_norm'] = (df['loudness'] + 60) / 60.0  # 将约-60到0的范围归一化到0-1
            df['loudness_norm'] = df['loudness_norm'].clip(0, 1)
            advanced_features.append(df['loudness_norm'].values.reshape(-1, 1))
            
        # 如果没有任何高级特征
        if not advanced_features:
            return np.zeros((len(df), 1))
            
        # 合并所有高级特征
        return np.hstack(advanced_features)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=1024, early_stopping=True, verbose=1):
        """
        训练混合推荐系统
        
        参数:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            epochs: 训练轮数
            batch_size: 批次大小
            early_stopping: 是否启用早停
            verbose: 显示详细程度
            
        返回:
            训练历史记录
        """
        start_time = time.time()
        logger.info("开始训练混合推荐系统...")
        
        # 预处理数据
        X_train_processed, y_train_processed = self.preprocess_features(X_train, y_train, is_training=True)
        
        if X_val is not None and y_val is not None:
            X_val_processed, y_val_processed = self.preprocess_features(X_val, y_val, is_training=False)
            validation_data = ([X_val_processed['user_input'], 
                               X_val_processed['item_input'], 
                               X_val_processed['feature_input']], 
                              y_val_processed)
        else:
            validation_data = None
        
        # 训练深度模型
        callbacks = []
        if early_stopping:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=5,
                restore_best_weights=True
            ))
            
        # 添加课程学习回调
        callbacks.append(self._create_curriculum_callback())
            
        history = self.deep_model.fit(
            [X_train_processed['user_input'], 
             X_train_processed['item_input'], 
             X_train_processed['feature_input']],
            y_train_processed,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # 训练GBDT模型
        # 从深度模型中提取特征
        deep_features = self._extract_deep_features(X_train_processed)
        
        # 将所有特征合并为GBDT输入
        gbdt_features = np.hstack([
            X_train_processed['user_input'].reshape(-1, 1),
            X_train_processed['item_input'].reshape(-1, 1),
            X_train_processed['feature_input'],
            deep_features
        ])
        
        # 训练GBDT模型
        # 对于LGBMRanker，需要分组信息
        # 假设每个用户是一个组
        group_sizes = X_train.groupby('user_idx').size().values
        
        self.gbdt.fit(
            gbdt_features, 
            y_train_processed,
            group=group_sizes,
            verbose=50
        )
        
        # 记录训练耗时
        training_time = time.time() - start_time
        logger.info(f"混合推荐系统训练完成，耗时: {training_time:.2f} 秒")
        
        # 保存训练历史
        self.history = {
            'deep_history': history.history,
            'gbdt_feature_importance': self.gbdt.feature_importances_,
            'training_time': training_time
        }
        
        return self.history
    
    def _create_curriculum_callback(self):
        """创建课程学习回调函数"""
        # 定义阶段
        stages = [
            {'epoch': 0, 'lr': 0.001, 'sample_weight': 0.3},
            {'epoch': 10, 'lr': 0.0005, 'sample_weight': 0.5},
            {'epoch': 20, 'lr': 0.0001, 'sample_weight': 0.8},
            {'epoch': 30, 'lr': 0.00005, 'sample_weight': 1.0}
        ]
        
        class CurriculumLearningCallback(tf.keras.callbacks.Callback):
            def __init__(self, stages):
                super().__init__()
                self.stages = stages
                
            def on_epoch_begin(self, epoch, logs=None):
                # 找到当前阶段
                current_stage = next((s for s in self.stages if s['epoch'] <= epoch), self.stages[-1])
                
                # 更新学习率
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, current_stage['lr'])
                
                # 记录当前阶段
                logger.info(f"当前课程学习阶段: 轮次={epoch}, 学习率={current_stage['lr']}, 样本权重={current_stage['sample_weight']}")
                
        return CurriculumLearningCallback(stages)
    
    def _extract_deep_features(self, X_processed):
        """
        从深度模型中提取特征
        
        参数:
            X_processed: 预处理后的特征
            
        返回:
            深度特征
        """
        # 创建一个临时模型，输出最后一个隐藏层
        feature_layer_name = 'dense_2'  # 假设最后一个隐藏层的名称
        feature_extractor = Model(
            inputs=self.deep_model.inputs,
            outputs=self.deep_model.get_layer(feature_layer_name).output
        )
        
        # 提取特征
        deep_features = feature_extractor.predict([
            X_processed['user_input'], 
            X_processed['item_input'], 
            X_processed['feature_input']
        ])
        
        return deep_features
    
    def predict(self, X):
        """
        预测评分
        
        参数:
            X: 特征数据
            
        返回:
            预测评分
        """
        # 预处理特征
        X_processed = self.preprocess_features(X)
        
        # 深度模型预测
        deep_pred = self.deep_model.predict([
            X_processed['user_input'], 
            X_processed['item_input'], 
            X_processed['feature_input']
        ])
        
        # 从深度模型中提取特征
        deep_features = self._extract_deep_features(X_processed)
        
        # 将所有特征合并为GBDT输入
        gbdt_features = np.hstack([
            X_processed['user_input'].reshape(-1, 1),
            X_processed['item_input'].reshape(-1, 1),
            X_processed['feature_input'],
            deep_features
        ])
        
        # GBDT模型预测
        gbdt_pred = self.gbdt.predict(gbdt_features)
        
        # 融合预测结果
        # 使用动态权重融合
        weights = self._get_dynamic_weights(X)
        final_pred = weights[0] * deep_pred.flatten() + weights[1] * gbdt_pred
        
        return final_pred
    
    def _get_dynamic_weights(self, X):
        """
        获取动态融合权重
        根据输入特征调整权重
        
        参数:
            X: 特征数据
            
        返回:
            [深度模型权重, GBDT权重]
        """
        # 简单情况下使用固定权重
        return [self.deep_weight, self.gbdt_weight]
        
        # TODO: 实现动态权重调整，例如基于用户历史交互或歌曲特征
    
    def recommend_for_user(self, user_id, item_pool, X_features, top_n=10):
        """
        为用户推荐歌曲
        
        参数:
            user_id: 用户ID
            item_pool: 候选歌曲池
            X_features: 特征数据
            top_n: 推荐数量
            
        返回:
            推荐结果列表
        """
        # 准备预测数据
        pred_data = []
        for item_id in item_pool:
            # 为每个用户-物品对创建特征行
            row = X_features[(X_features['user_idx'] == user_id) & 
                            (X_features['item_idx'] == item_id)]
            if not row.empty:
                pred_data.append(row)
            
        # 如果没有找到任何特征数据，返回空列表
        if not pred_data:
            return []
            
        pred_df = pd.concat(pred_data, ignore_index=True)
        
        # 预测评分
        pred_scores = self.predict(pred_df)
        
        # 创建结果DataFrame
        results = pd.DataFrame({
            'item_idx': pred_df['item_idx'],
            'score': pred_scores
        })
        
        # 排序并获取前N个结果
        top_items = results.sort_values('score', ascending=False).head(top_n)['item_idx'].tolist()
        
        return top_items
    
    def evaluate(self, X_test, y_test, k=10):
        """
        评估模型性能
        
        参数:
            X_test: 测试特征
            y_test: 测试标签
            k: 评估的推荐数量
            
        返回:
            评估指标字典
        """
        # 预测评分
        y_pred = self.predict(X_test)
        
        # 按用户分组
        metrics = {}
        grouped = X_test.groupby('user_idx')
        
        ndcg_scores = []
        recall_scores = []
        precision_scores = []
        
        for user_id, group in grouped:
            # 获取此用户的真实评分和预测评分
            true_ratings = y_test.iloc[group.index].values
            pred_ratings = y_pred[group.index]
            
            # 只考虑有正面评价的物品
            true_relevant = true_ratings > 0.5
            
            if np.sum(true_relevant) > 0:
                # 计算NDCG@k
                try:
                    ndcg = ndcg_score([true_ratings], [pred_ratings], k=min(k, len(pred_ratings)))
                    ndcg_scores.append(ndcg)
                except Exception as e:
                    logger.warning(f"计算NDCG时出错: {str(e)}")
                
                # 按预测评分排序
                sorted_indices = np.argsort(pred_ratings)[::-1][:k]
                
                # 计算Recall@k
                n_rel_and_rec_k = np.sum(true_relevant[sorted_indices])
                n_rel = np.sum(true_relevant)
                recall = n_rel_and_rec_k / n_rel if n_rel > 0 else 0
                recall_scores.append(recall)
                
                # 计算Precision@k
                precision = n_rel_and_rec_k / min(k, len(sorted_indices)) if len(sorted_indices) > 0 else 0
                precision_scores.append(precision)
        
        # 计算平均指标
        metrics['NDCG@10'] = np.mean(ndcg_scores) if ndcg_scores else 0
        metrics['Recall@10'] = np.mean(recall_scores) if recall_scores else 0
        metrics['Precision@10'] = np.mean(precision_scores) if precision_scores else 0
        
        # 输出结果
        logger.info(f"评估结果: NDCG@10={metrics['NDCG@10']:.4f}, Recall@10={metrics['Recall@10']:.4f}, Precision@10={metrics['Precision@10']:.4f}")
        
        return metrics
    
    def save(self, path):
        """
        保存模型
        
        参数:
            path: 保存路径
        """
        os.makedirs(path, exist_ok=True)
        
        # 保存深度模型
        self.deep_model.save(os.path.join(path, 'deep_model'))
        
        # 保存GBDT模型
        lgb.Booster(model_file=self.gbdt).save_model(os.path.join(path, 'gbdt_model.txt'))
        
        # 保存特征处理组件
        import joblib
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # 保存配置和权重
        config = {
            'user_num': self.user_num,
            'item_num': self.item_num,
            'feature_dim': self.feature_dim,
            'embedding_dim': self.embedding_dim,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'deep_weight': self.deep_weight,
            'gbdt_weight': self.gbdt_weight
        }
        import json
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f)
            
        logger.info(f"模型已保存至: {path}")
        
    @classmethod
    def load(cls, path):
        """
        加载模型
        
        参数:
            path: 模型路径
            
        返回:
            加载的混合推荐系统模型
        """
        import json
        with open(os.path.join(path, 'config.json'), 'r') as f:
            config = json.load(f)
            
        # 创建模型实例
        model = cls(
            user_num=config['user_num'],
            item_num=config['item_num'],
            feature_dim=config['feature_dim'],
            embedding_dim=config['embedding_dim'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )
        
        # 加载深度模型
        model.deep_model = tf.keras.models.load_model(os.path.join(path, 'deep_model'))
        
        # 加载GBDT模型
        model.gbdt = lgb.Booster(model_file=os.path.join(path, 'gbdt_model.txt'))
        
        # 加载特征处理组件
        import joblib
        model.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        
        # 设置权重
        model.deep_weight = config['deep_weight']
        model.gbdt_weight = config['gbdt_weight']
        
        logger.info(f"模型已从{path}加载")
        
        return model


class DynamicWeightAdjuster:
    """动态权重调整器"""
    
    def __init__(self, n_models, momentum=0.9):
        """
        初始化动态权重调整器
        
        参数:
            n_models: 模型数量
            momentum: 动量系数
        """
        self.weights = np.ones(n_models) / n_models
        self.momentum = momentum
        self.history = []
    
    def update_weights(self, model_performance):
        """
        基于模型性能更新权重
        
        参数:
            model_performance: 模型性能评估结果列表
            
        返回:
            更新后的权重
        """
        # 基于模型近期表现调整权重
        delta = (model_performance - np.mean(model_performance)) * 0.1
        self.weights = self.momentum * self.weights + (1 - self.momentum) * delta
        
        # 使用softmax归一化
        self.weights = np.exp(self.weights) / np.sum(np.exp(self.weights))
        
        # 记录历史
        self.history.append(self.weights.copy())
        
        return self.weights


def feature_crossing(inputs, dim=64):
    """
    使用Transformer进行特征交叉
    
    参数:
        inputs: 输入特征
        dim: 特征维度
        
    返回:
        交叉后的特征
    """
    # 创建Query、Key、Value
    query = Dense(dim, activation='swish')(inputs)
    key = Dense(dim, activation='swish')(inputs)
    value = Dense(dim, activation='swish')(inputs)
    
    # 计算注意力
    attention = tf.matmul(query, key, transpose_b=True)
    attention = tf.nn.softmax(attention / tf.sqrt(float(dim)))
    
    # 应用注意力
    crossed_features = tf.matmul(attention, value)
    
    return crossed_features


def create_advanced_features(df):
    """
    创建高级音乐特征
    
    参数:
        df: 特征数据框
        
    返回:
        添加了高级特征的数据框
    """
    # 节奏复杂度
    if 'beats' in df.columns and 'duration' in df.columns:
        df['rhythm_complexity'] = df['beats'] / (df['duration'] + 1e-6)
    
    # 情感能量比
    if 'energy' in df.columns and 'valence' in df.columns:
        df['energy_ratio'] = df['energy'] * df['valence']
    
    # 时间衰减因子（偏好近期行为）
    if 'release_year' in df.columns:
        df['time_decay'] = np.exp(-0.1 * (2023 - df['release_year']))
    
    return df


class MultiTaskModel(Model):
    """
    多任务学习模型
    同时预测评分和播放时长
    """
    
    def __init__(self, user_num, item_num, feature_dim=64):
        super().__init__()
        # 共享编码器
        self.shared_encoder = Dense(256, activation='gelu')
        
        # 用户和物品嵌入层
        self.user_embedding = Embedding(user_num, 64)
        self.item_embedding = Embedding(item_num, 64)
        
        # 任务特定层
        self.rating_head = Dense(1, activation='sigmoid', name='rating')
        self.playtime_head = Dense(1, activation='relu', name='playtime')
    
    def call(self, inputs):
        # 解包输入
        user_input, item_input, feature_input = inputs
        
        # 嵌入
        user_embed = Flatten()(self.user_embedding(user_input))
        item_embed = Flatten()(self.item_embedding(item_input))
        
        # 连接所有特征
        concat = Concatenate()([user_embed, item_embed, feature_input])
        
        # 通过共享编码器
        shared = self.shared_encoder(concat)
        
        # 任务特定预测
        rating = self.rating_head(shared)
        playtime = self.playtime_head(shared)
        
        return {
            'rating': rating,
            'playtime': playtime
        }


def contrastive_loss(y_true, y_pred, temperature=0.1):
    """
    对比学习损失函数
    
    参数:
        y_true: 真实标签
        y_pred: 预测向量
        temperature: 温度参数
        
    返回:
        对比损失
    """
    # 计算样本间相似度
    similarities = tf.matmul(y_pred, y_pred, transpose_b=True)
    logits = similarities / temperature
    
    # 创建标签 - 每个样本只与自己匹配
    labels = tf.range(tf.shape(y_pred)[0])
    
    # 计算交叉熵损失
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, 
        logits=logits
    )
    
    return loss 