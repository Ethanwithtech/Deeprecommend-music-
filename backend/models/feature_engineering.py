#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高级特征工程模块 - 为混合推荐系统创建高级特征
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureGenerator:
    """高级音乐特征生成器"""
    
    def __init__(self):
        """初始化特征生成器"""
        self.scalers = {}
        self.pca = None
        self.feature_stats = {}
    
    def create_audio_features(self, songs_df):
        """
        创建高级音频特征
        
        参数:
            songs_df: 包含基础音频特征的歌曲数据框
            
        返回:
            增强了高级特征的数据框
        """
        logger.info("创建高级音频特征...")
        df = songs_df.copy()
        
        # 1. 节奏特征
        self._create_rhythm_features(df)
        
        # 2. 情感特征
        self._create_emotion_features(df)
        
        # 3. 复杂度特征
        self._create_complexity_features(df)
        
        # 4. 时间特征
        self._create_temporal_features(df)
        
        # 5. 标准化所有新特征
        self._normalize_features(df)
        
        # 6. 特征降维（可选）
        # self._reduce_dimensions(df)
        
        # 记录特征统计信息
        self._compute_feature_stats(df)
        
        logger.info(f"创建了 {len(df.columns) - len(songs_df.columns)} 个高级特征")
        return df
    
    def _create_rhythm_features(self, df):
        """创建节奏相关特征"""
        # 检查必要的列
        if 'tempo' not in df.columns:
            logger.warning("缺少tempo列，无法创建所有节奏特征")
            df['tempo'] = 120.0  # 默认值
        
        if 'duration' not in df.columns:
            logger.warning("缺少duration列，无法创建所有节奏特征")
            df['duration'] = 240.0  # 默认值
            
        if 'beats' in df.columns:
            # 节奏复杂度 - 每秒钟的节拍数
            df['rhythm_complexity'] = df['beats'] / df['duration'].clip(1.0)
        elif 'tempo' in df.columns:
            # 估算节奏复杂度
            df['rhythm_complexity'] = (df['tempo'] / 60.0)
        
        # 节奏强度 - 基于tempo
        df['rhythm_intensity'] = df['tempo'] / 200.0  # 归一化到约0-1的范围
        df['rhythm_intensity'] = df['rhythm_intensity'].clip(0, 1)
        
        # BPM类别
        df['tempo_category'] = pd.cut(
            df['tempo'], 
            bins=[0, 70, 120, 160, 1000],
            labels=['slow', 'medium', 'fast', 'very_fast']
        ).astype(str)
        
        # 创建节奏类别的独热编码
        tempo_dummies = pd.get_dummies(df['tempo_category'], prefix='tempo')
        df = pd.concat([df, tempo_dummies], axis=1)
        
    def _create_emotion_features(self, df):
        """创建情感相关特征"""
        # 检查必要的列
        if 'valence' not in df.columns:
            logger.warning("缺少valence列，无法创建所有情感特征")
            df['valence'] = 0.5  # 默认值
            
        if 'energy' not in df.columns:
            logger.warning("缺少energy列，无法创建所有情感特征")
            df['energy'] = 0.5  # 默认值
            
        if 'mode' not in df.columns:
            logger.warning("缺少mode列，无法创建所有情感特征")
            df['mode'] = 1  # 默认值（大调）
            
        # 情感能量比 - 结合效价(valence)和能量
        df['emotion_energy_ratio'] = df['valence'] * df['energy']
        
        # 情感反差 - 效价与能量的差距
        df['emotion_contrast'] = np.abs(df['valence'] - df['energy'])
        
        # 音乐情绪预测
        # 基于Russell的情绪环形模型(Circumplex Model of Affect)
        # 高价高能量 = 兴奋(Excited)
        # 高价低能量 = 平静(Calm)
        # 低价高能量 = 愤怒(Angry)
        # 低价低能量 = 忧郁(Sad)
        
        # 计算象限
        df['emotion_quadrant'] = 0
        df.loc[(df['valence'] >= 0.5) & (df['energy'] >= 0.5), 'emotion_quadrant'] = 1  # 兴奋
        df.loc[(df['valence'] >= 0.5) & (df['energy'] < 0.5), 'emotion_quadrant'] = 2   # 平静
        df.loc[(df['valence'] < 0.5) & (df['energy'] >= 0.5), 'emotion_quadrant'] = 3   # 愤怒
        df.loc[(df['valence'] < 0.5) & (df['energy'] < 0.5), 'emotion_quadrant'] = 4    # 忧郁
        
        # 将象限转换为情绪标签
        emotion_map = {1: 'excited', 2: 'calm', 3: 'angry', 4: 'sad'}
        df['emotion_label'] = df['emotion_quadrant'].map(emotion_map)
        
        # 创建情绪标签的独热编码
        emotion_dummies = pd.get_dummies(df['emotion_label'], prefix='emotion')
        df = pd.concat([df, emotion_dummies], axis=1)
        
        # 模式(调式)特征 - 大调(1)通常感觉更积极，小调(0)感觉更消极
        if 'mode' in df.columns:
            df['major_mode'] = (df['mode'] == 1).astype(int)
        
    def _create_complexity_features(self, df):
        """创建音乐复杂度相关特征"""
        # 检查必要的列
        has_acoustic_features = all(col in df.columns for col in ['acousticness', 'instrumentalness'])
        
        if not has_acoustic_features:
            logger.warning("缺少声学特征，无法创建所有复杂度特征")
        
        # 音频复杂度指标（如果有足够的特征）
        if has_acoustic_features and 'loudness' in df.columns:
            # 计算一个音频复杂度指标
            df['audio_complexity'] = (
                0.4 * (1 - df['acousticness']) +  # 非原声程度
                0.3 * df['instrumentalness'] +    # 器乐程度
                0.3 * ((df['loudness'] + 60) / 60).clip(0, 1)  # 归一化响度
            )
        else:
            # 如果缺少一些列，创建一个基本的复杂度度量
            complexity_cols = [col for col in ['acousticness', 'instrumentalness', 'loudness'] 
                             if col in df.columns]
            
            if complexity_cols:
                # 使用可用列的平均值
                df['audio_complexity'] = df[complexity_cols].mean(axis=1)
            else:
                # 没有可用的列，设置默认值
                df['audio_complexity'] = 0.5
        
        # 音乐结构复杂度
        # 这通常需要更详细的音频分析，这里使用简化的估计
        if 'tempo' in df.columns and 'key' in df.columns and 'time_signature' in df.columns:
            # 创建一个复杂度指标融合多个特征
            df['structure_complexity'] = (
                0.4 * (df['tempo'] / 200) +  # 节奏复杂度（归一化）
                0.3 * (df['key'] / 11) +     # 调式复杂度（归一化）
                0.3 * (df['time_signature'] / 7)  # 拍号复杂度（归一化）
            )
            df['structure_complexity'] = df['structure_complexity'].clip(0, 1)
        
    def _create_temporal_features(self, df):
        """创建时间相关特征"""
        # 检查必要的列
        if 'year' not in df.columns and 'release_year' not in df.columns:
            logger.warning("缺少年份列，无法创建所有时间特征")
            df['year'] = 2000  # 默认值
        
        # 统一列名
        year_col = 'year' if 'year' in df.columns else 'release_year'
        
        # 时间衰减因子（给近期歌曲更高的权重）
        current_year = 2023
        df['time_decay'] = np.exp(-0.1 * (current_year - df[year_col]))
        
        # 时代分类
        df['era'] = pd.cut(
            df[year_col], 
            bins=[0, 1970, 1980, 1990, 2000, 2010, current_year+1],
            labels=['pre_1970s', '1970s', '1980s', '1990s', '2000s', '2010s_plus']
        ).astype(str)
        
        # 创建时代的独热编码
        era_dummies = pd.get_dummies(df['era'], prefix='era')
        df = pd.concat([df, era_dummies], axis=1)
        
        # 歌曲持续时间类别
        if 'duration' in df.columns:
            df['duration_category'] = pd.cut(
                df['duration'], 
                bins=[0, 120, 210, 300, 600, float('inf')],
                labels=['very_short', 'short', 'medium', 'long', 'very_long']
            ).astype(str)
            
            # 创建持续时间类别的独热编码
            duration_dummies = pd.get_dummies(df['duration_category'], prefix='duration')
            df = pd.concat([df, duration_dummies], axis=1)
    
    def _normalize_features(self, df):
        """标准化所有数值特征"""
        # 获取所有数值列
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 排除ID列和年份列
        exclude_cols = ['song_id', 'track_id', 'artist_id', 'album_id', 'year', 'release_year']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 为每个特征创建一个标准化器
        for col in numeric_cols:
            # 跳过已经是归一化形式的列
            if '_norm' in col:
                continue
                
            # 创建一个缩放器
            scaler = MinMaxScaler()
            
            # 应用缩放
            df[f"{col}_norm"] = scaler.fit_transform(df[[col]]).flatten()
            
            # 存储缩放器以便将来使用
            self.scalers[col] = scaler
    
    def _reduce_dimensions(self, df, n_components=10):
        """使用PCA降低特征维度"""
        # 获取所有归一化特征
        norm_cols = [col for col in df.columns if col.endswith('_norm')]
        
        if len(norm_cols) > n_components:
            # 创建PCA对象
            self.pca = PCA(n_components=n_components)
            
            # 应用PCA
            pca_features = self.pca.fit_transform(df[norm_cols])
            
            # 添加PCA特征到数据框
            for i in range(n_components):
                df[f'pca_feature_{i}'] = pca_features[:, i]
                
            logger.info(f"应用PCA降维，将 {len(norm_cols)} 个特征降至 {n_components} 维")
    
    def _compute_feature_stats(self, df):
        """计算并存储特征统计信息"""
        # 获取所有特征列
        feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        # 计算每个特征的统计信息
        for col in feature_cols:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median()
            }
    
    def create_user_features(self, interactions_df, songs_df):
        """
        创建用户特征
        
        参数:
            interactions_df: 用户-歌曲交互数据框
            songs_df: 歌曲特征数据框
            
        返回:
            用户特征数据框
        """
        logger.info("创建用户特征...")
        
        # 提取唯一用户ID
        user_ids = interactions_df['user_id'].unique()
        
        # 创建用户特征数据框
        user_features = []
        
        for user_id in user_ids:
            # 获取此用户的所有交互
            user_interactions = interactions_df[interactions_df['user_id'] == user_id]
            
            # 确保有评分列
            if 'rating' not in user_interactions.columns and 'play_count' in user_interactions.columns:
                user_interactions['rating'] = user_interactions['play_count']
            
            # 跳过没有评分的用户
            if 'rating' not in user_interactions.columns:
                continue
                
            # 获取用户评分的歌曲
            rated_songs = songs_df[songs_df['song_id'].isin(user_interactions['song_id'])]
            
            # 用户特征字典
            user_feature = {
                'user_id': user_id,
                'interaction_count': len(user_interactions),
                'avg_rating': user_interactions['rating'].mean(),
                'rating_std': user_interactions['rating'].std(),
                'max_rating': user_interactions['rating'].max(),
                'min_rating': user_interactions['rating'].min()
            }
            
            # 如果有足够的交互，计算偏好特征
            if len(rated_songs) > 0:
                # 计算用户偏好的平均音频特征
                audio_feature_cols = [col for col in rated_songs.columns 
                                    if col.endswith('_norm') 
                                    or col in ['audio_complexity', 'structure_complexity']]
                
                for col in audio_feature_cols:
                    # 基于评分加权的平均值
                    if len(user_interactions) > 0 and 'rating' in user_interactions.columns:
                        # 合并歌曲特征和评分
                        songs_with_ratings = rated_songs.merge(
                            user_interactions[['song_id', 'rating']], 
                            on='song_id'
                        )
                        
                        # 如果成功合并
                        if len(songs_with_ratings) > 0 and col in songs_with_ratings.columns:
                            # 计算加权平均值
                            weights = songs_with_ratings['rating']
                            values = songs_with_ratings[col]
                            user_feature[f'pref_{col}'] = np.average(values, weights=weights)
                    
                # 计算情感偏好占比
                if 'emotion_label' in rated_songs.columns:
                    emotion_counts = rated_songs['emotion_label'].value_counts(normalize=True)
                    for emotion, ratio in emotion_counts.items():
                        user_feature[f'pref_emotion_{emotion}'] = ratio
                
                # 计算时代偏好占比
                if 'era' in rated_songs.columns:
                    era_counts = rated_songs['era'].value_counts(normalize=True)
                    for era, ratio in era_counts.items():
                        user_feature[f'pref_era_{era}'] = ratio
            
            user_features.append(user_feature)
        
        # 创建数据框
        user_features_df = pd.DataFrame(user_features)
        
        # 填充可能的NaN值
        user_features_df = user_features_df.fillna(0)
        
        logger.info(f"创建了 {len(user_features_df.columns) - 1} 个用户特征")
        return user_features_df


def create_interaction_features(interactions_df, songs_df, users_df=None):
    """
    创建交互特征
    
    参数:
        interactions_df: 用户-歌曲交互数据框
        songs_df: 歌曲特征数据框
        users_df: 用户特征数据框（可选）
        
    返回:
        交互特征数据框
    """
    logger.info("创建交互特征...")
    
    # 复制交互数据
    interactions = interactions_df.copy()
    
    # 添加时间特征（如果存在）
    if 'timestamp' in interactions.columns:
        # 将时间戳转换为日期时间
        interactions['datetime'] = pd.to_datetime(interactions['timestamp'], unit='s')
        
        # 提取时间相关特征
        interactions['hour'] = interactions['datetime'].dt.hour
        interactions['day_of_week'] = interactions['datetime'].dt.dayofweek
        interactions['is_weekend'] = interactions['day_of_week'].isin([5, 6]).astype(int)
        
        # 时段类别（早上、下午、晚上、深夜）
        interactions['time_period'] = pd.cut(
            interactions['hour'], 
            bins=[0, 6, 12, 18, 24],
            labels=['night', 'morning', 'afternoon', 'evening']
        ).astype(str)
        
        # 创建时段的独热编码
        time_dummies = pd.get_dummies(interactions['time_period'], prefix='time')
        interactions = pd.concat([interactions, time_dummies], axis=1)
    
    # 合并歌曲特征
    interactions_with_features = interactions.merge(songs_df, on='song_id', how='left')
    
    # 如果提供了用户特征，也合并这些特征
    if users_df is not None:
        interactions_with_features = interactions_with_features.merge(users_df, on='user_id', how='left')
    
    # 创建混合特征
    if 'valence' in songs_df.columns and 'user_id' in interactions_with_features.columns:
        # 基于用户和歌曲特征的交叉特征
        # 例如：用户评分与歌曲效价的交互
        if 'rating' in interactions_with_features.columns and 'valence' in interactions_with_features.columns:
            interactions_with_features['rating_valence_interaction'] = (
                interactions_with_features['rating'] * interactions_with_features['valence']
            )
    
    logger.info(f"创建了 {len(interactions_with_features.columns) - len(interactions_df.columns)} 个交互特征")
    return interactions_with_features 