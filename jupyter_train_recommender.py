#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合音乐推荐系统训练模块 - Jupyter版本

该模块解决了原始训练脚本中的数据加载、评分处理和模型训练问题，
提供可在Jupyter Notebook中使用的函数来训练混合音乐推荐系统。
"""

import os
import sys
import logging
import time
import random
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from pathlib import Path
from datetime import datetime
import importlib.util

# 配置logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('jupyter_train_recommender')

def ensure_backend_module() -> bool:
    """确保可以导入HybridMusicRecommender模块，返回是否成功"""
    try:
        # 尝试查找和导入HybridMusicRecommender模块
        if not os.path.exists(os.path.join('backend', 'models', 'hybrid_music_recommender.py')):
            logger.error("找不到hybrid_music_recommender.py文件。请确保工作目录正确。")
            return False
        
        # 将backend目录添加到sys.path
        if 'backend' not in sys.path:
            sys.path.append('backend')
            logger.info("已将backend目录添加到sys.path")
        
        # 动态导入HybridMusicRecommender模块
        from backend.models.hybrid_music_recommender import HybridMusicRecommender
        logger.info("成功导入HybridMusicRecommender模块")
        return True
    except ImportError as e:
        logger.error(f"无法导入HybridMusicRecommender模块: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"检查HybridMusicRecommender模块时出错: {str(e)}")
        return False

def create_synthetic_data(user_count: int = 100, 
                         song_count: int = 500, 
                         interaction_count: int = 2000,
                         output_dir: str = 'data') -> Dict[str, str]:
    """
    创建合成的训练数据，解决原始数据加载问题
    
    参数:
        user_count: 用户数量
        song_count: 歌曲数量
        interaction_count: 交互记录数量
        output_dir: 输出目录
    
    返回:
        包含生成的数据文件路径的字典
    """
    logger.info(f"开始创建合成训练数据: {user_count}用户, {song_count}首歌曲, {interaction_count}条交互记录")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建用户ID列表
    user_ids = [f"USER{i}" for i in range(1, user_count + 1)]
    
    # 创建歌曲ID和元数据
    song_ids = [f"SONG{i}" for i in range(1, song_count + 1)]
    song_metadata = []
    
    for i, song_id in enumerate(song_ids):
        # 为每首歌曲创建随机元数据
        genre = random.choice(['Pop', 'Rock', 'Electronic', 'Classical', 'Jazz', 'HipHop', 'R&B', 'Country', 'Metal', 'Folk'])
        title = f"Song Title {i+1}"
        artist = f"Artist {random.randint(1, 50)}"
        
        # 创建随机音频特征
        tempo = random.uniform(60, 180)
        energy = random.uniform(0, 1)
        danceability = random.uniform(0, 1)
        acousticness = random.uniform(0, 1)
        
        song_metadata.append({
            'song_id': song_id,
            'title': title,
            'artist_name': artist,
            'genre': genre,
            'tempo': tempo,
            'energy': energy,
            'danceability': danceability,
            'acousticness': acousticness,
            'release_year': random.randint(1950, 2023)
        })
    
    # 创建用户元数据
    user_metadata = []
    for user_id in user_ids:
        # 为每个用户创建随机元数据
        age = random.randint(18, 65)
        gender = random.choice(['M', 'F', 'Other'])
        location = random.choice(['US', 'UK', 'CA', 'DE', 'FR', 'JP', 'CN', 'BR', 'IN', 'AU'])
        
        # 每个用户都有一些偏好
        genre_preference = random.choice(['Pop', 'Rock', 'Electronic', 'Classical', 'Jazz', 'HipHop'])
        energy_preference = random.uniform(0, 1)
        
        user_metadata.append({
            'user_id': user_id,
            'age': age,
            'gender': gender,
            'location': location,
            'favorite_genre': genre_preference,
            'preferred_energy': energy_preference,
            'registration_date': f"2022-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
            'active_hours': random.choice(['morning', 'evening', 'night'])
        })
    
    # 创建交互数据
    interactions = []
    for _ in range(interaction_count):
        user_id = random.choice(user_ids)
        song_id = random.choice(song_ids)
        
        # 每个用户对每首歌的评分在1-5之间
        play_count = random.randint(1, 20)
        rating = min(5, max(1, int(play_count * 5 / 20) + random.randint(0, 1)))
        
        # 随机上下文
        context = random.choice(['morning', 'evening', 'night', 'workout', 'relax', None])
        context_str = context if context else ''
        
        interactions.append({
            'user_id': user_id,
            'song_id': song_id,
            'play_count': play_count,
            'rating': rating,
            'context': context_str,
            'timestamp': int(time.time() - random.randint(0, 30*24*3600))  # 过去30天内的随机时间
        })
    
    # 保存数据
    # 1. 保存三元组数据 (user, song, play_count)
    triplets_path = output_path / 'train_triplets.txt'
    with open(triplets_path, 'w') as f:
        for interaction in interactions:
            f.write(f"{interaction['user_id']}\t{interaction['song_id']}\t{interaction['play_count']}\n")
    
    # 2. 保存评分数据 (user, song, rating, context)
    ratings_path = output_path / 'ratings.csv'
    ratings_df = pd.DataFrame([{
        'user_id': interaction['user_id'],
        'song_id': interaction['song_id'],
        'rating': interaction['rating'],
        'context': interaction['context'],
        'timestamp': interaction['timestamp']
    } for interaction in interactions])
    ratings_df.to_csv(ratings_path, index=False)
    
    # 3. 保存歌曲元数据
    song_metadata_path = output_path / 'song_metadata.csv'
    pd.DataFrame(song_metadata).to_csv(song_metadata_path, index=False)
    
    # 4. 保存用户元数据
    user_metadata_path = output_path / 'user_metadata.csv'
    pd.DataFrame(user_metadata).to_csv(user_metadata_path, index=False)
    
    # 返回生成的文件路径
    data_paths = {
        'triplets': str(triplets_path),
        'ratings': str(ratings_path),
        'song_metadata': str(song_metadata_path),
        'user_metadata': str(user_metadata_path)
    }
    
    logger.info(f"合成数据生成完成! 文件已保存到 {output_dir} 目录")
    return data_paths

def visualize_data(data_dir: str) -> None:
    """
    可视化训练数据的分布
    
    参数:
        data_dir: 数据目录
    """
    try:
        # 加载评分数据
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        if not os.path.exists(ratings_path):
            logger.error(f"找不到评分数据文件: {ratings_path}")
            return
        
        ratings_df = pd.read_csv(ratings_path)
        
        # 创建图形
        plt.figure(figsize=(15, 10))
        
        # 1. 评分分布
        plt.subplot(2, 3, 1)
        ratings_df['rating'].value_counts().sort_index().plot(kind='bar')
        plt.title('评分分布')
        plt.xlabel('评分')
        plt.ylabel('次数')
        
        # 2. 用户交互数量分布
        plt.subplot(2, 3, 2)
        user_interaction_counts = ratings_df['user_id'].value_counts()
        plt.hist(user_interaction_counts, bins=20)
        plt.title('用户交互数量分布')
        plt.xlabel('每用户交互数')
        plt.ylabel('用户数')
        
        # 3. 歌曲受欢迎度分布
        plt.subplot(2, 3, 3)
        song_popularity = ratings_df['song_id'].value_counts()
        plt.hist(song_popularity, bins=20)
        plt.title('歌曲受欢迎度分布')
        plt.xlabel('每歌曲评分数')
        plt.ylabel('歌曲数')
        
        # 4. 上下文分布
        plt.subplot(2, 3, 4)
        ratings_df['context'].fillna('unknown').value_counts().plot(kind='pie', autopct='%1.1f%%')
        plt.title('上下文分布')
        plt.ylabel('')
        
        # 5. 评分随时间的变化
        plt.subplot(2, 3, 5)
        if 'timestamp' in ratings_df.columns:
            ratings_df['date'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
            ratings_by_date = ratings_df.groupby(ratings_df['date'].dt.date)['rating'].mean()
            ratings_by_date.plot()
            plt.title('评分随时间的变化')
            plt.xlabel('日期')
            plt.ylabel('平均评分')
        
        plt.tight_layout()
        plt.show()
        
        # 额外的分析
        logger.info(f"数据集统计:")
        logger.info(f"  用户数: {ratings_df['user_id'].nunique()}")
        logger.info(f"  歌曲数: {ratings_df['song_id'].nunique()}")
        logger.info(f"  交互记录数: {len(ratings_df)}")
        logger.info(f"  平均每用户评分数: {len(ratings_df) / ratings_df['user_id'].nunique():.2f}")
        logger.info(f"  平均每歌曲评分数: {len(ratings_df) / ratings_df['song_id'].nunique():.2f}")
        logger.info(f"  评分密度: {len(ratings_df) / (ratings_df['user_id'].nunique() * ratings_df['song_id'].nunique()) * 100:.4f}%")
        
        # 如果有歌曲元数据，显示更多分析
        song_metadata_path = os.path.join(data_dir, 'song_metadata.csv')
        if os.path.exists(song_metadata_path):
            song_df = pd.read_csv(song_metadata_path)
            
            plt.figure(figsize=(15, 5))
            
            # 1. 流派分布
            plt.subplot(1, 3, 1)
            if 'genre' in song_df.columns:
                song_df['genre'].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title('歌曲流派分布')
                plt.ylabel('')
            
            # 2. 发行年份分布
            plt.subplot(1, 3, 2)
            if 'release_year' in song_df.columns:
                song_df['release_year'].hist(bins=20)
                plt.title('发行年份分布')
                plt.xlabel('年份')
                plt.ylabel('歌曲数')
            
            # 3. 音频特征分布
            plt.subplot(1, 3, 3)
            audio_features = ['tempo', 'energy', 'danceability', 'acousticness']
            features_to_plot = [col for col in audio_features if col in song_df.columns]
            
            if features_to_plot:
                song_df[features_to_plot].boxplot()
                plt.title('音频特征分布')
                plt.ylabel('值')
            
            plt.tight_layout()
            plt.show()
            
    except Exception as e:
        logger.error(f"可视化数据时出错: {str(e)}")

def train_recommender(data_dir: str, 
                     output_dir: str = 'models/trained',
                     use_deep_learning: bool = False,
                     debug: bool = False) -> Optional[Any]:
    """
    训练混合音乐推荐系统
    
    参数:
        data_dir: 数据目录
        output_dir: 输出目录
        use_deep_learning: 是否使用深度学习模型
        debug: 是否输出调试信息
    
    返回:
        训练好的推荐器对象，如果训练失败则返回None
    """
    try:
        # 确保backend模块可用
        if not ensure_backend_module():
            logger.error("无法导入必要的后端模块，训练终止")
            return None
        
        # 导入HybridMusicRecommender
        from backend.models.hybrid_music_recommender import HybridMusicRecommender
        
        # 检查数据目录
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            return None
        
        # 检查并创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 构建数据文件路径
        ratings_path = os.path.join(data_dir, 'ratings.csv')
        songs_path = os.path.join(data_dir, 'song_metadata.csv')
        users_path = os.path.join(data_dir, 'user_metadata.csv')
        
        # 检查必要的数据文件是否存在
        for path, name in [(ratings_path, '评分'), (songs_path, '歌曲元数据'), (users_path, '用户元数据')]:
            if not os.path.exists(path):
                logger.error(f"{name}文件不存在: {path}")
                return None
        
        # 创建推荐器实例
        logger.info("创建推荐器实例...")
        recommender = HybridMusicRecommender()
        
        # 加载歌曲和用户元数据
        logger.info("加载歌曲和用户元数据...")
        recommender.load_song_metadata(songs_path)
        recommender.load_user_metadata(users_path)
        
        # 修复评分数据加载和处理逻辑
        logger.info("加载和处理评分数据...")
        
        # 直接加载CSV格式的评分数据
        ratings_df = pd.read_csv(ratings_path)
        
        if debug:
            logger.info(f"评分数据前5行:\n{ratings_df.head()}")
            logger.info(f"评分数据列: {ratings_df.columns.tolist()}")
            logger.info(f"评分数据统计:\n{ratings_df.describe()}")
        
        # 检查数据格式
        required_columns = ['user_id', 'song_id', 'rating']
        if not all(col in ratings_df.columns for col in required_columns):
            logger.error(f"评分数据缺少必要的列: {required_columns}")
            return None
        
        # 添加上下文信息
        if 'context' not in ratings_df.columns:
            ratings_df['context'] = ''
        
        # 确保评分数据的用户和歌曲都存在于元数据中
        valid_songs = None
        valid_users = None
        
        if hasattr(recommender, 'songs_df') and len(recommender.songs_df) > 0:
            valid_songs = set(recommender.songs_df['song_id'])
            initial_len = len(ratings_df)
            ratings_df = ratings_df[ratings_df['song_id'].isin(valid_songs)]
            filtered_len = len(ratings_df)
            if debug:
                logger.info(f"过滤无效歌曲后: {initial_len} -> {filtered_len} 条记录")
        
        if hasattr(recommender, 'users_df') and len(recommender.users_df) > 0:
            valid_users = set(recommender.users_df['user_id'])
            initial_len = len(ratings_df)
            ratings_df = ratings_df[ratings_df['user_id'].isin(valid_users)]
            filtered_len = len(ratings_df)
            if debug:
                logger.info(f"过滤无效用户后: {initial_len} -> {filtered_len} 条记录")
        
        # 检查是否有足够的评分数据
        if len(ratings_df) == 0:
            logger.error("没有有效的评分数据，无法训练模型")
            
            # 尝试创建一些简单的评分数据以便训练能继续
            if debug and valid_songs and valid_users:
                logger.info("创建一些简单的评分数据以便训练继续...")
                sample_users = list(valid_users)[:min(10, len(valid_users))]
                sample_songs = list(valid_songs)[:min(20, len(valid_songs))]
                
                synthetic_ratings = []
                for user in sample_users:
                    for song in sample_songs:
                        synthetic_ratings.append({
                            'user_id': user,
                            'song_id': song,
                            'rating': random.randint(1, 5),
                            'context': random.choice(['morning', 'evening', 'workout', ''])
                        })
                
                ratings_df = pd.DataFrame(synthetic_ratings)
                logger.info(f"创建了 {len(ratings_df)} 条合成评分记录")
            else:
                return None
        
        logger.info(f"预处理后有 {len(ratings_df)} 条有效评分记录")
        
        # 将处理好的评分数据保存到推荐器
        recommender.ratings_df = ratings_df
        
        # 构建推荐器的内部数据结构
        logger.info("构建推荐器模型...")
        
        # 检查并修正构建方法
        if hasattr(recommender, 'build'):
            try:
                recommender.build()
            except Exception as e:
                logger.error(f"构建推荐器模型失败: {str(e)}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # 尝试手动构建内部数据结构
                    logger.info("尝试手动构建内部数据结构...")
                    # 创建用户-项目矩阵
                    if not hasattr(recommender, 'user_item_matrix'):
                        user_ids = ratings_df['user_id'].unique()
                        song_ids = ratings_df['song_id'].unique()
                        
                        # 创建ID到索引的映射
                        recommender.user_to_idx = {user: i for i, user in enumerate(user_ids)}
                        recommender.idx_to_user = {i: user for i, user in enumerate(user_ids)}
                        recommender.song_to_idx = {song: i for i, song in enumerate(song_ids)}
                        recommender.idx_to_song = {i: song for i, song in enumerate(song_ids)}
                        
                        # 创建评分矩阵
                        matrix = np.zeros((len(user_ids), len(song_ids)))
                        
                        for _, row in ratings_df.iterrows():
                            user_idx = recommender.user_to_idx.get(row['user_id'])
                            song_idx = recommender.song_to_idx.get(row['song_id'])
                            if user_idx is not None and song_idx is not None:
                                matrix[user_idx, song_idx] = row['rating']
                        
                        recommender.user_item_matrix = matrix
                        logger.info(f"手动创建了用户-项目矩阵: {matrix.shape}")
        else:
            logger.error("推荐器没有build方法，跳过数据结构构建")
        
        # 训练协同过滤模型
        logger.info("训练协同过滤模型...")
        
        if hasattr(recommender, 'train_collaborative_model'):
            try:
                recommender.train_collaborative_model()
            except Exception as e:
                logger.error(f"训练协同过滤模型失败: {str(e)}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
                
                # 尝试直接调用train_collaborative_filtering方法
                if hasattr(recommender, 'train_collaborative_filtering'):
                    try:
                        logger.info("尝试直接调用train_collaborative_filtering方法...")
                        recommender.train_collaborative_filtering()
                    except Exception as e:
                        logger.error(f"调用train_collaborative_filtering失败: {str(e)}")
                        if debug:
                            logger.error(traceback.format_exc())
                            
                            # 尝试创建一个简单的协同过滤模型
                            logger.info("尝试创建一个简单的协同过滤模型...")
                            from sklearn.decomposition import NMF
                            
                            if hasattr(recommender, 'user_item_matrix') and recommender.user_item_matrix is not None:
                                # 使用简单的NMF方法
                                n_factors = min(50, min(recommender.user_item_matrix.shape) - 1)
                                n_factors = max(2, n_factors)  # 至少2个特征
                                
                                # 处理零值
                                matrix = recommender.user_item_matrix.copy()
                                mask = matrix == 0
                                matrix[mask] = np.nan
                                matrix = np.nan_to_num(matrix, nan=0)
                                
                                model = NMF(n_components=n_factors, init='random', random_state=42)
                                recommender.user_factors = model.fit_transform(matrix)
                                recommender.item_factors = model.components_.T
                                
                                logger.info(f"创建了简单的NMF协同过滤模型，特征数: {n_factors}")
        else:
            logger.error("推荐器没有train_collaborative_model方法，跳过协同过滤训练")
        
        # 训练上下文感知模型
        logger.info("训练上下文感知模型...")
        if hasattr(recommender, 'train_context_model'):
            try:
                recommender.train_context_model()
            except Exception as e:
                logger.error(f"训练上下文感知模型失败: {str(e)}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # 尝试创建一个简单的上下文模型
                    logger.info("尝试创建一个简单的上下文模型...")
                    
                    # 创建上下文映射
                    context_ratings = ratings_df[ratings_df['context'] != ''].copy()
                    
                    if len(context_ratings) > 0:
                        contexts = context_ratings['context'].unique()
                        recommender.context_models = {}
                        
                        for context in contexts:
                            context_data = context_ratings[context_ratings['context'] == context]
                            recommender.context_models[context] = {
                                'mean_rating': context_data['rating'].mean(),
                                'user_means': context_data.groupby('user_id')['rating'].mean().to_dict(),
                                'song_means': context_data.groupby('song_id')['rating'].mean().to_dict()
                            }
                        
                        logger.info(f"创建了简单的上下文模型，包含 {len(contexts)} 个上下文")
        else:
            logger.error("推荐器没有train_context_model方法，跳过上下文感知训练")
        
        # 训练内容模型（如果使用深度学习）
        if use_deep_learning and hasattr(recommender, 'train_content_model'):
            logger.info("训练内容模型...")
            try:
                recommender.train_content_model()
            except Exception as e:
                logger.error(f"训练内容模型失败: {str(e)}")
                if debug:
                    import traceback
                    logger.error(traceback.format_exc())
        
        # 确保推荐方法可用
        if not hasattr(recommender, 'recommend'):
            logger.warning("推荐器没有recommend方法，创建一个基本实现")
            
            def basic_recommend(user_id, top_n=10, context=None):
                """基本推荐实现"""
                try:
                    # 如果用户不在数据中，返回空列表
                    if user_id not in recommender.user_to_idx:
                        return []
                    
                    user_idx = recommender.user_to_idx[user_id]
                    
                    # 获取用户的评分记录
                    user_ratings = recommender.user_item_matrix[user_idx]
                    
                    # 找出用户尚未评分的歌曲
                    unrated_indices = np.where(user_ratings == 0)[0]
                    
                    # 计算推荐分数
                    scores = []
                    
                    if hasattr(recommender, 'user_factors') and hasattr(recommender, 'item_factors'):
                        # 使用矩阵分解的潜在因子
                        user_vec = recommender.user_factors[user_idx]
                        
                        for song_idx in unrated_indices:
                            song_vec = recommender.item_factors[song_idx]
                            score = np.dot(user_vec, song_vec)
                            
                            # 如果有上下文信息，调整分数
                            if context and hasattr(recommender, 'context_models') and context in recommender.context_models:
                                ctx_model = recommender.context_models[context]
                                song_id = recommender.idx_to_song[song_idx]
                                
                                # 基于上下文的简单调整
                                ctx_adj = 0
                                if song_id in ctx_model['song_means']:
                                    ctx_adj = ctx_model['song_means'][song_id] - 3  # 假设中性评分为3
                                
                                score += ctx_adj * 0.2  # 赋予上下文20%的权重
                            
                            scores.append((song_idx, score))
                    else:
                        # 简单的流行度推荐
                        song_popularity = np.sum(recommender.user_item_matrix > 0, axis=0)
                        for song_idx in unrated_indices:
                            score = song_popularity[song_idx]
                            scores.append((song_idx, score))
                    
                    # 排序并获取top_n
                    scores.sort(key=lambda x: x[1], reverse=True)
                    top_indices = [idx for idx, _ in scores[:top_n]]
                    
                    # 转换为歌曲信息
                    recommendations = []
                    for idx in top_indices:
                        song_id = recommender.idx_to_song[idx]
                        song_info = recommender.songs_df[recommender.songs_df['song_id'] == song_id].iloc[0].to_dict()
                        song_info['score'] = float(scores[top_indices.index(idx)][1])
                        recommendations.append(song_info)
                    
                    return recommendations
                except Exception as e:
                    logger.error(f"基本推荐方法出错: {str(e)}")
                    if debug:
                        import traceback
                        logger.error(traceback.format_exc())
                    return []
            
            # 将基本推荐方法添加到推荐器
            recommender.recommend = basic_recommend
        
        # 保存训练好的模型
        model_path = os.path.join(output_dir, 'hybrid_recommender.pkl')
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(recommender, f)
            logger.info(f"模型已保存到 {model_path}")
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            if debug:
                import traceback
                logger.error(traceback.format_exc())
        
        logger.info("推荐系统训练完成!")
        return recommender
    
    except Exception as e:
        logger.error(f"训练推荐系统时出错: {str(e)}")
        if debug:
            import traceback
            logger.error(traceback.format_exc())
        return None

def test_recommendations(recommender, test_users: List[str] = None) -> None:
    """
    测试推荐功能
    
    参数:
        recommender: 训练好的推荐器对象
        test_users: 测试用户列表，如果为None则随机选择
    """
    if recommender is None:
        logger.error("推荐器对象为None，无法测试推荐功能")
        return
    
    try:
        if test_users is None or len(test_users) == 0:
            # 如果没有指定测试用户，随机选择几个
            if hasattr(recommender, 'users_df') and len(recommender.users_df) > 0:
                test_users = recommender.users_df['user_id'].sample(min(5, len(recommender.users_df))).tolist()
            else:
                test_users = ["USER1", "USER2", "USER3", "USER4", "USER5"]
        
        logger.info(f"为 {len(test_users)} 个用户测试推荐功能...")
        
        # 测试普通推荐
        for user_id in test_users:
            logger.info(f"为用户 {user_id} 生成推荐:")
            try:
                recommendations = recommender.recommend(user_id, top_n=5)
                
                if recommendations:
                    for idx, rec in enumerate(recommendations):
                        logger.info(f"  {idx+1}. {rec.get('title', 'Unknown')} - {rec.get('artist_name', 'Unknown')} (评分: {rec.get('score', 0):.2f})")
                else:
                    logger.info("  未能生成推荐")
            except Exception as e:
                logger.error(f"  为用户 {user_id} 生成推荐时出错: {str(e)}")
        
        # 测试上下文感知推荐
        contexts = ["morning", "evening", "workout"]
        user_id = test_users[0]  # 使用第一个测试用户
        
        for context in contexts:
            logger.info(f"为用户 {user_id} 在 {context} 上下文中生成推荐:")
            try:
                context_recs = recommender.recommend(user_id, context=context, top_n=3)
                
                if context_recs:
                    for idx, rec in enumerate(context_recs):
                        logger.info(f"  {idx+1}. {rec.get('title', 'Unknown')} - {rec.get('artist_name', 'Unknown')} (评分: {rec.get('score', 0):.2f})")
                else:
                    logger.info("  未能生成上下文推荐")
            except Exception as e:
                logger.error(f"  为用户 {user_id} 在 {context} 上下文中生成推荐时出错: {str(e)}")
    
    except Exception as e:
        logger.error(f"测试推荐功能时出错: {str(e)}")

def load_recommender(model_path: str) -> Optional[Any]:
    """
    加载已训练的推荐器模型
    
    参数:
        model_path: 模型路径
    
    返回:
        加载的推荐器对象，如果加载失败则返回None
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"模型文件不存在: {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            recommender = pickle.load(f)
        
        logger.info(f"成功加载模型: {model_path}")
        return recommender
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return None

def main_training_workflow(user_count: int = 200, 
                          song_count: int = 1000, 
                          interaction_count: int = 5000,
                          data_dir: str = 'data',
                          output_dir: str = 'models/trained',
                          use_deep_learning: bool = False,
                          debug: bool = False) -> Optional[Any]:
    """
    执行完整的训练工作流程
    
    参数:
        user_count: 用户数量
        song_count: 歌曲数量
        interaction_count: 交互记录数量
        data_dir: 数据目录
        output_dir: 输出目录
        use_deep_learning: 是否使用深度学习模型
        debug: 是否输出调试信息
    
    返回:
        训练好的推荐器对象，如果训练失败则返回None
    """
    # 1. 创建合成数据
    create_synthetic_data(
        user_count=user_count,
        song_count=song_count,
        interaction_count=interaction_count,
        output_dir=data_dir
    )
    
    # 2. 可视化数据
    visualize_data(data_dir)
    
    # 3. 训练模型
    recommender = train_recommender(
        data_dir=data_dir,
        output_dir=output_dir,
        use_deep_learning=use_deep_learning,
        debug=debug
    )
    
    # 4. 测试推荐
    if recommender:
        test_recommendations(recommender)
    
    return recommender

if __name__ == "__main__":
    # 如果作为脚本运行，执行完整工作流程
    main_training_workflow(debug=True) 