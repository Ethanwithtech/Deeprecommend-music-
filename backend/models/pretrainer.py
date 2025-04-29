#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合推荐模型预训练脚本 - 使用MSD数据集
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
from tqdm import tqdm
import random

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.msd_processor import MSDDataProcessor
from backend.models.hybrid_recommender import HybridRecommender

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入TensorFlow并检查GPU可用性
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
    # 检查GPU可用性
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_info = f"找到 {len(gpus)} 个 GPU: "
        for gpu in gpus:
            gpu_info += f"{gpu.name} "
        logger.info(gpu_info)
        # 设置内存增长
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            else:
        logger.info("未找到 GPU，将使用 CPU 进行训练")
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("未安装 TensorFlow，无法训练深度学习模型")

def load_processed_data(data_dir):
    """从处理好的MSD数据目录加载数据
    
    参数:
        data_dir: 包含处理好的数据文件的目录
        
    返回:
        包含songs, interactions, audio_features和user_features的字典
    """
    logger.info(f"从 {data_dir} 加载处理好的数据...")
    
    # 尝试加载不同格式的文件
    def try_load_file(basename):
        # 先尝试parquet格式
        parquet_path = os.path.join(data_dir, f"{basename}.parquet")
        if os.path.exists(parquet_path):
            try:
                return pd.read_parquet(parquet_path)
            except Exception as e:
                logger.warning(f"无法加载parquet文件 {parquet_path}: {e}")
        
        # 再尝试csv格式
        csv_path = os.path.join(data_dir, f"{basename}.csv")
        if os.path.exists(csv_path):
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                logger.warning(f"无法加载csv文件 {csv_path}: {e}")
        
        logger.error(f"找不到有效的 {basename} 数据文件")
        return None
    
    # 加载各种数据文件
    songs = try_load_file("songs")
    interactions = try_load_file("interactions")
    audio_features = try_load_file("audio_features")
    user_features = try_load_file("user_features")
    
    # 检查是否所有必要的数据都加载成功
    if songs is None or interactions is None:
        logger.error("缺少必要的歌曲或交互数据，无法继续")
        return None
    
    # 检查交互数据的列并确保存在必要的列
    logger.info(f"交互数据列: {interactions.columns.tolist()}")
    
    # 确保交互数据中有必要的列
    required_cols = ['user_id', 'song_id']
    for col in required_cols:
        if col not in interactions.columns:
            logger.error(f"交互数据中缺少必要的列: {col}")
            return None
    
    # 如果没有rating列但有其他可能的评分列，尝试重命名
    if 'rating' not in interactions.columns:
        if 'plays' in interactions.columns:
            # 如果有plays列但没有rating列，则从plays创建rating
            logger.info("从plays列创建rating列")
            interactions['rating'] = interactions['plays'].apply(
                lambda x: min(5, max(1, int(np.log2(x) + 1)))
            )
        elif 'score' in interactions.columns:
            logger.info("将score列重命名为rating列")
            interactions['rating'] = interactions['score']
    
    # 如果没有plays列但是需要计算用户特征
    if user_features is None and 'user_id' in interactions.columns:
        logger.info("从交互数据创建简单的用户特征...")
        
        # 确定要聚合的列
        agg_dict = {'song_id': 'nunique'}
        
        # 添加可选列的聚合方式
        if 'rating' in interactions.columns:
            agg_dict['rating'] = ['count', 'mean', 'std', 'max']
        
        # 添加plays列的聚合方式(如果存在)
        if 'plays' in interactions.columns:
            agg_dict['plays'] = ['count', 'mean', 'std', 'max']
        
        # 执行聚合操作
        user_stats = interactions.groupby('user_id').agg(agg_dict)
        
        # 整理列名
        if isinstance(user_stats.columns, pd.MultiIndex):
            user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
        
        user_features = user_stats.fillna(0)
    
    # 如果没有加载到音频特征，尝试从songs中提取
    if audio_features is None and 'tempo' in songs.columns:
        logger.info("从歌曲数据中提取音频特征...")
        feature_cols = ['tempo', 'loudness', 'duration', 'key', 'mode', 'energy_ratio', 'tempo_norm']
        # 保留存在的列
        valid_cols = [col for col in feature_cols if col in songs.columns]
        audio_features = songs[valid_cols].copy()
        audio_features.index = songs['song_id'] if 'song_id' in songs.columns else songs.index
    
    logger.info(f"成功加载数据: {len(songs)} 首歌曲, {len(interactions)} 条交互")
    
    return {
        'songs': songs,
        'interactions': interactions,
        'audio_features': audio_features,
        'user_features': user_features
    }

def create_sample_data(n_songs=500, n_users=100, n_interactions=2000):
    """创建样本数据用于测试"""
    logger.info(f"创建样本数据: {n_songs} 首歌曲, {n_users} 用户, {n_interactions} 互动")
    
    # 生成歌曲数据
    song_ids = [f'S{i:06d}' for i in range(n_songs)]
    songs = pd.DataFrame({
        'song_id': song_ids,
        'title': [f'Song Title {i}' for i in range(n_songs)],
        'artist_name': [f'Artist {i % 50}' for i in range(n_songs)],
        'duration': np.random.uniform(120, 300, n_songs),
        'tempo': np.random.uniform(60, 180, n_songs),
        'loudness': np.random.uniform(-20, 0, n_songs),
        'key': np.random.randint(0, 12, n_songs),
        'mode': np.random.randint(0, 2, n_songs),
        'energy_ratio': np.random.uniform(0, 1, n_songs),
        'tempo_norm': np.random.uniform(0, 1, n_songs)
    })
    
    # 生成用户数据
    user_ids = [f'U{i:06d}' for i in range(n_users)]
    
    # 生成交互数据
    interactions = []
    for _ in range(n_interactions):
        user_id = random.choice(user_ids)
        song_id = random.choice(song_ids)
        plays = int(np.random.exponential(5)) + 1
        rating = min(5, max(1, int(plays * 0.7 + np.random.normal(0, 0.5))))
        interactions.append({
            'user_id': user_id,
            'song_id': song_id,
            'plays': plays,
            'rating': rating
        })
    
    interactions_df = pd.DataFrame(interactions)
    
    # 处理重复项
    interactions_df = interactions_df.groupby(['user_id', 'song_id']).agg({
        'plays': 'sum',
        'rating': 'max'
    }).reset_index()
    
    # 生成音频特征
    audio_features = songs[['song_id', 'tempo', 'loudness', 'duration', 'key', 'mode', 'energy_ratio', 'tempo_norm']]
    
    # 生成用户特征
    user_features = []
    for user_id in user_ids:
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        if not user_interactions.empty:
            total_plays = user_interactions['plays'].sum()
            avg_plays = user_interactions['plays'].mean()
            std_plays = user_interactions['plays'].std() if len(user_interactions) > 1 else 0
            max_plays = user_interactions['plays'].max()
            unique_songs = len(user_interactions)
            
            user_features.append({
                'user_id': user_id,
                'total_plays': total_plays,
                'avg_plays': avg_plays,
                'std_plays': std_plays,
                'max_plays': max_plays,
                'unique_songs': unique_songs
            })
    
    user_features_df = pd.DataFrame(user_features)
    
    return songs, interactions_df, audio_features, user_features_df

def train_hybrid_recommender(interactions, audio_features, songs, user_features=None, train_deep_model=False, test_size=0.2):
    """训练混合推荐模型"""
    logger.info("开始训练混合推荐模型...")
        
        # 分割训练集和测试集
    train_interactions, test_interactions = train_test_split(
        interactions, test_size=test_size, random_state=42
    )
    
    logger.info(f"训练集: {len(train_interactions)} 条记录, 测试集: {len(test_interactions)} 条记录")
    
    # 创建并训练混合推荐模型
    recommender = HybridRecommender()
    
    start_time = time.time()
    recommender.train(
        train_interactions, 
        audio_features, 
        songs, 
        user_features=user_features,
        train_deep_model=train_deep_model
    )
    training_time = time.time() - start_time
    
    logger.info(f"模型训练完成，耗时: {training_time:.2f} 秒")
    
    # 为测试用户生成推荐
    if not test_interactions.empty:
        test_user = test_interactions['user_id'].iloc[0]
        recommendations = recommender.recommend(test_user, top_n=5)
        
        logger.info(f"测试用户 {test_user} 的前5个推荐:")
        for i, rec in enumerate(recommendations, 1):
            song_id = rec['song_id']
            score = rec['score']
            cf_score = rec['cf_score']
            content_score = rec['content_score']
            context_score = rec['context_score']
            deep_score = rec.get('deep_score', 0.0)
            
            song_info = songs[songs['song_id'] == song_id]
            if not song_info.empty:
                title = song_info['title'].iloc[0]
                artist = song_info['artist_name'].iloc[0]
                logger.info(f"{i}. {title} - {artist} (分数: {score:.4f}, CF: {cf_score:.4f}, 内容: {content_score:.4f}, 上下文: {context_score:.4f}, 深度: {deep_score:.4f})")
    
    return recommender

def train_and_save_model(output_dir, train_deep_model=False):
    """训练并保存模型"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建样本数据
    songs, interactions, audio_features, user_features = create_sample_data()
    
    # 训练模型
    recommender = train_hybrid_recommender(
        interactions, 
        audio_features, 
        songs, 
        user_features=user_features,
        train_deep_model=train_deep_model
    )
    
    # 保存模型
    model_path = os.path.join(output_dir, 'hybrid_recommender.pkl')
    recommender.save_model(model_path)
    logger.info(f"模型已保存至: {model_path}")

def main():
    parser = argparse.ArgumentParser(description="MSD数据处理与混合推荐系统预训练工具")
    parser.add_argument("--msd_path", type=str, help="MSD数据路径")
    parser.add_argument("--h5_file", type=str, help="MSD的h5文件路径")
    parser.add_argument("--triplets_file", type=str, help="MSD的triplets文件路径")
    parser.add_argument("--processed_data_dir", type=str, help="已处理的MSD数据目录")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="处理后数据的输出目录")
    parser.add_argument("--model_path", type=str, default="models/hybrid_model.pkl", help="模型保存路径")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据")
    parser.add_argument("--use_processed", action="store_true", help="使用已处理好的数据训练模型")
    parser.add_argument("--chunk_limit", type=int, default=None, help="处理的数据块数量限制")
    parser.add_argument("--train_deep_models", action="store_true", help="是否训练深度学习模型")
    args = parser.parse_args()
    
    # 打印环境信息
    print("="*50)
    print("MSD数据处理与混合推荐系统预训练工具")
    print("="*50)
    
    # 设置默认的MSD文件路径
    if args.msd_path and not args.h5_file:
        args.h5_file = os.path.join(args.msd_path, "msd_summary_file.h5")
    if args.msd_path and not args.triplets_file:
        args.triplets_file = os.path.join(args.msd_path, "train_triplets.txt")
    
    # 确保模型目录存在
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # 如果指定了创建示例数据
    if args.create_sample:
        print("创建示例数据...")
        data = create_sample_data()
        print("训练示例模型...")
        train_and_save_model(args.output_dir, args.train_deep_models)
        return
    
    # 如果指定使用已处理的数据
    if args.use_processed:
        data_dir = args.processed_data_dir or args.output_dir
        print(f"使用已处理的数据 {data_dir} 训练模型...")
        data = load_processed_data(data_dir)
        if data:
            print("训练模型...")
            train_and_save_model(args.output_dir, args.train_deep_models)
        return
    
    # 处理MSD数据
    if args.h5_file and args.triplets_file:
        print(f"处理MSD数据...")
        processor = MSDDataProcessor(output_dir=args.output_dir)
        data = processor.process_msd_data(args.h5_file, args.triplets_file, chunk_limit=args.chunk_limit)
        
        print("训练模型...")
        train_and_save_model(args.output_dir, args.train_deep_models)
    else:
        print("错误: 需要提供MSD数据路径或创建示例数据")
        parser.print_help()

if __name__ == "__main__":
    main() 