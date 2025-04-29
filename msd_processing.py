#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Million Song Dataset (MSD) 处理与训练脚本

本脚本用于处理MSD数据集并训练推荐模型，然后将模型保存到文件并自动下载到本地。
设计用于在Google Colab环境中运行。
"""

import os
import sys
import time
import pickle
import json
import numpy as np
import pandas as pd
import logging
import zipfile
import requests
from sklearn.model_selection import train_test_split
from pathlib import Path

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)

# 导入混合推荐模型
from backend.models.hybrid_music_recommender import HybridMusicRecommender
from backend.models.model_evaluator import ModelEvaluator

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量定义
MSD_DATA_DIR = "msd_data"
MSD_PROCESSED_DIR = os.path.join(MSD_DATA_DIR, "processed")
MSD_MODELS_DIR = os.path.join(MSD_DATA_DIR, "models")
MSD_EVALUATION_DIR = os.path.join(MSD_DATA_DIR, "evaluation")

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"创建目录: {directory}")

def download_data(from_drive=True, drive_msd_path=None, drive_triplets_path=None):
    """
    下载MSD数据集
    
    参数:
        from_drive: 是否从Google Drive下载数据
        drive_msd_path: Google Drive上MSD元数据文件路径
        drive_triplets_path: Google Drive上MSD用户-歌曲-播放次数三元组文件路径
    """
    logger.info("下载MSD数据集...")
    
    # 确保数据目录存在
    ensure_dir(MSD_DATA_DIR)
    
    try:
        if from_drive and drive_msd_path and drive_triplets_path:
            # 从Google Drive下载文件
            from google.colab import drive
            drive.mount('/content/drive')
            
            # 复制MSD元数据
            os.system(f"cp '{drive_msd_path}' '{MSD_DATA_DIR}/msd_metadata.csv'")
            
            # 复制三元组数据
            os.system(f"cp '{drive_triplets_path}' '{MSD_DATA_DIR}/msd_triplets.csv'")
            
            logger.info("从Google Drive成功下载MSD数据集")
        else:
            # 从公开源下载数据
            metadata_url = "https://cdn.jsdelivr.net/gh/taylorlu/Music-Recommender@master/data/msd_metadata.csv"
            triplets_url = "https://cdn.jsdelivr.net/gh/taylorlu/Music-Recommender@master/data/msd_triplets.csv"
            
            # 下载元数据
            logger.info(f"从 {metadata_url} 下载MSD元数据...")
            metadata_file = f"{MSD_DATA_DIR}/msd_metadata.csv"
            with requests.get(metadata_url, stream=True) as r:
                r.raise_for_status()
                with open(metadata_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            # 下载三元组数据
            logger.info(f"从 {triplets_url} 下载MSD三元组数据...")
            triplets_file = f"{MSD_DATA_DIR}/msd_triplets.csv"
            with requests.get(triplets_url, stream=True) as r:
                r.raise_for_status()
                with open(triplets_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            
            logger.info("成功下载MSD数据集")
        
        return True
    except Exception as e:
        logger.error(f"下载MSD数据集失败: {str(e)}")
        return False

def process_msd_data(sample_size=None, min_interactions=5):
    """
    处理MSD数据并生成训练文件
    
    参数:
        sample_size: 用户样本数量，None表示使用全部数据
        min_interactions: 最少交互次数
    """
    logger.info("处理MSD数据集...")
    
    # 确保处理目录存在
    ensure_dir(MSD_PROCESSED_DIR)
    
    try:
        # 加载元数据
        logger.info("加载歌曲元数据...")
        metadata_file = f"{MSD_DATA_DIR}/msd_metadata.csv"
        songs_df = pd.read_csv(metadata_file)
        logger.info(f"加载了 {len(songs_df)} 首歌曲元数据")
        
        # 加载三元组数据
        logger.info("加载用户-歌曲交互数据...")
        triplets_file = f"{MSD_DATA_DIR}/msd_triplets.csv"
        triplets_df = pd.read_csv(triplets_file)
        logger.info(f"加载了 {len(triplets_df)} 条用户-歌曲交互记录")
        
        # 对用户进行采样
        if sample_size and sample_size < triplets_df['user_id'].nunique():
            logger.info(f"对用户进行采样，目标用户数: {sample_size}")
            user_counts = triplets_df['user_id'].value_counts()
            # 选择交互次数多于最低阈值的用户
            qualified_users = user_counts[user_counts >= min_interactions].index.tolist()
            # 从合格用户中随机选择
            if len(qualified_users) > sample_size:
                qualified_users = np.random.choice(qualified_users, sample_size, replace=False)
            # 筛选这些用户的数据
            triplets_df = triplets_df[triplets_df['user_id'].isin(qualified_users)]
            logger.info(f"采样后数据: {len(triplets_df)} 条记录, {triplets_df['user_id'].nunique()} 名用户")
        
        # 将播放次数转换为评分
        logger.info("将播放次数转换为评分...")
        # 按用户分组，将播放次数归一化到1-5
        ratings = []
        for user, group in triplets_df.groupby('user_id'):
            # 如果用户只有一条记录，设为5分
            if len(group) == 1:
                group = group.copy()
                group['rating'] = 5.0
                ratings.append(group)
                continue
                
            # 否则归一化到1-5分
            plays = group['play_count'].values
            min_play = plays.min()
            max_play = plays.max()
            
            # 如果最大和最小相同，全部设为5分
            if max_play == min_play:
                normalized_plays = np.ones_like(plays) * 5.0
            else:
                # 否则进行线性归一化
                normalized_plays = 1.0 + (plays - min_play) * 4.0 / (max_play - min_play)
            
            group = group.copy()
            group['rating'] = normalized_plays
            ratings.append(group)
        
        # 合并归一化后的评分
        ratings_df = pd.concat(ratings, ignore_index=True)
        
        # 重命名列以符合推荐系统格式
        ratings_df = ratings_df.rename(columns={
            'user_id': 'user_id',
            'song_id': 'song_id',
            'play_count': 'count',
            'rating': 'rating'
        })
        
        # 准备歌曲元数据
        songs_df = songs_df.rename(columns={
            'song_id': 'song_id',
            'title': 'title',
            'artist_name': 'artist',
            'release': 'album',
            'year': 'year'
        })
        
        # 添加地区、语言和风格信息（用于内容特征训练）
        if 'genre' not in songs_df.columns:
            genres = ['rock', 'pop', 'electronic', 'folk', 'jazz', 'classical', 'hiphop', 'country']
            songs_df['genre'] = np.random.choice(genres, size=len(songs_df))
        
        if 'language' not in songs_df.columns:
            languages = ['english', 'spanish', 'french', 'japanese', 'korean', 'chinese']
            songs_df['language'] = np.random.choice(languages, size=len(songs_df))
        
        if 'region' not in songs_df.columns:
            regions = ['us', 'uk', 'europe', 'asia', 'latin', 'africa']
            songs_df['region'] = np.random.choice(regions, size=len(songs_df))
            
        # 创建一个简化的用户信息表（用于用户特征训练）
        user_ids = ratings_df['user_id'].unique()
        
        users_data = []
        for user_id in user_ids:
            # 随机生成用户特征
            gender = np.random.choice(['male', 'female'])
            age = np.random.randint(18, 65)
            country = np.random.choice(['US', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'China'])
            
            users_data.append({
                'user_id': user_id,
                'gender': gender,
                'age': age,
                'country': country
            })
        
        users_df = pd.DataFrame(users_data)
        
        # 保存处理后的数据
        logger.info("保存处理后的数据...")
        
        # 保存评分数据
        ratings_file = os.path.join(MSD_PROCESSED_DIR, "ratings.csv")
        ratings_df.to_csv(ratings_file, index=False)
        logger.info(f"评分数据已保存到 {ratings_file}")
        
        # 保存歌曲数据
        songs_file = os.path.join(MSD_PROCESSED_DIR, "songs.csv")
        songs_df.to_csv(songs_file, index=False)
        logger.info(f"歌曲数据已保存到 {songs_file}")
        
        # 保存用户数据
        users_file = os.path.join(MSD_PROCESSED_DIR, "users.csv")
        users_df.to_csv(users_file, index=False)
        logger.info(f"用户数据已保存到 {users_file}")
        
        # 返回处理后的数据
        return ratings_df, songs_df, users_df
    
    except Exception as e:
        logger.error(f"处理MSD数据失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def train_cf_model(triplets_df):
    """训练协同过滤模型并寻找最佳参数"""
    logger.info("训练协同过滤模型并寻找最佳参数...")
    
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import GridSearchCV
    
    # 使用Surprise库加载数据
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(triplets_df[['user_id', 'song_id', 'rating']], reader)
    
    # 定义参数网格
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.005, 0.01],
        'reg_all': [0.02, 0.1]
    }
    
    # 执行网格搜索
    gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)
    gs.fit(data)
    
    # 获取最佳参数
    best_params = gs.best_params['rmse']
    logger.info(f"最佳SVD参数: {best_params}")
    
    # 使用最佳参数训练模型
    best_model = SVD(
        n_factors=best_params['n_factors'],
        n_epochs=best_params['n_epochs'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all']
    )
    
    # 在全部数据上训练
    trainset = data.build_full_trainset()
    best_model.fit(trainset)
    
    # 返回训练好的模型
    return best_model

def train_hybrid_model(data_dir=MSD_PROCESSED_DIR, use_msd=True, test_size=0.2):
    """
    训练混合推荐模型
    
    参数:
        data_dir: 训练数据目录
        use_msd: 是否使用MSD数据集增强训练
        test_size: 测试集比例
    
    返回:
        训练好的混合推荐模型实例
    """
    logger.info("训练混合推荐模型...")
    
    # 确保模型目录存在
    ensure_dir(MSD_MODELS_DIR)
    
    try:
        # 初始化混合推荐系统
        recommender = HybridMusicRecommender(data_dir=data_dir, use_msd=use_msd)
        
        # 加载必要的训练数据
        recommender.load_data()
        
        # 预处理数据
        recommender.preprocess_data()
        
        # 如果使用MSD数据集，进行预训练
        if use_msd:
            logger.info("使用MSD数据集进行预训练...")
            recommender.pretrain_with_msd()
        
        # 训练各个组件模型
        logger.info("训练协同过滤模型...")
        recommender.train_collaborative_filtering()
        
        logger.info("训练内容特征模型...")
        recommender.train_content_based()
        
        logger.info("训练上下文感知模型...")
        recommender.train_context_aware()
        
        logger.info("训练深度学习模型...")
        recommender.train_deep_learning_models()
        
        # 保存训练好的模型
        model_path = os.path.join(MSD_MODELS_DIR, "hybrid_recommender.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(recommender, f)
        logger.info(f"混合推荐模型已保存到 {model_path}")
        
        return recommender
    
    except Exception as e:
        logger.error(f"训练混合推荐模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_hybrid_model(recommender=None, data_dir=MSD_PROCESSED_DIR, test_size=0.2):
    """
    评估混合推荐模型性能，并与单一算法进行对比分析
    
    参数:
        recommender: 已训练好的推荐器实例，如果为None则从文件加载
        data_dir: 数据目录
        test_size: 测试集比例
    """
    logger.info("评估混合推荐模型性能...")
    
    # 确保评估目录存在
    ensure_dir(MSD_EVALUATION_DIR)
    
    try:
        # 如果没有提供推荐器实例，则从文件加载
        if recommender is None:
            model_path = os.path.join(MSD_MODELS_DIR, "hybrid_recommender.pkl")
            if os.path.exists(model_path):
                logger.info(f"从文件加载混合推荐模型: {model_path}")
                with open(model_path, 'rb') as f:
                    recommender = pickle.load(f)
            else:
                logger.error(f"未找到混合推荐模型文件: {model_path}")
                return None
        
        # 初始化评估器
        evaluator = ModelEvaluator(data_dir=data_dir, test_size=test_size)
        
        # 加载数据
        logger.info("加载评估数据...")
        if not evaluator.load_data():
            logger.error("评估数据加载失败")
            return None
        
        # 设置要评估的推荐器
        logger.info("设置评估推荐器...")
        evaluator.recommender = recommender
        
        # 评估所有算法并比较性能
        logger.info("开始评估所有算法...")
        evaluator.evaluate_all_algorithms()
        
        # 执行t-test比较算法性能差异
        logger.info("执行t-test统计显著性测试...")
        evaluator.statistical_significance_test()
        
        # 返回评估结果
        return evaluator.results
    
    except Exception as e:
        logger.error(f"评估混合推荐模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MSD数据处理与混合推荐模型训练脚本")
    
    parser.add_argument('--download', action='store_true',
                        help='下载MSD数据集')
    parser.add_argument('--process', action='store_true',
                        help='处理MSD数据')
    parser.add_argument('--train', action='store_true',
                        help='训练混合推荐模型')
    parser.add_argument('--evaluate', action='store_true',
                        help='评估模型性能')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='用户样本数量')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    
    args = parser.parse_args()
    
    # 下载数据
    if args.download:
        download_data()
    
    # 处理数据
    ratings_df = None
    if args.process:
        ratings_df, songs_df, users_df = process_msd_data(sample_size=args.sample_size)
    
    # 训练模型
    recommender = None
    if args.train:
        recommender = train_hybrid_model(test_size=args.test_size)
    
    # 评估模型
    if args.evaluate:
        evaluate_hybrid_model(recommender, test_size=args.test_size)
    
    logger.info("脚本执行完毕")

if __name__ == "__main__":
    main() 