#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSD（Million Song Dataset）模型训练主脚本
提供训练MSD数据集的入口点
"""

import os
import sys
import logging
import argparse
import time
import numpy as np
import pickle
import pandas as pd
import h5py
from collections import defaultdict, Counter
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('msd_training.log')
    ]
)
logger = logging.getLogger('msd_model')

# 导入处理模块
try:
    from process_msd_data import main as process_data
except ImportError:
    # 如果导入失败，尝试调整路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(current_dir)
    sys.path.append(parent_dir)
    from process_msd_data import main as process_data

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练MSD数据集推荐模型')
    
    parser.add_argument('--h5_dir', type=str, required=True,
                        help='H5文件所在目录路径')
    parser.add_argument('--triplets_file', type=str, required=True,
                        help='triplets文件路径（用户-歌曲-播放计数）')
    parser.add_argument('--output', type=str, default='../models/msd_model.pkl',
                        help='输出模型文件路径')
    parser.add_argument('--max_songs_per_user', type=int, default=20,
                        help='每个用户最多处理的歌曲数量')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='用户最少需要的交互数')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--random_state', type=int, default=42,
                        help='随机种子')
    
    return parser.parse_args()

def load_h5_data(h5_file):
    """从H5文件加载歌曲数据"""
    logger.info(f"加载H5文件: {h5_file}")
    try:
        with h5py.File(h5_file, 'r') as f:
            # 提取歌曲ID和特征
            song_ids = []
            features = []
            titles = []
            artists = []
            
            # 假设H5文件有特定结构，根据实际文件结构调整这部分代码
            for song_id in f.keys():
                song_ids.append(song_id)
                
                # 特征数据
                if 'analysis' in f[song_id] and 'features' in f[song_id]['analysis']:
                    feat = f[song_id]['analysis']['features'][:]
                else:
                    feat = np.zeros(10)  # 使用默认特征
                features.append(feat)
                
                # 元数据
                if 'metadata' in f[song_id]:
                    metadata = f[song_id]['metadata']
                    if 'song_title' in metadata:
                        titles.append(metadata['song_title'][()])
                    else:
                        titles.append("未知歌曲")
                    
                    if 'artist_name' in metadata:
                        artists.append(metadata['artist_name'][()])
                    else:
                        artists.append("未知艺术家")
                else:
                    titles.append("未知歌曲")
                    artists.append("未知艺术家")
            
            # 转换为数组
            features = np.array(features)
            
            # 创建歌曲信息DataFrame
            songs_df = pd.DataFrame({
                'song_id': song_ids,
                'title': titles,
                'artist': artists
            })
            
            return songs_df, features
            
    except Exception as e:
        logger.error(f"加载H5文件时出错: {e}")
        raise

def load_triplets(triplets_file, sample_size=None):
    """加载用户-歌曲-播放次数三元组数据"""
    logger.info(f"加载三元组文件: {triplets_file}")
    
    try:
        if not os.path.exists(triplets_file):
            logger.warning(f"三元组文件不存在: {triplets_file}")
            logger.info("生成模拟数据以演示")
            
            # 生成模拟数据
            user_ids = [f"user_{i}" for i in range(1000)]
            song_ids = [f"song_{i}" for i in range(500)]
            
            np.random.seed(42)
            interactions = []
            for i in range(sample_size or 50000):
                user_id = np.random.choice(user_ids)
                song_id = np.random.choice(song_ids)
                play_count = np.random.randint(1, 100)
                interactions.append((user_id, song_id, play_count))
                
            triplets_df = pd.DataFrame(interactions, columns=['user_id', 'song_id', 'play_count'])
        else:
            # 读取真实的三元组文件
            # 假设文件格式为: user_id, song_id, play_count，以逗号分隔
            triplets_df = pd.read_csv(triplets_file, header=None, names=['user_id', 'song_id', 'play_count'])
            
            # 如果需要的话，对数据进行采样
            if sample_size and len(triplets_df) > sample_size:
                triplets_df = triplets_df.sample(n=sample_size, random_state=42)
        
        logger.info(f"三元组数据大小: {len(triplets_df)}")
        return triplets_df
        
    except Exception as e:
        logger.error(f"加载三元组数据时出错: {e}")
        raise

def train_model(triplets_df, n_factors=100, n_epochs=20):
    """使用Surprise库训练SVD模型"""
    logger.info("开始训练协同过滤模型")
    
    # 创建Surprise数据集
    reader = Reader(rating_scale=(0, 1000))  # 假设播放次数范围为0-1000
    data = Dataset.load_from_df(triplets_df[['user_id', 'song_id', 'play_count']], reader)
    
    # 使用SVD算法
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, verbose=True)
    
    # 交叉验证以评估模型
    logger.info("执行交叉验证...")
    cv_results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    
    logger.info(f"交叉验证结果: {cv_results}")
    
    # 使用全部数据训练最终模型
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    
    logger.info("模型训练完成")
    return algo

def build_recommendation_data(model, songs_df, triplets_df):
    """构建用于推荐的数据结构"""
    logger.info("构建推荐数据结构")
    
    # 创建用户播放历史
    user_history = defaultdict(list)
    for _, row in triplets_df.iterrows():
        user_history[row['user_id']].append(row['song_id'])
    
    # 创建歌曲相似度矩阵（基于模型潜在因子）
    song_factors = {}
    for song_id in songs_df['song_id']:
        if song_id in model.trainset.to_inner_iid:
            inner_id = model.trainset.to_inner_iid(song_id)
            song_factors[song_id] = model.qi[inner_id]
    
    # 构建song_id到歌曲信息的映射
    song_metadata = songs_df.set_index('song_id').to_dict(orient='index')
    
    # 创建推荐数据结构
    recommendation_data = {
        'user_history': dict(user_history),
        'song_factors': song_factors,
        'song_metadata': song_metadata,
        'model': model
    }
    
    return recommendation_data

def parse_triplets_file(triplets_file, max_songs_per_user=20):
    """
    解析用户-歌曲-播放次数三元组文件
    
    参数:
        triplets_file: triplets文件路径
        max_songs_per_user: 每个用户最多保留的歌曲数
        
    返回:
        users_songs: 用户听过的歌曲字典，格式为 {user_id: [song_id1, song_id2, ...]}
        song_play_count: 歌曲播放次数字典，格式为 {song_id: play_count}
        triplets_df: triplets数据的DataFrame
    """
    logger.info(f"解析triplets文件: {triplets_file}")
    
    users_songs = defaultdict(list)
    song_play_count = Counter()
    
    # 使用分块读取处理大文件
    chunk_size = 100000
    reader = pd.read_csv(triplets_file, sep='\t', header=None, 
                          names=['user_id', 'song_id', 'play_count'],
                          chunksize=chunk_size)
    
    all_chunks = []
    
    for i, chunk in enumerate(reader):
        all_chunks.append(chunk)
        logger.info(f"已处理 {(i+1)*chunk_size} 行")
    
    triplets_df = pd.concat(all_chunks)
    
    # 统计歌曲播放次数
    song_counts = triplets_df['song_id'].value_counts()
    song_play_count.update(song_counts.to_dict())
    
    # 对每个用户，按播放次数排序并最多保留max_songs_per_user首歌
    for user_id, user_data in triplets_df.groupby('user_id'):
        top_songs = user_data.sort_values('play_count', ascending=False)
        top_songs = top_songs.head(max_songs_per_user)
        users_songs[user_id] = top_songs['song_id'].tolist()
    
    logger.info(f"共有 {len(users_songs)} 位用户")
    logger.info(f"共有 {len(song_play_count)} 首歌曲")
    logger.info(f"共有 {len(triplets_df)} 条播放记录")
    
    return users_songs, song_play_count, triplets_df

def extract_song_metadata(h5_files_dir):
    """
    从H5文件中提取歌曲元数据
    
    参数:
        h5_files_dir: 包含H5文件的目录路径
        
    返回:
        song_metadata: 歌曲元数据字典，格式为 {song_id: {'title': title, 'artist': artist, ...}}
        song_features: 歌曲特征字典，格式为 {song_id: np.array([feature1, feature2, ...])}
    """
    logger.info(f"从目录提取歌曲元数据: {h5_files_dir}")
    
    song_metadata = {}
    song_features = {}
    
    # 遍历目录获取所有H5文件
    h5_files = []
    for root, dirs, files in os.walk(h5_files_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    logger.info(f"发现 {len(h5_files)} 个H5文件")
    
    # 处理每个H5文件
    for i, h5_file in enumerate(h5_files):
        if i % 100 == 0:
            logger.info(f"正在处理第 {i}/{len(h5_files)} 个H5文件")
        
        try:
            with h5py.File(h5_file, 'r') as f:
                # 提取歌曲ID
                song_id = f['metadata/songs'].attrs['song_id'].decode('utf-8')
                
                # 提取元数据
                metadata = {}
                metadata['title'] = f['metadata/songs'].attrs.get('title', b'').decode('utf-8')
                metadata['artist'] = f['metadata/songs'].attrs.get('artist_name', b'').decode('utf-8')
                metadata['release'] = f['metadata/songs'].attrs.get('release', b'').decode('utf-8')
                metadata['year'] = int(f['metadata/songs'].attrs.get('year', 0))
                
                # 提取特征
                if 'analysis/songs' in f:
                    features = []
                    # 添加音频特征
                    if 'analysis/segments_timbre' in f:
                        timbre = np.array(f['analysis/segments_timbre'])
                        timbre_avg = np.mean(timbre, axis=0)
                        features.extend(timbre_avg)
                    
                    # 可以添加更多特征...
                    
                    if features:
                        song_features[song_id] = np.array(features)
                
                song_metadata[song_id] = metadata
        except Exception as e:
            logger.warning(f"处理文件 {h5_file} 时出错: {e}")
    
    logger.info(f"已提取 {len(song_metadata)} 首歌曲的元数据")
    logger.info(f"已提取 {len(song_features)} 首歌曲的特征")
    
    return song_metadata, song_features

def train_svd_model(triplets_df, test_size=0.2):
    """
    训练SVD推荐模型
    
    参数:
        triplets_df: triplets数据的DataFrame
        test_size: 测试集比例
        
    返回:
        model: 训练好的SVD模型
        trainset: 训练集
        testset: 测试集
    """
    logger.info("准备训练数据")
    
    # 创建surprise数据集
    reader = Reader(rating_scale=(0, 1000))  # 假设最大播放次数为1000
    data = Dataset.load_from_df(triplets_df[['user_id', 'song_id', 'play_count']], reader)
    
    # 划分训练集和测试集
    trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
    
    # 训练SVD模型
    logger.info("开始训练SVD模型")
    start_time = time.time()
    
    model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
    model.fit(trainset)
    
    elapsed_time = time.time() - start_time
    logger.info(f"模型训练完成，耗时 {elapsed_time:.2f} 秒")
    
    # 评估模型
    logger.info("评估模型")
    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions)
    mae = accuracy.mae(predictions)
    
    logger.info(f"测试集RMSE: {rmse}")
    logger.info(f"测试集MAE: {mae}")
    
    return model, trainset, testset

def extract_song_factors(model, trainset):
    """
    从模型中提取歌曲隐因子
    
    参数:
        model: 训练好的SVD模型
        trainset: 训练集
        
    返回:
        song_factors: 歌曲隐因子字典，格式为 {song_id: np.array([factor1, factor2, ...])}
    """
    logger.info("从模型中提取歌曲隐因子")
    
    song_factors = {}
    
    # 获取所有歌曲ID
    all_songs = set(trainset._raw2inner_id_items.keys())
    
    # 提取每首歌曲的隐因子
    for song_id in all_songs:
        if song_id in trainset._raw2inner_id_items:
            song_inner_id = trainset.to_inner_iid(song_id)
            factors = model.qi[song_inner_id]
            song_factors[song_id] = factors
    
    logger.info(f"已提取 {len(song_factors)} 首歌曲的隐因子")
    
    return song_factors

def save_model(model, users_songs, song_factors, song_metadata, output_path):
    """
    保存模型和相关数据
    
    参数:
        model: 训练好的SVD模型
        users_songs: 用户听过的歌曲字典
        song_factors: 歌曲隐因子字典
        song_metadata: 歌曲元数据字典
        output_path: 输出文件路径
    """
    logger.info(f"保存模型到: {output_path}")
    
    # 创建输出目录（如果不存在）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备保存的数据
    data_to_save = {
        'model': model,
        'user_history': users_songs,
        'song_factors': song_factors,
        'song_metadata': song_metadata
    }
    
    # 保存数据
    with open(output_path, 'wb') as f:
        pickle.dump(data_to_save, f)
    
    logger.info("模型和数据保存成功")

def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # 加载数据
    songs_df, features = load_h5_data(args.h5_file)
    triplets_df = load_triplets(args.triplets_file, args.sample_size)
    
    # 训练模型
    model = train_model(triplets_df, args.n_factors, args.n_epochs)
    
    # 构建推荐数据
    recommendation_data = build_recommendation_data(model, songs_df, triplets_df)
    
    # 保存模型和数据
    logger.info(f"保存模型到: {args.model_path}")
    with open(args.model_path, 'wb') as f:
        pickle.dump(recommendation_data, f)
    
    # 保存处理后的数据
    songs_df.to_csv(os.path.join(args.output_dir, 'songs.csv'), index=False)
    np.save(os.path.join(args.output_dir, 'features.npy'), features)
    triplets_df.to_csv(os.path.join(args.output_dir, 'triplets.csv'), index=False)
    
    logger.info("处理完成，所有数据和模型已保存")

    # 处理triplets文件
    users_songs, song_play_count, triplets_df = parse_triplets_file(
        args.triplets_file, args.max_songs_per_user)
    
    # 提取歌曲元数据
    song_metadata, song_features = extract_song_metadata(args.h5_file)
    
    # 训练推荐模型
    model, trainset, testset = train_svd_model(triplets_df, args.test_size)
    
    # 提取歌曲隐因子
    song_factors = extract_song_factors(model, trainset)
    
    # 保存模型和数据
    save_model(model, users_songs, song_factors, song_metadata, args.model_path)
    
    logger.info("处理完成")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 