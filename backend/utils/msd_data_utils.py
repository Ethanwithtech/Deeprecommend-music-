#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSD数据处理工具

该模块提供用于处理Million Song Dataset的各种工具函数。
"""

import os
import logging
import pandas as pd
import numpy as np
import json
import pickle
from collections import defaultdict
from tqdm import tqdm
from scipy.sparse import csr_matrix

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('msd_data_utils')

def load_msd_triplets(file_path, subset=None):
    """
    加载MSD用户-歌曲-播放次数三元组数据
    
    参数:
        file_path: 数据文件路径
        subset: 加载的最大行数（None表示所有）
        
    返回:
        包含user_id, song_id, play_count列的DataFrame
    """
    logger.info(f"从 {file_path} 加载MSD三元组数据")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    # 根据文件扩展名确定加载方法
    if file_path.endswith('.txt') or file_path.endswith('.tsv'):
        # 加载原始三元组文件
        column_names = ['user_id', 'song_id', 'play_count']
        
        # 如果需要子集，使用nrows参数
        if subset:
            df = pd.read_csv(file_path, sep='\t', header=None, names=column_names, nrows=subset)
        else:
            df = pd.read_csv(file_path, sep='\t', header=None, names=column_names)
    
    elif file_path.endswith('.csv'):
        # 假设CSV文件有标题
        if subset:
            df = pd.read_csv(file_path, nrows=subset)
        else:
            df = pd.read_csv(file_path)
        
        # 确保列名一致
        df = df.rename(columns={
            'user': 'user_id',
            'song': 'song_id',
            'count': 'play_count',
            'plays': 'play_count'
        })
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    # 将play_count列转换为整数
    df['play_count'] = df['play_count'].astype(int)
    
    logger.info(f"加载了 {len(df)} 条记录")
    logger.info(f"唯一用户数: {df['user_id'].nunique()}")
    logger.info(f"唯一歌曲数: {df['song_id'].nunique()}")
    
    return df

def load_msd_metadata(file_path):
    """
    加载MSD歌曲元数据
    
    参数:
        file_path: 元数据文件路径
        
    返回:
        song_id到元数据的字典
    """
    logger.info(f"从 {file_path} 加载歌曲元数据")
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    metadata_dict = {}
    
    # 根据文件扩展名确定加载方法
    if file_path.endswith('.json'):
        # 加载JSON格式的元数据
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
    
    elif file_path.endswith('.csv'):
        # 加载CSV格式的元数据
        df = pd.read_csv(file_path)
        
        # 将DataFrame转换为字典
        for _, row in df.iterrows():
            song_id = row['song_id']
            # 删除song_id列，使用其余列作为元数据
            metadata = row.drop('song_id').to_dict()
            metadata_dict[song_id] = metadata
    
    elif file_path.endswith('.pickle') or file_path.endswith('.pkl'):
        # 加载pickle格式的元数据
        with open(file_path, 'rb') as f:
            metadata_dict = pickle.load(f)
    
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")
    
    logger.info(f"加载了 {len(metadata_dict)} 首歌曲的元数据")
    
    return metadata_dict

def preprocess_ratings(play_counts, min_count=1, max_count=None, scaling='log'):
    """
    将播放次数转换为评分
    
    参数:
        play_counts: 播放次数Series或列表
        min_count: 最小播放次数阈值
        max_count: 最大播放次数阈值（None表示不限制）
        scaling: 缩放方法，可以是'linear'、'log'或'binary'
        
    返回:
        转换后的评分Series或numpy数组
    """
    # 转换为numpy数组以便处理
    counts = np.array(play_counts)
    
    # 应用最小播放次数阈值
    counts[counts < min_count] = 0
    
    # 应用最大播放次数阈值（如果指定）
    if max_count is not None:
        counts[counts > max_count] = max_count
    
    # 根据缩放方法转换评分
    if scaling == 'binary':
        # 二元：只要播放过就是1，否则是0
        ratings = (counts > 0).astype(float)
    
    elif scaling == 'log':
        # 对数缩放：log(1 + play_count)
        # 首先将0播放次数设为0评分
        ratings = np.zeros_like(counts, dtype=float)
        nonzero_mask = counts > 0
        ratings[nonzero_mask] = np.log1p(counts[nonzero_mask])
    
    elif scaling == 'linear':
        # 线性缩放：直接使用播放次数
        ratings = counts.astype(float)
    
    else:
        raise ValueError(f"不支持的缩放方法: {scaling}")
    
    # 如果输入是pandas Series，返回Series；否则返回numpy数组
    if isinstance(play_counts, pd.Series):
        return pd.Series(ratings, index=play_counts.index)
    else:
        return ratings

def create_user_song_matrix(triplets_df, min_ratings=0):
    """
    从三元组数据创建用户-歌曲矩阵
    
    参数:
        triplets_df: 包含user_id, song_id, rating列的DataFrame
        min_ratings: 用户的最小评分数（过滤冷启动用户）
        
    返回:
        用户-歌曲矩阵和ID映射
    """
    logger.info("创建用户-歌曲矩阵")
    
    # 检查必要的列
    required_columns = ['user_id', 'song_id']
    if 'rating' not in triplets_df.columns:
        if 'play_count' in triplets_df.columns:
            # 如果没有rating列但有play_count列，使用play_count作为rating
            logger.info("使用play_count作为rating")
            triplets_df['rating'] = preprocess_ratings(triplets_df['play_count'])
        else:
            raise ValueError("缺少rating或play_count列")
    
    # 过滤冷启动用户
    if min_ratings > 0:
        user_counts = triplets_df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_ratings].index
        filtered_df = triplets_df[triplets_df['user_id'].isin(valid_users)]
        logger.info(f"过滤后的用户数: {len(valid_users)}")
    else:
        filtered_df = triplets_df
    
    # 创建用户和歌曲的映射
    unique_users = filtered_df['user_id'].unique()
    unique_songs = filtered_df['song_id'].unique()
    
    user_to_index = {user: i for i, user in enumerate(unique_users)}
    song_to_index = {song: i for i, song in enumerate(unique_songs)}
    
    index_to_user = {i: user for user, i in user_to_index.items()}
    index_to_song = {i: song for song, i in song_to_index.items()}
    
    # 创建矩阵的行、列和值
    row_indices = [user_to_index[user] for user in filtered_df['user_id']]
    col_indices = [song_to_index[song] for song in filtered_df['song_id']]
    ratings = filtered_df['rating'].values
    
    # 创建稀疏矩阵
    user_song_sparse = csr_matrix(
        (ratings, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_songs))
    )
    
    # 转换为密集矩阵
    user_song_matrix = user_song_sparse.toarray()
    
    logger.info(f"矩阵大小: {user_song_matrix.shape}")
    
    # 返回矩阵和映射
    mappings = {
        'user_to_index': user_to_index,
        'song_to_index': song_to_index,
        'index_to_user': index_to_user,
        'index_to_song': index_to_song
    }
    
    return user_song_matrix, mappings

def create_song_song_similarity(user_song_matrix, song_mapping, top_k=50):
    """
    计算歌曲之间的余弦相似度
    
    参数:
        user_song_matrix: 用户-歌曲矩阵
        song_mapping: 歌曲ID映射
        top_k: 每首歌保留的最相似歌曲数量
        
    返回:
        歌曲相似度字典，格式为 {song_id: [(similar_song_id, similarity), ...]}
    """
    logger.info("计算歌曲相似度")
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 获取歌曲的特征（用户-歌曲矩阵的转置）
    song_user_matrix = user_song_matrix.T
    
    # 计算余弦相似度
    logger.info("计算余弦相似度...")
    similarity_matrix = cosine_similarity(song_user_matrix)
    
    # 为每首歌找到最相似的歌曲
    song_similarities = {}
    
    logger.info(f"为每首歌找到前 {top_k} 个最相似的歌曲...")
    for i in tqdm(range(similarity_matrix.shape[0])):
        # 获取当前歌曲的相似度
        song_sim = similarity_matrix[i]
        
        # 获取当前歌曲的ID
        song_id = song_mapping['index_to_song'][i]
        
        # 将自己的相似度设为0
        song_sim[i] = 0
        
        # 找到最相似的top_k首歌曲
        most_similar_indices = np.argsort(song_sim)[-top_k:][::-1]
        most_similar_values = song_sim[most_similar_indices]
        
        # 转换为歌曲ID
        similar_songs = [
            (song_mapping['index_to_song'][idx], float(sim))
            for idx, sim in zip(most_similar_indices, most_similar_values)
            if sim > 0  # 只保留正相似度
        ]
        
        # 保存结果
        song_similarities[song_id] = similar_songs
    
    logger.info(f"计算了 {len(song_similarities)} 首歌曲的相似度")
    
    return song_similarities

def save_song_similarities(song_similarities, output_file):
    """
    保存歌曲相似度结果
    
    参数:
        song_similarities: 歌曲相似度字典
        output_file: 输出文件路径
    """
    logger.info(f"保存歌曲相似度到 {output_file}")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 根据文件扩展名选择保存格式
    if output_file.endswith('.pickle') or output_file.endswith('.pkl'):
        with open(output_file, 'wb') as f:
            pickle.dump(song_similarities, f)
    
    elif output_file.endswith('.json'):
        # JSON格式需要将numpy类型转换为原生Python类型
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(song_similarities, f)
    
    else:
        raise ValueError(f"不支持的文件格式: {output_file}")
    
    logger.info("歌曲相似度保存成功")

def split_data(triplets_df, test_ratio=0.2, validation_ratio=0.05, seed=42):
    """
    将数据分割为训练集、验证集和测试集
    
    参数:
        triplets_df: 包含user_id, song_id, play_count/rating列的DataFrame
        test_ratio: 测试集比例
        validation_ratio: 验证集比例
        seed: 随机种子
        
    返回:
        (train_df, val_df, test_df) 元组
    """
    logger.info("分割数据集")
    
    # 确保随机性可重现
    np.random.seed(seed)
    
    # 按用户分组
    user_groups = triplets_df.groupby('user_id')
    
    train_data = []
    val_data = []
    test_data = []
    
    for user_id, user_data in tqdm(user_groups):
        # 如果用户只有1或2条记录，全部放入训练集
        if len(user_data) <= 2:
            train_data.append(user_data)
            continue
        
        # 随机打乱数据
        user_data = user_data.sample(frac=1).reset_index(drop=True)
        
        # 计算分割点
        test_size = max(1, int(len(user_data) * test_ratio))
        val_size = max(1, int(len(user_data) * validation_ratio))
        
        # 确保至少有一条记录进入训练集
        if test_size + val_size >= len(user_data):
            test_size = max(1, len(user_data) // 3)
            val_size = max(1, len(user_data) // 6)
            
        train_size = len(user_data) - test_size - val_size
        
        # 分割数据
        user_train = user_data.iloc[:train_size]
        user_val = user_data.iloc[train_size:train_size+val_size]
        user_test = user_data.iloc[train_size+val_size:]
        
        # 添加到各个数据集
        train_data.append(user_train)
        val_data.append(user_val)
        test_data.append(user_test)
    
    # 合并各个数据集
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    logger.info(f"训练集大小: {len(train_df)}")
    logger.info(f"验证集大小: {len(val_df)}")
    logger.info(f"测试集大小: {len(test_df)}")
    
    return train_df, val_df, test_df 