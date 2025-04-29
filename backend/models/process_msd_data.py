#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理MSD（Million Song Dataset）数据集
提取元数据、用户交互数据并训练推荐模型
"""

import os
import sys
import logging
import argparse
import h5py
import numpy as np
import pandas as pd
import time
import pickle
import random
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from models.msd_processor import MSDDataProcessor
from models.hybrid_recommender import HybridRecommender
from sklearn.model_selection import train_test_split
from pathlib import Path

# 确保backend模块可以被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.models.hybrid_music_recommender import HybridMusicRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('msd_processing.log')
    ]
)
logger = logging.getLogger('msd_process')

def process_h5_file(h5_path, limit=None):
    """
    从H5文件提取歌曲元数据
    
    参数:
        h5_path: H5文件的路径
        limit: 要处理的歌曲数量上限
        
    返回:
        包含歌曲元数据的DataFrame
    """
    logger.info(f"正在处理H5文件: {h5_path}")
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # 获取所有歌曲ID
            song_ids = list(f['metadata']['songs']['song_id'])
            # 如果有limit限制，只取前limit个
            if limit is not None:
                song_ids = song_ids[:limit]
                
            # 提取基础元数据
            titles = []
            artists = []
            album_names = []
            durations = []
            years = []
            
            # 音频特征
            tempos = []
            loudness = []
            keys = []
            modes = []
            
            for song_id in song_ids:
                song_idx = list(f['metadata']['songs']['song_id']).index(song_id)
                
                # 基础元数据
                titles.append(f['metadata']['songs']['title'][song_idx].decode('utf-8', errors='replace'))
                artists.append(f['metadata']['songs']['artist_name'][song_idx].decode('utf-8', errors='replace'))
                album_names.append(f['metadata']['songs']['release'][song_idx].decode('utf-8', errors='replace'))
                
                try:
                    duration = float(f['analysis']['songs']['duration'][song_idx])
                    durations.append(duration)
                except (KeyError, IndexError):
                    durations.append(None)
                    
                try:
                    year = int(f['musicbrainz']['songs']['year'][song_idx])
                    years.append(year if year > 0 else None)
                except (KeyError, IndexError, ValueError):
                    years.append(None)
                
                # 音频分析特征
                try:
                    tempo = float(f['analysis']['songs']['tempo'][song_idx])
                    tempos.append(tempo)
                except (KeyError, IndexError):
                    tempos.append(None)
                    
                try:
                    loud = float(f['analysis']['songs']['loudness'][song_idx])
                    loudness.append(loud)
                except (KeyError, IndexError):
                    loudness.append(None)
                    
                try:
                    key = int(f['analysis']['songs']['key'][song_idx])
                    keys.append(key)
                except (KeyError, IndexError):
                    keys.append(None)
                    
                try:
                    mode = int(f['analysis']['songs']['mode'][song_idx])
                    modes.append(mode)
                except (KeyError, IndexError):
                    modes.append(None)
            
            # 创建DataFrame
            songs_df = pd.DataFrame({
                'song_id': song_ids,
                'title': titles,
                'artist_name': artists,
                'album_name': album_names,
                'duration': durations,
                'year': years,
                'tempo': tempos,
                'loudness': loudness,
                'key': keys,
                'mode': modes
            })
            
            # 确保song_id列为字符串类型
            songs_df['song_id'] = songs_df['song_id'].astype(str)
            
            logger.info(f"从H5文件提取了 {len(songs_df)} 首歌曲的元数据")
            return songs_df

    except Exception as e:
        logger.error(f"处理H5文件时出错: {str(e)}")
        return pd.DataFrame()

def process_triplets_file(triplets_path, limit=None):
    """
    处理triplets文件中的用户播放数据
    
    参数:
        triplets_path: triplets文件的路径
        limit: 要处理的记录数量上限
        
    返回:
        包含用户互动数据的DataFrame
    """
    logger.info(f"正在处理triplets文件: {triplets_path}")
    
    try:
        # 以分块方式读取大文件
        chunks = pd.read_csv(triplets_path, sep='\t', header=None, 
            names=['user_id', 'song_id', 'play_count'],
                            chunksize=100000)
        
        all_data = []
        total_records = 0
        
        for chunk in chunks:
            # 确保ID列为字符串类型
            chunk['user_id'] = chunk['user_id'].astype(str)
            chunk['song_id'] = chunk['song_id'].astype(str)
            
            all_data.append(chunk)
            total_records += len(chunk)
            
            # 打印进度
            logger.debug(f"已处理 {total_records} 条记录")
            
            # 如果达到limit，停止处理
            if limit is not None and total_records >= limit:
                logger.info(f"已达到设定的记录数量上限: {limit}")
                break
        
        # 合并所有数据块
        triplets_df = pd.concat(all_data, ignore_index=True)
        
        # 如果超过limit，只保留前limit条
        if limit is not None and len(triplets_df) > limit:
            triplets_df = triplets_df.iloc[:limit]
        
        # 确保play_count为数值型
        triplets_df['play_count'] = pd.to_numeric(triplets_df['play_count'], errors='coerce')
        
        # 删除play_count为NaN的行
        triplets_df = triplets_df.dropna(subset=['play_count'])
        
        logger.info(f"从triplets文件提取了 {len(triplets_df)} 条用户播放记录")
        return triplets_df

    except Exception as e:
        logger.error(f"处理triplets文件时出错: {str(e)}")
        return pd.DataFrame()

def create_mock_songs_data(n_songs=1000):
    """创建模拟歌曲数据用于测试"""
    logger.info(f"创建 {n_songs} 首模拟歌曲数据")
    
    np.random.seed(42)
    
    # 生成唯一的歌曲ID
    song_ids = [f"S{i:06d}" for i in range(n_songs)]
    
    # 创建模拟数据
    titles = [f"Song Title {i}" for i in range(n_songs)]
    artists = [f"Artist {i % 100}" for i in range(n_songs)]  # 100个不同的艺术家
    albums = [f"Album {i % 200}" for i in range(n_songs)]    # 200个不同的专辑
    
    # 模拟特征
    durations = np.random.uniform(120, 420, n_songs)  # 2-7分钟
    years = np.random.randint(1960, 2023, n_songs)
    tempo = np.random.uniform(60, 180, n_songs)
    loudness = np.random.uniform(-20, 0, n_songs)
    key = np.random.randint(0, 12, n_songs)
    mode = np.random.randint(0, 2, n_songs)
    
    # 创建DataFrame
    songs_df = pd.DataFrame({
        'song_id': song_ids,
        'title': titles,
        'artist_name': artists,
        'album_name': albums,
        'duration': durations,
        'year': years,
        'tempo': tempo,
        'loudness': loudness,
        'key': key,
        'mode': mode
    })
    
    return songs_df

def create_mock_interactions_data(songs_df, n_users=500, interactions_per_user=(5, 50)):
    """创建模拟用户互动数据用于测试"""
    logger.info(f"创建 {n_users} 个用户的模拟互动数据")
    
    np.random.seed(42)
    
    # 生成唯一的用户ID
    user_ids = [f"U{i:06d}" for i in range(n_users)]
    
    # 确保songs_df中的song_id是字符串类型
    song_ids = songs_df['song_id'].astype(str).tolist()
    
    interactions = []
    
    for user_id in user_ids:
        # 为每个用户随机选择一些歌曲进行互动
        n_interactions = np.random.randint(interactions_per_user[0], interactions_per_user[1])
        user_songs = np.random.choice(song_ids, size=n_interactions, replace=False)
        
        # 生成播放次数
        play_counts = np.random.randint(1, 100, size=n_interactions)
        
        for song_id, play_count in zip(user_songs, play_counts):
            interactions.append({
            'user_id': user_id,
                'song_id': song_id,
                'play_count': play_count
            })
    
    interactions_df = pd.DataFrame(interactions)
    logger.info(f"创建了 {len(interactions_df)} 条模拟互动数据")
    
    return interactions_df

def create_users_df(triplets_df):
    """从triplets数据创建用户数据框"""
    
    # 确保user_id为字符串类型
    triplets_df = triplets_df.copy()
    triplets_df['user_id'] = triplets_df['user_id'].astype(str)
    
    # 获取唯一用户ID
    user_ids = triplets_df['user_id'].unique()
    logger.info(f"找到 {len(user_ids)} 个唯一用户")
    
    # 创建用户数据框
    users_df = pd.DataFrame({'user_id': user_ids})
    
    # 为每个用户生成模拟特征（如果需要）
    # 这里可以添加任何其他用户特征的生成逻辑
    
    return users_df

def create_simulation_data(n_songs=1000, n_users=500):
    """创建完整的模拟数据集用于测试"""
    logger.info("创建模拟数据集用于测试")
    
    # 创建歌曲数据
    songs_df = create_mock_songs_data(n_songs)
    
    # 创建用户互动数据
    interactions_df = create_mock_interactions_data(songs_df, n_users)
    
    # 创建用户数据
    users_df = create_users_df(interactions_df)
    
    # 模拟音频特征
    audio_features = []
    for song_id in songs_df['song_id']:
        # 创建模拟音频特征
        features = {
            'song_id': song_id,
            'acousticness': np.random.uniform(0, 1),
            'danceability': np.random.uniform(0, 1),
            'energy': np.random.uniform(0, 1),
            'instrumentalness': np.random.uniform(0, 1),
            'liveness': np.random.uniform(0, 1),
            'speechiness': np.random.uniform(0, 1),
            'valence': np.random.uniform(0, 1)
        }
        audio_features.append(features)
    
    audio_features_df = pd.DataFrame(audio_features)
    
    return {
        'songs': songs_df,
        'interactions': interactions_df,
        'users': users_df,
        'audio_features': audio_features_df
    }

def process_msd_data(h5_path=None, triplets_path=None, processed_dir='processed_data', 
                     chunk_limit=None, test_size=0.2, random_state=42, force_process=False):
    """
    处理MSD数据集并准备用于推荐系统训练
    
    参数:
        h5_path: H5文件路径
        triplets_path: triplets文件路径
        processed_dir: 处理后数据存储目录
        chunk_limit: 限制处理的记录数量
        test_size: 测试集比例
        random_state: 随机种子
        force_process: 强制重新处理数据
        
    返回:
        包含处理后数据和训练/测试集的字典
    """
    # 创建处理目录（如果不存在）
    os.makedirs(processed_dir, exist_ok=True)
    
    # 检查是否有保存的处理数据
    processed_file = os.path.join(processed_dir, 'processed_msd_data.pkl')
    if os.path.exists(processed_file) and not force_process:
        logger.info(f"加载已处理的数据: {processed_file}")
        try:
            data = pd.read_pickle(processed_file)
            
            # 确保所有ID列被处理为字符串类型
            if 'songs' in data:
                data['songs']['song_id'] = data['songs']['song_id'].astype(str)
            if 'interactions' in data:
                data['interactions']['user_id'] = data['interactions']['user_id'].astype(str)
                data['interactions']['song_id'] = data['interactions']['song_id'].astype(str)
            if 'audio_features' in data and 'song_id' in data['audio_features'].columns:
                data['audio_features']['song_id'] = data['audio_features']['song_id'].astype(str)
            if 'user_features' in data and 'user_id' in data['user_features'].columns:
                data['user_features']['user_id'] = data['user_features']['user_id'].astype(str)
                
            return data
        except Exception as e:
            logger.error(f"加载处理后数据时出错: {str(e)}")
            logger.info("重新处理数据...")
    
    # 处理实际数据或创建模拟数据
    if h5_path and triplets_path:
        # 处理H5文件提取歌曲元数据
        songs_df = process_h5_file(h5_path, limit=chunk_limit)
        
        # 处理triplets文件提取用户互动数据
        triplets_df = process_triplets_file(triplets_path, limit=chunk_limit)
        
        # 如果数据为空，则使用模拟数据
        if songs_df.empty or triplets_df.empty:
            logger.warning("处理实际数据失败，创建模拟数据代替")
            sim_data = create_simulation_data()
            songs_df = sim_data['songs']
            triplets_df = sim_data['interactions']
    else:
        # 使用模拟数据
        logger.info("未提供实际数据路径，创建模拟数据")
        sim_data = create_simulation_data()
        songs_df = sim_data['songs']
        triplets_df = sim_data['interactions']
    
    # 确保ID列是字符串类型
    songs_df['song_id'] = songs_df['song_id'].astype(str)
    triplets_df['user_id'] = triplets_df['user_id'].astype(str)
    triplets_df['song_id'] = triplets_df['song_id'].astype(str)
    
    # 将play_count转换为rating (1-5的分数)
    # 这里只转换play_count列为数值型，确保song_id和user_id保持为字符串
    triplets_df['play_count'] = pd.to_numeric(triplets_df['play_count'], errors='coerce')
    triplets_df = triplets_df.dropna(subset=['play_count'])
    
    # 筛选存在于songs_df中的歌曲
    valid_songs = set(songs_df['song_id'])
    triplets_df = triplets_df[triplets_df['song_id'].isin(valid_songs)]
    
    # 获取有效的用户ID
    valid_users = set(triplets_df['user_id'])
    
    # 创建用户数据框
    users_df = pd.DataFrame({'user_id': list(valid_users)})
    
    # 确保用户ID是字符串类型
    users_df['user_id'] = users_df['user_id'].astype(str)
    
    # 创建音频特征
    try:
        audio_features_df = create_audio_features(songs_df)
        # 确保song_id是字符串类型
        audio_features_df['song_id'] = audio_features_df['song_id'].astype(str)
    except Exception as e:
        logger.error(f"创建音频特征时出错: {str(e)}")
        # 创建一个只有song_id的空特征DataFrame
        audio_features_df = pd.DataFrame({'song_id': songs_df['song_id']})
    
    # 创建数据字典
    data = {
        'songs': songs_df,
        'interactions': triplets_df,
        'audio_features': audio_features_df,
        'user_features': create_user_features(triplets_df)
    }
    
    # 分割为训练集和测试集
    train_interactions, test_interactions = train_test_split(
        triplets_df, test_size=test_size, random_state=random_state
    )
    
    data['train_interactions'] = train_interactions
    data['test_interactions'] = test_interactions
    
    # 保存处理后的数据
    try:
        pd.to_pickle(data, processed_file)
        logger.info(f"已保存处理后的数据到: {processed_file}")
    except Exception as e:
        logger.error(f"保存处理后数据时出错: {str(e)}")
    
    return data

def create_audio_features(songs_df):
    """
    从歌曲数据中提取或创建音频特征
    """
    # 确保song_id是字符串类型
    songs_df = songs_df.copy()
    songs_df['song_id'] = songs_df['song_id'].astype(str)
    
    # 使用现有的音频特征，或创建模拟特征
    features = []
    
    # 检查songs_df中是否已有音频特征
    audio_cols = ['tempo', 'loudness', 'key', 'mode']
    has_audio_features = all(col in songs_df.columns for col in audio_cols)
    
    for _, song in songs_df.iterrows():
        song_id = song['song_id']
        feature = {'song_id': song_id}
        
        if has_audio_features:
            # 使用已有特征
            for col in audio_cols:
                if col in song:
                    feature[col] = song[col]
        
        # 添加模拟特征
        feature['acousticness'] = np.random.uniform(0, 1)
        feature['danceability'] = np.random.uniform(0, 1)
        feature['energy'] = np.random.uniform(0, 1)
        feature['instrumentalness'] = np.random.uniform(0, 1)
        feature['liveness'] = np.random.uniform(0, 1)
        feature['speechiness'] = np.random.uniform(0, 1)
        feature['valence'] = np.random.uniform(0, 1)
        
        features.append(feature)
    
    return pd.DataFrame(features)

def create_user_features(interactions_df):
    """
    创建用户特征
    """
    # 确保用户ID是字符串类型
    interactions_df = interactions_df.copy()
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)

        # 获取唯一用户ID
        user_ids = interactions_df['user_id'].unique()

    # 创建用户特征
    user_features = []
    
    for user_id in user_ids:
        # 获取用户的互动数据
        user_interactions = interactions_df[interactions_df['user_id'] == user_id]
        
        # 计算用户特征
        total_plays = user_interactions['play_count'].sum()
        avg_plays = user_interactions['play_count'].mean()
        num_songs = len(user_interactions)
        
        # 创建用户特征字典
        feature = {
            'user_id': user_id,
            'total_plays': total_plays,
            'avg_plays': avg_plays,
            'num_songs': num_songs
        }
        
        user_features.append(feature)
    
    return pd.DataFrame(user_features)

def main():
    """
    测试数据处理函数
    """
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 处理参数
    h5_path = None
    triplets_path = None
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
    if len(sys.argv) > 2:
        triplets_path = sys.argv[2]
    
    # 处理数据
    data = process_msd_data(h5_path, triplets_path, chunk_limit=1000)
    
    # 打印数据概况
    for key, df in data.items():
        if isinstance(df, pd.DataFrame):
            print(f"{key}: {len(df)} 行")
    
    # 保存样本数据用于检查
    for key, df in data.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            sample_file = f"sample_{key}.csv"
            df.head(100).to_csv(sample_file, index=False)
            print(f"样本数据已保存到 {sample_file}")

if __name__ == "__main__":
    main()