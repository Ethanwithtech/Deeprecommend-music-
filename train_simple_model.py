#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的混合推荐系统训练脚本
使用模拟数据进行训练和测试
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import tensorflow as tf
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 配置日志
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('simple_trainer')

def create_mock_songs(n_songs=1000):
    """创建模拟歌曲数据"""
    logger.info(f"创建 {n_songs} 首模拟歌曲")
    
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

def create_mock_interactions(songs_df, n_users=500, min_items=5, max_items=50):
    """创建模拟用户互动数据"""
    logger.info(f"创建 {n_users} 个用户的模拟互动数据")
    
    np.random.seed(42)
    
    # 生成唯一的用户ID
    user_ids = [f"U{i:06d}" for i in range(n_users)]
    
    # 获取所有歌曲ID
    all_song_ids = songs_df['song_id'].values
    
    # 生成互动数据
    interactions = []
    
    for user_id in user_ids:
        # 每个用户与随机数量的歌曲互动
        n_items = np.random.randint(min_items, max_items+1)
        # 随机选择歌曲
        selected_songs = np.random.choice(all_song_ids, size=n_items, replace=False)
        
        for song_id in selected_songs:
            # 生成播放次数
            play_count = np.random.randint(1, 100)
            interactions.append({
                'user_id': user_id,
                'song_id': song_id,
                'play_count': play_count,
                'rating': np.log1p(play_count) / np.log1p(100)  # 归一化评分
            })
    
    # 创建DataFrame
    interactions_df = pd.DataFrame(interactions)
    
    return interactions_df

def create_audio_features(songs_df, n_features=10):
    """创建模拟音频特征"""
    logger.info("创建模拟音频特征")
    
    np.random.seed(42)
    
    # 使用歌曲的基本属性生成相关特征
    features = []
    
    for _, song in songs_df.iterrows():
        # 基于歌曲的tempo和loudness生成特征
        tempo = song.get('tempo', np.random.uniform(60, 180))
        loudness = song.get('loudness', np.random.uniform(-20, 0))
        
        # 生成基本特征
        feature_dict = {
            'song_id': song['song_id'],
            'energy': min(1.0, max(0.0, 0.5 + loudness/30 + np.random.normal(0, 0.1))),
            'acousticness': min(1.0, max(0.0, np.random.beta(2, 5))),
            'danceability': min(1.0, max(0.0, 0.5 + (tempo-120)/200 + np.random.normal(0, 0.15))),
            'instrumentalness': min(1.0, max(0.0, np.random.beta(1, 3))),
            'valence': min(1.0, max(0.0, np.random.beta(5, 5)))
        }
        
        # 添加随机额外特征
        for i in range(5, n_features):
            feature_dict[f'feature_{i}'] = np.random.random()
        
        features.append(feature_dict)
    
    # 创建DataFrame
    audio_features_df = pd.DataFrame(features)
    
    return audio_features_df

def save_data(data_dir):
    """保存生成的模拟数据"""
    logger.info(f"保存数据到目录: {data_dir}")
    
    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    
    # 创建模拟数据
    songs_df = create_mock_songs(1000)
    interactions_df = create_mock_interactions(songs_df, 500)
    audio_features_df = create_audio_features(songs_df)
    
    # 保存数据
    songs_df.to_csv(os.path.join(data_dir, 'songs.csv'), index=False)
    interactions_df.to_csv(os.path.join(data_dir, 'interactions.csv'), index=False)
    audio_features_df.to_csv(os.path.join(data_dir, 'audio_features.csv'), index=False)
    
    logger.info(f"保存了 {len(songs_df)} 首歌曲, {len(interactions_df)} 条交互, {len(audio_features_df)} 条音频特征")
    
    return songs_df, interactions_df, audio_features_df

def build_simple_model(n_users, n_items, embedding_dim=16):
    """构建简单的协同过滤模型"""
    logger.info(f"构建协同过滤模型: {n_users} 用户, {n_items} 项目, {embedding_dim} 维嵌入")
    
    # 用户输入
    user_input = tf.keras.layers.Input(shape=(1,))
    user_embedding = tf.keras.layers.Embedding(n_users, embedding_dim)(user_input)
    user_vec = tf.keras.layers.Flatten()(user_embedding)
    
    # 物品输入
    item_input = tf.keras.layers.Input(shape=(1,))
    item_embedding = tf.keras.layers.Embedding(n_items, embedding_dim)(item_input)
    item_vec = tf.keras.layers.Flatten()(item_embedding)
    
    # 特征交叉
    dot_product = tf.keras.layers.Dot(axes=1)([user_vec, item_vec])
    
    # 构建模型
    model = tf.keras.models.Model(inputs=[user_input, item_input], outputs=dot_product)
    
    # 编译模型
    model.compile(optimizer='adam', loss='mse')
    
    return model

def train_model(data_dir='data/processed', output_dir='models'):
    """训练简单的推荐模型"""
    logger.info("开始训练简单推荐模型")
    
    # 如果目录不存在，创建并保存模拟数据
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        songs_df, interactions_df, audio_features_df = save_data(data_dir)
    else:
        # 加载数据
        songs_df = pd.read_csv(os.path.join(data_dir, 'songs.csv'))
        interactions_df = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
        audio_features_df = pd.read_csv(os.path.join(data_dir, 'audio_features.csv'))
    
    # 创建用户和项目ID映射
    user_ids = interactions_df['user_id'].unique()
    song_ids = songs_df['song_id'].unique()
    
    user_id_map = {u_id: idx for idx, u_id in enumerate(user_ids)}
    song_id_map = {s_id: idx for idx, s_id in enumerate(song_ids)}
    
    # 为交互数据添加索引
    interactions_df['user_idx'] = interactions_df['user_id'].map(user_id_map)
    interactions_df['item_idx'] = interactions_df['song_id'].map(song_id_map)
    
    # 为歌曲数据添加索引
    songs_df['item_idx'] = songs_df['song_id'].map(song_id_map)
    
    # 准备训练数据
    X_user = interactions_df['user_idx'].values
    X_item = interactions_df['item_idx'].values
    y = interactions_df['rating'].values
    
    # 分割训练集和测试集
    X_user_train, X_user_test, X_item_train, X_item_test, y_train, y_test = train_test_split(
        X_user, X_item, y, test_size=0.2, random_state=42
    )
    
    # 构建模型
    model = build_simple_model(len(user_ids), len(song_ids))
    
    # 训练模型
    logger.info("开始训练...")
    history = model.fit(
        [X_user_train, X_item_train],
        y_train,
        batch_size=64,
        epochs=5,
        validation_data=([X_user_test, X_item_test], y_test),
        verbose=1
    )
    
    # 评估模型
    loss = model.evaluate([X_user_test, X_item_test], y_test)
    logger.info(f"测试集损失: {loss:.4f}")
    
    # 保存模型
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, 'simple_model.h5'))
    logger.info(f"模型已保存到 {output_dir}/simple_model.h5")
    
    # 测试推荐
    test_user = X_user_test[0]
    test_items = np.array(list(range(len(song_ids))))
    test_users = np.full_like(test_items, test_user)
    
    # 预测评分
    pred_ratings = model.predict([test_users, test_items])
    
    # 获取前10个推荐
    top_indices = np.argsort(pred_ratings.flatten())[-10:][::-1]
    
    # 获取对应的歌曲ID
    inv_song_map = {idx: song_id for song_id, idx in song_id_map.items()}
    top_songs = [inv_song_map[idx] for idx in top_indices]
    
    logger.info(f"为用户 {list(user_id_map.keys())[test_user]} 的推荐歌曲:")
    for i, song_id in enumerate(top_songs, 1):
        song_info = songs_df[songs_df['song_id'] == song_id].iloc[0]
        logger.info(f"{i}. {song_info['title']} - {song_info['artist_name']} (评分: {pred_ratings.flatten()[top_indices[i-1]]:.2f})")
    
    return model, history

if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "="*60)
    print("   开始训练简单混合推荐模型")
    print("="*60)
    
    model, history = train_model()
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"训练完成! 耗时: {elapsed:.2f} 秒")
    print("="*60) 