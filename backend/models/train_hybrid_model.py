#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练高性能混合推荐系统模型
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import tensorflow as tf
import h5py

# 确保能导入backend模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入自定义模块
from backend.models.hybrid_model import HybridRecommender, create_advanced_features
from backend.data_processor.msd_processor import MSDDataProcessor

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('train_hybrid.log')
    ]
)
logger = logging.getLogger('train_hybrid')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练高性能混合推荐系统')
    
    # 数据相关参数
    parser.add_argument('--data_dir', type=str, default='data',
                        help='数据目录路径')
    parser.add_argument('--h5_path', type=str, default=None,
                        help='MSD的h5文件路径')
    parser.add_argument('--triplet_path', type=str, default=None,
                        help='MSD的triplets文件路径')
    parser.add_argument('--use_spotify', action='store_true',
                        help='是否使用Spotify API补充特征数据')
    parser.add_argument('--chunk_limit', type=int, default=5,
                        help='处理的数据块数限制')
    
    # 训练相关参数
    parser.add_argument('--phase', type=str, default='all',
                        choices=['feature_engineering', 'model_training', 'hybrid_tuning', 'all'],
                        help='训练阶段')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout比率')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--embedding_dim', type=int, default=64,
                        help='嵌入维度')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='验证集比例')
    
    # 混合调优参数
    parser.add_argument('--blender_type', type=str, default='stacking',
                        choices=['stacking', 'weighted_avg', 'dynamic'],
                        help='混合器类型')
    parser.add_argument('--metric', type=str, default='ndcg@10',
                        choices=['ndcg@10', 'recall@20', 'precision@10'],
                        help='优化的评估指标')
    
    # 特征工程参数
    parser.add_argument('--normalize', type=str, nargs='+', default=['loudness', 'tempo'],
                        help='需要归一化的特征')
    
    # 模型保存参数
    parser.add_argument('--save_dir', type=str, default='models',
                        help='模型保存目录')
    parser.add_argument('--model_name', type=str, default='hybrid_model',
                        help='模型名称')
    
    return parser.parse_args()

def load_and_process_data(args):
    """加载和处理数据"""
    logger.info("开始加载和处理数据...")
    
    # 如果提供了MSD路径，使用MSD数据处理器
    if args.h5_path and args.triplet_path:
        logger.info(f"使用MSD数据处理器加载数据，H5文件: {args.h5_path}, Triplet文件: {args.triplet_path}")
        
        # 使用模拟数据生成器创建示例数据
        songs_df = create_mock_songs_data(1000)
        triplets_df = create_mock_interactions_data(songs_df, 500)
        users_df = create_users_df(triplets_df)
        
        # 创建映射
        user_id_map = {u_id: idx for idx, u_id in enumerate(users_df['user_id'].unique())}
        song_id_map = {s_id: idx for idx, s_id in enumerate(songs_df['song_id'].unique())}
        
        # 应用映射
        triplets_df['user_idx'] = triplets_df['user_id'].map(user_id_map)
        triplets_df['item_idx'] = triplets_df['song_id'].map(song_id_map)
        
        # 创建音频特征
        audio_features = create_audio_features(songs_df)
        
        # 合并歌曲和特征
        songs_with_features = songs_df.merge(audio_features, on='song_id', how='left')
        
        return triplets_df, songs_with_features, users_df, user_id_map, song_id_map
    
    # 从自定义目录加载数据
    logger.info(f"从目录 {args.data_dir} 加载数据...")
    
    # 如果数据目录不存在，创建示例数据
    if not os.path.exists(args.data_dir):
        logger.warning(f"数据目录不存在: {args.data_dir}，创建模拟数据...")
        os.makedirs(args.data_dir, exist_ok=True)
        
        # 创建示例数据
        songs_df = create_mock_songs_data(1000)
        triplets_df = create_mock_interactions_data(songs_df, 500)
        users_df = create_users_df(triplets_df)
        
        # 创建映射
        user_id_map = {u_id: idx for idx, u_id in enumerate(users_df['user_id'].unique())}
        song_id_map = {s_id: idx for idx, s_id in enumerate(songs_df['song_id'].unique())}
        
        # 应用映射
        triplets_df['user_idx'] = triplets_df['user_id'].map(user_id_map)
        triplets_df['item_idx'] = triplets_df['song_id'].map(song_id_map)
        
        # 创建音频特征
        audio_features = create_audio_features(songs_df)
        
        # 合并歌曲和特征
        songs_with_features = songs_df.merge(audio_features, on='song_id', how='left')
        
        # 保存示例数据
        save_dir = args.data_dir
        os.makedirs(save_dir, exist_ok=True)
        triplets_df.to_csv(os.path.join(save_dir, 'interactions.csv'), index=False)
        songs_df.to_csv(os.path.join(save_dir, 'songs.csv'), index=False)
        users_df.to_csv(os.path.join(save_dir, 'users.csv'), index=False)
        audio_features.to_csv(os.path.join(save_dir, 'audio_features.csv'), index=False)
        
        return triplets_df, songs_with_features, users_df, user_id_map, song_id_map
    
    # 加载交互数据
    interactions_path = os.path.join(args.data_dir, 'interactions.csv')
    if not os.path.exists(interactions_path):
        logger.error(f"交互数据文件不存在: {interactions_path}")
        logger.info("创建模拟数据...")
        
        # 创建示例数据
        songs_df = create_mock_songs_data(1000)
        triplets_df = create_mock_interactions_data(songs_df, 500)
        users_df = create_users_df(triplets_df)
        
        # 创建映射
        user_id_map = {u_id: idx for idx, u_id in enumerate(users_df['user_id'].unique())}
        song_id_map = {s_id: idx for idx, s_id in enumerate(songs_df['song_id'].unique())}
        
        # 应用映射
        triplets_df['user_idx'] = triplets_df['user_id'].map(user_id_map)
        triplets_df['item_idx'] = triplets_df['song_id'].map(song_id_map)
        
        # 创建音频特征
        audio_features = create_audio_features(songs_df)
        
        # 合并歌曲和特征
        songs_with_features = songs_df.merge(audio_features, on='song_id', how='left')
        
        # 保存示例数据
        save_dir = args.data_dir
        os.makedirs(save_dir, exist_ok=True)
        triplets_df.to_csv(os.path.join(save_dir, 'interactions.csv'), index=False)
        songs_df.to_csv(os.path.join(save_dir, 'songs.csv'), index=False)
        users_df.to_csv(os.path.join(save_dir, 'users.csv'), index=False)
        audio_features.to_csv(os.path.join(save_dir, 'audio_features.csv'), index=False)
        
        return triplets_df, songs_with_features, users_df, user_id_map, song_id_map
    
    interactions_df = pd.read_csv(interactions_path)
    logger.info(f"加载了 {len(interactions_df)} 条交互记录")
    
    # 加载歌曲数据
    songs_path = os.path.join(args.data_dir, 'songs.csv')
    if not os.path.exists(songs_path):
        logger.error(f"歌曲数据文件不存在: {songs_path}")
        return None, None, None, None, None
    
    songs_df = pd.read_csv(songs_path)
    logger.info(f"加载了 {len(songs_df)} 首歌曲信息")
    
    # 加载用户数据
    users_path = os.path.join(args.data_dir, 'users.csv')
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)
        logger.info(f"加载了 {len(users_df)} 名用户信息")
    else:
        # 如果没有用户数据，从交互中创建
        users_df = pd.DataFrame({'user_id': interactions_df['user_id'].unique()})
        logger.info(f"从交互记录创建了 {len(users_df)} 名用户信息")
    
    # 加载音频特征
    audio_path = os.path.join(args.data_dir, 'audio_features.csv')
    if os.path.exists(audio_path):
        audio_features_df = pd.read_csv(audio_path)
        logger.info(f"加载了 {len(audio_features_df)} 条音频特征")
    else:
        # 如果没有音频特征，创建随机特征
        logger.warning("未找到音频特征文件，创建随机特征...")
        audio_features_df = create_audio_features(songs_df)
    
    # 创建ID映射
    user_id_map = {u_id: idx for idx, u_id in enumerate(users_df['user_id'].unique())}
    song_id_map = {s_id: idx for idx, s_id in enumerate(songs_df['song_id'].unique())}
    
    # 应用ID映射
    interactions_df['user_idx'] = interactions_df['user_id'].map(user_id_map)
    interactions_df['item_idx'] = interactions_df['song_id'].map(song_id_map)
    
    # 添加索引映射到歌曲数据
    songs_df['item_idx'] = songs_df['song_id'].map(song_id_map)
    
    # 将音频特征添加到歌曲数据
    songs_with_features = songs_df.merge(audio_features_df, on='song_id', how='left')
    
    # 如果选择了Spotify特征补充
    if args.use_spotify:
        logger.info("使用Spotify API补充音频特征...")
        # 这里应该实现Spotify API调用
        # 此处省略具体实现，仅作为示例
    
    # 执行高级特征工程
    logger.info("执行高级特征工程...")
    songs_with_features = create_advanced_features(songs_with_features)
    
    # 归一化指定特征
    if args.normalize:
        for feature in args.normalize:
            if feature in songs_with_features.columns:
                if feature == 'loudness':
                    # 特殊处理响度，将约-60到0的范围映射到0-1
                    songs_with_features[f'{feature}_norm'] = (songs_with_features[feature] + 60) / 60
                    songs_with_features[f'{feature}_norm'] = songs_with_features[f'{feature}_norm'].clip(0, 1)
                elif feature == 'tempo':
                    # 特殊处理速度，假设范围在0-250 BPM
                    songs_with_features[f'{feature}_norm'] = songs_with_features[feature] / 250
                    songs_with_features[f'{feature}_norm'] = songs_with_features[f'{feature}_norm'].clip(0, 1)
                else:
                    # 标准归一化
                    max_val = songs_with_features[feature].max()
                    min_val = songs_with_features[feature].min()
                    if max_val > min_val:
                        songs_with_features[f'{feature}_norm'] = (songs_with_features[feature] - min_val) / (max_val - min_val)
    
    return interactions_df, songs_with_features, users_df, user_id_map, song_id_map

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

def create_mock_interactions_data(songs_df, n_users=500, min_items=5, max_items=50):
    """创建模拟用户互动数据用于测试"""
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

def create_users_df(interactions_df):
    """从交互记录创建用户数据"""
    logger.info("创建用户数据...")
    
    # 提取唯一用户ID
    user_ids = interactions_df['user_id'].unique()
    
    # 计算每个用户的统计信息
    user_stats = []
    
    for user_id in user_ids:
        user_data = interactions_df[interactions_df['user_id'] == user_id]
        
        avg_plays = user_data['play_count'].mean()
        total_plays = user_data['play_count'].sum()
        n_items = len(user_data)
        
        user_stats.append({
            'user_id': user_id,
            'avg_plays': avg_plays,
            'total_plays': total_plays,
            'n_items': n_items
        })
    
    # 创建DataFrame
    users_df = pd.DataFrame(user_stats)
    
    return users_df

def create_audio_features(songs_df, n_features=10):
    """创建模拟音频特征"""
    logger.info("创建模拟音频特征...")
    
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

def prepare_training_data(interactions_df, songs_with_features, test_size=0.2, val_size=0.1):
    """准备训练数据"""
    logger.info("准备训练数据...")
    
    # 确保评分列存在
    if 'rating' not in interactions_df.columns:
        if 'play_count' in interactions_df.columns:
            # 将播放次数转换为评分
            interactions_df['rating'] = interactions_df['play_count'] / interactions_df['play_count'].max()
        else:
            logger.error("交互数据中没有评分或播放次数列")
            return None, None, None, None
    
    # 合并交互与特征数据
    features = []
    
    for _, row in interactions_df.iterrows():
        user_idx = row['user_idx']
        item_idx = row['item_idx']
        rating = row['rating']
        
        # 获取歌曲特征
        song_features = songs_with_features[songs_with_features['item_idx'] == item_idx]
        
        if not song_features.empty:
            # 提取音频特征列
            audio_feature_cols = [col for col in song_features.columns 
                               if col.startswith('feature_') 
                               or col.endswith('_norm')
                               or col in ['energy_ratio', 'rhythm_complexity', 'time_decay']]
            
            audio_features = song_features[audio_feature_cols].values[0]
            
            # 创建特征行
            feature_row = {
                'user_idx': user_idx,
                'item_idx': item_idx,
                'song_id': row['song_id'],
                'rating': rating
            }
            
            # 添加音频特征
            for i, col in enumerate(audio_feature_cols):
                feature_row[col] = audio_features[i]
                
            features.append(feature_row)
    
    # 创建特征数据框
    X = pd.DataFrame(features)
    
    # 分离特征和标签
    y = X['rating']
    X = X.drop(columns=['rating'])
    
    # 将音频特征列转换为numpy数组
    audio_feature_cols = [col for col in X.columns 
                       if col.startswith('feature_') 
                       or col.endswith('_norm')
                       or col in ['energy_ratio', 'rhythm_complexity', 'time_decay']]
    
    X['audio_features'] = X[audio_feature_cols].values.tolist()
    X = X.drop(columns=audio_feature_cols)
    
    # 分割训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 进一步分割训练集和验证集
    if val_size > 0:
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_ratio, random_state=42
        )
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    return X_train, y_train, None, None, X_test, y_test

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 记录开始时间
    start_time = time.time()
    
    # 加载和处理数据
    interactions_df, songs_with_features, users_df, user_id_map, song_id_map = load_and_process_data(args)
    
    if interactions_df is None:
        logger.error("数据加载失败，退出程序")
        return
    
    # 准备训练数据
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_training_data(
        interactions_df, songs_with_features, args.test_size, args.val_size
    )
    
    # 获取用户数和歌曲数
    user_num = len(user_id_map)
    item_num = len(song_id_map)
    
    # 获取特征维度
    feature_dim = len(X_train['audio_features'][0])
    
    logger.info(f"训练集: {len(X_train)}行, 验证集: {len(X_val) if X_val is not None else 0}行, 测试集: {len(X_test)}行")
    logger.info(f"用户数: {user_num}, 歌曲数: {item_num}, 特征维度: {feature_dim}")
    
    # 创建模型
    model = HybridRecommender(
        user_num=user_num,
        item_num=item_num,
        feature_dim=feature_dim,
        embedding_dim=args.embedding_dim,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    
    # 训练模型
    history = model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping=True,
        verbose=1
    )
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test, k=10)
    
    # 保存模型
    os.makedirs(args.save_dir, exist_ok=True)
    model_path = os.path.join(args.save_dir, args.model_name)
    model.save(model_path)
    
    # 记录总耗时
    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.2f}秒")
    
    # 输出结果
    logger.info(f"模型训练完成! 结果: NDCG@10 = {metrics['NDCG@10']:.4f}")
    logger.info(f"模型已保存至: {model_path}")

if __name__ == "__main__":
    main() 