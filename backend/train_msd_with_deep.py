#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MSD深度学习预训练脚本
处理Million Song Dataset并训练混合推荐模型，包含深度学习组件
"""

import os
import sys
import time
import argparse
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入自定义模块
from backend.models.msd_processor import MSDDataProcessor
from backend.models.hybrid_recommender import HybridRecommender
from backend.models.deep_learning import DeepRecommender

# 配置日志
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f'deep_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

# 创建日志
logger = logging.getLogger('msd_deep_trainer')
logger.setLevel(logging.INFO)

# 创建控制台处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)

# 创建文件处理器
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)

# 添加处理器到日志
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Spotify API默认凭证
DEFAULT_SPOTIFY_CLIENT_ID = 'bdfa10b0a8bf49a3a413ba67d2ff1706'
DEFAULT_SPOTIFY_CLIENT_SECRET = 'b8e97ad8e96043b4b0d768d3e3c568b4'


def check_gpu():
    """检查GPU是否可用，并设置TensorFlow使用GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置TensorFlow使用第一个GPU
            tf.config.set_visible_devices(gpus[0], 'GPU')
            # 允许GPU内存增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"已检测到 {len(gpus)} 个GPU设备，将使用第一个GPU: {gpus[0]}")
            return True
        except RuntimeError as e:
            logger.error(f"设置GPU时出错: {e}")
            return False
    else:
        logger.warning("未检测到GPU，将使用CPU训练（可能较慢）")
        return False


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MSD深度学习推荐训练")
    
    # 数据参数
    parser.add_argument('--h5_file', type=str, required=True,
                       help='MSD元数据HDF5文件路径')
    parser.add_argument('--triplet_file', type=str, required=True,
                       help='MSD三元组播放数据文件路径')
    parser.add_argument('--output_dir', type=str, default='processed_data',
                       help='处理后数据保存目录')
    
    # 处理参数
    parser.add_argument('--force_process', action='store_true',
                       help='强制重新处理数据')
    parser.add_argument('--chunk_limit', type=int, default=None,
                       help='处理的数据块数量限制(用于测试)')
    parser.add_argument('--rating_method', type=str, default='log',
                       choices=['log', 'linear', 'percentile'],
                       help='播放次数转评分方法')
    parser.add_argument('--max_interactions', type=int, default=None,
                       help='训练使用的最大交互记录数量')
    
    # Spotify参数
    parser.add_argument('--no_spotify', action='store_true',
                       help='不使用Spotify API丰富数据')
    parser.add_argument('--use_spotify', action='store_true',
                       help='使用Spotify API丰富数据')
    parser.add_argument('--spotify_client_id', type=str, default=None,
                       help='Spotify API的Client ID')
    parser.add_argument('--spotify_client_secret', type=str, default=None,
                       help='Spotify API的Client Secret')
    parser.add_argument('--spotify_max_songs', type=int, default=1000,
                       help='使用Spotify处理的最大歌曲数')
    parser.add_argument('--spotify_batch_size', type=int, default=50,
                       help='Spotify API批处理大小')
    parser.add_argument('--spotify_workers', type=int, default=5,
                       help='Spotify并行处理线程数')
    parser.add_argument('--spotify_strategy', type=str, default='popular',
                       choices=['all', 'popular', 'diverse'],
                       help='Spotify处理策略(all=全部, popular=热门优先, diverse=多样性优先)')
    parser.add_argument('--spotify_cache_file', type=str, default=None,
                       help='Spotify缓存文件路径')
    
    # 训练参数
    parser.add_argument('--skip_deep', action='store_true',
                       help='跳过深度学习模型训练')
    parser.add_argument('--skip_hybrid', action='store_true',
                       help='跳过混合模型训练')
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='学习率')
    parser.add_argument('--embedding_dim', type=int, default=64,
                       help='嵌入维度')
    parser.add_argument('--layers', type=str, default='[256,128,64]',
                       help='神经网络层结构，JSON格式')
    
    # 优化参数
    parser.add_argument('--cf_weight', type=float, default=0.4,
                       help='协同过滤权重')
    parser.add_argument('--content_weight', type=float, default=0.3,
                       help='内容推荐权重')
    parser.add_argument('--context_weight', type=float, default=0.1,
                       help='上下文推荐权重')
    parser.add_argument('--deep_weight', type=float, default=0.2,
                       help='深度学习推荐权重')
    
    # 添加日志级别参数
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    # 添加用户采样参数
    parser.add_argument('--user_sample', type=int, default=None,
                       help='限制使用的用户数量，随机采样指定数量的用户')
    parser.add_argument('--no_filter_inactive_users', action='store_true',
                       help='不过滤不活跃用户，保留所有用户')
    
    args = parser.parse_args()
    
    # 确保权重总和为1
    weights_sum = args.cf_weight + args.content_weight + args.context_weight + args.deep_weight
    if abs(weights_sum - 1.0) > 0.01:
        print(f"警告：权重总和 ({weights_sum}) 不为1，自动归一化")
        args.cf_weight /= weights_sum
        args.content_weight /= weights_sum
        args.context_weight /= weights_sum
        args.deep_weight /= weights_sum
    
    return args


def process_msd_data(args):
    """处理MSD数据集"""
    print("\n" + "="*80)
    print("步骤1: 数据处理")
    print("="*80)
    
    # 检查数据文件是否存在
    if not os.path.exists(args.h5_file):
        raise FileNotFoundError(f"找不到HDF5元数据文件: {args.h5_file}")
        
    if not os.path.exists(args.triplet_file):
        raise FileNotFoundError(f"找不到三元组数据文件: {args.triplet_file}")
    
    # 是否使用Spotify
    use_spotify = args.use_spotify and not args.no_spotify
    
    # 获取Spotify API凭证 - 优先使用命令行参数提供的凭证
    spotify_client_id = args.spotify_client_id or os.environ.get('SPOTIFY_CLIENT_ID') or DEFAULT_SPOTIFY_CLIENT_ID
    spotify_client_secret = args.spotify_client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET') or DEFAULT_SPOTIFY_CLIENT_SECRET
    
    # 验证凭证是否有效
    if use_spotify and (not spotify_client_id or not spotify_client_secret):
        print("警告: 未找到有效的Spotify API凭证，禁用Spotify功能")
        use_spotify = False
    else:
        print(f"使用Spotify API凭证: {spotify_client_id[:5]}...{spotify_client_id[-5:]}")
    
    # 初始化数据处理器
    processor = MSDDataProcessor(
        output_dir=args.output_dir,
        use_spotify=use_spotify,
        spotify_client_id=spotify_client_id,
        spotify_client_secret=spotify_client_secret,
        spotify_batch_size=args.spotify_batch_size,
        spotify_workers=args.spotify_workers,
        spotify_strategy=args.spotify_strategy,
        spotify_cache_file=args.spotify_cache_file,
        force_process=args.force_process
    )
    
    # 处理数据
    songs, interactions, audio_features, user_features = processor.process_msd_data(
        h5_file=args.h5_file,
        triplet_file=args.triplet_file,
        output_dir=args.output_dir,
        chunk_limit=args.chunk_limit,
        max_spotify_songs=args.spotify_max_songs if use_spotify else None,
        rating_method=args.rating_method,
        force_process=args.force_process,
        spotify_batch_size=args.spotify_batch_size,
        spotify_workers=args.spotify_workers,
        spotify_strategy=args.spotify_strategy,
        user_sample=args.user_sample,
        no_filter_inactive_users=args.no_filter_inactive_users
    )
    
    print(f"数据处理完成: {len(songs)}首歌曲, {len(interactions)}条交互记录")
    return songs, interactions, audio_features, user_features


def load_processed_data(processed_dir):
    """加载已处理的数据"""
    processed_dir = Path(processed_dir)
    
    logger.info("加载已处理的数据...")
    songs = pd.read_parquet(processed_dir / 'songs.parquet')
    interactions = pd.read_parquet(processed_dir / 'interactions.parquet')
    audio_features = pd.read_parquet(processed_dir / 'audio_features.parquet')
    user_features = pd.read_parquet(processed_dir / 'user_features.parquet')
    
    # 确保ID字段为字符串类型
    if 'song_id' in songs.columns:
        songs['song_id'] = songs['song_id'].astype(str)
    if 'song_id' in interactions.columns:
        interactions['song_id'] = interactions['song_id'].astype(str)
    if 'user_id' in interactions.columns:
        interactions['user_id'] = interactions['user_id'].astype(str)
    if 'song_id' in audio_features.columns:
        audio_features['song_id'] = audio_features['song_id'].astype(str)
    if 'user_id' in user_features.columns:
        user_features['user_id'] = user_features['user_id'].astype(str)
    
    logger.info(f"数据加载完成: {len(songs)} 首歌曲, {len(interactions)} 条交互, {len(user_features)} 位用户")
    return songs, interactions, audio_features, user_features


def train_deep_model(interactions_train, songs, audio_features, args):
    """训练深度学习模型"""
    logger.info("开始训练深度学习模型...")
    start_time = time.time()
    
    # 创建用户和物品ID映射
    user_ids = interactions_train['user_id'].unique()
    song_ids = songs['song_id'].unique()
    
    user_to_idx = {user_id: i for i, user_id in enumerate(user_ids)}
    song_to_idx = {song_id: i for i, song_id in enumerate(song_ids)}
    
    logger.info(f"创建了用户映射 ({len(user_to_idx)} 个用户) 和歌曲映射 ({len(song_to_idx)} 首歌曲)")
    
    # 准备训练数据
    # 确保所有ID能够在映射中找到，过滤掉不存在的ID
    valid_data = interactions_train[
        interactions_train['user_id'].isin(user_to_idx.keys()) & 
        interactions_train['song_id'].isin(song_to_idx.keys())
    ]
    
    if len(valid_data) == 0:
        logger.error("没有有效的训练数据，无法训练深度学习模型")
        return None, None
    
    user_indices = np.array([user_to_idx[user_id] for user_id in valid_data['user_id']])
    item_indices = np.array([song_to_idx[song_id] for song_id in valid_data['song_id']])
    ratings = np.array(valid_data['rating'].astype(float))
    
    # 准备音频特征（如果可用）
    item_features = None
    if audio_features is not None and not audio_features.empty:
        # 创建特征矩阵 - 需要与song_to_idx映射对应
        # 选择所有特征列（除了song_id）
        feature_columns = [col for col in audio_features.columns if col != 'song_id']
        
        if not feature_columns:
            logger.warning("没有可用的音频特征列，将不使用特征")
        else:
            # 创建特征矩阵
            item_features = np.zeros((len(song_to_idx), len(feature_columns)))
            
            # 使用索引进行快速查找
            audio_features_indexed = audio_features.set_index('song_id')
            
            # 为每个歌曲ID创建特征向量
            for song_id, idx in song_to_idx.items():
                if song_id in audio_features_indexed.index:
                    try:
                        feature_values = audio_features_indexed.loc[song_id, feature_columns]
                        
                        # 检查是否返回了Series或DataFrame
                        if isinstance(feature_values, pd.Series):
                            # 直接使用Series的values属性
                            item_features[idx] = feature_values.values
                        elif isinstance(feature_values, pd.DataFrame):
                            # 如果是DataFrame，取第一行
                            logger.debug(f"歌曲 {song_id} 有多行特征，仅使用第一行")
                            item_features[idx] = feature_values.iloc[0].values
                        else:
                            logger.debug(f"歌曲 {song_id} 的特征格式异常: {type(feature_values)}")
                    except Exception as e:
                        logger.warning(f"处理歌曲 {song_id} 特征时出错: {e}")
                        # 使用0值填充
                        item_features[idx] = np.zeros(len(feature_columns))
            
            logger.info(f"准备了音频特征矩阵: {item_features.shape}")
    
    # 创建并训练模型
    model = DeepRecommender(
        n_users=len(user_to_idx),
        n_items=len(song_to_idx),
        embedding_dim=args.embedding_dim,
        item_features=item_features
    )
    
    # 设置用户和物品映射
    model.user_map = user_to_idx
    model.item_map = song_to_idx
    model.reverse_user_map = {v: k for k, v in user_to_idx.items()}
    model.reverse_item_map = {v: k for k, v in song_to_idx.items()}
    
    # 编译模型
    model.compile_model(learning_rate=args.learning_rate)
    
    # 训练模型
    history = model.fit(
        user_indices=user_indices,
        item_indices=item_indices,
        ratings=ratings,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1
    )
    
    training_time = time.time() - start_time
    logger.info(f"深度学习模型训练完成，耗时 {training_time:.2f} 秒")
    
    # 保存训练历史
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'mae': [float(x) for x in history.history['mae']],
        'val_mae': [float(x) for x in history.history['val_mae']]
    }
    
    return model, history_dict


def train_hybrid_model(interactions_train, songs, audio_features, user_features, deep_model=None):
    """训练混合推荐模型"""
    logger.info("开始训练混合推荐模型...")
    start_time = time.time()
    
    # 创建混合推荐模型
    model = HybridRecommender()
    
    # 训练模型
    model.train(
        interactions=interactions_train,
        songs=songs,
        audio_features=audio_features,
        user_features=user_features,
        train_deep_model=False  # 不在这里训练深度模型
    )
    
    # 单独设置深度模型
    if deep_model:
        model.deep_model = deep_model
        model.has_deep_model = True
        # 增加深度模型权重
        model.deep_weight = 0.35  # 提高深度学习权重到35%
        # 调整其他权重
        model.cf_weight = 0.3    # 协同过滤降至30%
        model.content_weight = 0.2  # 内容模型20%
        model.context_weight = 0.1  # 上下文模型10%
        model.mood_weight = 0.05   # 情绪模型5%
        
        logger.info(f"已集成深度学习模型，权重配置: 深度学习={model.deep_weight}, 协同过滤={model.cf_weight}, "
                   f"内容={model.content_weight}, 上下文={model.context_weight}, 情绪={model.mood_weight}")
    
    training_time = time.time() - start_time
    logger.info(f"混合推荐模型训练完成，耗时 {training_time:.2f} 秒")
    
    return model


def evaluate_recommender(model, interactions_test, deep_model=None, top_n=10):
    """评估推荐模型性能"""
    logger.info(f"开始评估模型性能（使用 top-{top_n} 推荐）...")
    
    # 按用户分组
    user_groups = interactions_test.groupby('user_id')
    
    # 初始化评估指标
    precision_sum = 0
    recall_sum = 0
    ndcg_sum = 0
    user_count = 0
    
    # 对每个用户评估
    for user_id, group in tqdm(user_groups, desc="评估用户", ncols=100):
        # 获取用户实际交互的歌曲集合
        actual_songs = set(group['song_id'].values)
        
        # 生成推荐
        try:
            if deep_model and hasattr(model, 'recommend_with_deep'):
                recommendations = model.recommend_with_deep(user_id, n=top_n, deep_model=deep_model)
            else:
                # 检查recommend方法的参数
                import inspect
                recommend_params = inspect.signature(model.recommend).parameters
                if 'top_n' in recommend_params:
                    recommendations = model.recommend(user_id, top_n=top_n)
                elif 'n' in recommend_params:
                    recommendations = model.recommend(user_id, n=top_n)
                else:
                    recommendations = model.recommend(user_id)
                    # 如果recommend方法不接受数量参数，手动截取前N个
                    if recommendations and len(recommendations) > top_n:
                        recommendations = recommendations[:top_n]
        except Exception as e:
            logger.warning(f"为用户 {user_id} 生成推荐时出错: {str(e)}")
            continue
        
        if not recommendations:
            continue
        
        # 提取推荐的歌曲ID，处理不同的推荐结果格式
        recommended_songs = []
        for rec in recommendations:
            if isinstance(rec, dict) and 'song_id' in rec:
                # 字典格式 {'song_id': id, 'score': score}
                recommended_songs.append(rec['song_id'])
            elif isinstance(rec, tuple) and len(rec) >= 1:
                # 元组格式 (song_id, score)
                recommended_songs.append(rec[0])
            elif isinstance(rec, str):
                # 直接是歌曲ID
                recommended_songs.append(rec)
        
        # 计算精确率和召回率
        hits = len(set(recommended_songs) & actual_songs)
        precision = hits / len(recommended_songs) if recommended_songs else 0
        recall = hits / len(actual_songs) if actual_songs else 0
        
        # 计算NDCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual_songs), top_n)))
        dcg = 0
        for i, song_id in enumerate(recommended_songs):
            if song_id in actual_songs:
                dcg += 1.0 / np.log2(i + 2)
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # 累加指标
        precision_sum += precision
        recall_sum += recall
        ndcg_sum += ndcg
        user_count += 1
    
    # 计算平均指标
    avg_precision = precision_sum / user_count if user_count > 0 else 0
    avg_recall = recall_sum / user_count if user_count > 0 else 0
    avg_ndcg = ndcg_sum / user_count if user_count > 0 else 0
    
    logger.info(f"评估结果 ({user_count} 用户):")
    logger.info(f"  精确率 (Precision@{top_n}): {avg_precision:.4f}")
    logger.info(f"  召回率 (Recall@{top_n}): {avg_recall:.4f}")
    logger.info(f"  NDCG@{top_n}: {avg_ndcg:.4f}")
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'ndcg': avg_ndcg,
        'user_count': user_count
    }


def save_models(deep_model, hybrid_model, output_dir, deep_history=None):
    """保存训练好的模型"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存深度学习模型
    if deep_model:
        deep_model_path = output_dir / 'deep_model'
        deep_model.save(str(deep_model_path))
        logger.info(f"深度学习模型已保存到 {deep_model_path}")
        
        # 保存训练历史
        if deep_history:
            with open(output_dir / 'deep_model_history.json', 'w', encoding='utf-8') as f:
                json.dump(deep_history, f, indent=2)
    
    # 保存混合模型
    if hybrid_model:
        hybrid_model_path = output_dir / 'hybrid_model.pkl'
        hybrid_model.save_model(str(hybrid_model_path))
        logger.info(f"混合推荐模型已保存到 {hybrid_model_path}")


def main():
    """程序主入口，解析命令行参数并执行训练流程"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='使用MSD数据集训练深度学习推荐模型')
    
    # 数据相关参数
    parser.add_argument('--h5_file', type=str, required=True, help='MSD HDF5文件路径')
    parser.add_argument('--triplet_file', type=str, required=True, help='三元组文件路径')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='处理后数据输出目录')
    parser.add_argument('--user_sample', type=int, default=None, help='用于训练的随机用户样本数量')
    parser.add_argument('--chunk_limit', type=int, default=None, help='限制处理的数据块(用于测试)')
    parser.add_argument('--no_filter_inactive', action='store_true', help='不过滤不活跃用户')
    parser.add_argument('--force_process', action='store_true', help='强制重新处理数据')
    
    # Spotify相关参数
    parser.add_argument('--use_spotify', action='store_true', help='使用Spotify API获取额外特征')
    parser.add_argument('--spotify_client_id', type=str, default=None, help='Spotify API客户端ID')
    parser.add_argument('--spotify_client_secret', type=str, default=None, help='Spotify API客户端密钥')
    parser.add_argument('--spotify_max_songs', type=int, default=10000, help='要处理的Spotify歌曲最大数量')
    parser.add_argument('--spotify_batch_size', type=int, default=50, help='Spotify API批处理大小')
    parser.add_argument('--spotify_workers', type=int, default=5, help='Spotify API并行工作线程数')
    parser.add_argument('--spotify_strategy', type=str, default='all', 
                        choices=['all', 'popular', 'diverse'], help='Spotify处理策略')
    parser.add_argument('--spotify_cache_file', type=str, default=None, help='自定义Spotify缓存文件')
    
    # 模型训练参数
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=1024, help='训练批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--embedding_size', type=int, default=64, help='嵌入向量维度')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--save_model', action='store_true', help='保存训练好的模型')
    parser.add_argument('--model_dir', type=str, default='models', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 打印控制台参数
    print("开始训练过程，使用以下参数:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # 如果指定了force_process，则打印相关信息
    if args.force_process:
        print("强制重新处理数据，忽略缓存")
    
    # 初始化数据处理器
    processor = MSDDataProcessor(
        output_dir=args.output_dir,
        use_spotify=args.use_spotify,
        spotify_client_id=args.spotify_client_id,
        spotify_client_secret=args.spotify_client_secret,
        spotify_batch_size=args.spotify_batch_size,
        spotify_workers=args.spotify_workers,
        spotify_strategy=args.spotify_strategy,
        spotify_cache_file=args.spotify_cache_file,
        force_process=args.force_process
    )
    
    # 处理MSD数据
    songs, interactions, audio_features, user_features = processor.process_msd_data(
        h5_file=args.h5_file,
        triplet_file=args.triplet_file,
        output_dir=args.output_dir,
        chunk_limit=args.chunk_limit,
        max_spotify_songs=args.spotify_max_songs,
        force_process=args.force_process,  # 传递force_process参数
        spotify_batch_size=args.spotify_batch_size,
        spotify_workers=args.spotify_workers,
        spotify_strategy=args.spotify_strategy,
        user_sample=args.user_sample,
        no_filter_inactive_users=args.no_filter_inactive
    )
    
    # 如果指定了最大交互数，限制数据量
    if hasattr(args, 'max_interactions') and args.max_interactions is not None and args.max_interactions > 0:
        logger.info(f"限制交互数据量到 {args.max_interactions} 条")
        # 确保保留热门交互
        if 'play_count' in interactions.columns:
            interactions = interactions.sort_values(by='play_count', ascending=False)
        elif 'count' in interactions.columns:
            interactions = interactions.sort_values(by='count', ascending=False)
        else:
            interactions = interactions.sort_values(by='rating', ascending=False)
            
        interactions = interactions.head(args.max_interactions)
        # 确保用户和歌曲ID在限制后的数据中仍然存在
        user_ids = interactions['user_id'].unique()
        song_ids = interactions['song_id'].unique()
        logger.info(f"限制后的数据包含 {len(user_ids)} 个用户和 {len(song_ids)} 首歌曲")
    
    # 分割训练集和测试集
    interactions_train, interactions_test = train_test_split(
        interactions, test_size=0.2, random_state=42
    )
    
    logger.info(f"训练集大小: {len(interactions_train)} 条记录, 测试集大小: {len(interactions_test)} 条记录")
    
    # 处理深度学习模型
    deep_model = None
    deep_history = None
    
    if not hasattr(args, 'skip_deep') or not args.skip_deep:
        # 训练深度学习模型
        logger.info("开始训练深度学习模型...")
        deep_model, deep_history = train_deep_model(interactions_train, songs, audio_features, args)
    else:
        logger.info("跳过深度学习模型训练")
    
    # 训练混合推荐模型
    if not hasattr(args, 'skip_hybrid') or not args.skip_hybrid:
        logger.info("开始训练混合推荐模型...")
        hybrid_model = train_hybrid_model(interactions_train, songs, audio_features, user_features, deep_model)
        
        # 评估混合推荐模型
        logger.info("评估混合推荐模型...")
        metrics = evaluate_recommender(hybrid_model, interactions_test)
        precision, recall, ndcg = metrics['precision'], metrics['recall'], metrics['ndcg']
        
        # 如果有深度学习模型，评估包含深度学习的混合推荐
        if deep_model:
            logger.info("使用深度学习模型进行混合推荐评估...")
            dl_metrics = evaluate_recommender(
                hybrid_model, interactions_test, deep_model=deep_model
            )
            dl_precision, dl_recall, dl_ndcg = dl_metrics['precision'], dl_metrics['recall'], dl_metrics['ndcg']
            logger.info(f"深度学习混合推荐结果:")
            logger.info(f"  精确率 (Precision@10): {dl_precision:.4f}")
            logger.info(f"  召回率 (Recall@10): {dl_recall:.4f}")
            logger.info(f"  NDCG@10: {dl_ndcg:.4f}")
        
        # 保存模型
        logger.info("保存模型...")
        models_dir = Path(args.output_dir) / 'models'
        save_models(deep_model, hybrid_model, models_dir, deep_history)
    else:
        logger.info("跳过混合模型训练")
    
    total_time = time.time() - start_time
    logger.info(f"总耗时: {total_time:.2f} 秒")
    logger.info("训练流程完成！")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 