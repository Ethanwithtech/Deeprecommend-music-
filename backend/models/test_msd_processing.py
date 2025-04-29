#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试MSD数据处理脚本

用于测试Million Song Dataset数据处理和模型训练
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import time
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入处理模块
from backend.models.process_msd_data import process_h5_file, create_mock_songs_data
from backend.models.hybrid_music_recommender import HybridMusicRecommender

def test_msd_processing(msd_path=None, output_dir="processed_data", sample_size=10000):
    """
    测试MSD数据处理

    参数:
        msd_path: MSD数据集路径
        output_dir: 输出目录
        sample_size: 处理的样本数量
    """
    logger.info("开始测试MSD数据处理...")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 如果未提供MSD路径，使用模拟数据
    if not msd_path:
        logger.info("未提供MSD路径，使用模拟数据")
        songs_data = create_mock_songs_data(sample_size)

        # 保存模拟歌曲数据
        songs_df = pd.DataFrame(songs_data)
        songs_df.to_csv(os.path.join(output_dir, "songs.csv"), index=False)
        logger.info(f"已保存 {len(songs_df)} 首模拟歌曲数据")

        # 创建模拟用户数据
        users_data = []
        for i in range(1000):
            user_id = f"U{i:06d}"
            user = {
                'user_id': user_id,
                'age': np.random.randint(18, 65),
                'gender': np.random.choice(['M', 'F']),
                'country': np.random.choice(['US', 'UK', 'JP', 'CN', 'DE', 'FR']),
                'registration_time': int(time.time()) - np.random.randint(0, 365*24*3600),
                'preferred_genres': np.random.choice(['pop', 'rock', 'jazz', 'classical', 'electronic'],
                                                   size=np.random.randint(1, 4)).tolist(),
                'emotional_state': np.random.choice(['happy', 'sad', 'excited', 'relaxed']),
                'activity_context': np.random.choice(['studying', 'working', 'exercising', 'relaxing', 'commuting']),
                'time_preference': np.random.choice(['morning', 'afternoon', 'evening', 'night']),
                'device_type': np.random.choice(['mobile', 'desktop', 'tablet', 'speaker'])
            }
            users_data.append(user)

        # 保存模拟用户数据
        users_df = pd.DataFrame(users_data)
        users_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)
        logger.info(f"已保存 {len(users_df)} 名模拟用户数据")

        # 创建模拟评分数据
        ratings_data = []
        for i in range(50000):
            user_idx = np.random.randint(0, len(users_data))
            song_idx = np.random.randint(0, len(songs_df))

            user_id = users_data[user_idx]['user_id']
            song_id = songs_df.iloc[song_idx]['song_id']

            # 基本评分数据
            rating = {
                'user_id': user_id,
                'song_id': song_id,
                'rating': np.random.randint(1, 6),
                'timestamp': int(time.time()) - np.random.randint(0, 30*24*3600)
            }

            # 添加上下文信息
            rating['emotional_state'] = np.random.choice(['happy', 'sad', 'excited', 'relaxed'])
            rating['activity'] = np.random.choice(['studying', 'working', 'exercising', 'relaxing', 'commuting'])
            rating['time_of_day'] = np.random.choice(['morning', 'afternoon', 'evening', 'night'])
            rating['device'] = np.random.choice(['mobile', 'desktop', 'tablet', 'speaker'])

            # 添加交互指标
            rating['skip_count'] = np.random.randint(0, 5)
            rating['completion_rate'] = np.random.uniform(0, 1)
            rating['repeat_count'] = np.random.randint(0, 10)

            ratings_data.append(rating)

        # 保存模拟评分数据
        ratings_df = pd.DataFrame(ratings_data)
        ratings_df.to_csv(os.path.join(output_dir, "ratings.csv"), index=False)
        logger.info(f"已保存 {len(ratings_df)} 条模拟评分数据")

    else:
        # 处理真实MSD数据
        logger.info(f"处理MSD数据: {msd_path}")

        # 检查路径是否存在
        if not os.path.exists(msd_path):
            logger.error(f"MSD路径不存在: {msd_path}")
            return False

        # 查找h5文件
        h5_files = []
        if os.path.isdir(msd_path):
            for root, dirs, files in os.walk(msd_path):
                for file in files:
                    if file.endswith('.h5'):
                        h5_files.append(os.path.join(root, file))
        elif os.path.isfile(msd_path) and msd_path.endswith('.h5'):
            h5_files = [msd_path]

        if not h5_files:
            logger.error(f"未找到h5文件: {msd_path}")
            return False

        logger.info(f"找到 {len(h5_files)} 个h5文件")

        # 处理h5文件
        all_songs = []
        for i, h5_file in enumerate(h5_files[:10]):  # 限制处理的文件数量
            logger.info(f"处理文件 {i+1}/{len(h5_files[:10])}: {h5_file}")
            songs = process_h5_file(h5_file, max_songs=sample_size//10)
            all_songs.extend(songs)

            # 如果已经处理了足够的歌曲，停止处理
            if len(all_songs) >= sample_size:
                all_songs = all_songs[:sample_size]
                break

        # 保存处理后的歌曲数据
        songs_df = pd.DataFrame(all_songs)
        songs_df.to_csv(os.path.join(output_dir, "songs.csv"), index=False)
        logger.info(f"已保存 {len(songs_df)} 首歌曲数据")

        # 创建模拟用户和评分数据（因为MSD不包含这些）
        # 这部分与上面的模拟数据创建相同
        # ...

    logger.info("MSD数据处理测试完成")
    return True

def test_msd_training(data_dir="processed_data"):
    """
    测试使用处理后的MSD数据训练模型

    参数:
        data_dir: 数据目录
    """
    logger.info("开始测试MSD数据训练...")

    # 检查数据目录
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return False

    # 检查必要的文件
    required_files = ["songs.csv", "users.csv", "ratings.csv"]
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            logger.error(f"缺少必要的文件: {file}")
            return False

    # 创建推荐模型
    recommender = HybridMusicRecommender(data_dir=data_dir, use_msd=True)

    # 加载数据
    logger.info("加载数据...")
    if not recommender.load_data():
        logger.error("加载数据失败")
        return False

    # 训练模型
    logger.info("训练模型...")
    if not recommender.train():
        logger.error("训练模型失败")
        return False

    # 测试推荐
    logger.info("测试推荐...")
    user_id = recommender.users_df['user_id'].iloc[0]

    # 测试不同上下文下的推荐
    contexts = [
        {"emotion": "happy", "activity": "exercising"},
        {"emotion": "sad", "activity": "relaxing"},
        {"emotion": "relaxed", "activity": "studying"},
        {"emotion": "excited", "activity": "socializing"}
    ]

    for i, context in enumerate(contexts):
        logger.info(f"\n测试上下文 {i+1}: {context}")
        recommendations = recommender.recommend(user_id, context=context, top_n=5)

        logger.info(f"上下文: {context}")
        logger.info("推荐结果:")
        for j, rec in enumerate(recommendations):
            song_id = rec.get('song_id', 'unknown')
            score = rec.get('predicted_score', 0.0)
            title = rec.get('title', 'Unknown Title')
            artist = rec.get('artist_name', 'Unknown Artist')
            explanation = rec.get('explanation', 'No explanation available')
            logger.info(f"{j+1}. {title} by {artist} (ID: {song_id}, 分数: {score:.2f}) - {explanation}")

    # 保存模型
    model_path = os.path.join(data_dir, "hybrid_model.pkl")
    if recommender.save_model(model_path):
        logger.info(f"模型已保存到: {model_path}")

    logger.info("MSD数据训练测试完成")
    return True

if __name__ == "__main__":
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="测试MSD数据处理和模型训练")
    parser.add_argument("--msd_path", help="MSD数据集路径", default=None)
    parser.add_argument("--output_dir", help="输出目录", default="processed_data")
    parser.add_argument("--sample_size", help="处理的样本数量", type=int, default=10000)
    parser.add_argument("--skip_processing", help="跳过数据处理", action="store_true")
    parser.add_argument("--skip_training", help="跳过模型训练", action="store_true")

    args = parser.parse_args()

    # 测试数据处理
    if not args.skip_processing:
        test_msd_processing(args.msd_path, args.output_dir, args.sample_size)

    # 测试模型训练
    if not args.skip_training:
        test_msd_training(args.output_dir)
