#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用Million Song Dataset训练推荐模型

此脚本用于处理MSD数据集并训练混合推荐模型
"""

import os
import sys
import logging
import argparse
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
from backend.models.extract_msd_content_features import process_msd_directory, normalize_features, reduce_dimensions
from backend.models.hybrid_music_recommender import HybridMusicRecommender

def process_msd_data(msd_path, output_dir, max_songs=None, max_files=None):
    """
    处理MSD数据集

    参数:
        msd_path: MSD数据集路径
        output_dir: 输出目录
        max_songs: 最大处理歌曲数量
        max_files: 最大处理文件数量

    返回:
        处理成功返回True，否则返回False
    """
    logger.info(f"处理MSD数据集: {msd_path}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    try:
        # 检查路径是否存在
        if not os.path.exists(msd_path):
            logger.error(f"MSD路径不存在: {msd_path}")
            return False

        # 直接检查指定的h5文件
        summary_file = os.path.join(msd_path, "msd_summary_file.h5")
        if os.path.exists(summary_file):
            logger.info(f"找到指定的摘要文件: {summary_file}")
            h5_files = [summary_file]
        else:
            # 如果指定文件不存在，搜索其他可能的h5文件
            logger.warning(f"指定的摘要文件不存在: {summary_file}，搜索其他h5文件")
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

        # 限制处理文件数量
        if max_files and len(h5_files) > max_files:
            h5_files = h5_files[:max_files]
            logger.info(f"限制处理 {max_files} 个文件")

        # 处理h5文件
        all_songs = []
        batch_size = min(1000, max_songs // 10 if max_songs else 1000)

        for i, h5_file in enumerate(h5_files):
            logger.info(f"处理文件 {i+1}/{len(h5_files)}: {h5_file}")
            try:
                # 调用process_h5_file函数
                songs_df = process_h5_file(h5_file)
                # 如果需要限制歌曲数量，取前batch_size首
                songs = songs_df.head(batch_size).to_dict('records') if len(songs_df) > 0 else []
                all_songs.extend(songs)

                # 如果已经处理了足够的歌曲，停止处理
                if max_songs and len(all_songs) >= max_songs:
                    all_songs = all_songs[:max_songs]
                    break

                # 每处理10个文件，保存一次中间结果
                if (i + 1) % 10 == 0:
                    temp_df = pd.DataFrame(all_songs)
                    temp_file = os.path.join(output_dir, f"songs_temp_{i+1}.csv")
                    temp_df.to_csv(temp_file, index=False)
                    logger.info(f"保存中间结果: {temp_file}")
            except Exception as e:
                logger.error(f"处理h5文件 {h5_file} 时出错: {str(e)}")
                logger.exception("详细错误信息:")
                continue

        # 保存处理后的歌曲数据
        songs_df = pd.DataFrame(all_songs)

        # 如果没有提取到歌曲数据，创建一些模拟数据
        if len(songs_df) == 0:
            logger.warning("未从h5文件提取到歌曲数据，创建模拟数据")
            from backend.models.process_msd_data import create_mock_songs_data
            songs_df = create_mock_songs_data(10000)

        # 确保songs_df有clean_song_id列
        if 'clean_song_id' not in songs_df.columns:
            logger.info("添加clean_song_id列用于匹配")
            songs_df['clean_song_id'] = songs_df['song_id'].apply(
                lambda x: x[-7:] if isinstance(x, str) and len(x) > 7 else x
            )

        songs_file = os.path.join(output_dir, "songs.csv")
        songs_df.to_csv(songs_file, index=False)
        logger.info(f"已保存 {len(songs_df)} 首歌曲数据到 {songs_file}")

        # 提取音频特征
        logger.info("提取音频特征...")
        try:
            features_file = os.path.join(output_dir, "audio_features.csv")
            features_df = process_msd_directory(msd_path, features_file, max_files)

            # 标准化特征
            logger.info("标准化音频特征...")
            norm_file = os.path.join(output_dir, "audio_features_normalized.csv")
            norm_features_df = normalize_features(features_df, norm_file)

            # 降维特征
            logger.info("降维音频特征...")
            reduced_file = os.path.join(output_dir, "audio_features_reduced.csv")
            reduced_features_df = reduce_dimensions(norm_features_df, 20, reduced_file)

            # 合并特征与元数据
            logger.info("合并特征与元数据...")
            merged_file = os.path.join(output_dir, "songs_with_features.csv")
            merged_df = pd.merge(songs_df, reduced_features_df, on='song_id', how='left')
            merged_df.to_csv(merged_file, index=False)
            logger.info(f"已保存合并数据到 {merged_file}")

            # 更新songs_df为合并后的数据
            songs_df = merged_df
        except Exception as e:
            logger.error(f"处理音频特征时出错: {str(e)}")
            logger.exception("详细错误信息:")
            # 继续处理，使用原始songs_df

        # 使用MSD数据集中的真实用户数据
        logger.info("处理MSD用户数据...")

        # 从MSD数据中提取用户信息
        # 尝试从Taste Profile数据集中提取用户信息
        taste_profile_path = os.path.join(os.path.dirname(msd_path), "taste_profile")
        if os.path.exists(taste_profile_path):
            logger.info(f"找到Taste Profile数据: {taste_profile_path}")
            # 处理Taste Profile数据
            user_data = extract_user_data_from_taste_profile(taste_profile_path, output_dir)
        else:
            # 如果没有Taste Profile数据，从交互数据中提取用户
            logger.info("未找到Taste Profile数据，从交互数据中提取用户")
            user_data = extract_users_from_interactions(msd_path, output_dir)

        # 保存用户数据
        users_file = os.path.join(output_dir, "users.csv")
        user_data.to_csv(users_file, index=False)
        logger.info(f"已保存 {len(user_data)} 名用户数据到 {users_file}")

        # 处理评分/交互数据
        logger.info("处理MSD评分数据...")
        ratings_df = extract_ratings_from_msd(msd_path, user_data, songs_df, output_dir)

        # 保存评分数据
        ratings_file = os.path.join(output_dir, "ratings.csv")
        ratings_df.to_csv(ratings_file, index=False)
        logger.info(f"已保存 {len(ratings_df)} 条评分数据到 {ratings_file}")

        return True

    except Exception as e:
        logger.error(f"处理MSD数据集时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return False

def extract_user_data_from_taste_profile(taste_profile_path, output_dir):
    """从Taste Profile数据集中提取用户信息"""
    logger.info("从Taste Profile提取用户数据")

    # 查找用户数据文件
    user_files = []
    for root, dirs, files in os.walk(taste_profile_path):
        for file in files:
            if file.endswith('.txt') or file.endswith('.csv') or file.endswith('.dat'):
                user_files.append(os.path.join(root, file))

    if not user_files:
        logger.warning("未找到Taste Profile用户数据文件")
        return pd.DataFrame({'user_id': [f"U{i:06d}" for i in range(100)]})

    # 提取用户ID
    user_ids = set()
    for file in user_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        user_ids.add(parts[0])
        except Exception as e:
            logger.warning(f"处理文件 {file} 时出错: {str(e)}")

    # 创建用户数据
    users_data = []
    for user_id in user_ids:
        user = {
            'user_id': user_id,
            'registration_time': int(time.time()) - np.random.randint(0, 365*24*3600)
        }
        users_data.append(user)

    return pd.DataFrame(users_data)

def extract_users_from_interactions(msd_path, output_dir):
    """从交互数据中提取用户"""
    logger.info("从交互数据中提取用户")

    # 查找交互数据文件
    interaction_files = []
    for root, dirs, files in os.walk(msd_path):
        for file in files:
            if 'triplets' in file.lower() or 'interaction' in file.lower():
                interaction_files.append(os.path.join(root, file))

    if not interaction_files:
        logger.warning("未找到交互数据文件")
        return pd.DataFrame({'user_id': [f"U{i:06d}" for i in range(100)]})

    # 提取用户ID
    user_ids = set()
    for file in interaction_files[:5]:  # 限制处理文件数量
        try:
            with open(file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i > 100000:  # 限制处理行数
                        break
                    parts = line.strip().split()
                    if len(parts) >= 1:
                        user_ids.add(parts[0])
        except Exception as e:
            logger.warning(f"处理文件 {file} 时出错: {str(e)}")

    # 如果没有找到用户，创建一些默认用户
    if not user_ids:
        user_ids = [f"U{i:06d}" for i in range(100)]

    # 创建用户数据
    users_data = []
    for user_id in user_ids:
        user = {
            'user_id': user_id,
            'registration_time': int(time.time()) - np.random.randint(0, 365*24*3600)
        }
        users_data.append(user)

    return pd.DataFrame(users_data)

def extract_ratings_from_msd(msd_path, users_df, songs_df, output_dir):
    """从MSD数据中提取评分/交互数据"""
    logger.info("从MSD数据中提取评分数据")

    # 直接检查指定的triplets文件
    triplets_file = os.path.join(msd_path, "train_triplets.txt", "train_triplets.txt")
    if os.path.exists(triplets_file):
        logger.info(f"找到指定的triplets文件: {triplets_file}")
        rating_files = [triplets_file]
    else:
        # 如果指定文件不存在，搜索其他可能的文件
        logger.warning(f"指定的triplets文件不存在: {triplets_file}，搜索其他文件")
        rating_files = []
        for root, dirs, files in os.walk(msd_path):
            for file in files:
                if 'triplets' in file.lower() or 'interaction' in file.lower() or 'rating' in file.lower():
                    rating_files.append(os.path.join(root, file))

    # 如果没有找到评分文件，尝试查找Last.fm数据
    if not rating_files:
        lastfm_path = os.path.join(os.path.dirname(msd_path), "lastfm")
        if os.path.exists(lastfm_path):
            for root, dirs, files in os.walk(lastfm_path):
                for file in files:
                    if file.endswith('.txt') or file.endswith('.csv') or file.endswith('.dat'):
                        rating_files.append(os.path.join(root, file))

    # 如果仍然没有找到评分文件，创建一些模拟评分
    if not rating_files:
        logger.warning("未找到评分数据文件，创建模拟评分")
        return create_simulated_ratings(users_df, songs_df)

    # 提取评分数据
    ratings_data = []
    valid_user_ids = set(users_df['user_id'])
    valid_song_ids = set(songs_df['song_id'])

    for file in rating_files[:3]:  # 限制处理文件数量
        try:
            logger.info(f"处理评分文件: {file}")

            # 检测文件编码和格式
            encoding = 'utf-8'
            try:
                with open(file, 'r', encoding=encoding) as f:
                    sample_line = f.readline()
            except UnicodeDecodeError:
                # 如果UTF-8解码失败，尝试其他编码
                encoding = 'latin-1'
                with open(file, 'r', encoding=encoding) as f:
                    sample_line = f.readline()

            logger.info(f"使用编码: {encoding}")
            logger.info(f"样本行: {sample_line}")

            # 检测分隔符
            if '\t' in sample_line:
                separator = '\t'
            elif ',' in sample_line:
                separator = ','
            else:
                separator = None  # 使用空白字符作为分隔符

            logger.info(f"使用分隔符: {separator}")

            # 处理特殊格式的triplets文件
            with open(file, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i > 500000:  # 限制处理行数，但处理更多行以获取足够数据
                        break

                    # 处理可能的字节字符串格式
                    if line.startswith("b'") and line.endswith("',\n"):
                        # 移除字节字符串标记和末尾的逗号
                        line = line[2:-3]
                        # 替换转义的制表符和换行符
                        line = line.replace('\\t', '\t').replace('\\n', '')

                    # 分割行
                    if separator:
                        parts = line.strip().split(separator)
                    else:
                        parts = line.strip().split()

                    if len(parts) >= 3:
                        user_id = parts[0]
                        song_id = parts[1]

                        # 检查用户ID和歌曲ID是否有效
                        # 对于triplets文件，我们可能需要添加用户
                        if user_id not in valid_user_ids:
                            # 添加新用户
                            new_user = {'user_id': user_id}
                            users_df = pd.concat([users_df, pd.DataFrame([new_user])], ignore_index=True)
                            valid_user_ids.add(user_id)

                        # 对于歌曲ID，我们需要确保它在songs_df中
                        # 首先尝试直接匹配
                        song_match = songs_df[songs_df['song_id'] == song_id]

                        # 如果没有直接匹配，尝试使用clean_song_id
                        if len(song_match) == 0 and 'clean_song_id' in songs_df.columns:
                            clean_id = song_id[-7:] if len(song_id) > 7 else song_id
                            song_match = songs_df[songs_df['clean_song_id'] == clean_id]

                        if len(song_match) > 0:
                            # 使用匹配的歌曲ID
                            matched_song_id = song_match.iloc[0]['song_id']

                            # 尝试提取播放次数并转换为评分
                            try:
                                play_count = int(parts[2])
                                # 将播放次数转换为评分 (1-5)
                                # 使用对数缩放以处理大范围的播放次数
                                if play_count > 0:
                                    rating_value = 1 + min(4, np.log1p(play_count) / np.log(10) * 2)
                                else:
                                    rating_value = 1.0
                            except:
                                # 如果无法解析为播放次数，使用默认评分
                                rating_value = 3.0

                            # 创建评分记录
                            rating = {
                                'user_id': user_id,
                                'song_id': matched_song_id,
                                'rating': rating_value,
                                'play_count': int(parts[2]) if parts[2].isdigit() else 1,
                                'timestamp': int(time.time()) - np.random.randint(0, 30*24*3600)
                            }

                            # 添加上下文信息 (基于歌曲特征推断)
                            song_row = song_match.iloc[0]
                            if song_row is not None:
                                # 基于歌曲特征推断情感状态
                                if 'energy' in song_row:
                                    energy = song_row['energy']
                                    if energy > 0.8:
                                        rating['emotional_state'] = 'excited'
                                    elif energy > 0.6:
                                        rating['emotional_state'] = 'happy'
                                    elif energy < 0.4:
                                        rating['emotional_state'] = 'relaxed'
                                    else:
                                        rating['emotional_state'] = 'neutral'
                                else:
                                    rating['emotional_state'] = np.random.choice(['happy', 'sad', 'excited', 'relaxed'])

                                # 基于歌曲特征推断活动
                                if 'tempo' in song_row:
                                    tempo = song_row['tempo']
                                    if tempo > 120:
                                        rating['activity'] = 'exercising'
                                    elif tempo < 80:
                                        rating['activity'] = 'relaxing'
                                    else:
                                        rating['activity'] = 'studying'
                                else:
                                    rating['activity'] = np.random.choice(['studying', 'working', 'exercising', 'relaxing', 'commuting'])
                            else:
                                rating['emotional_state'] = np.random.choice(['happy', 'sad', 'excited', 'relaxed'])
                                rating['activity'] = np.random.choice(['studying', 'working', 'exercising', 'relaxing', 'commuting'])

                            # 添加时间和设备信息
                            rating['time_of_day'] = np.random.choice(['morning', 'afternoon', 'evening', 'night'])
                            rating['device'] = np.random.choice(['mobile', 'desktop', 'tablet', 'speaker'])

                            ratings_data.append(rating)

                            # 定期打印进度
                            if len(ratings_data) % 10000 == 0:
                                logger.info(f"已处理 {len(ratings_data)} 条评分数据")
        except Exception as e:
            logger.warning(f"处理文件 {file} 时出错: {str(e)}")
            logger.exception("详细错误信息:")

    # 如果没有提取到足够的评分，补充一些模拟评分
    if len(ratings_data) < 1000:
        logger.warning(f"只提取到 {len(ratings_data)} 条评分，补充模拟评分")
        simulated_ratings = create_simulated_ratings(users_df, songs_df, 10000 - len(ratings_data))
        ratings_data.extend(simulated_ratings.to_dict('records'))
    else:
        logger.info(f"成功提取了 {len(ratings_data)} 条评分数据")

    return pd.DataFrame(ratings_data)

def create_simulated_ratings(users_df, songs_df, num_ratings=10000):
    """创建模拟评分数据"""
    logger.info(f"创建 {num_ratings} 条模拟评分数据")

    ratings_data = []
    for i in range(num_ratings):
        if i % 1000 == 0:
            logger.info(f"已创建 {i} 条模拟评分")

        user_idx = np.random.randint(0, len(users_df))
        song_idx = np.random.randint(0, len(songs_df))

        user_id = users_df.iloc[user_idx]['user_id']
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

        ratings_data.append(rating)

    return pd.DataFrame(ratings_data)

def train_model(data_dir, model_path):
    """
    训练推荐模型

    参数:
        data_dir: 数据目录
        model_path: 模型保存路径

    返回:
        训练成功返回True，否则返回False
    """
    logger.info(f"使用数据目录 {data_dir} 训练模型")

    try:
        # 检查数据目录
        if not os.path.exists(data_dir):
            logger.error(f"数据目录不存在: {data_dir}")
            return False

        # 检查必要的文件
        required_files = ["songs.csv", "users.csv", "ratings.csv"]
        for file in required_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                logger.error(f"缺少必要的文件: {file_path}")
                return False

        # 创建推荐模型
        recommender = HybridMusicRecommender(
            data_dir=data_dir,
            use_msd=True,
            use_cf=True,
            use_content=True,
            use_context=True,
            use_deep_learning=False  # 深度学习模型训练较慢，默认不启用
        )

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

        # 保存模型
        logger.info(f"保存模型到 {model_path}")
        if not recommender.save_model(model_path):
            logger.error("保存模型失败")
            return False

        # 测试模型
        logger.info("测试模型...")

        # 安全获取用户ID
        test_user_id = None
        if recommender.users_df is not None and len(recommender.users_df) > 0:
            test_user_id = recommender.users_df['user_id'].iloc[0]
        else:
            # 如果没有用户数据，使用默认用户ID
            test_user_id = "U000000"
            logger.warning(f"没有找到用户数据，使用默认用户ID: {test_user_id}")

        # 测试不同上下文下的推荐
        contexts = [
            {"emotion": "happy", "activity": "exercising"},
            {"emotion": "sad", "activity": "relaxing"},
            {"emotion": "relaxed", "activity": "studying"},
            {"emotion": "excited", "activity": "socializing"}
        ]

        for i, context in enumerate(contexts):
            logger.info(f"\n测试上下文 {i+1}: {context}")
            try:
                recommendations = recommender.recommend(test_user_id, context=context, top_n=5)

                logger.info(f"上下文: {context}")
                logger.info("推荐结果:")
                for j, rec in enumerate(recommendations):
                    song_id = rec.get('song_id', 'unknown')
                    score = rec.get('score', 0.0)
                    title = rec.get('title', 'Unknown Title')
                    artist = rec.get('artist_name', 'Unknown Artist')
                    explanation = rec.get('explanation', 'No explanation available')
                    logger.info(f"{j+1}. {title} by {artist} (ID: {song_id}, 分数: {score:.2f}) - {explanation}")
            except Exception as e:
                logger.error(f"测试上下文 {context} 时出错: {str(e)}")
                logger.exception("详细错误信息:")

        return True

    except Exception as e:
        logger.error(f"训练模型时出错: {str(e)}")
        logger.exception("详细错误信息:")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="使用Million Song Dataset训练推荐模型")
    parser.add_argument("--msd_path", help="MSD数据集路径", required=True)
    parser.add_argument("--output_dir", help="输出目录", default="msd_processed")
    parser.add_argument("--model_path", help="模型保存路径", default="msd_model.pkl")
    parser.add_argument("--max_songs", help="最大处理歌曲数量", type=int, default=10000)
    parser.add_argument("--max_files", help="最大处理文件数量", type=int, default=100)
    parser.add_argument("--skip_processing", help="跳过数据处理", action="store_true")
    parser.add_argument("--skip_training", help="跳过模型训练", action="store_true")

    args = parser.parse_args()

    # 处理MSD数据
    if not args.skip_processing:
        logger.info("开始处理MSD数据...")
        if not process_msd_data(args.msd_path, args.output_dir, args.max_songs, args.max_files):
            logger.error("处理MSD数据失败")
            return 1

    # 训练模型
    if not args.skip_training:
        logger.info("开始训练模型...")
        if not train_model(args.output_dir, args.model_path):
            logger.error("训练模型失败")
            return 1

    logger.info("处理完成")
    return 0

if __name__ == "__main__":
    sys.exit(main())
