#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化版MSD模型训练脚本 - 从处理好的数据直接训练模型
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import time
import argparse

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models.hybrid_recommender import HybridRecommender

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_dir):
    """加载处理好的数据文件"""
    logger.info(f"从 {data_dir} 加载数据...")
    
    files = {
        'songs': None,
        'interactions': None,
        'audio_features': None,
        'user_features': None
    }
    
    # 尝试加载CSV文件
    for name in files.keys():
        path = os.path.join(data_dir, f"{name}.csv")
        if os.path.exists(path):
            logger.info(f"加载 {path}")
            try:
                files[name] = pd.read_csv(path)
                logger.info(f"成功加载 {name}: {len(files[name])} 行")
            except Exception as e:
                logger.error(f"加载 {path} 时出错: {str(e)}")
    
    # 必须有歌曲和交互数据
    if files['songs'] is None or files['interactions'] is None:
        logger.error("必须提供歌曲和交互数据")
        return None
    
    # 查看交互数据的列并进行处理
    logger.info(f"交互数据列: {files['interactions'].columns.tolist()}")
    
    # 确保有必要的列: user_id 和 song_id
    required_cols = ['user_id', 'song_id']
    for col in required_cols:
        if col not in files['interactions'].columns:
            logger.error(f"交互数据缺少必要的列: {col}")
            return None
    
    # 创建或转换rating列
    if 'rating' not in files['interactions'].columns:
        if 'plays' in files['interactions'].columns:
            logger.info("从plays创建rating列")
            try:
                files['interactions']['rating'] = files['interactions']['plays'].apply(
                    lambda x: min(5, max(1, int(np.log2(float(x) if x > 0 else 1) + 1)))
                )
            except Exception as e:
                logger.error(f"从plays创建rating时出错: {str(e)}")
                # 简单回退方案: 使用固定值
                files['interactions']['rating'] = 3
        else:
            logger.warning("交互数据中缺少rating和plays列，使用默认评分")
            files['interactions']['rating'] = 3  # 默认中等评分
    else:
        # 确保rating是数值型
        try:
            files['interactions']['rating'] = pd.to_numeric(files['interactions']['rating'])
            logger.info("成功将rating转换为数值型")
        except Exception as e:
            logger.error(f"转换rating列为数值型时出错: {str(e)}")
            # 尝试使用正则表达式提取数字
            try:
                files['interactions']['rating'] = files['interactions']['rating'].astype(str).str.extract('(\d+)').astype(float)
                logger.info("通过提取数字成功转换rating列")
            except:
                # 最终回退方案
                logger.warning("无法转换rating列，使用默认值")
                files['interactions']['rating'] = 3
    
    # 检查rating范围是否在1-5之间
    rating_min = files['interactions']['rating'].min()
    rating_max = files['interactions']['rating'].max()
    logger.info(f"Rating范围: {rating_min} - {rating_max}")
    
    if rating_min < 1 or rating_max > 5:
        logger.warning(f"Rating范围异常: {rating_min}-{rating_max}，将进行归一化")
        try:
            # 重新缩放到1-5范围
            if rating_min == rating_max:
                files['interactions']['rating'] = 3  # 如果所有评分相同，设置为中等评分
            else:
                files['interactions']['rating'] = 1 + 4 * (files['interactions']['rating'] - rating_min) / (rating_max - rating_min)
        except Exception as e:
            logger.error(f"归一化rating时出错: {str(e)}")
            files['interactions']['rating'] = 3  # 默认中等评分
    
    # 如果没有音频特征，从歌曲数据提取
    if files['audio_features'] is None:
        try:
            feature_cols = ['tempo', 'loudness', 'duration', 'key', 'mode']
            # 添加可能存在的派生特征
            if 'energy_ratio' in files['songs'].columns:
                feature_cols.append('energy_ratio')
            if 'tempo_norm' in files['songs'].columns:
                feature_cols.append('tempo_norm')
            
            # 获取实际存在的列
            valid_cols = [col for col in feature_cols if col in files['songs'].columns]
            
            if valid_cols:
                logger.info(f"从歌曲数据创建音频特征，使用列: {valid_cols}")
                files['audio_features'] = files['songs'][valid_cols].copy()
                # 设置索引
                if 'song_id' in files['songs'].columns:
                    files['audio_features'].index = files['songs']['song_id']
            else:
                logger.warning("歌曲数据中没有有效的特征列，创建最小特征集")
                # 创建一个简单的特征集
                files['audio_features'] = pd.DataFrame(
                    np.random.randn(len(files['songs']), 1),
                    index=files['songs']['song_id'],
                    columns=['feature1']
                )
        except Exception as e:
            logger.error(f"创建音频特征时出错: {str(e)}")
            # 创建一个最小化特征集
            files['audio_features'] = pd.DataFrame(
                np.ones((len(files['songs']), 1)),
                index=files['songs']['song_id'] if 'song_id' in files['songs'].columns else range(len(files['songs'])),
                columns=['feature1']
            )
    
    # 如果没有用户特征，创建基本特征
    if files['user_features'] is None:
        try:
            logger.info("创建基本用户特征")
            # 确定要聚合的列
            agg_dict = {'song_id': 'nunique'}
            
            if 'rating' in files['interactions'].columns:
                agg_dict['rating'] = ['count', 'mean']
            
            if 'plays' in files['interactions'].columns:
                agg_dict['plays'] = ['sum', 'mean']
            
            # 执行分组聚合
            user_stats = files['interactions'].groupby('user_id').agg(agg_dict)
            
            # 整理列名
            if isinstance(user_stats.columns, pd.MultiIndex):
                user_stats.columns = ['_'.join(col).strip() for col in user_stats.columns.values]
            
            files['user_features'] = user_stats
        except Exception as e:
            logger.error(f"创建用户特征时出错: {str(e)}")
            # 创建最简单的用户特征
            unique_users = files['interactions']['user_id'].unique()
            files['user_features'] = pd.DataFrame(
                np.ones((len(unique_users), 1)),
                index=unique_users,
                columns=['activity']
            )
    
    # 返回处理后的数据
    logger.info("数据加载和处理完成")
    return files

def train_model(data, model_path, train_deep_model=False):
    """训练并保存模型"""
    logger.info("分割训练集和测试集...")
    train_data, test_data = train_test_split(data['interactions'], test_size=0.2, random_state=42)
    
    logger.info(f"训练集大小: {len(train_data)}, 测试集大小: {len(test_data)}")
    
    # 初始化模型
    logger.info("初始化混合推荐模型...")
    model = HybridRecommender()
    
    # 训练模型
    start_time = time.time()
    if train_deep_model:
        logger.info("将训练包含深度学习模型的完整混合推荐系统...")
        model.train(
            interactions=train_data,
            audio_features=data['audio_features'],
            songs=data['songs'],
            user_features=data['user_features'],
            train_deep_model=True
        )
    else:
        logger.info("训练基础混合推荐模型(不含深度学习)...")
        model.train(
            interactions=train_data,
            audio_features=data['audio_features'],
            songs=data['songs'],
            user_features=data['user_features']
        )
    
    # 记录训练时间
    elapsed = time.time() - start_time
    logger.info(f"模型训练完成，耗时 {elapsed:.2f} 秒")
    
    # 保存模型
    logger.info(f"保存模型到 {model_path}")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save_model(model_path)
    
    # 简单评估
    try:
        logger.info("测试模型...")
        test_user = test_data['user_id'].iloc[0]
        recommendations = model.recommend(test_user, top_n=5)
        logger.info(f"为用户 {test_user} 的推荐例子: {recommendations}")
    except Exception as e:
        logger.error(f"测试模型时出错: {str(e)}")
    
    return model

def main():
    parser = argparse.ArgumentParser(description="从处理好的MSD数据训练混合推荐模型")
    parser.add_argument("--data_dir", type=str, default="data/msd_processed", help="处理好的数据目录")
    parser.add_argument("--model_path", type=str, default="models/hybrid_model.pkl", help="模型保存路径")
    parser.add_argument("--train_deep_models", action="store_true", help="是否训练深度学习模型")
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("MSD混合推荐模型训练")
    logger.info("=" * 50)
    
    if args.train_deep_models:
        logger.info("将训练包含深度学习组件的完整模型")
        try:
            import tensorflow as tf
            logger.info(f"TensorFlow版本: {tf.__version__}")
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"找到 {len(gpus)} 个GPU设备")
                for gpu in gpus:
                    logger.info(f"  {gpu}")
                # 设置内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.info("未找到GPU，将使用CPU训练(可能较慢)")
        except ImportError:
            logger.error("未安装TensorFlow，无法训练深度学习模型")
            logger.error("请安装TensorFlow: pip install tensorflow")
            return
        except Exception as e:
            logger.error(f"配置TensorFlow时出错: {str(e)}")
    
    # 加载数据
    data = load_data(args.data_dir)
    if data is None:
        logger.error("加载数据失败，无法继续")
        return
    
    # 训练模型
    train_model(data, args.model_path, args.train_deep_models)
    
    logger.info("=" * 50)
    logger.info("训练完成!")
    logger.info("=" * 50)

if __name__ == "__main__":
    main() 