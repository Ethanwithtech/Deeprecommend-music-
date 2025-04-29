#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推荐结果保存工具
从训练好的混合推荐模型生成推荐结果，并保存为JSON格式以便进行可视化
"""

import os
import sys
import json
import pickle
import random
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_hybrid_model(model_path):
    """加载混合推荐模型"""
    logger.info(f"正在加载混合推荐模型: {model_path}")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("模型加载成功")
        return model
    except Exception as e:
        logger.error(f"加载模型时出错: {str(e)}")
        return None

def load_test_interactions(test_file):
    """加载测试交互数据"""
    logger.info(f"正在加载测试交互数据: {test_file}")
    try:
        interactions = pd.read_parquet(test_file)
        logger.info(f"加载了 {len(interactions)} 条测试交互记录")
        return interactions
    except Exception as e:
        logger.error(f"加载测试数据时出错: {str(e)}")
        return None

def generate_recommendations(model, interactions=None, user_count=20, recs_per_user=10):
    """生成推荐结果
    
    参数:
        model: 混合推荐模型
        interactions: 测试交互数据（如果有）
        user_count: 要生成推荐的用户数量
        recs_per_user: 每个用户的推荐数量
    """
    logger.info(f"为 {user_count} 个用户生成每人 {recs_per_user} 条推荐...")
    
    recommendations = {}
    
    # 如果有测试交互数据，使用其中的用户
    if interactions is not None and len(interactions) > 0:
        # 获取唯一用户ID
        unique_users = interactions['user_id'].unique()
        # 如果用户太多，随机选取一部分
        if len(unique_users) > user_count:
            selected_users = np.random.choice(unique_users, user_count, replace=False)
        else:
            selected_users = unique_users
            
        logger.info(f"从测试数据中选择了 {len(selected_users)} 个用户")
    else:
        # 如果没有测试数据，从模型中选择随机用户
        if hasattr(model, 'user_id_map') and model.user_id_map:
            all_users = list(model.user_id_map.keys())
            # 如果用户太多，随机选取一部分
            if len(all_users) > user_count:
                selected_users = random.sample(all_users, user_count)
            else:
                selected_users = all_users
                
            logger.info(f"从模型中选择了 {len(selected_users)} 个用户")
        else:
            logger.error("无法获取用户列表")
            return {}
    
    # 为每个用户生成推荐
    for user_id in selected_users:
        try:
            # 尝试生成推荐
            user_recs = model.recommend(user_id, top_n=recs_per_user)
            
            if user_recs:
                recommendations[user_id] = user_recs
                logger.debug(f"为用户 {user_id} 生成了 {len(user_recs)} 条推荐")
        except Exception as e:
            logger.warning(f"为用户 {user_id} 生成推荐时出错: {str(e)}")
    
    logger.info(f"成功为 {len(recommendations)} 个用户生成了推荐")
    return recommendations

def save_recommendations(recommendations, output_file):
    """保存推荐结果为JSON文件"""
    logger.info(f"将推荐结果保存到: {output_file}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 将NumPy和Pandas数据类型转换为原生Python类型
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd._libs.tslibs.timestamps.Timestamp)):
            return obj.isoformat()
        else:
            return obj
    
    # 遍历并转换所有数据
    serializable_recs = {}
    for user_id, recs in recommendations.items():
        # 确保用户ID是字符串
        user_key = str(user_id)
        serializable_recs[user_key] = []
        
        for rec in recs:
            # 转换每个推荐项
            serializable_rec = {}
            for key, value in rec.items():
                serializable_rec[key] = convert_to_serializable(value)
            serializable_recs[user_key].append(serializable_rec)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_recs, f, ensure_ascii=False, indent=2)
        logger.info(f"推荐结果已保存到 {output_file}")
        return True
    except Exception as e:
        logger.error(f"保存推荐结果时出错: {str(e)}")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成并保存推荐结果')
    parser.add_argument('--model', type=str, required=True,
                      help='混合推荐模型文件路径')
    parser.add_argument('--test-data', type=str, default=None,
                      help='测试交互数据文件路径 (Parquet格式)')
    parser.add_argument('--output', type=str, default='recommendations.json',
                      help='输出的推荐结果JSON文件路径')
    parser.add_argument('--user-count', type=int, default=20,
                      help='要生成推荐的用户数量')
    parser.add_argument('--recs-per-user', type=int, default=10,
                      help='每个用户的推荐数量')
    parser.add_argument('--log-level', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                      default='INFO', help='日志级别')
    
    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    logger.setLevel(getattr(logging, args.log_level))
    
    # 加载混合推荐模型
    model = load_hybrid_model(args.model)
    if model is None:
        return 1
    
    # 加载测试交互数据（如果有）
    interactions = None
    if args.test_data:
        interactions = load_test_interactions(args.test_data)
    
    # 生成推荐
    recommendations = generate_recommendations(
        model, 
        interactions=interactions,
        user_count=args.user_count,
        recs_per_user=args.recs_per_user
    )
    
    if not recommendations:
        logger.error("未能生成任何推荐")
        return 1
    
    # 保存推荐结果
    if save_recommendations(recommendations, args.output):
        logger.info("推荐结果生成并保存成功")
        return 0
    else:
        logger.error("保存推荐结果失败")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 