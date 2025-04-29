#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合推荐模型评估与算法对比分析
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.hybrid_music_recommender import HybridMusicRecommender

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """推荐模型评估工具"""
    
    def __init__(self, data_dir="processed_data", test_size=0.2, random_state=42):
        """
        初始化评估器
        
        参数:
            data_dir: 数据目录
            test_size: 测试集比例
            random_state: 随机种子
        """
        self.data_dir = data_dir
        self.test_size = test_size
        self.random_state = random_state
        self.ratings_df = None
        self.train_data = None
        self.test_data = None
        self.recommender = None
        self.results = {}
        
        # 创建结果目录
        self.results_dir = os.path.join(data_dir, "evaluation_results")
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self):
        """加载和分割数据"""
        logger.info("加载评分数据...")
        
        ratings_file = os.path.join(self.data_dir, "ratings.csv")
        if not os.path.exists(ratings_file):
            logger.error(f"评分数据文件不存在: {ratings_file}")
            return False
        
        self.ratings_df = pd.read_csv(ratings_file)
        logger.info(f"加载了 {len(self.ratings_df)} 条评分数据")
        
        # 划分训练集和测试集
        train_data, test_data = train_test_split(
            self.ratings_df, 
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        self.train_data = train_data.reset_index(drop=True)
        self.test_data = test_data.reset_index(drop=True)
        
        logger.info(f"训练集: {len(self.train_data)} 条, 测试集: {len(self.test_data)} 条")
        return True
    
    def initialize_recommender(self, use_msd=True):
        """初始化推荐器"""
        logger.info("初始化混合推荐系统...")
        
        # 训练数据目录
        train_dir = os.path.join(self.results_dir, "train_data")
        os.makedirs(train_dir, exist_ok=True)
        
        # 保存训练集
        self.train_data.to_csv(os.path.join(train_dir, "ratings.csv"), index=False)
        
        # 复制歌曲和用户数据
        songs_file = os.path.join(self.data_dir, "songs.csv")
        if os.path.exists(songs_file):
            songs_df = pd.read_csv(songs_file)
            songs_df.to_csv(os.path.join(train_dir, "songs.csv"), index=False)
        
        users_file = os.path.join(self.data_dir, "users.csv")
        if os.path.exists(users_file):
            users_df = pd.read_csv(users_file)
            users_df.to_csv(os.path.join(train_dir, "users.csv"), index=False)
        
        # 初始化推荐器
        self.recommender = HybridMusicRecommender(data_dir=train_dir, use_msd=use_msd)
        
        # 预训练
        logger.info("预训练推荐模型...")
        self.recommender.pretrain_with_msd()
        
        return True
    
    def evaluate_algorithm(self, algorithm, algorithm_name):
        """评估单个算法性能"""
        logger.info(f"评估 {algorithm_name} 算法性能...")
        
        predictions = []
        start_time = time.time()
        
        for idx, row in self.test_data.iterrows():
            user_id = row['user_id']
            song_id = row['song_id']
            actual_rating = row['rating']
            
            # 根据不同算法获取预测评分
            predicted_rating = None
            
            if algorithm == 'hybrid':
                # 使用混合推荐
                recs = self.recommender.recommend(user_id, top_n=100)
                for rec in recs:
                    if rec['song_id'] == song_id:
                        predicted_rating = rec['score'] * 5  # 归一化到1-5分
                        break
            
            elif algorithm == 'svdpp':
                # 使用SVD++
                algorithm_input = self.recommender._prepare_svdpp_input(
                    user_id, 
                    self.recommender.user_vectors.get(user_id, {'interaction_history': {'explicit_ratings': {}}})
                )
                svdpp_recs = self.recommender._svdpp_recommend(algorithm_input, top_n=100)
                for rec in svdpp_recs:
                    if rec['song_id'] == song_id:
                        predicted_rating = rec['predicted_score']
                        break
            
            elif algorithm == 'content_based':
                # 使用内容特征
                if user_id in self.recommender.user_vectors:
                    algorithm_input = self.recommender._prepare_content_input(self.recommender.user_vectors[user_id])
                    content_recs = self.recommender._content_based_recommend(algorithm_input, top_n=100)
                    for rec in content_recs:
                        if rec['song_id'] == song_id:
                            predicted_rating = rec['predicted_score'] * 5
                            break
            
            elif algorithm == 'context_aware':
                # 使用上下文感知
                if user_id in self.recommender.user_vectors:
                    algorithm_input = self.recommender._prepare_context_input(self.recommender.user_vectors[user_id])
                    context_recs = self.recommender._context_aware_recommend(algorithm_input, top_n=100)
                    for rec in context_recs:
                        if rec['song_id'] == song_id:
                            predicted_rating = rec['predicted_score'] * 5
                            break
            
            elif algorithm == 'mlp':
                # 使用MLP
                mlp_recs = self.recommender._mlp_recommend(user_id, top_n=100)
                for rec in mlp_recs:
                    if rec['song_id'] == song_id:
                        predicted_rating = rec['predicted_score'] * 5
                        break
            
            elif algorithm == 'ncf':
                # 使用NCF
                ncf_recs = self.recommender._ncf_recommend(user_id, top_n=100)
                for rec in ncf_recs:
                    if rec['song_id'] == song_id:
                        predicted_rating = rec['predicted_score'] * 5
                        break
            
            if predicted_rating is not None:
                predictions.append({
                    'user_id': user_id,
                    'song_id': song_id,
                    'actual_rating': actual_rating,
                    'predicted_rating': predicted_rating,
                    'error': abs(actual_rating - predicted_rating)  # 添加误差值用于后续t-test
                })
                
            # 每100个样本打印进度
            if (idx + 1) % 100 == 0:
                logger.info(f"已处理 {idx + 1}/{len(self.test_data)} 个测试样本...")
        
        # 计算评估指标
        if predictions:
            df = pd.DataFrame(predictions)
            
            # 计算性能指标
            mae = mean_absolute_error(df['actual_rating'], df['predicted_rating'])
            rmse = np.sqrt(mean_squared_error(df['actual_rating'], df['predicted_rating']))
            
            # 计算覆盖率（有多少测试样本能产生预测）
            coverage = len(predictions) / len(self.test_data) * 100
            
            # 计算执行时间
            execution_time = time.time() - start_time
            
            # 保存结果
            result = {
                'algorithm': algorithm_name,
                'mae': mae,
                'rmse': rmse,
                'coverage': coverage,
                'execution_time': execution_time,
                'predictions': df.to_dict('records'),
                'errors': df['error'].values.tolist()  # 保存误差列表用于统计测试
            }
            
            self.results[algorithm] = result
            
            logger.info(f"{algorithm_name} 评估结果: MAE={mae:.4f}, RMSE={rmse:.4f}, 覆盖率={coverage:.2f}%, 执行时间={execution_time:.2f}秒")
            
            # 保存预测结果
            output_file = os.path.join(self.results_dir, f"{algorithm}_predictions.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"预测结果已保存到 {output_file}")
            
            return result
        else:
            logger.warning(f"{algorithm_name} 没有产生任何预测结果")
            return None
    
    def evaluate_all_algorithms(self):
        """评估所有算法并比较性能"""
        algorithms = {
            'hybrid': '混合算法',
            'svdpp': 'SVD++协同过滤',
            'content_based': '内容特征匹配',
            'context_aware': '上下文感知',
            'mlp': 'MLP深度学习',
            'ncf': 'NCF神经协同过滤'
        }
        
        for algo, name in algorithms.items():
            self.evaluate_algorithm(algo, name)
        
        # 比较算法性能
        self.compare_algorithms()
        
        # 执行算法性能的统计显著性测试
        self.statistical_significance_test()
    
    def statistical_significance_test(self):
        """
        执行t-test统计显著性测试，比较混合算法与单一算法的性能差异
        """
        logger.info("执行统计显著性测试（t-test）...")
        
        if 'hybrid' not in self.results:
            logger.error("缺少混合算法的结果，无法执行统计测试")
            return
        
        hybrid_errors = self.results['hybrid']['errors']
        
        # 准备统计结果表格
        stats_results = []
        
        # 对每个单一算法执行t-test
        for algo in self.results:
            if algo == 'hybrid':
                continue
                
            algo_errors = self.results[algo]['errors']
            
            # 确保两组数据长度相同，取最小长度
            min_length = min(len(hybrid_errors), len(algo_errors))
            if min_length == 0:
                logger.warning(f"算法 {algo} 没有足够的数据进行比较")
                continue
                
            # 裁剪数据
            hybrid_sample = hybrid_errors[:min_length]
            algo_sample = algo_errors[:min_length]
            
            # 执行配对t-test (双尾测试)
            t_stat, p_value = stats.ttest_rel(hybrid_sample, algo_sample)
            
            # 判断显著性 (p < 0.05 表示有显著差异)
            is_significant = p_value < 0.05
            
            # 计算效应量 (Cohen's d)
            mean_diff = np.mean(hybrid_sample) - np.mean(algo_sample)
            pooled_std = np.sqrt((np.std(hybrid_sample)**2 + np.std(algo_sample)**2) / 2)
            effect_size = abs(mean_diff) / pooled_std if pooled_std > 0 else 0
            
            # 确定优势方
            better_algorithm = '混合算法' if np.mean(hybrid_sample) < np.mean(algo_sample) else self.results[algo]['algorithm']
            
            # 记录结果
            stats_results.append({
                'comparison': f'混合算法 vs {self.results[algo]["algorithm"]}',
                'hybrid_mean_error': np.mean(hybrid_sample),
                'algo_mean_error': np.mean(algo_sample),
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': is_significant,
                'effect_size': effect_size,
                'effect_strength': self._interpret_effect_size(effect_size),
                'better_algorithm': better_algorithm
            })
            
            logger.info(f"混合算法 vs {self.results[algo]['algorithm']}: t={t_stat:.4f}, p={p_value:.4f}, " +
                       f"显著性: {'是' if is_significant else '否'}, 效应量: {effect_size:.4f}")
        
        # 将结果保存为CSV
        if stats_results:
            df = pd.DataFrame(stats_results)
            output_file = os.path.join(self.results_dir, "statistical_test_results.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"统计测试结果已保存到 {output_file}")
            
            # 创建统计测试结果可视化
            self._generate_statistical_test_chart(df)
    
    def _interpret_effect_size(self, d):
        """解释Cohen's d效应量大小"""
        if d < 0.2:
            return "微小"
        elif d < 0.5:
            return "小"
        elif d < 0.8:
            return "中等"
        else:
            return "大"
    
    def _generate_statistical_test_chart(self, stats_df):
        """生成统计测试结果图表"""
        plt.figure(figsize=(12, 8))
        
        # 准备数据
        comparisons = stats_df['comparison'].tolist()
        p_values = stats_df['p_value'].tolist()
        effect_sizes = stats_df['effect_size'].tolist()
        
        # 设置颜色 - 显著性差异为红色
        colors = ['r' if p < 0.05 else 'gray' for p in p_values]
        
        # 绘制显著性和效应量
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # p值柱状图
        ax1.bar(comparisons, p_values, color=colors)
        ax1.axhline(y=0.05, color='black', linestyle='--', alpha=0.7, label='显著性水平 (p=0.05)')
        ax1.set_title('统计显著性 (p-value)')
        ax1.set_xticklabels(comparisons, rotation=45, ha='right')
        ax1.set_ylabel('p-value')
        ax1.legend()
        
        # 效应量柱状图
        ax2.bar(comparisons, effect_sizes, color=colors)
        ax2.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, label='微小效应 (d=0.2)')
        ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='小效应 (d=0.5)')
        ax2.axhline(y=0.8, color='blue', linestyle='--', alpha=0.7, label='中等效应 (d=0.8)')
        ax2.set_title('效应量 (Cohen\'s d)')
        ax2.set_xticklabels(comparisons, rotation=45, ha='right')
        ax2.set_ylabel('Effect Size (d)')
        ax2.legend()
        
        plt.tight_layout()
        
        # 保存图表
        output_file = os.path.join(self.results_dir, "statistical_test_chart.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"统计测试图表已保存到 {output_file}")
        plt.close()
        
    def compare_algorithms(self):
        """比较所有算法的性能"""
        logger.info("比较所有算法性能...")
        
        if not self.results:
            logger.warning("没有算法结果可供比较")
            return
        
        # 提取性能指标
        algorithms = []
        maes = []
        rmses = []
        coverages = []
        execution_times = []
        
        for algo, result in self.results.items():
            algorithms.append(result['algorithm'])
            maes.append(result['mae'])
            rmses.append(result['rmse'])
            coverages.append(result['coverage'])
            execution_times.append(result['execution_time'])
        
        # 创建比较表格
        comparison_df = pd.DataFrame({
            '算法': algorithms,
            'MAE': maes,
            'RMSE': rmses,
            '覆盖率(%)': coverages,
            '执行时间(秒)': execution_times
        })
        
        # 按MAE排序
        comparison_df = comparison_df.sort_values('MAE')
        
        # 保存比较结果
        output_file = os.path.join(self.results_dir, "algorithm_comparison.csv")
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"算法比较结果已保存到 {output_file}")
        
        # 生成性能对比图表
        self._generate_performance_charts()
        
        # 输出结果摘要
        logger.info("\n算法性能比较摘要:")
        logger.info(comparison_df.to_string(index=False))
        
        # 确定最佳算法
        best_algo = comparison_df.iloc[0]['算法']
        logger.info(f"\n在MAE指标上表现最佳的算法是: {best_algo}")
        
        return comparison_df 