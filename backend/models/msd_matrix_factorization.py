#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSD矩阵分解模型

该模块实现了基于矩阵分解的推荐系统，用于处理Million Song Dataset数据。
"""

import os
import time
import logging
import numpy as np
import pickle
from tqdm import tqdm
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('msd_matrix_factorization')

class MSDMatrixFactorization:
    """
    基于矩阵分解的MSD推荐模型
    """
    
    def __init__(self, n_factors=100, max_iter=100, regularization=0.01, random_state=42):
        """
        初始化MSD矩阵分解模型
        
        参数:
            n_factors: 潜在因子数量
            max_iter: 最大迭代次数
            regularization: 正则化参数
            random_state: 随机种子
        """
        self.n_factors = n_factors
        self.max_iter = max_iter
        self.regularization = regularization
        self.random_state = random_state
        self.model = None
        self.user_factors = None
        self.song_factors = None
        self.user_mapping = None
        self.song_mapping = None
        self.song_metadata = None
        self.user_history = None
    
    def fit(self, user_song_matrix, user_mapping, song_mapping, song_metadata=None):
        """
        训练矩阵分解模型
        
        参数:
            user_song_matrix: 用户-歌曲评分矩阵
            user_mapping: 用户ID到矩阵索引的映射
            song_mapping: 歌曲ID到矩阵索引的映射
            song_metadata: 歌曲元数据字典
        
        返回:
            self
        """
        logger.info(f"开始训练矩阵分解模型，参数: n_factors={self.n_factors}, max_iter={self.max_iter}")
        
        start_time = time.time()
        
        # 保存映射关系
        self.user_mapping = user_mapping
        self.song_mapping = song_mapping
        self.song_metadata = song_metadata or {}
        
        # 准备用户历史数据
        self.user_history = {}
        for user_id, user_idx in user_mapping['user_to_index'].items():
            # 找出该用户评分非零的歌曲
            indices = user_song_matrix[user_idx].nonzero()[1]
            songs = [song_mapping['index_to_song'][idx] for idx in indices]
            self.user_history[user_id] = songs
        
        # 将密集矩阵转换为稀疏矩阵以提高性能
        sparse_matrix = csr_matrix(user_song_matrix)
        
        # 初始化并训练NMF模型
        self.model = NMF(
            n_components=self.n_factors,
            init='random',
            max_iter=self.max_iter,
            alpha=self.regularization,
            l1_ratio=0,
            random_state=self.random_state,
            verbose=0
        )
        
        logger.info(f"矩阵形状: {sparse_matrix.shape}")
        
        # 训练模型
        self.user_factors = self.model.fit_transform(sparse_matrix)
        self.song_factors = self.model.components_.T
        
        train_time = time.time() - start_time
        logger.info(f"模型训练完成，耗时: {train_time:.2f}秒")
        
        return self
    
    def save_model(self, output_file):
        """
        保存模型到文件
        
        参数:
            output_file: 输出文件路径
        """
        logger.info(f"保存模型到 {output_file}")
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 准备用户因子和歌曲因子的字典
        user_factors_dict = {}
        song_factors_dict = {}
        
        # 将用户因子转换为字典
        for user_id, user_idx in self.user_mapping['user_to_index'].items():
            user_factors_dict[user_id] = self.user_factors[user_idx]
        
        # 将歌曲因子转换为字典
        for song_id, song_idx in self.song_mapping['song_to_index'].items():
            song_factors_dict[song_id] = self.song_factors[song_idx]
        
        # 需要保存的数据
        model_data = {
            'user_factors': user_factors_dict,
            'song_factors': song_factors_dict,
            'song_metadata': self.song_metadata,
            'user_history': self.user_history,
            'model_params': {
                'n_factors': self.n_factors,
                'max_iter': self.max_iter,
                'regularization': self.regularization
            }
        }
        
        # 保存模型
        try:
            with open(output_file, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info("模型保存成功")
        except Exception as e:
            logger.error(f"保存模型时出错: {e}")
            raise
    
    @staticmethod
    def load_model(input_file):
        """
        从文件加载模型
        
        参数:
            input_file: 输入文件路径
            
        返回:
            加载的模型对象
        """
        logger.info(f"从 {input_file} 加载模型")
        
        try:
            with open(input_file, 'rb') as f:
                model_data = pickle.load(f)
            
            # 获取模型参数
            params = model_data['model_params']
            
            # 创建新模型
            model = MSDMatrixFactorization(
                n_factors=params['n_factors'],
                max_iter=params['max_iter'],
                regularization=params['regularization']
            )
            
            # 设置模型数据
            # 注意：这里不设置user_factors和song_factors矩阵，因为我们直接使用字典形式的因子
            model.user_mapping = None  # 不再需要映射
            model.song_mapping = None  # 不再需要映射
            model.song_metadata = model_data['song_metadata']
            model.user_history = model_data['user_history']
            
            logger.info("模型加载成功")
            return model, model_data
        except Exception as e:
            logger.error(f"加载模型时出错: {e}")
            raise

def train_msd_model(triplets_df, metadata_dict=None, output_file=None, **model_params):
    """
    训练MSD推荐模型的便捷函数
    
    参数:
        triplets_df: 包含user_id, song_id, play_count列的DataFrame
        metadata_dict: 歌曲元数据字典
        output_file: 模型输出文件路径
        **model_params: 模型参数
        
    返回:
        训练好的模型和用户-歌曲矩阵
    """
    from backend.utils.msd_data_utils import create_user_song_matrix, preprocess_ratings
    
    logger.info("准备训练数据")
    
    # 将播放次数转换为评分
    triplets_df['rating'] = preprocess_ratings(triplets_df['play_count'])
    
    # 创建用户-歌曲矩阵
    matrix, mappings = create_user_song_matrix(triplets_df)
    
    logger.info(f"用户-歌曲矩阵大小: {matrix.shape}")
    
    # 训练模型
    model = MSDMatrixFactorization(**model_params)
    model.fit(matrix, 
              {
                  'user_to_index': mappings['user_to_index'], 
                  'index_to_user': mappings['index_to_user']
              }, 
              {
                  'song_to_index': mappings['song_to_index'], 
                  'index_to_song': mappings['index_to_song']
              }, 
              metadata_dict
             )
    
    # 如果指定了输出文件，保存模型
    if output_file:
        model.save_model(output_file)
    
    return model, matrix 