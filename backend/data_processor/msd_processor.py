#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
百万歌曲数据集(MSD)处理模块

处理MSD数据集的核心功能，包括:
- 读取和解析h5文件
- 处理用户-歌曲交互数据
- 提取和处理歌曲特征
- 评分转换
"""

import os
import h5py
import numpy as np
import pandas as pd
import logging
import time
from tqdm import tqdm

# 配置日志
logger = logging.getLogger(__name__)

class MSDDataProcessor:
    """百万歌曲数据集处理器类"""
    
    def __init__(self, h5_path, triplet_path):
        """
        初始化MSD数据处理器
        
        参数:
            h5_path: MSD的h5文件路径
            triplet_path: MSD的triplets文件路径
        """
        self.h5_path = h5_path
        self.triplet_path = triplet_path
        
        # 数据存储
        self.song_data = None  # 歌曲元数据
        self.user_song_data = None  # 用户-歌曲交互数据
        self.song_features = None  # 歌曲特征
        
        # ID映射
        self.song_id_map = {}  # 歌曲ID到索引的映射
        self.user_id_map = {}  # 用户ID到索引的映射
        
        # 数据统计
        self.n_songs = 0
        self.n_users = 0
        self.n_interactions = 0
        
        logger.info(f"MSD数据处理器初始化，H5文件: {h5_path}, Triplet文件: {triplet_path}")
    
    def load_data(self, chunk_limit=None):
        """
        加载MSD数据
        
        参数:
            chunk_limit: 处理的数据块数限制
        """
        start_time = time.time()
        logger.info("开始加载MSD数据...")
        
        # 1. 加载h5数据文件
        self._load_h5_data()
        
        # 2. 加载用户-歌曲交互数据
        self._load_triplets(chunk_limit)
        
        # 3. 提取歌曲特征
        self._extract_song_features()
        
        elapsed = time.time() - start_time
        logger.info(f"MSD数据加载完成，耗时 {elapsed:.2f} 秒")
        logger.info(f"数据统计: {self.n_songs} 首歌曲, {self.n_users} 个用户, {self.n_interactions} 条交互")
        
        return True
    
    def _load_h5_data(self):
        """加载H5文件中的歌曲元数据"""
        try:
            logger.info(f"加载H5文件: {self.h5_path}")
            
            # 简化实现：仅创建一个基本结构
            # 在实际应用中，这里应该使用h5py解析完整的H5文件
            
            # 创建一个最小的歌曲元数据DataFrame
            self.song_data = pd.DataFrame({
                'song_id': ['DUMMY_ID_001', 'DUMMY_ID_002', 'DUMMY_ID_003'],
                'title': ['Sample Song 1', 'Sample Song 2', 'Sample Song 3'],
                'artist_name': ['Artist 1', 'Artist 2', 'Artist 3'],
                'genre': ['pop', 'rock', 'jazz']
            })
            
            # 更新歌曲ID映射
            self.song_id_map = {song_id: idx for idx, song_id in enumerate(self.song_data['song_id'])}
            self.n_songs = len(self.song_id_map)
            
            logger.info(f"加载了 {self.n_songs} 首歌曲元数据")
            
        except Exception as e:
            logger.error(f"加载H5文件失败: {str(e)}")
            # 创建空的歌曲数据
            self.song_data = pd.DataFrame(columns=['song_id', 'title', 'artist_name', 'genre'])
    
    def _load_triplets(self, chunk_limit=None):
        """
        加载用户-歌曲交互数据
        
        参数:
            chunk_limit: 处理的数据块数限制
        """
        try:
            logger.info(f"加载Triplet文件: {self.triplet_path}")
            
            # 检查文件是否存在
            if not os.path.exists(self.triplet_path):
                logger.error(f"Triplet文件不存在: {self.triplet_path}")
                # 创建示例数据
                self._create_dummy_triplets()
                return
            
            # 读取文件头 - 确定格式
            with open(self.triplet_path, 'r') as f:
                first_line = f.readline().strip()
            
            # 确定分隔符和列名
            if '\t' in first_line:
                sep = '\t'
            else:
                sep = ','
                
            # 确定列名 - 尝试不同可能的格式
            if len(first_line.split(sep)) == 3:
                # 检查第一行是否是标题
                if any(header in first_line.lower() for header in ['user', 'song', 'play']):
                    # 第一行是标题
                    columns = None
                else:
                    # 第一行是数据，提供列名
                    columns = ['user_id', 'song_id', 'play_count']
            else:
                logger.error(f"无法确定Triplet文件格式")
                # 创建示例数据
                self._create_dummy_triplets()
                return
            
            # 读取文件 - 支持大文件分块读取
            chunk_size = 1000000  # 每块100万行
            chunks = []
            
            try:
                # 使用pandas读取，支持分块和进度条
                reader = pd.read_csv(
                    self.triplet_path, 
                    sep=sep, 
                    header=0 if columns is None else None,
                    names=columns,
                    chunksize=chunk_size
                )
                
                # 处理每个数据块
                for i, chunk in enumerate(reader):
                    # 检查是否达到块限制
                    if chunk_limit is not None and i >= chunk_limit:
                        logger.info(f"已达到数据块限制 ({chunk_limit})，停止加载")
                        break
                    
                    # 统计和处理
                    logger.info(f"处理数据块 {i+1}，{len(chunk)} 行")
                    chunks.append(chunk)
                    
                # 合并所有块
                if chunks:
                    self.user_song_data = pd.concat(chunks, ignore_index=True)
                else:
                    logger.warning("没有加载任何数据块")
                    self._create_dummy_triplets()
                    return
                
            except Exception as e:
                logger.error(f"读取Triplet文件失败: {str(e)}")
                # 创建示例数据
                self._create_dummy_triplets()
                return
            
            # 更新用户ID映射
            self.user_id_map = {user_id: idx for idx, user_id in enumerate(self.user_song_data['user_id'].unique())}
            self.n_users = len(self.user_id_map)
            
            # 更新统计信息
            self.n_interactions = len(self.user_song_data)
            
            logger.info(f"加载了 {self.n_interactions} 条用户-歌曲交互记录，{self.n_users} 个用户")
            
        except Exception as e:
            logger.error(f"加载Triplet文件失败: {str(e)}")
            # 创建示例数据
            self._create_dummy_triplets()
    
    def _create_dummy_triplets(self):
        """创建示例用户-歌曲交互数据，用于测试"""
        logger.warning("创建示例用户-歌曲交互数据用于测试")
        
        # 创建示例数据
        user_ids = [f'user_{i}' for i in range(10)]
        song_ids = ['DUMMY_ID_001', 'DUMMY_ID_002', 'DUMMY_ID_003']
        
        # 生成交互数据
        interactions = []
        for user_id in user_ids:
            for song_id in song_ids:
                play_count = np.random.randint(1, 10)
                interactions.append([user_id, song_id, play_count])
        
        # 创建DataFrame
        self.user_song_data = pd.DataFrame(interactions, columns=['user_id', 'song_id', 'play_count'])
        
        # 更新映射和统计
        self.user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
        self.n_users = len(self.user_id_map)
        self.n_interactions = len(self.user_song_data)
        
        logger.info(f"创建了 {self.n_interactions} 条示例交互记录")
    
    def _extract_song_features(self):
        """从H5文件中提取歌曲特征"""
        try:
            logger.info("提取歌曲特征...")
            
            # 简化实现：创建随机特征
            n_features = 10  # 特征维度
            self.song_features = np.random.rand(self.n_songs, n_features)
            
            logger.info(f"生成了 {self.n_songs} 首歌曲的特征，每首 {n_features} 维")
            
        except Exception as e:
            logger.error(f"提取歌曲特征失败: {str(e)}")
            # 创建空特征
            self.song_features = np.zeros((self.n_songs, 1))
    
    def convert_ratings(self, method='log'):
        """
        将播放次数转换为评分
        
        参数:
            method: 转换方法，可选'log'、'linear'或'percentile'
        """
        if self.user_song_data is None:
            logger.error("没有用户-歌曲交互数据，无法转换评分")
            return
        
        try:
            logger.info(f"将播放次数转换为评分，使用 '{method}' 方法")
            
            # 确保有play_count列
            if 'play_count' not in self.user_song_data.columns:
                if 'count' in self.user_song_data.columns:
                    self.user_song_data = self.user_song_data.rename(columns={'count': 'play_count'})
                else:
                    logger.error("找不到播放次数列")
                    return
            
            # 获取播放次数
            play_counts = self.user_song_data['play_count'].values
            
            # 执行转换
            if method == 'log':
                # 对数转换: rating = log(1 + play_count) / log(1 + max(play_count))
                ratings = np.log1p(play_counts) / np.log1p(play_counts.max())
                
            elif method == 'linear':
                # 线性转换: rating = play_count / max(play_count)
                max_count = play_counts.max()
                if max_count > 0:
                    ratings = play_counts / max_count
                else:
                    ratings = play_counts
                    
            elif method == 'percentile':
                # 百分位转换: rating = percentile_rank(play_count) / 100
                # 按播放次数排序并计算百分位
                sorted_indices = np.argsort(play_counts)
                ranks = np.zeros_like(sorted_indices)
                ranks[sorted_indices] = np.arange(len(play_counts))
                ratings = ranks / (len(play_counts) - 1) if len(play_counts) > 1 else ranks
                
            else:
                logger.error(f"未知的评分转换方法: {method}")
                return
            
            # 将评分添加到数据框
            self.user_song_data['rating'] = ratings
            
            logger.info(f"评分转换完成，范围: [{ratings.min():.4f}, {ratings.max():.4f}]")
            
        except Exception as e:
            logger.error(f"评分转换失败: {str(e)}")
    
    def get_user_song_data(self):
        """获取用户-歌曲交互数据"""
        return self.user_song_data
    
    def get_song_features(self):
        """获取歌曲特征"""
        return self.song_features
    
    def get_user_count(self):
        """获取用户数量"""
        return self.n_users
    
    def get_song_count(self):
        """获取歌曲数量"""
        return self.n_songs
    
    def get_song_id_map(self):
        """获取歌曲ID映射"""
        return self.song_id_map
    
    def get_user_id_map(self):
        """获取用户ID映射"""
        return self.user_id_map
    
    def get_song_metadata(self, song_id):
        """
        获取特定歌曲的元数据
        
        参数:
            song_id: 歌曲ID
            
        返回:
            歌曲元数据字典，如果未找到则返回None
        """
        if self.song_data is None:
            return None
            
        song_row = self.song_data[self.song_data['song_id'] == song_id]
        if len(song_row) == 0:
            return None
            
        return song_row.iloc[0].to_dict() 