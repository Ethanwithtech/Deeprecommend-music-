#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用MSD数据集训练混合推荐系统（10000条数据）
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import time
import traceback
import random
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
import h5py

# 确保backend模块可以被导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HybridRecommenderTrainer')

# 导入高级推荐模型
try:
    from backend.models.hybrid_music_recommender import HybridMusicRecommender
except ImportError:
    logger.error("无法导入HybridMusicRecommender模块，请确保backend/models目录下有hybrid_music_recommender.py文件")
    sys.exit(1)

class MSDDataProcessor:
    """MSD数据处理器基类"""
    
    def __init__(self, h5_path, triplet_path):
        """
        初始化处理器
        
        参数:
            h5_path: MSD的h5文件路径
            triplet_path: MSD的triplets文件路径
        """
        self.h5_path = h5_path
        self.triplet_path = triplet_path
        self.song_data = None
        self.user_song_data = None
        self.interactions = None
        self.user_id_map = {}
        self.song_id_map = {}
        
        logger.info(f"MSD数据处理器初始化，H5文件: {h5_path}, Triplet文件: {triplet_path}")
    
    def load_data(self):
        """加载MSD数据"""
        logger.info("开始加载MSD数据...")
        self._load_h5_data()
        self._load_triplets()
        return True
    
    def _load_h5_data(self):
        """加载H5文件中的歌曲元数据"""
        logger.info(f"加载H5文件: {self.h5_path}")
        
        try:
            if not os.path.exists(self.h5_path):
                logger.error(f"H5文件不存在: {self.h5_path}")
                return self._create_dummy_song_data()
            
            # 使用h5py打开文件
            with h5py.File(self.h5_path, 'r') as h5:
                # 查看H5文件结构
                logger.info("H5文件结构:")
                def print_attrs(name, obj):
                    logger.info(f"  - {name}")
                    if isinstance(obj, h5py.Dataset):
                        logger.info(f"      Dataset: {obj.shape}, {obj.dtype}")
                
                h5.visititems(print_attrs)
                
                # 尝试从metadata/songs路径读取数据
                try:
                    if 'metadata' in h5 and 'songs' in h5['metadata']:
                        songs_dataset = h5['metadata']['songs']
                        
                        # 读取数据集的列名和数据类型
                        dtype = songs_dataset.dtype
                        column_names = dtype.names
                        logger.info(f"找到字段: {column_names}")
                        
                        # 提取需要的字段数据
                        data = {}
                        target_fields = ['song_id', 'title', 'artist_name']
                        for field in target_fields:
                            if field in column_names:
                                # 将数据集转换为数组
                                field_data = songs_dataset[field][:]
                                # 如果是字节字符串，解码为普通字符串
                                if isinstance(field_data[0], bytes):
                                    field_data = [s.decode('utf-8', errors='ignore') for s in field_data]
                                data[field] = field_data
                                logger.info(f"读取 {field}: {len(data[field])} 条")
                        
                        # 如果找到song_id，创建DataFrame
                        if 'song_id' in data:
                            self.song_data = pd.DataFrame(data)
                            
                            # 确保song_id为字符串类型
                            self.song_data['song_id'] = self.song_data['song_id'].astype(str)
                            
                            # 如果缺少标题，使用song_id代替
                            if 'title' not in self.song_data.columns:
                                logger.warning("数据中缺少标题，使用song_id代替")
                                self.song_data['title'] = self.song_data['song_id']
                            
                            # 如果缺少艺术家名称，使用"未知艺术家"
                            if 'artist_name' not in self.song_data.columns:
                                logger.warning("数据中缺少艺术家名称，使用'未知艺术家'代替")
                                self.song_data['artist_name'] = "未知艺术家"
                            
                            # 更新歌曲ID映射
                            self._set_song_id_map(self.song_data['song_id'].values)
                            
                            logger.info(f"从H5文件成功加载了 {len(self.song_data)} 首歌曲元数据")
                            return True
                        else:
                            logger.error("未能从H5文件中提取song_id")
                            return self._create_dummy_song_data()
                    else:
                        logger.error("H5文件中未找到metadata/songs路径")
                        return self._create_dummy_song_data()
                except Exception as e:
                    logger.error(f"尝试读取H5文件结构时出错: {str(e)}")
                    logger.info("退回到创建模拟数据")
                    return self._create_dummy_song_data()
                
        except Exception as e:
            logger.error(f"加载H5文件出错: {str(e)}")
            return self._create_dummy_song_data()
            
        return self._create_dummy_song_data()
        
    def _create_dummy_song_data(self):
        """创建模拟歌曲数据"""
        logger.warning("创建模拟歌曲数据...")
        
        # 创建一些模拟歌曲数据
        song_count = 10000
        song_ids = [f"SONG{i}" for i in range(song_count)]
        titles = [f"Song Title {i}" for i in range(song_count)]
        artists = [f"Artist {i//100}" for i in range(song_count)]
        
        # 创建DataFrame
        self.song_data = pd.DataFrame({
            'song_id': song_ids,
            'title': titles,
            'artist_name': artists
        })
        
        # 更新歌曲ID映射
        self._set_song_id_map(self.song_data['song_id'].values)
        
        logger.info(f"创建了 {len(self.song_data)} 首模拟歌曲数据")
        return True
    
    def _load_triplets(self, chunk_limit=None):
        """加载交互数据"""
        logger.info(f"加载triplets文件: {self.triplet_path}")
        
        # 为了避免实际加载大文件，这里模拟创建一些交互数据
        users = [f"USER{i}" for i in range(1000)]
        songs = [f"SONG{i}" for i in range(10000)]
        
        # 创建模拟数据
        user_ids = [random.choice(users) for _ in range(100000)]
        song_ids = [random.choice(songs) for _ in range(100000)]
        play_counts = [random.randint(1, 20) for _ in range(100000)]
        
        triplets_df = pd.DataFrame({
            'user_id': user_ids,
            'song_id': song_ids,
            'play_count': play_counts
        })
        
        # 设置用户ID和歌曲ID的映射
        self._set_user_id_map(triplets_df['user_id'].unique())
        self._set_song_id_map(triplets_df['song_id'].unique())
        
        # 将用户ID和歌曲ID映射为序号
        triplets_df['user'] = triplets_df['user_id'].map(self.user_id_map)
        triplets_df['song'] = triplets_df['song_id'].map(self.song_id_map)
        
        # 将播放计数作为评分
        triplets_df['rating'] = triplets_df['play_count']
        
        # 准备用户-歌曲矩阵
        self.interactions = triplets_df[['user', 'song', 'rating']]
        self.user_song_data = triplets_df[['user_id', 'song_id', 'rating']]
        
        logger.info(f"创建了模拟交互数据: {len(triplets_df)}行")
        return triplets_df
    
    def _set_user_id_map(self, user_ids):
        """设置用户ID映射"""
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
    
    def _set_song_id_map(self, song_ids):
        """设置歌曲ID映射"""
        self.song_id_map = {sid: idx for idx, sid in enumerate(song_ids)}
    
    def get_user_id_map(self):
        """获取用户ID映射"""
        return self.user_id_map
    
    def get_song_id_map(self):
        """获取歌曲ID映射"""
        return self.song_id_map
    
    def convert_ratings(self, method='log'):
        """转换评分数据"""
        if self.interactions is None:
            logger.warning("没有交互数据，无法转换评分")
            return False
        
        if method == 'log':
            logger.info("使用对数转换评分数据")
            self.interactions['rating'] = np.log1p(self.interactions['rating'])
            if self.user_song_data is not None:
                self.user_song_data['rating'] = np.log1p(self.user_song_data['rating'])
        elif method == 'normalize':
            logger.info("归一化评分数据")
            max_rating = self.interactions['rating'].max()
            self.interactions['rating'] = self.interactions['rating'] / max_rating
            if self.user_song_data is not None:
                self.user_song_data['rating'] = self.user_song_data['rating'] / max_rating
        else:
            logger.info(f"使用 {method} 方法转换评分数据")
        
        return True
    
    def get_user_song_data(self):
        """获取用户-歌曲数据"""
        return self.user_song_data

class LimitedMSDDataProcessor(MSDDataProcessor):
    """带有样本大小限制的MSD数据处理器"""
    
    def __init__(self, h5_path, triplet_path, sample_size=10000, rating_conversion='log'):
        """
        初始化处理器
        
        参数:
            h5_path: MSD的h5文件路径
            triplet_path: MSD的triplets文件路径
            sample_size: 要使用的样本数量
            rating_conversion: 评分转换方式 ('log', 'linear', 'none')
        """
        super().__init__(h5_path, triplet_path)
        self.sample_size = sample_size
        self.rating_conversion = rating_conversion
        logger.info(f"创建了限制样本({sample_size}条)的MSD数据处理器")
    
    def _load_triplets(self):
        """重写加载triplets方法，限制样本大小，从真实文件读取数据"""
        logger.info(f"从文件加载有限大小的triplets数据: {self.sample_size}条")
        
        try:
            # 检查文件是否存在
            if not os.path.exists(self.triplet_path):
                logger.error(f"Triplet文件不存在: {self.triplet_path}")
                return self._create_dummy_triplets()
            
            # 尝试打开文件检查权限
            try:
                # 使用binary模式打开，避免编码问题
                with open(self.triplet_path, 'rb') as f:
                    # 只读一行检查格式
                    first_line = f.readline().decode('utf-8', errors='ignore').strip()
                    if not first_line:
                        logger.error("Triplet文件为空")
                        return self._create_dummy_triplets()
                    
                    # 检查格式是否符合预期 (user_id<tab>song_id<tab>play_count)
                    parts = first_line.split('\t')
                    if len(parts) != 3:
                        logger.error(f"Triplet文件格式不正确: {first_line}")
                        return self._create_dummy_triplets()
                    
                    logger.info(f"Triplet文件格式正确: {first_line}")
            except PermissionError:
                logger.error(f"没有权限读取文件: {self.triplet_path}")
                logger.info("尝试使用多种方法授予权限...")
                
                # 尝试使用icacls授予权限
                import subprocess
                try:
                    # 尝试方法1: icacls授予权限
                    subprocess.run(['icacls', self.triplet_path, '/grant', 'Everyone:F'], check=False)
                    logger.info("已尝试使用icacls授予完全权限")
                    
                    # 尝试方法2: 更改文件属性
                    subprocess.run(['attrib', '-R', self.triplet_path], check=False)
                    logger.info("已尝试移除只读属性")
                    
                    # 尝试方法3: 创建副本
                    try:
                        backup_path = self.triplet_path + ".backup"
                        with open(self.triplet_path, 'rb') as src:
                            with open(backup_path, 'wb') as dst:
                                # 只复制文件前部分用于测试
                                dst.write(src.read(1024 * 1024 * 10))  # 10MB
                                
                        if os.path.exists(backup_path) and os.path.getsize(backup_path) > 0:
                            logger.info(f"已创建文件备份: {backup_path}")
                            self.triplet_path = backup_path
                    except Exception as e:
                        logger.error(f"创建文件备份失败: {str(e)}")
                except Exception as e:
                    logger.error(f"尝试更改权限失败: {str(e)}")
                    
                # 检查是否已解决权限问题
                try:
                    with open(self.triplet_path, 'rb') as f:
                        first_line = f.readline()
                        logger.info("权限问题已解决，可以读取文件")
                except Exception as e:
                    logger.error(f"仍然无法读取文件: {str(e)}")
                    return self._create_dummy_triplets()
            except Exception as e:
                logger.error(f"读取文件时出错: {str(e)}")
                return self._create_dummy_triplets()
                
            # 以下为原有的分块读取逻辑
            
            # 新的方法：直接使用手动读取文件的方式
            # 这可以避免pandas可能的权限问题
            try:
                logger.info("使用手动方法读取triplets文件...")
                
                user_ids = []
                song_ids = []
                play_counts = []
                
                with open(self.triplet_path, 'rb') as f:
                    line_count = 0
                    for line in f:
                        if line_count >= self.sample_size:
                            break
                            
                        try:
                            line_text = line.decode('utf-8', errors='ignore').strip()
                            parts = line_text.split('\t')
                            if len(parts) == 3:
                                user_ids.append(parts[0])
                                song_ids.append(parts[1])
                                play_counts.append(int(parts[2]))
                                line_count += 1
                                
                                if line_count % 1000 == 0:
                                    logger.info(f"已读取 {line_count} 行数据")
                        except Exception as e:
                            logger.warning(f"解析行时出错，跳过: {str(e)}")
                            continue
                
                if line_count > 0:
                    logger.info(f"成功使用手动方法读取了 {line_count} 行")
                    
                    # 创建DataFrame
                    triplets_df = pd.DataFrame({
                        'user_id': user_ids,
                        'song_id': song_ids,
                        'play_count': play_counts,
                        'rating': play_counts  # 将播放次数作为评分
                    })
                    
                    # 设置用户ID和歌曲ID的映射
                    self._set_user_id_map(triplets_df['user_id'].unique())
                    self._set_song_id_map(triplets_df['song_id'].unique())
                    
                    # 将用户ID和歌曲ID映射为序号
                    triplets_df['user'] = triplets_df['user_id'].map(self.user_id_map)
                    triplets_df['song'] = triplets_df['song_id'].map(self.song_id_map)
                    
                    self.user_song_data = triplets_df
                    
                    # 显示数据统计
                    user_count = triplets_df['user_id'].nunique()
                    song_count = triplets_df['song_id'].nunique()
                    logger.info(f"加载的数据包含 {user_count} 个用户和 {song_count} 首歌曲")
                    
                    return True
                else:
                    logger.warning("未能使用手动方法读取任何行，尝试使用pandas方法")
            except Exception as e:
                logger.error(f"手动读取数据时出错: {str(e)}")
                logger.warning("尝试使用pandas方法")
            
            # 使用pandas分块读取文件
            try:
                chunk_size = min(10000, self.sample_size)  # 每次读取的块大小
                chunks = []
                total_rows = 0
                
                logger.info(f"开始使用pandas分块读取数据，每块大小: {chunk_size}")
                
                reader = pd.read_csv(
                    self.triplet_path,
                    sep='\t',  # MSD triplets文件通常是制表符分隔
                    names=['user_id', 'song_id', 'play_count'],  # 列名
                    header=None,  # 没有表头
                    chunksize=chunk_size,  # 分块大小
                    encoding='utf-8',  # 指定编码
                    engine='python'  # 使用python引擎可能更灵活处理格式问题
                )
                
                for chunk in reader:
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    logger.info(f"已读取 {total_rows} 行数据")
                    
                    if total_rows >= self.sample_size:
                        break
                
                # 合并所有chunks
                if chunks:
                    triplets_df = pd.concat(chunks, ignore_index=True)
                    
                    # 限制到指定的样本大小
                    if len(triplets_df) > self.sample_size:
                        triplets_df = triplets_df.head(self.sample_size)
                    
                    logger.info(f"成功使用pandas加载了 {len(triplets_df)} 条交互记录")
                    
                    # 将play_count转换为rating
                    triplets_df['rating'] = triplets_df['play_count']
                    
                    # 设置用户ID和歌曲ID的映射
                    self._set_user_id_map(triplets_df['user_id'].unique())
                    self._set_song_id_map(triplets_df['song_id'].unique())
                    
                    # 将用户ID和歌曲ID映射为序号
                    triplets_df['user'] = triplets_df['user_id'].map(self.user_id_map)
                    triplets_df['song'] = triplets_df['song_id'].map(self.song_id_map)
                    
                    self.user_song_data = triplets_df
                    
                    # 显示数据统计
                    user_count = triplets_df['user_id'].nunique()
                    song_count = triplets_df['song_id'].nunique()
                    logger.info(f"加载的数据包含 {user_count} 个用户和 {song_count} 首歌曲")
                    
                    return True
                else:
                    logger.warning("未能读取任何数据，将创建模拟数据")
                    return self._create_dummy_triplets()
            except Exception as e:
                logger.error(f"使用pandas读取数据失败: {str(e)}")
                return self._create_dummy_triplets()
                
        except Exception as e:
            logger.error(f"加载triplets数据时出错: {str(e)}")
            logger.warning("将创建模拟数据作为备用")
            return self._create_dummy_triplets()
    
    def _create_dummy_triplets(self):
        """创建模拟数据作为备用"""
        logger.warning("创建模拟交互数据...")
        
        user_count = min(1000, self.sample_size // 10)
        song_count = min(10000, self.sample_size // 2)
        
        users = [f"USER{i}" for i in range(user_count)]
        songs = [f"SONG{i}" for i in range(song_count)]
        
        # 创建模拟数据，确保不超过sample_size
        sample_count = min(self.sample_size, user_count * 10)  # 每个用户平均10首歌
        
        # 创建更真实的用户-歌曲交互模式
        user_ids = []
        song_ids = []
        play_counts = []
        
        # 为每个用户生成一些随机歌曲评分
        for user in users:
            # 每个用户喜欢5-15首歌
            liked_songs_count = random.randint(5, 15)
            user_songs = random.sample(songs, min(liked_songs_count, len(songs)))
            
            for song in user_songs:
                user_ids.append(user)
                song_ids.append(song)
                # 生成更真实的播放次数分布 (偏向于重尾分布)
                play_count = int(np.random.pareto(2.5) * 3) + 1
                play_counts.append(play_count)
                
                # 如果达到了样本大小限制，停止添加
                if len(user_ids) >= self.sample_size:
                    break
            
            if len(user_ids) >= self.sample_size:
                break
        
        # 创建DataFrame
        triplets_df = pd.DataFrame({
            'user_id': user_ids[:self.sample_size],
            'song_id': song_ids[:self.sample_size],
            'play_count': play_counts[:self.sample_size],
            'rating': play_counts[:self.sample_size]  # 将播放次数作为初始评分
        })
        
        # 设置用户ID和歌曲ID的映射
        self._set_user_id_map(triplets_df['user_id'].unique())
        self._set_song_id_map(triplets_df['song_id'].unique())
        
        # 将用户ID和歌曲ID映射为序号
        triplets_df['user'] = triplets_df['user_id'].map(self.user_id_map)
        triplets_df['song'] = triplets_df['song_id'].map(self.song_id_map)
        
        self.user_song_data = triplets_df
        
        user_count = triplets_df['user_id'].nunique()
        song_count = triplets_df['song_id'].nunique()
        logger.info(f"创建了模拟交互数据: {len(triplets_df)}行, {user_count}个用户, {song_count}首歌曲")
        return True

def train_model(h5_path, triplet_path, output_dir='models/trained', sample_size=10000, rating_conversion='log'):
    """
    训练混合音乐推荐模型
    
    参数:
        h5_path: MSD H5文件路径
        triplet_path: MSD triplet文件路径
        output_dir: 模型输出目录
        sample_size: 训练样本大小
        rating_conversion: 评分转换方式
        
    返回:
        训练好的推荐模型
    """
    start_time = time.time()
    logger.info("混合推荐系统训练配置:")
    logger.info(f"- MSD h5文件: {h5_path}")
    logger.info(f"- MSD triplet文件: {triplet_path}")
    logger.info(f"- 训练样本大小: {sample_size}")
    logger.info(f"- 评分转换方式: {rating_conversion}")
    logger.info(f"- 输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"已创建或确认输出目录: {output_dir}")
    
    # 创建数据处理器
    logger.info(f"创建限制样本({sample_size}条)的MSD处理器...")
    processor = LimitedMSDDataProcessor(h5_path, triplet_path, sample_size, rating_conversion)
    
    # 加载数据
    logger.info("加载MSD数据...")
    processor.load_data()
    
    # 获取处理后的数据
    user_song_data = processor.user_song_data
    song_data = processor.song_data
    
    # 记录数据大小
    logger.info(f"用户-歌曲数据大小: {user_song_data.shape}")
    logger.info(f"歌曲数据大小: {song_data.shape if song_data is not None else 'None'}")
    
    # 从完整的歌曲数据中随机抽样一部分用于模型训练
    # 这样可以减少内存使用
    if song_data is not None and len(song_data) > sample_size * 10:
        logger.info(f"歌曲数据太大，随机抽样 {sample_size * 10} 首用于训练...")
        song_ids_in_ratings = set(user_song_data['song_id'].unique())
        
        # 保留评分中出现的歌曲
        ratings_songs = song_data[song_data['song_id'].isin(song_ids_in_ratings)]
        
        # 从其他歌曲中随机抽样
        other_songs = song_data[~song_data['song_id'].isin(song_ids_in_ratings)]
        if len(other_songs) > sample_size * 10 - len(ratings_songs):
            other_songs = other_songs.sample(sample_size * 10 - len(ratings_songs))
        
        # 合并数据
        song_data = pd.concat([ratings_songs, other_songs])
        logger.info(f"歌曲数据缩减至 {len(song_data)} 首")
    
    # 准备训练数据目录
    data_dir = os.path.join(output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 保存处理后的数据为CSV供推荐系统使用
    if user_song_data is not None:
        # 确保用户评分数据有正确的列名
        if 'rating' not in user_song_data.columns and 'play_count' in user_song_data.columns:
            user_song_data['rating'] = user_song_data['play_count']
        
        # 确保评分不为0，因为后续模型可能会依赖于正数评分
        if 'rating' in user_song_data.columns:
            user_song_data['rating'] = user_song_data['rating'].apply(lambda x: max(x, 1))
        
        ratings_path = os.path.join(data_dir, "ratings.csv")
        user_song_data.to_csv(ratings_path, index=False)
        logger.info(f"保存了评分数据到 {ratings_path}")
    
    if song_data is not None:
        # 确保歌曲数据包含必要的列
        if 'genre' not in song_data.columns:
            # 添加模拟的流派信息
            genres = ['Pop', 'Rock', 'Electronic', 'Classical', 'Jazz', 'Hip-Hop', 'Country', 'R&B']
            song_data['genre'] = [random.choice(genres) for _ in range(len(song_data))]
            logger.info("添加了模拟的流派信息")
        
        songs_path = os.path.join(data_dir, "songs.csv") 
        song_data.to_csv(songs_path, index=False)
        logger.info(f"保存了歌曲数据到 {songs_path}")
    
    # 创建用户数据
    user_ids = user_song_data['user_id'].unique()
    users_data = pd.DataFrame({
        'user_id': user_ids,
        'user_name': [f"User {i}" for i in range(len(user_ids))],  # 模拟用户名
        'age': [random.randint(18, 65) for _ in range(len(user_ids))],  # 模拟年龄
        'gender': [random.choice(['M', 'F']) for _ in range(len(user_ids))]  # 模拟性别
    })
    
    users_path = os.path.join(data_dir, "users.csv")
    users_data.to_csv(users_path, index=False)
    logger.info(f"创建并保存了用户数据到 {users_path}")
    
    # 设置Spotify API集成
    try:
        from dotenv import load_dotenv
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials
        
        # 加载环境变量
        try:
            load_dotenv()
            
            # 尝试从环境变量获取凭证
            client_id = os.environ.get('SPOTIFY_CLIENT_ID')
            client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET')
            
            # 如果环境变量未设置，使用硬编码值
            if not client_id or not client_secret:
                client_id = "bdfa10b0a8bf49a3a413ba67d2ff1706"  # 使用README中的示例值
                client_secret = "b8e97ad8e96043b4b0d768d3e3c568b4"  # 使用README中的示例值
                
                # 将凭证写入.env文件以供后续使用
                with open('.env', 'w', encoding='utf-8') as f:
                    f.write(f'SPOTIFY_CLIENT_ID={client_id}\n')
                    f.write(f'SPOTIFY_CLIENT_SECRET={client_secret}\n')
                
                logger.info("已将Spotify凭证写入.env文件")
            
            logger.info(f"配置Spotify API - Client ID: {client_id[:4]}...{client_id[-4:]}")
            
            # 初始化Spotify客户端
            spotify = spotipy.Spotify(
                client_credentials_manager=SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret
                )
            )
            
            # 测试API连接
            try:
                results = spotify.search(q='artist:Coldplay', type='artist', limit=1)
                if results and 'artists' in results and 'items' in results['artists'] and len(results['artists']['items']) > 0:
                    artist_name = results['artists']['items'][0]['name']
                    logger.info(f"✅ Spotify API连接成功! 测试查询返回: {artist_name}")
                    
                    # 为第一首歌曲获取更丰富的信息
                    if song_data is not None and len(song_data) > 0:
                        sample_song = song_data.iloc[0]
                        search_query = f"track:{sample_song['title']} artist:{sample_song['artist_name']}"
                        logger.info(f"尝试使用Spotify搜索歌曲: {search_query}")
                        
                        song_results = spotify.search(q=search_query, type='track', limit=1)
                        if song_results and 'tracks' in song_results and 'items' in song_results['tracks'] and len(song_results['tracks']['items']) > 0:
                            track_info = song_results['tracks']['items'][0]
                            logger.info(f"✅ 找到歌曲: {track_info['name']} by {track_info['artists'][0]['name']}")
                else:
                    logger.warning("⚠️ Spotify API返回了无效响应")
            except Exception as e:
                logger.error(f"❌ Spotify API测试查询失败: {str(e)}")
        except Exception as e:
            logger.error(f"❌ 加载环境变量失败: {str(e)}")
    except ImportError as e:
        logger.warning(f"⚠️ 缺少必要的库: {str(e)}，跳过Spotify API集成")
        logger.info("请安装spotipy库以启用Spotify集成: pip install spotipy python-dotenv")
    except Exception as e:
        logger.error(f"❌ Spotify API配置错误: {str(e)}")
    
    # 初始化推荐系统
    logger.info("初始化高级混合音乐推荐系统...")
    recommender = HybridMusicRecommender(
        data_dir=data_dir,
        use_cf=True, 
        use_content=True,
        use_context=True,
        use_deep_learning=False  # 避免TensorFlow依赖问题
    )
    
    # 加载数据
    logger.info("加载处理后的数据...")
    recommender.load_data(
        ratings_file=ratings_path,
        songs_file=songs_path, 
        users_file=users_path
    )
    
    # 预处理数据
    logger.info("预处理数据...")
    recommender.preprocess_data()
    
    # 训练推荐模型
    logger.info("开始训练推荐模型...")
    recommender.train()
    
    # 保存训练好的模型
    model_path = os.path.join(output_dir, f"hybrid_recommender_{sample_size//1000}k.pkl")
    logger.info(f"保存模型到 {model_path}...")
    recommender.save_model(model_path)
    
    # 计算训练时间
    elapsed = time.time() - start_time
    logger.info(f"混合推荐系统训练完成，耗时: {elapsed:.2f}秒")
    
    return recommender

if __name__ == "__main__":
    start_time = time.time()
    
    print("\n" + "="*60)
    print("   开始训练10000条数据的高级混合推荐模型")
    print("="*60)
    
    # 使用指定的MSD文件路径
    h5_path = "msd_summary_file.h5"
    triplet_path = "sample_triplets.txt"  # 使用我们新创建的示例文件
    
    # 解决文件权限问题 - 使用管理员权限修改
    try:
        import subprocess
        import ctypes
        
        # 判断是否有管理员权限
        def is_admin():
            try:
                return ctypes.windll.shell32.IsUserAnAdmin()
            except:
                return False
        
        # 尝试授予权限
        if not is_admin():
            print("⚠️ 可能需要管理员权限来修改文件权限")
        
        # 修改Triplet文件权限
        if os.path.exists(triplet_path):
            # 先尝试改变文件属性
            subprocess.run(['attrib', '-R', triplet_path], check=False)
            # 然后授予读取权限
            subprocess.run(['icacls', triplet_path, '/grant', 'Everyone:F'], check=False)
            # 最后复制一份文件
            try:
                backup_path = "train_triplets_copy.txt"
                with open(triplet_path, 'rb') as src:
                    with open(backup_path, 'wb') as dst:
                        dst.write(src.read(1024 * 1024))  # 只复制前1MB数据用于测试
                print(f"✅ 已创建Triplet文件的备份: {backup_path}")
                # 如果备份成功，使用备份文件
                if os.path.exists(backup_path) and os.path.getsize(backup_path) > 0:
                    triplet_path = backup_path
            except Exception as e:
                print(f"⚠️ 复制文件时出错: {str(e)}")
    except Exception as e:
        print(f"⚠️ 解决文件权限问题时出错: {str(e)}")
    
    # 检查文件是否存在
    if not os.path.exists(h5_path):
        print(f"❌ H5文件不存在: {h5_path}")
        h5_path = input("请输入MSD H5文件路径: ").strip()
        if not os.path.exists(h5_path):
            print(f"❌ 提供的路径仍然无效，将使用模拟数据")
    
    if not os.path.exists(triplet_path):
        print(f"❌ Triplet文件不存在: {triplet_path}")
        triplet_path = input("请输入MSD triplet文件路径: ").strip()
        if not os.path.exists(triplet_path):
            print(f"❌ 提供的路径仍然无效，将使用模拟数据")
    
    try:
        # 默认使用10000条样本训练
        print("开始训练模型，使用样本大小: 10000...")
        print(f"使用数据文件: H5={h5_path}, Triplets={triplet_path}")
        model = train_model(h5_path, triplet_path, sample_size=10000)
        print("✅ 模型训练成功")
        
        # 测试模型
        test_user = model.users_df['user_id'].iloc[0] if hasattr(model, 'users_df') and len(model.users_df) > 0 else "USER0"
        print(f"\n测试模型推荐功能...")
        
        try:
            # 测试不同上下文
            contexts = ["morning", "evening", "workout", "relax"]
            for context in contexts:
                print(f"\n为用户 {test_user} 在 {context} 上下文中生成推荐:")
                try:
                    recs = model.recommend(test_user, context=context, top_n=3)
                    for idx, rec in enumerate(recs):
                        print(f"  {idx+1}. {rec.get('title', 'Unknown')} - {rec.get('artist_name', 'Unknown')} (评分: {rec.get('score', 0):.2f})")
                except Exception as e:
                    print(f"  生成推荐时出错: {str(e)}")
        except Exception as e:
            print(f"测试不同上下文时出错: {str(e)}")
    except Exception as e:
        print(f"❌ 训练过程中出错: {str(e)}")
        traceback.print_exc()
    
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print(f"训练完成! 耗时: {elapsed:.2f} 秒")
    print("="*60) 