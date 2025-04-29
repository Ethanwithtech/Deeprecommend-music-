#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSD数据集处理工具
"""

import os
import numpy as np
import pandas as pd
import pickle
import logging
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('msd_utils')

class MSDRecommender:
    """MSD数据集推荐系统类"""
    
    def __init__(self, model_path=None):
        """
        初始化MSD推荐器
        
        参数:
            model_path: 模型文件路径
        """
        self.model_path = model_path
        self.model = None
        self.user_history = {}
        self.song_factors = {}
        self.song_metadata = {}
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        加载已训练的模型
        
        参数:
            model_path: 模型文件路径
        """
        logger.info(f"加载模型: {model_path}")
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
                
            self.model = data.get('model')
            self.user_history = data.get('user_history', {})
            self.song_factors = data.get('song_factors', {})
            self.song_metadata = data.get('song_metadata', {})
            
            logger.info(f"模型加载成功，包含 {len(self.song_metadata)} 首歌曲和 {len(self.user_history)} 位用户")
            return True
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_song_info(self, song_id):
        """
        获取歌曲信息
        
        参数:
            song_id: 歌曲ID
            
        返回:
            歌曲信息字典
        """
        return self.song_metadata.get(song_id, {})
    
    def get_user_history(self, user_id):
        """
        获取用户听歌历史
        
        参数:
            user_id: 用户ID
            
        返回:
            用户听过的歌曲ID列表
        """
        return self.user_history.get(user_id, [])
    
    def recommend_for_user(self, user_id, n=10, exclude_listened=True):
        """
        为用户推荐歌曲
        
        参数:
            user_id: 用户ID
            n: 推荐数量
            exclude_listened: 是否排除已听过的歌曲
            
        返回:
            推荐歌曲列表，每个元素为(歌曲ID, 分数)
        """
        if not self.model:
            logger.error("模型未加载")
            return []
        
        try:
            # 检查用户是否存在于训练集中
            if user_id not in self.user_history:
                logger.warning(f"用户 {user_id} 不在训练集中，使用冷启动推荐")
                return self.recommend_popular(n)
            
            # 获取用户已听过的歌曲
            listened_songs = set(self.user_history.get(user_id, []))
            
            # 为用户预测所有歌曲的评分
            predictions = []
            
            # 确保用户在训练集中
            if user_id in self.model.trainset._raw2inner_id_users:
                u_inner = self.model.trainset.to_inner_uid(user_id)
                
                for song_id in self.song_metadata:
                    # 如果排除已听过的歌曲且歌曲在已听列表中，则跳过
                    if exclude_listened and song_id in listened_songs:
                        continue
                    
                    # 检查歌曲是否在训练集中
                    if song_id in self.model.trainset._raw2inner_id_items:
                        i_inner = self.model.trainset.to_inner_iid(song_id)
                        pred = self.model.predict(u_inner, i_inner, clip=False, verbose=False).est
                        predictions.append((song_id, pred))
            
            # 按预测分数排序
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            # 返回前n个推荐
            return predictions[:n]
        except Exception as e:
            logger.error(f"为用户 {user_id} 生成推荐时出错: {e}")
            return []
    
    def recommend_similar_songs(self, song_id, n=10):
        """
        推荐与给定歌曲相似的歌曲
        
        参数:
            song_id: 歌曲ID
            n: 推荐数量
            
        返回:
            相似歌曲列表，每个元素为(歌曲ID, 相似度分数)
        """
        if song_id not in self.song_factors:
            logger.warning(f"歌曲 {song_id} 不在数据集中")
            return []
        
        # 获取目标歌曲的因子
        target_factors = self.song_factors[song_id]
        
        # 计算所有歌曲与目标歌曲的余弦相似度
        similarities = []
        for s_id, factors in self.song_factors.items():
            if s_id != song_id:
                # 计算余弦相似度
                similarity = np.dot(target_factors, factors) / (np.linalg.norm(target_factors) * np.linalg.norm(factors))
                similarities.append((s_id, similarity))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前n个最相似的歌曲
        return similarities[:n]
    
    def recommend_popular(self, n=10):
        """
        推荐热门歌曲（冷启动推荐）
        
        参数:
            n: 推荐数量
            
        返回:
            热门歌曲列表，每个元素为(歌曲ID, 热门度分数)
        """
        # 统计每首歌曲被播放的用户数量
        song_popularity = defaultdict(int)
        for user_id, songs in self.user_history.items():
            for song_id in songs:
                song_popularity[song_id] += 1
        
        # 按播放用户数量排序
        popular_songs = [(song_id, count) for song_id, count in song_popularity.items()]
        popular_songs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前n个热门歌曲
        return popular_songs[:n]
    
    def enrich_recommendations(self, recommendations):
        """
        用元数据充实推荐结果
        
        参数:
            recommendations: 推荐列表，每个元素为(歌曲ID, 分数)
            
        返回:
            充实后的推荐列表，每个元素为包含元数据的字典
        """
        result = []
        for song_id, score in recommendations:
            # 获取歌曲元数据
            metadata = self.get_song_info(song_id)
            
            # 创建推荐项
            item = {
                'song_id': song_id,
                'score': float(score),
                'title': metadata.get('title', '未知歌曲'),
                'artist': metadata.get('artist', '未知艺术家')
            }
            
            result.append(item)
        
        return result

def load_msd_recommender(model_path):
    """
    加载MSD推荐器的便捷函数
    
    参数:
        model_path: 模型文件路径
        
    返回:
        MSDRecommender实例
    """
    recommender = MSDRecommender(model_path)
    return recommender

if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) < 2:
        print("用法: python msd_utils.py <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    recommender = load_msd_recommender(model_path)
    
    # 打印一些统计信息
    print(f"加载了 {len(recommender.song_metadata)} 首歌曲")
    print(f"有 {len(recommender.user_history)} 位用户的听歌历史")
    
    # 获取示例用户
    if recommender.user_history:
        example_user = next(iter(recommender.user_history.keys()))
        print(f"\n为用户 {example_user} 生成推荐:")
        
        recommendations = recommender.recommend_for_user(example_user, n=5)
        enriched_recs = recommender.enrich_recommendations(recommendations)
        
        for i, rec in enumerate(enriched_recs, 1):
            print(f"{i}. {rec['title']} - {rec['artist']} (分数: {rec['score']:.2f})")
    
    # 获取示例歌曲
    if recommender.song_metadata:
        example_song = next(iter(recommender.song_metadata.keys()))
        song_info = recommender.get_song_info(example_song)
        print(f"\n歌曲 {example_song} 信息:")
        print(f"标题: {song_info.get('title', '未知')}")
        print(f"艺术家: {song_info.get('artist', '未知')}")
        
        print(f"\n与歌曲 {song_info.get('title', example_song)} 相似的歌曲:")
        similar_songs = recommender.recommend_similar_songs(example_song, n=5)
        enriched_similar = recommender.enrich_recommendations(similar_songs)
        
        for i, song in enumerate(enriched_similar, 1):
            print(f"{i}. {song['title']} - {song['artist']} (相似度: {song['score']:.2f})") 