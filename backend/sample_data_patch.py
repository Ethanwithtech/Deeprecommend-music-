#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
为HybridMusicRecommender类添加sample_size参数支持的补丁文件

用法:
1. 在引入HybridMusicRecommender类后导入此补丁
2. 调用apply_patch()函数

示例:
```python
from backend.models.hybrid_music_recommender import HybridMusicRecommender
from backend.sample_data_patch import apply_patch
apply_patch()
```
"""

import logging
import numpy as np

logger = logging.getLogger("SampleDataPatch")

def apply_patch():
    """应用补丁，给HybridMusicRecommender类添加对sample_size参数的支持"""
    
    from backend.models.hybrid_music_recommender import HybridMusicRecommender
    
    # 保存原始方法的引用
    original_pretrain_with_msd = HybridMusicRecommender.pretrain_with_msd
    
    # 定义新的方法
    def patched_pretrain_with_msd(self, msd_dir=None, spotify_data_path=None, chunk_limit=5, 
                                rating_style="log", sample_size=None):
        """
        扩展的pretrain_with_msd方法，添加了sample_size参数
        
        参数:
            msd_dir: MSD数据集目录
            spotify_data_path: Spotify API数据文件路径
            chunk_limit: 处理的数据块数限制
            rating_style: 评分转换方式
            sample_size: 要训练的数据样本数量，如果指定，将随机抽取这么多条记录
        """
        # 记录是否指定了sample_size
        self._sample_size = sample_size
        if sample_size:
            logger.info(f"将使用 {sample_size} 条随机样本进行训练")
        
        # 调用原始方法
        result = original_pretrain_with_msd(self, msd_dir, spotify_data_path, chunk_limit, rating_style)
        
        # 如果指定了sample_size，对数据进行抽样
        if result and sample_size and hasattr(self, 'ratings_df') and self.ratings_df is not None:
            # 确保sample_size不超过可用数据量
            available_samples = len(self.ratings_df)
            if sample_size > available_samples:
                logger.warning(f"请求的样本数 {sample_size} 超过了可用数据量 {available_samples}，将使用全部数据")
                return result
            
            logger.info(f"从 {available_samples} 条记录中随机抽取 {sample_size} 条")
            
            # 随机抽样
            self.ratings_df = self.ratings_df.sample(n=sample_size, random_state=42)
            
            # 更新用户和歌曲ID集合
            self.user_id_map = {uid: i for i, uid in enumerate(self.ratings_df['user_id'].unique())}
            self.song_id_map = {sid: i for i, sid in enumerate(self.ratings_df['song_id'].unique())}
            
            logger.info(f"抽样后的数据: {len(self.ratings_df)} 条记录, {len(self.user_id_map)} 用户, {len(self.song_id_map)} 歌曲")
        
        return result
    
    # 替换方法
    HybridMusicRecommender.pretrain_with_msd = patched_pretrain_with_msd
    logger.info("已应用sample_size参数支持补丁")
    
    return True 