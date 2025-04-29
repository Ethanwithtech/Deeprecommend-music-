#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音乐预览工具模块

提供音乐预览功能的工具函数和类，用于获取、管理和缓存歌曲的预览URL。
支持从多个来源(如Spotify)获取预览，并能够批量处理多个歌曲。
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

# 导入Spotify集成
from backend.spotify_integration import SpotifyManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('preview_utils')

class PreviewManager:
    """音乐预览管理器
    
    管理歌曲预览URL的获取和缓存，支持从不同来源获取预览。
    提供批量处理和推荐数据扩充功能。
    """
    
    def __init__(self, cache_file='preview_cache.json', cache_days=30):
        """初始化预览管理器
        
        参数:
            cache_file: 缓存文件路径，默认为当前目录下的preview_cache.json
            cache_days: 缓存有效期(天)，默认30天
        """
        # 缓存相关
        self.cache_file = cache_file
        self.cache_days = cache_days
        self.cache = {}
        
        # 统计信息
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 尝试加载缓存
        self._load_cache()
        
        # 初始化Spotify管理器
        self.spotify = SpotifyManager()
    
    def _load_cache(self):
        """从文件加载预览URL缓存"""
        if not os.path.exists(self.cache_file):
            logger.info(f"缓存文件 {self.cache_file} 不存在，将创建新缓存")
            return
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                self.cache = json.load(f)
                
            # 清理过期缓存
            self._clean_expired_cache()
            
            cache_size = len(self.cache)
            logger.info(f"已从 {self.cache_file} 加载 {cache_size} 条预览URL缓存")
            
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"加载缓存文件失败: {e}")
            self.cache = {}
    
    def _save_cache(self):
        """保存预览URL缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
                
            logger.info(f"已保存 {len(self.cache)} 条预览URL缓存到 {self.cache_file}")
            
        except IOError as e:
            logger.error(f"保存缓存文件失败: {e}")
    
    def _clean_expired_cache(self):
        """清理过期的缓存条目"""
        now = datetime.now()
        expiry_date = now - timedelta(days=self.cache_days)
        expiry_timestamp = expiry_date.timestamp()
        
        # 统计清理前缓存数量
        before_count = len(self.cache)
        
        # 过滤出未过期的缓存
        self.cache = {
            k: v for k, v in self.cache.items() 
            if v.get('timestamp', 0) > expiry_timestamp
        }
        
        # 计算清理数量
        cleaned_count = before_count - len(self.cache)
        if cleaned_count > 0:
            logger.info(f"已清理 {cleaned_count} 条过期的预览URL缓存")
    
    def _generate_cache_key(self, track_name: str, artist_name: str = None) -> str:
        """生成缓存键
        
        参数:
            track_name: 歌曲名称
            artist_name: 艺术家名称，可选
            
        返回:
            缓存键字符串
        """
        if artist_name:
            return f"{track_name.lower()}:{artist_name.lower()}"
        return track_name.lower()
    
    def get_preview_url(self, track_name: str, artist_name: str = None, force_refresh: bool = False) -> Optional[str]:
        """获取歌曲预览URL
        
        参数:
            track_name: 歌曲名称
            artist_name: 艺术家名称，可选
            force_refresh: 是否强制刷新缓存，默认False
            
        返回:
            预览URL，如果无法获取则返回None
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(track_name, artist_name)
        
        # 检查缓存
        if not force_refresh and cache_key in self.cache:
            self.cache_hits += 1
            logger.debug(f"缓存命中: {track_name} - {artist_name}")
            return self.cache[cache_key].get('preview_url')
        
        # 缓存未命中
        self.cache_misses += 1
        logger.debug(f"缓存未命中: {track_name} - {artist_name}")
        
        # 从Spotify获取预览URL
        preview_url = self._get_spotify_preview(track_name, artist_name)
        
        # 更新缓存
        if preview_url:
            self.cache[cache_key] = {
                'preview_url': preview_url,
                'timestamp': datetime.now().timestamp()
            }
            # 每10次缓存更新保存一次文件
            if (self.cache_hits + self.cache_misses) % 10 == 0:
                self._save_cache()
        
        return preview_url
    
    def _get_spotify_preview(self, track_name: str, artist_name: str = None) -> Optional[str]:
        """从Spotify获取预览URL
        
        参数:
            track_name: 歌曲名称
            artist_name: 艺术家名称，可选
            
        返回:
            预览URL，如果无法获取则返回None
        """
        try:
            # 搜索歌曲
            tracks = self.spotify.search_track(track_name, artist_name, limit=1)
            
            if not tracks:
                logger.warning(f"未在Spotify找到歌曲: {track_name} - {artist_name}")
                return None
            
            track = tracks[0]
            preview_url = track.get('preview_url')
            
            if not preview_url:
                logger.warning(f"歌曲没有预览URL: {track_name} - {artist_name}")
                return None
                
            logger.info(f"已获取预览URL: {track_name} - {artist_name}")
            return preview_url
            
        except Exception as e:
            logger.error(f"获取Spotify预览URL失败: {e}")
            return None
    
    def get_preview_urls_batch(self, songs: List[Dict]) -> Dict[str, str]:
        """批量获取多首歌曲的预览URL
        
        参数:
            songs: 歌曲信息列表，每个字典包含'name'和可选的'artist'键
            
        返回:
            预览URL字典，键为歌曲名:艺术家，值为预览URL
        """
        result = {}
        
        for song in songs:
            track_name = song.get('name')
            artist_name = song.get('artist')
            
            if not track_name:
                continue
                
            # 生成缓存键和结果键
            result_key = self._generate_cache_key(track_name, artist_name)
            
            # 获取预览URL
            preview_url = self.get_preview_url(track_name, artist_name)
            
            if preview_url:
                result[result_key] = preview_url
        
        # 保存更新的缓存
        self._save_cache()
        
        return result
    
    def enrich_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """为推荐结果添加预览URL
        
        参数:
            recommendations: 推荐结果列表，每个字典包含'name'和可选的'artist'键
            
        返回:
            扩充了预览URL的推荐结果列表
        """
        for rec in recommendations:
            track_name = rec.get('name')
            artist_name = rec.get('artist')
            
            if not track_name:
                continue
                
            # 获取预览URL
            preview_url = self.get_preview_url(track_name, artist_name)
            
            # 添加预览URL
            rec['preview_url'] = preview_url
        
        # 保存更新的缓存
        self._save_cache()
        
        return recommendations
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息
        
        返回:
            包含缓存统计信息的字典
        """
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'cache_size': len(self.cache),
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': f"{hit_rate:.2f}%",
            'total_requests': total_requests
        }


if __name__ == "__main__":
    # 测试代码
    preview_manager = PreviewManager()
    
    # 测试获取单个预览URL
    print("获取单个歌曲预览:")
    preview_url = preview_manager.get_preview_url("Shape of You", "Ed Sheeran")
    print(f"预览URL: {preview_url}")
    
    # 测试批量获取预览URL
    print("\n批量获取预览:")
    songs = [
        {'name': 'Bad Guy', 'artist': 'Billie Eilish'},
        {'name': 'Blinding Lights', 'artist': 'The Weeknd'},
        {'name': 'Dance Monkey', 'artist': 'Tones and I'}
    ]
    
    preview_urls = preview_manager.get_preview_urls_batch(songs)
    for song, url in preview_urls.items():
        print(f"{song}: {url}")
    
    # 测试扩充推荐结果
    print("\n扩充推荐结果:")
    recommendations = [
        {'name': 'Thunder', 'artist': 'Imagine Dragons', 'score': 0.95},
        {'name': 'Believer', 'artist': 'Imagine Dragons', 'score': 0.92},
        {'name': 'Something Just Like This', 'artist': 'The Chainsmokers', 'score': 0.88}
    ]
    
    enriched_recommendations = preview_manager.enrich_recommendations(recommendations)
    for rec in enriched_recommendations:
        print(f"{rec['name']} - {rec['artist']}: {rec.get('preview_url')}")
    
    # 显示缓存统计
    print("\n缓存统计:")
    stats = preview_manager.get_cache_stats()
    for key, value in stats.items():
        print(f"{key}: {value}") 