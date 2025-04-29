#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify预览工具模块

此模块提供获取Spotify歌曲预览URL的功能，包括缓存机制，
用于在音乐推荐系统中快速获取歌曲预览片段。
"""

import os
import json
import logging
import time
from datetime import datetime
import sys

# 将当前目录添加到Python路径，以便导入父目录的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spotify_integration import SpotifyManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spotify_preview')

class SpotifyPreviewHelper:
    """Spotify预览URL获取器
    
    提供获取并缓存Spotify歌曲预览URL的功能，
    用于增强音乐推荐系统的用户体验。
    """
    
    def __init__(self, client_id=None, client_secret=None, cache_file=None):
        """初始化Spotify预览助手
        
        参数:
            client_id: Spotify应用客户端ID
            client_secret: Spotify应用客户端密钥
            cache_file: 预览URL缓存文件路径
        """
        # 初始化Spotify管理器
        self.spotify = SpotifyManager(client_id, client_secret)
        
        # 设置缓存文件路径
        self.cache_file = cache_file or os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'processed_data', 
            'preview_cache.json'
        )
        
        # 初始化缓存
        self.cache = self._load_cache()
        
        # 缓存统计
        self.cache_hits = 0
        self.cache_misses = 0
    
    def _load_cache(self):
        """从文件加载预览URL缓存
        
        返回:
            加载的缓存字典
        """
        try:
            if os.path.exists(self.cache_file):
                logger.info(f"从 {self.cache_file} 加载预览缓存")
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                logger.info(f"成功加载 {len(cache_data)} 条预览缓存记录")
                return cache_data
            else:
                logger.info(f"预览缓存文件 {self.cache_file} 不存在，创建新缓存")
                # 确保目录存在
                os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
                return {}
        except Exception as e:
            logger.error(f"加载预览缓存出错: {e}")
            return {}
    
    def _save_cache(self):
        """保存预览URL缓存到文件"""
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            
            logger.info(f"保存 {len(self.cache)} 条预览缓存记录")
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存预览缓存出错: {e}")
    
    def get_preview_url(self, track_name, artist_name):
        """获取歌曲的预览URL
        
        首先尝试从缓存中获取，如果没有则从Spotify API获取
        
        参数:
            track_name: 歌曲名称
            artist_name: 艺术家名称
            
        返回:
            包含预览URL和元数据的字典，如果未找到则返回None
        """
        # 构建缓存键
        cache_key = f"{track_name}|{artist_name}".lower()
        
        # 尝试从缓存获取
        if cache_key in self.cache:
            # 检查缓存项是否已过期（超过30天）
            cache_entry = self.cache[cache_key]
            cache_time = datetime.fromisoformat(cache_entry.get('timestamp', '2000-01-01'))
            current_time = datetime.now()
            
            # 如果缓存未过期且有预览URL
            if (current_time - cache_time).days < 30 and cache_entry.get('preview_url'):
                self.cache_hits += 1
                logger.debug(f"缓存命中: {track_name} - {artist_name}")
                return cache_entry
        
        # 缓存未命中，从Spotify获取
        self.cache_misses += 1
        logger.info(f"缓存未命中，搜索Spotify: {track_name} - {artist_name}")
        
        try:
            # 搜索歌曲
            tracks = self.spotify.search_track(track_name, artist_name, limit=1)
            
            if not tracks:
                logger.warning(f"未找到歌曲: {track_name} - {artist_name}")
                # 记录未找到的歌曲到缓存，避免重复搜索
                self.cache[cache_key] = {
                    'track_name': track_name,
                    'artist_name': artist_name,
                    'preview_url': None,
                    'found': False,
                    'timestamp': datetime.now().isoformat()
                }
                self._save_cache()
                return None
            
            # 获取歌曲信息
            track = tracks[0]
            preview_data = {
                'track_name': track['name'],
                'artist_name': track['artists'][0]['name'],
                'preview_url': track.get('preview_url'),
                'track_id': track['id'],
                'album_name': track['album']['name'],
                'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
                'found': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果没有预览URL，尝试获取详细信息
            if not preview_data['preview_url']:
                logger.info(f"歌曲没有预览URL，获取详细信息: {track['id']}")
                track_info = self.spotify.get_track_info(track['id'])
                if track_info:
                    preview_data['preview_url'] = track_info.get('preview_url')
            
            # 添加到缓存
            self.cache[cache_key] = preview_data
            self._save_cache()
            
            return preview_data
            
        except Exception as e:
            logger.error(f"获取预览URL时出错: {e}")
            return None
    
    def enrich_recommendations(self, recommendations):
        """用预览URL丰富推荐结果
        
        参数:
            recommendations: 推荐结果列表，每个推荐应包含track_name和artist_name字段
            
        返回:
            丰富后的推荐结果列表
        """
        if not recommendations:
            return []
        
        enriched_recommendations = []
        
        for rec in recommendations:
            track_name = rec.get('track_name')
            artist_name = rec.get('artist_name')
            
            if not track_name or not artist_name:
                enriched_recommendations.append(rec)
                continue
            
            # 获取预览URL
            preview_data = self.get_preview_url(track_name, artist_name)
            
            if preview_data and preview_data.get('found'):
                # 合并原始推荐和预览数据
                merged_rec = {**rec}
                
                # 添加预览数据
                for key in ['preview_url', 'album_cover', 'track_id']:
                    if key in preview_data and preview_data[key]:
                        merged_rec[key] = preview_data[key]
                
                enriched_recommendations.append(merged_rec)
            else:
                # 如果没有找到预览，保持原样
                enriched_recommendations.append(rec)
        
        # 缓存统计日志
        total_requests = self.cache_hits + self.cache_misses
        if total_requests > 0:
            hit_rate = (self.cache_hits / total_requests) * 100
            logger.info(f"预览URL缓存命中率: {hit_rate:.2f}% ({self.cache_hits}/{total_requests})")
        
        return enriched_recommendations
    
    def get_track_data(self, track_id):
        """直接通过Spotify track_id获取歌曲数据
        
        参数:
            track_id: Spotify歌曲ID
            
        返回:
            包含歌曲信息和预览URL的字典
        """
        try:
            track_info = self.spotify.get_track_info(track_id)
            
            if not track_info:
                logger.warning(f"未找到歌曲ID: {track_id}")
                return None
            
            track_data = {
                'track_name': track_info.get('name'),
                'artist_name': track_info.get('artists', [{}])[0].get('name'),
                'preview_url': track_info.get('preview_url'),
                'track_id': track_id,
                'album_name': track_info.get('album', {}).get('name'),
                'album_cover': track_info.get('album', {}).get('images', [{}])[0].get('url') if track_info.get('album', {}).get('images') else None,
                'found': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # 添加到缓存
            cache_key = f"{track_data['track_name']}|{track_data['artist_name']}".lower()
            self.cache[cache_key] = track_data
            self._save_cache()
            
            return track_data
            
        except Exception as e:
            logger.error(f"通过ID获取歌曲数据时出错: {e}")
            return None
    
    def get_batch_previews(self, songs):
        """批量获取多首歌曲的预览URL
        
        参数:
            songs: 歌曲列表，每首歌曲应为字典，包含track_name和artist_name字段
            
        返回:
            包含预览URL的歌曲列表
        """
        if not songs:
            return []
        
        results = []
        total = len(songs)
        
        for i, song in enumerate(songs, 1):
            if i % 10 == 0:
                logger.info(f"正在获取预览 {i}/{total}...")
            
            track_name = song.get('track_name')
            artist_name = song.get('artist_name')
            
            if not track_name or not artist_name:
                results.append(song)
                continue
            
            # 获取预览URL
            preview_data = self.get_preview_url(track_name, artist_name)
            
            if preview_data and preview_data.get('found'):
                # 合并原始数据和预览数据
                merged_data = {**song}
                
                # 添加预览数据
                for key in ['preview_url', 'album_cover', 'track_id']:
                    if key in preview_data and preview_data[key]:
                        merged_data[key] = preview_data[key]
                
                results.append(merged_data)
            else:
                # 如果没有找到预览，保持原样
                results.append(song)
            
            # 每20个请求暂停一下，避免触发API速率限制
            if i % 20 == 0 and i < total:
                time.sleep(1)
        
        return results


if __name__ == "__main__":
    # 测试代码
    preview_helper = SpotifyPreviewHelper()
    
    # 测试获取单个歌曲的预览URL
    test_songs = [
        {"track_name": "Shape of You", "artist_name": "Ed Sheeran"},
        {"track_name": "Blinding Lights", "artist_name": "The Weeknd"},
        {"track_name": "Dance Monkey", "artist_name": "Tones and I"},
        {"track_name": "Uptown Funk", "artist_name": "Mark Ronson"}
    ]
    
    print("获取歌曲预览URL:")
    for song in test_songs:
        preview = preview_helper.get_preview_url(song["track_name"], song["artist_name"])
        if preview and preview.get('found'):
            print(f"{preview['track_name']} - {preview['artist_name']}: {preview['preview_url']}")
        else:
            print(f"{song['track_name']} - {song['artist_name']}: 未找到预览")
    
    # 测试丰富推荐
    print("\n测试丰富推荐:")
    recommendations = [
        {"track_name": "Shivers", "artist_name": "Ed Sheeran", "score": 0.92},
        {"track_name": "Bad Habits", "artist_name": "Ed Sheeran", "score": 0.89},
        {"track_name": "Perfect", "artist_name": "Ed Sheeran", "score": 0.85}
    ]
    
    enriched = preview_helper.enrich_recommendations(recommendations)
    for rec in enriched:
        print(f"{rec['track_name']} - {rec['artist_name']}: {rec.get('preview_url', '无预览')}") 