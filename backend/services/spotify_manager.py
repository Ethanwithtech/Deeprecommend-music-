#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify API 管理器

处理与Spotify API的交互，包括获取音乐信息、搜索、推荐等功能
"""

import os
import sys
import json
import base64
import requests
import logging
import time
from urllib.parse import urlencode
from dotenv import load_dotenv

# 配置日志
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()

class SpotifyManager:
    """Spotify API管理器
    
    处理与Spotify API的所有交互，包括认证、搜索、获取推荐等
    """
    
    def __init__(self, client_id=None, client_secret=None):
        """初始化Spotify管理器
        
        Args:
            client_id: Spotify API客户端ID
            client_secret: Spotify API客户端密钥
        """
        # 尝试从环境变量获取认证信息
        self.client_id = client_id or os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        # API端点
        self.auth_url = 'https://accounts.spotify.com/api/token'
        self.base_url = 'https://api.spotify.com/v1'
        
        # 认证令牌
        self.access_token = None
        self.token_expiry = 0
        
        # 连接API
        self.connect()
        
        logger.info("Spotify管理器初始化完成")
        
    def connect(self):
        """连接到Spotify API，获取访问令牌
        
        Returns:
            bool: 是否成功连接
        """
        if not self.client_id or not self.client_secret:
            logger.warning("未提供Spotify API认证信息")
            return False
            
        try:
            logger.info("连接Spotify API...")
            
            # 如果令牌有效，直接返回
            if self.access_token and time.time() < self.token_expiry - 60:
                return True
                
            # 准备认证
            auth_header = base64.b64encode(f"{self.client_id}:{self.client_secret}".encode()).decode()
            
            headers = {
                'Authorization': f'Basic {auth_header}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            payload = {'grant_type': 'client_credentials'}
            
            # 发送认证请求
            response = requests.post(self.auth_url, headers=headers, data=payload)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data['access_token']
            self.token_expiry = time.time() + data['expires_in']
            
            logger.info("Spotify API连接成功")
            return True
            
        except Exception as e:
            logger.error(f"连接Spotify API时出错: {str(e)}")
            return False
    
    def is_connected(self):
        """检查是否成功连接到Spotify API
        
        Returns:
            bool: 是否已连接
        """
        return self.access_token is not None and time.time() < self.token_expiry - 60
    
    def _make_request(self, endpoint, method='GET', params=None, data=None):
        """向Spotify API发送请求
        
        Args:
            endpoint: API端点
            method: HTTP方法
            params: URL参数
            data: 请求体数据
            
        Returns:
            dict: 返回的JSON数据
        """
        # 确保连接
        if not self.is_connected():
            if not self.connect():
                logger.error("无法连接到Spotify API")
                return None
        
        url = f"{self.base_url}/{endpoint}"
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, params=params, data=json.dumps(data))
            else:
                logger.error(f"不支持的HTTP方法: {method}")
                return None
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                # 令牌过期，重新连接
                self.connect()
                return self._make_request(endpoint, method, params, data)
            else:
                logger.error(f"Spotify API请求错误: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"请求Spotify API时出错: {str(e)}")
            return None
    
    def search(self, query, search_type='track', limit=10, offset=0):
        """搜索Spotify音乐
        
        Args:
            query: 搜索关键词
            search_type: 搜索类型（track, album, artist, playlist）
            limit: 返回结果数量上限
            offset: 结果偏移量
            
        Returns:
            dict: 搜索结果
        """
        params = {
            'q': query,
            'type': search_type,
            'limit': limit,
            'offset': offset
        }
        
        return self._make_request('search', params=params)
    
    def get_track(self, track_id):
        """获取单个音轨的信息
        
        Args:
            track_id: Spotify音轨ID
            
        Returns:
            dict: 音轨信息
        """
        # 去除'spotify:track:'前缀
        if track_id.startswith('spotify:track:'):
            track_id = track_id.split(':')[-1]
            
        return self._make_request(f'tracks/{track_id}')
    
    def get_artist(self, artist_id):
        """获取艺术家信息
        
        Args:
            artist_id: Spotify艺术家ID
            
        Returns:
            dict: 艺术家信息
        """
        # 去除'spotify:artist:'前缀
        if artist_id.startswith('spotify:artist:'):
            artist_id = artist_id.split(':')[-1]
            
        return self._make_request(f'artists/{artist_id}')
    
    def get_artist_top_tracks(self, artist_id, country='US'):
        """获取艺术家的热门歌曲
        
        Args:
            artist_id: Spotify艺术家ID
            country: 国家/地区代码
            
        Returns:
            dict: 艺术家热门歌曲
        """
        # 去除'spotify:artist:'前缀
        if artist_id.startswith('spotify:artist:'):
            artist_id = artist_id.split(':')[-1]
            
        return self._make_request(f'artists/{artist_id}/top-tracks', params={'market': country})
    
    def get_recommendations(self, seed_tracks=None, seed_artists=None, seed_genres=None, limit=20, attributes=None):
        """获取音乐推荐
        
        Args:
            seed_tracks: 种子音轨ID列表
            seed_artists: 种子艺术家ID列表
            seed_genres: 种子流派列表
            limit: 返回结果数量
            attributes: 音乐特征参数字典
            
        Returns:
            dict: 推荐结果
        """
        params = {'limit': limit}
        
        # 添加种子
        if seed_tracks:
            params['seed_tracks'] = ','.join(seed_tracks)
        if seed_artists:
            params['seed_artists'] = ','.join(seed_artists)
        if seed_genres:
            params['seed_genres'] = ','.join(seed_genres)
            
        # 添加音乐特征参数
        if attributes:
            for key, value in attributes.items():
                params[f'target_{key}'] = value
        
        return self._make_request('recommendations', params=params)
    
    def get_genre_seeds(self):
        """获取可用的流派种子
        
        Returns:
            list: 可用的流派列表
        """
        result = self._make_request('recommendations/available-genre-seeds')
        if result and 'genres' in result:
            return result['genres']
        return []
    
    def get_new_releases(self, country='US', limit=20, offset=0):
        """获取新发行音乐
        
        Args:
            country: 国家/地区代码
            limit: 返回结果数量
            offset: 结果偏移量
            
        Returns:
            dict: 新发行音乐
        """
        params = {
            'country': country,
            'limit': limit,
            'offset': offset
        }
        
        return self._make_request('browse/new-releases', params=params)
    
    def get_similar_tracks(self, track_id, limit=10):
        """获取相似音轨
        
        Args:
            track_id: Spotify音轨ID
            limit: 返回结果数量
            
        Returns:
            dict: 相似音轨列表
        """
        # 去除'spotify:track:'前缀
        if track_id.startswith('spotify:track:'):
            track_id = track_id.split(':')[-1]
            
        # 先获取音轨信息
        track_info = self.get_track(track_id)
        if not track_info:
            return None
            
        # 获取艺术家ID和流派
        artist_id = track_info['artists'][0]['id'] if track_info.get('artists') else None
        
        # 使用音轨和艺术家作为种子获取推荐
        seed_tracks = [track_id]
        seed_artists = [artist_id] if artist_id else None
        
        return self.get_recommendations(seed_tracks=seed_tracks, seed_artists=seed_artists, limit=limit)
    
    def get_audio_features(self, track_ids):
        """获取音轨的音频特征
        
        Args:
            track_ids: 单个或多个Spotify音轨ID
            
        Returns:
            dict: 音频特征
        """
        if isinstance(track_ids, str):
            # 单个ID
            if track_ids.startswith('spotify:track:'):
                track_ids = track_ids.split(':')[-1]
            return self._make_request(f'audio-features/{track_ids}')
        else:
            # 多个ID
            clean_ids = []
            for track_id in track_ids:
                if track_id.startswith('spotify:track:'):
                    clean_ids.append(track_id.split(':')[-1])
                else:
                    clean_ids.append(track_id)
                    
            return self._make_request('audio-features', params={'ids': ','.join(clean_ids)})
    
    def get_track_preview_url(self, track_id):
        """获取音轨的预览URL
        
        Args:
            track_id: Spotify音轨ID
            
        Returns:
            str: 预览URL或None
        """
        track_info = self.get_track(track_id)
        if track_info and 'preview_url' in track_info:
            return track_info['preview_url']
        return None
    
    def get_artist_by_name(self, artist_name, limit=1):
        """根据名称搜索艺术家
        
        Args:
            artist_name: 艺术家名称
            limit: 返回结果数量
            
        Returns:
            dict: 艺术家信息
        """
        search_result = self.search(artist_name, search_type='artist', limit=limit)
        if not search_result or not search_result.get('artists') or not search_result['artists'].get('items'):
            return None
            
        return search_result['artists']['items'][0] if limit == 1 else search_result['artists']['items']
    
    def get_track_by_name(self, track_name, artist_name=None, limit=1):
        """根据名称搜索音轨
        
        Args:
            track_name: 音轨名称
            artist_name: 艺术家名称（可选）
            limit: 返回结果数量
            
        Returns:
            dict: 音轨信息
        """
        query = track_name
        if artist_name:
            query += f" artist:{artist_name}"
            
        search_result = self.search(query, search_type='track', limit=limit)
        if not search_result or not search_result.get('tracks') or not search_result['tracks'].get('items'):
            return None
            
        return search_result['tracks']['items'][0] if limit == 1 else search_result['tracks']['items']
        
    def get_tracks_by_mood(self, mood, limit=10):
        """根据情绪获取音轨
        
        Args:
            mood: 情绪标签（happy, sad, energetic, relaxed等）
            limit: 返回结果数量
            
        Returns:
            list: 音轨列表
        """
        # 情绪-音乐特征映射
        mood_features = {
            'happy': {'valence': 0.8, 'energy': 0.6, 'tempo': 120},
            'sad': {'valence': 0.2, 'energy': 0.4, 'tempo': 80},
            'energetic': {'energy': 0.9, 'tempo': 140},
            'relaxed': {'energy': 0.3, 'acousticness': 0.8},
            'romantic': {'valence': 0.6, 'energy': 0.5, 'acousticness': 0.5},
            'angry': {'energy': 0.8, 'valence': 0.3},
            'nostalgic': {'acousticness': 0.6, 'instrumentalness': 0.4}
        }
        
        # 情绪-流派映射
        mood_genres = {
            'happy': ['pop', 'dance', 'disco'],
            'sad': ['blues', 'jazz', 'folk', 'classical'],
            'energetic': ['rock', 'metal', 'edm', 'hip-hop'],
            'relaxed': ['ambient', 'classical', 'jazz', 'chill'],
            'romantic': ['r&b', 'soul', 'jazz'],
            'angry': ['metal', 'punk', 'hardcore'],
            'nostalgic': ['folk', 'classical', 'oldies']
        }
        
        # 获取情绪相关特征和流派
        features = mood_features.get(mood.lower(), {'valence': 0.5, 'energy': 0.5})
        genres = mood_genres.get(mood.lower(), ['pop'])
        
        # 获取推荐
        return self.get_recommendations(
            seed_genres=genres[:3],  # 最多使用3个流派种子
            limit=limit,
            attributes=features
        )
    
    def get_tracks_for_emotion(self, emotion, limit=5):
        """根据情感状态推荐适合的音乐
        
        Args:
            emotion: 情感状态（字符串）
            limit: 返回结果数量
            
        Returns:
            list: 推荐歌曲列表
        """
        # 将情感状态映射到情绪类别
        emotion_mood_map = {
            'happy': 'happy',
            'sad': 'sad',
            'angry': 'angry',
            'relaxed': 'relaxed',
            'excited': 'energetic',
            'nervous': 'energetic',
            'bored': 'energetic',
            'tired': 'relaxed',
            'calm': 'relaxed',
            'stressed': 'relaxed',
            'depressed': 'sad',
            'anxious': 'relaxed',
            'confused': 'relaxed',
            'lonely': 'sad',
            'love': 'romantic',
            'nostalgic': 'nostalgic'
        }
        
        # 获取对应的情绪类别
        mood = emotion_mood_map.get(emotion.lower(), 'relaxed')
        
        # 根据情绪获取推荐
        recommendations = self.get_tracks_by_mood(mood, limit)
        
        if not recommendations or not recommendations.get('tracks'):
            return []
            
        # 格式化结果
        results = []
        for track in recommendations['tracks']:
            preview_url = track.get('preview_url')
            if not preview_url:
                continue  # 跳过没有预览URL的歌曲
                
            # 整理专辑封面
            image_url = None
            if track.get('album') and track['album'].get('images') and len(track['album']['images']) > 0:
                image_url = track['album']['images'][0]['url']
                
            # 添加到结果
            results.append({
                'song_id': track['id'],
                'track_name': track['name'],
                'title': track['name'],
                'artist': track['artists'][0]['name'],
                'artist_name': track['artists'][0]['name'],
                'album_name': track['album']['name'],
                'preview_url': preview_url,
                'image_url': image_url,
                'genre': mood,
                'external_url': track['external_urls']['spotify'],
                'explanation': f"适合{emotion}情绪的{mood}类型音乐"
            })
            
        return results
        
    def get_audio_analysis(self, track_id):
        """获取音轨的详细音频分析
        
        Args:
            track_id: Spotify音轨ID
            
        Returns:
            dict: 音频分析数据
        """
        # 去除'spotify:track:'前缀
        if track_id.startswith('spotify:track:'):
            track_id = track_id.split(':')[-1]
            
        return self._make_request(f'audio-analysis/{track_id}')

# 如果作为独立脚本运行
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 初始化Spotify管理器
    spotify = SpotifyManager()
    
    if spotify.is_connected():
        print("Spotify API连接成功")
        
        # 测试搜索
        results = spotify.search("Shape of You", limit=1)
        if results and results.get('tracks') and results['tracks'].get('items'):
            track = results['tracks']['items'][0]
            print(f"找到歌曲: {track['name']} - {track['artists'][0]['name']}")
            
            # 获取预览URL
            preview_url = track.get('preview_url')
            if preview_url:
                print(f"预览URL: {preview_url}")
            else:
                print("无预览URL")
        else:
            print("未找到搜索结果")
    else:
        print("Spotify API连接失败，请检查认证信息") 