#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify API集成模块

提供与Spotify API交互的功能，包括认证、搜索、获取曲目信息、
音频特征和推荐等。为音乐推荐系统提供数据支持。
"""

import os
import time
import base64
import requests
import logging
from typing import Dict, List, Any, Optional, Union

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spotify_integration')

class SpotifyManager:
    """Spotify API管理器
    
    处理与Spotify API的交互，包括认证和各种API请求。
    提供对曲目、艺术家、专辑等信息的访问方法。
    """
    
    # Spotify API端点
    AUTH_URL = 'https://accounts.spotify.com/api/token'
    API_BASE_URL = 'https://api.spotify.com/v1'
    
    def __init__(self, client_id=None, client_secret=None):
        """初始化Spotify管理器
        
        参数:
            client_id: Spotify应用客户端ID，默认从环境变量获取
            client_secret: Spotify应用客户端密钥，默认从环境变量获取
        """
        # 优先使用传入的凭证，其次使用环境变量
        self.client_id = client_id or os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        if not self.client_id or not self.client_secret:
            logger.warning("未提供Spotify凭证，部分功能可能无法使用")
        
        # 访问令牌
        self.access_token = None
        self.token_expiry = 0
        
        # 尝试获取初始令牌
        self._get_access_token()
    
    def _get_access_token(self) -> bool:
        """获取Spotify访问令牌
        
        使用客户端凭证获取访问令牌，管理令牌过期
        
        返回:
            获取令牌是否成功
        """
        # 如果没有凭证，无法获取令牌
        if not self.client_id or not self.client_secret:
            logger.error("获取访问令牌失败: 未提供客户端凭证")
            return False
        
        # 检查令牌是否有效（考虑提前30秒刷新）
        current_time = time.time()
        if self.access_token and current_time < self.token_expiry - 30:
            return True
        
        try:
            # 准备认证头
            auth_string = f"{self.client_id}:{self.client_secret}"
            auth_bytes = auth_string.encode('utf-8')
            auth_base64 = base64.b64encode(auth_bytes).decode('utf-8')
            
            headers = {
                'Authorization': f'Basic {auth_base64}',
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            payload = {'grant_type': 'client_credentials'}
            
            response = requests.post(self.AUTH_URL, headers=headers, data=payload)
            response.raise_for_status()
            
            # 解析响应
            auth_data = response.json()
            self.access_token = auth_data['access_token']
            expires_in = auth_data['expires_in']
            self.token_expiry = current_time + expires_in
            
            logger.info(f"成功获取Spotify访问令牌，有效期 {expires_in} 秒")
            return True
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"获取访问令牌失败: HTTP错误 {e.response.status_code}")
            logger.error(f"错误详情: {e.response.text}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"获取访问令牌失败: 请求错误 {e}")
            return False
        except (KeyError, ValueError) as e:
            logger.error(f"获取访问令牌失败: 响应解析错误 {e}")
            return False
    
    def _make_api_request(self, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None) -> Optional[Dict]:
        """发送API请求到Spotify
        
        参数:
            endpoint: API端点路径
            method: HTTP方法（GET, POST等）
            params: URL查询参数
            data: 请求体数据
            
        返回:
            API响应的JSON数据，或None（请求失败时）
        """
        # 确保有有效的访问令牌
        if not self._get_access_token():
            logger.error(f"API请求失败: 无法获取有效的访问令牌")
            return None
        
        # 构建完整URL
        url = f"{self.API_BASE_URL}/{endpoint}"
        
        # 准备请求头
        headers = {'Authorization': f'Bearer {self.access_token}'}
        
        try:
            # 发送请求
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params)
            elif method.upper() == 'POST':
                headers['Content-Type'] = 'application/json'
                response = requests.post(url, headers=headers, params=params, json=data)
            else:
                logger.error(f"不支持的HTTP方法: {method}")
                return None
            
            # 处理401错误 - 令牌可能已过期
            if response.status_code == 401:
                logger.info("访问令牌已过期，尝试刷新...")
                # 重置令牌，强制刷新
                self.access_token = None
                self.token_expiry = 0
                
                # 重试请求
                if self._get_access_token():
                    return self._make_api_request(endpoint, method, params, data)
                else:
                    logger.error("刷新令牌失败，无法重试请求")
                    return None
            
            # 处理其他HTTP错误
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"API请求失败: HTTP错误 {e.response.status_code} - {endpoint}")
            logger.debug(f"错误详情: {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"API请求失败: 请求错误 {e} - {endpoint}")
            return None
        except (ValueError, KeyError) as e:
            logger.error(f"API请求失败: 响应解析错误 {e} - {endpoint}")
            return None
    
    def search_track(self, track_name: str, artist_name: str = None, limit: int = 5) -> List[Dict]:
        """搜索歌曲
        
        参数:
            track_name: 歌曲名称
            artist_name: 可选的艺术家名称
            limit: 返回结果数量限制
            
        返回:
            匹配歌曲的列表，如果未找到则为空列表
        """
        # 构建搜索查询
        query = f'track:"{track_name}"'
        if artist_name:
            query += f' artist:"{artist_name}"'
        
        # 设置参数
        params = {
            'q': query,
            'type': 'track',
            'limit': min(limit, 50)  # Spotify API限制最大为50
        }
        
        # 发送请求
        response = self._make_api_request('search', params=params)
        
        if not response or 'tracks' not in response:
            logger.warning(f"搜索歌曲失败: {track_name} - {artist_name}")
            return []
        
        # 提取歌曲列表
        tracks = response['tracks']['items']
        
        if not tracks:
            logger.info(f"未找到匹配的歌曲: {track_name} - {artist_name}")
            return []
        
        return tracks
    
    def get_track_info(self, track_id: str) -> Optional[Dict]:
        """获取歌曲详细信息
        
        参数:
            track_id: Spotify歌曲ID
            
        返回:
            歌曲详细信息字典，如果未找到则为None
        """
        return self._make_api_request(f'tracks/{track_id}')
    
    def get_audio_features(self, track_id: str) -> Optional[Dict]:
        """获取歌曲音频特征
        
        参数:
            track_id: Spotify歌曲ID
            
        返回:
            音频特征字典，如果未找到则为None
        """
        return self._make_api_request(f'audio-features/{track_id}')
    
    def get_recommendations(self, seed_tracks: List[str] = None, seed_artists: List[str] = None, 
                           seed_genres: List[str] = None, limit: int = 20, **kwargs) -> List[Dict]:
        """获取推荐歌曲
        
        参数:
            seed_tracks: 种子歌曲ID列表
            seed_artists: 种子艺术家ID列表
            seed_genres: 种子流派列表
            limit: 返回结果数量
            **kwargs: 其他推荐参数（如target_danceability等）
            
        返回:
            推荐歌曲列表，如果请求失败则为空列表
        """
        # 至少需要一个种子项
        if not seed_tracks and not seed_artists and not seed_genres:
            logger.error("获取推荐失败: 需要提供至少一个种子项")
            return []
        
        # 构建参数
        params = {'limit': min(limit, 100)}  # Spotify API限制最大为100
        
        if seed_tracks:
            params['seed_tracks'] = ','.join(seed_tracks[:5])  # 最多5个种子
        
        if seed_artists:
            params['seed_artists'] = ','.join(seed_artists[:5])  # 最多5个种子
        
        if seed_genres:
            params['seed_genres'] = ','.join(seed_genres[:5])  # 最多5个种子
        
        # 添加其他参数
        for key, value in kwargs.items():
            params[key] = value
        
        # 发送请求
        response = self._make_api_request('recommendations', params=params)
        
        if not response or 'tracks' not in response:
            logger.warning("获取推荐失败")
            return []
        
        return response['tracks']
    
    def get_artist_info(self, artist_id: str) -> Optional[Dict]:
        """获取艺术家信息
        
        参数:
            artist_id: Spotify艺术家ID
            
        返回:
            艺术家信息字典，如果未找到则为None
        """
        return self._make_api_request(f'artists/{artist_id}')
    
    def get_album_tracks(self, album_id: str, limit: int = 50) -> List[Dict]:
        """获取专辑曲目
        
        参数:
            album_id: Spotify专辑ID
            limit: 返回结果数量
            
        返回:
            专辑曲目列表，如果请求失败则为空列表
        """
        params = {'limit': min(limit, 50)}  # Spotify API限制最大为50
        
        response = self._make_api_request(f'albums/{album_id}/tracks', params=params)
        
        if not response or 'items' not in response:
            logger.warning(f"获取专辑曲目失败: {album_id}")
            return []
        
        return response['items']


if __name__ == "__main__":
    # 测试代码
    spotify = SpotifyManager()
    
    # 搜索歌曲
    print("搜索歌曲 'Shape of You':")
    tracks = spotify.search_track("Shape of You", "Ed Sheeran", limit=1)
    
    if tracks:
        track = tracks[0]
        print(f"歌曲: {track['name']}")
        print(f"艺术家: {track['artists'][0]['name']}")
        print(f"预览URL: {track.get('preview_url')}")
        print(f"专辑封面: {track['album']['images'][0]['url'] if track['album']['images'] else 'None'}")
        
        # 获取音频特征
        print("\n获取音频特征:")
        features = spotify.get_audio_features(track['id'])
        if features:
            print(f"舞曲性: {features['danceability']}")
            print(f"能量: {features['energy']}")
            print(f"节奏: {features['tempo']} BPM")
            print(f"调性: {features['key']}")
            print(f"语言可能性: {features['speechiness']}")
            print(f"乐器主导性: {features['instrumentalness']}")
            print(f"现场表演可能性: {features['liveness']}")
        
        # 获取推荐
        print("\n获取推荐:")
        recommendations = spotify.get_recommendations(
            seed_tracks=[track['id']], 
            limit=5
        )
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['name']} - {rec['artists'][0]['name']}")
    else:
        print("未找到歌曲") 