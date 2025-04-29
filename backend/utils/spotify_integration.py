import os
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
import time
from functools import lru_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('spotify_integration')

class SpotifyManager:
    """Spotify API集成管理器，处理音乐元数据和预览播放"""
    
    def __init__(self, client_id=None, client_secret=None):
        """初始化Spotify管理器
        
        参数:
            client_id: Spotify Client ID
            client_secret: Spotify Client Secret
        """
        # 优先使用传入的凭据，其次使用环境变量
        self.client_id = client_id or os.environ.get('SPOTIFY_CLIENT_ID', '4f1a2f4e1e034050ac432f8ebba72484')
        self.client_secret = client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET', '4abd4c31749748c8b89f7807c61a3f11')
        
        self.sp = None
        self.last_error = None
        self.connect()
        
    def connect(self):
        """连接到Spotify API"""
        try:
            logger.info("连接Spotify API...")
            auth_manager = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            logger.info("Spotify API连接成功")
            return True
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Spotify API连接失败: {e}")
            return False
    
    @lru_cache(maxsize=1000)
    def search_track(self, track_name, artist_name=None, limit=1):
        """搜索歌曲
        
        参数:
            track_name: 歌曲名称
            artist_name: 艺术家名称(可选)
            limit: 返回结果数量
            
        返回:
            匹配的歌曲列表
        """
        if not self.sp:
            logger.error("Spotify客户端未初始化")
            return []
        
        try:
            # 构建查询字符串
            query = f"track:{track_name}"
            if artist_name:
                query += f" artist:{artist_name}"
            
            logger.info(f"搜索Spotify歌曲: {query}")
            results = self.sp.search(q=query, type='track', limit=limit)
            
            if results and 'tracks' in results and 'items' in results['tracks']:
                return results['tracks']['items']
            return []
        except Exception as e:
            logger.error(f"搜索歌曲时出错: {e}")
            # 如果遇到速率限制，等待并重试
            if "429" in str(e):
                logger.warning("达到Spotify API速率限制，等待5秒后重试")
                time.sleep(5)
                return self.search_track(track_name, artist_name, limit)
            return []
    
    def get_track_info(self, track_id=None, track_name=None, artist_name=None):
        """获取歌曲详细信息
        
        可以通过track_id直接获取，或通过track_name和artist_name搜索获取
        
        参数:
            track_id: Spotify歌曲ID
            track_name: 歌曲名称(与track_id二选一)
            artist_name: 艺术家名称(可选)
            
        返回:
            歌曲详细信息字典
        """
        if not self.sp:
            logger.error("Spotify客户端未初始化")
            return {}
        
        try:
            # 如果没有track_id但有track_name，先搜索获取track_id
            if not track_id and track_name:
                tracks = self.search_track(track_name, artist_name, limit=1)
                if tracks:
                    track_id = tracks[0]['id']
                else:
                    logger.warning(f"未找到歌曲: {track_name} - {artist_name}")
                    return {}
            
            if not track_id:
                logger.error("未提供有效的track_id或track_name")
                return {}
            
            # 获取歌曲详情
            logger.info(f"获取歌曲详情: {track_id}")
            track_data = self.sp.track(track_id)
            
            # 提取艺术家ID
            artist_ids = [artist['id'] for artist in track_data['artists']]
            
            # 获取艺术家详情（包含流派信息）
            artists_data = self.sp.artists(artist_ids) if artist_ids else {'artists': []}
            
            # 提取所有艺术家的流派
            genres = []
            for artist in artists_data['artists']:
                genres.extend(artist.get('genres', []))
            
            # 构建增强的歌曲信息
            enhanced_info = {
                'track_id': track_data['id'],
                'track_name': track_data['name'],
                'artists': [{'id': artist['id'], 'name': artist['name']} for artist in track_data['artists']],
                'album_name': track_data['album']['name'],
                'album_id': track_data['album']['id'],
                'album_cover': track_data['album']['images'][0]['url'] if track_data['album']['images'] else None,
                'release_date': track_data['album'].get('release_date'),
                'popularity': track_data['popularity'],
                'preview_url': track_data.get('preview_url'),
                'external_url': track_data['external_urls'].get('spotify'),
                'genres': list(set(genres)),
                'duration_ms': track_data['duration_ms'],
                'explicit': track_data['explicit']
            }
            
            return enhanced_info
        except Exception as e:
            logger.error(f"获取歌曲信息时出错: {e}")
            return {}
    
    def get_artist_info(self, artist_id=None, artist_name=None):
        """获取艺术家详细信息
        
        参数:
            artist_id: Spotify艺术家ID
            artist_name: 艺术家名称(与artist_id二选一)
            
        返回:
            艺术家详细信息字典
        """
        if not self.sp:
            logger.error("Spotify客户端未初始化")
            return {}
        
        try:
            # 如果没有artist_id但有artist_name，先搜索获取artist_id
            if not artist_id and artist_name:
                results = self.sp.search(q=f"artist:{artist_name}", type='artist', limit=1)
                if results and 'artists' in results and 'items' in results['artists'] and results['artists']['items']:
                    artist_id = results['artists']['items'][0]['id']
                else:
                    logger.warning(f"未找到艺术家: {artist_name}")
                    return {}
            
            if not artist_id:
                logger.error("未提供有效的artist_id或artist_name")
                return {}
            
            # 获取艺术家详情
            logger.info(f"获取艺术家详情: {artist_id}")
            artist_data = self.sp.artist(artist_id)
            
            # 获取艺术家热门歌曲
            top_tracks = self.sp.artist_top_tracks(artist_id)
            
            # 获取相似艺术家
            related_artists = self.sp.artist_related_artists(artist_id)
            
            # 构建增强的艺术家信息
            enhanced_info = {
                'artist_id': artist_data['id'],
                'name': artist_data['name'],
                'genres': artist_data.get('genres', []),
                'popularity': artist_data['popularity'],
                'followers': artist_data['followers']['total'],
                'image': artist_data['images'][0]['url'] if artist_data['images'] else None,
                'external_url': artist_data['external_urls'].get('spotify'),
                'top_tracks': [
                    {
                        'id': track['id'],
                        'name': track['name'],
                        'album': track['album']['name'],
                        'preview_url': track.get('preview_url'),
                        'image': track['album']['images'][0]['url'] if track['album']['images'] else None
                    } for track in top_tracks.get('tracks', [])[:5]
                ],
                'related_artists': [
                    {
                        'id': artist['id'],
                        'name': artist['name'],
                        'image': artist['images'][0]['url'] if artist['images'] else None
                    } for artist in related_artists.get('artists', [])[:5]
                ]
            }
            
            return enhanced_info
        except Exception as e:
            logger.error(f"获取艺术家信息时出错: {e}")
            return {}
    
    def get_new_releases(self, limit=10, country='US'):
        """获取新发行专辑
        
        参数:
            limit: 返回结果数量
            country: 国家代码
            
        返回:
            新发行专辑列表
        """
        if not self.sp:
            logger.error("Spotify客户端未初始化")
            return []
        
        try:
            logger.info(f"获取新发行专辑，国家: {country}, 数量: {limit}")
            results = self.sp.new_releases(country=country, limit=limit)
            
            if 'albums' in results and 'items' in results['albums']:
                releases = []
                for album in results['albums']['items']:
                    releases.append({
                        'album_id': album['id'],
                        'album_name': album['name'],
                        'artists': [{'id': artist['id'], 'name': artist['name']} for artist in album['artists']],
                        'release_date': album.get('release_date'),
                        'image': album['images'][0]['url'] if album['images'] else None,
                        'external_url': album['external_urls'].get('spotify')
                    })
                return releases
            return []
        except Exception as e:
            logger.error(f"获取新发行专辑时出错: {e}")
            return []
    
    def enrich_recommendation(self, recommendation):
        """使用Spotify数据丰富推荐结果
        
        参数:
            recommendation: 原始推荐结果
            
        返回:
            丰富后的推荐结果
        """
        if not recommendation:
            return recommendation
        
        track_name = recommendation.get('track_name')
        artist_name = recommendation.get('artist_name')
        
        if not track_name or not artist_name:
            return recommendation
        
        # 从Spotify搜索匹配信息
        spotify_info = self.get_track_info(track_name=track_name, artist_name=artist_name)
        
        if spotify_info:
            # 添加Spotify数据
            enhanced_rec = {**recommendation}  # 创建推荐的副本
            
            # 添加Spotify特有的字段
            for key in ['album_cover', 'preview_url', 'external_url', 'genres', 'popularity']:
                if key in spotify_info and spotify_info[key]:
                    enhanced_rec[key] = spotify_info[key]
            
            return enhanced_rec
        
        return recommendation
    
    def get_similar_tracks(self, track_id=None, track_name=None, artist_name=None, limit=5):
        """获取相似歌曲推荐
        
        参数:
            track_id: Spotify歌曲ID
            track_name: 歌曲名称(与track_id二选一)
            artist_name: 艺术家名称(可选)
            limit: 返回结果数量
            
        返回:
            相似歌曲列表
        """
        if not self.sp:
            logger.error("Spotify客户端未初始化")
            return []
        
        try:
            # 如果没有track_id但有track_name，先搜索获取track_id
            if not track_id and track_name:
                tracks = self.search_track(track_name, artist_name, limit=1)
                if tracks:
                    track_id = tracks[0]['id']
                else:
                    logger.warning(f"未找到歌曲: {track_name} - {artist_name}")
                    return []
            
            if not track_id:
                logger.error("未提供有效的track_id或track_name")
                return []
            
            # 获取推荐
            logger.info(f"获取相似歌曲: {track_id}")
            results = self.sp.recommendations(seed_tracks=[track_id], limit=limit)
            
            if 'tracks' in results:
                similar_tracks = []
                for track in results['tracks']:
                    similar_tracks.append({
                        'track_id': track['id'],
                        'track_name': track['name'],
                        'artists': [{'id': artist['id'], 'name': artist['name']} for artist in track['artists']],
                        'album_name': track['album']['name'],
                        'album_cover': track['album']['images'][0]['url'] if track['album']['images'] else None,
                        'preview_url': track.get('preview_url'),
                        'external_url': track['external_urls'].get('spotify'),
                        'explanation': "Spotify推荐的相似歌曲"
                    })
                return similar_tracks
            return []
        except Exception as e:
            logger.error(f"获取相似歌曲时出错: {e}")
            return []

# 测试代码
if __name__ == "__main__":
    # 直接运行该文件时，测试功能是否正常
    spotify_manager = SpotifyManager()
    
    # 测试搜索歌曲
    test_track = spotify_manager.search_track("Shape of You", "Ed Sheeran")
    if test_track:
        print(f"找到歌曲: {test_track[0]['name']} - {test_track[0]['artists'][0]['name']}")
        
        # 测试获取歌曲详情
        track_id = test_track[0]['id']
        track_info = spotify_manager.get_track_info(track_id=track_id)
        print(f"歌曲详情: {track_info.get('track_name')} - 流派: {track_info.get('genres')}")
        
        # 测试相似歌曲
        similar_tracks = spotify_manager.get_similar_tracks(track_id=track_id)
        print(f"找到 {len(similar_tracks)} 首相似歌曲")
    else:
        print("未找到测试歌曲") 