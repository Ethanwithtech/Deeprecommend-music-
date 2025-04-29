"""
音乐预览URL获取模块

提供从不同来源获取音乐预览URL的功能，包括7digital、Spotify等。
作为音乐推荐系统的补充功能。
"""

import os
import logging
from .sevendigital_api import get_preview_url

logger = logging.getLogger(__name__)

# 检查是否禁用7digital功能
DISABLE_7DIGITAL = os.environ.get('DISABLE_7DIGITAL', '').lower() == 'true'

# 检查是否配置了7digital API密钥
SEVENDIGITAL_API_KEY = os.environ.get('SEVENDIGITAL_API_KEY')
SEVENDIGITAL_AVAILABLE = not DISABLE_7DIGITAL and SEVENDIGITAL_API_KEY is not None and SEVENDIGITAL_API_KEY != 'your_api_key_here'

if DISABLE_7DIGITAL:
    logger.info("7digital功能已被禁用")

# 检查是否配置了Spotify API密钥
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '4f1a2f4e1e034050ac432f8ebba72484')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '4abd4c31749748c8b89f7807c61a3f11')
SPOTIFY_AVAILABLE = True  # 强制启用Spotify服务

logger.info(f"Spotify服务状态: {'可用' if SPOTIFY_AVAILABLE else '不可用'}")

# 默认预览URL - 当无法获取实际预览时使用
DEFAULT_PREVIEW_URL = ""


def get_music_preview(track_name, artist_name, album_name=None, track_id=None, release_id=None, artist_id=None):
    """
    获取音乐预览URL，尝试多种来源
    
    参数:
        track_name: 歌曲名称
        artist_name: 艺术家名称
        album_name: 专辑名称(可选)
        track_id: 7digital的曲目ID(可选)
        release_id: 7digital的专辑ID(可选)
        artist_id: 7digital的艺术家ID(可选)
        
    返回:
        预览URL，如果无法获取则返回空字符串
    """
    preview_url = DEFAULT_PREVIEW_URL
    
    # 如果7digital功能被禁用，直接返回默认URL
    if DISABLE_7DIGITAL:
        logger.debug(f"7digital功能已禁用，无法获取预览: {track_name} - {artist_name}")
        return preview_url
    
    # 首先尝试7digital API
    if SEVENDIGITAL_AVAILABLE:
        try:
            preview_url = get_preview_url(
                track_name=track_name, 
                artist_name=artist_name,
                track_7digitalid=track_id if track_id else -1,
                release_7digitalid=release_id if release_id else -1,
                artist_7digitalid=artist_id if artist_id else -1
            )
            
            if preview_url:
                logger.info(f"7digital预览URL获取成功: {track_name} - {artist_name}")
                return preview_url
        except Exception as e:
            logger.error(f"7digital预览URL获取失败: {str(e)}")
            # 发生错误时返回默认URL
            return preview_url
    else:
        logger.debug("7digital API密钥未配置或功能已禁用，无法获取预览")
    
    # 后续可以添加其他音乐预览来源，如Spotify等
    
    # 如果所有来源都失败，返回默认URL
    return preview_url


def check_preview_services():
    """
    检查预览服务的可用性
    
    返回:
        服务状态信息的字典
    """
    services = {
        "7digital": {
            "available": SEVENDIGITAL_AVAILABLE,
            "message": "可用" if SEVENDIGITAL_AVAILABLE else ("已禁用" if DISABLE_7DIGITAL else "API密钥未配置")
        },
        "spotify": {
            "available": SPOTIFY_AVAILABLE,
            "message": "可用" if SPOTIFY_AVAILABLE else "API密钥未配置"
        }
    }
    
    return services 