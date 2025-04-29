"""
7digital API集成模块

用于从7digital服务获取歌曲预览URL。
基于百万歌曲数据集项目的代码进行改进和现代化。

原始版权说明:
Thierry Bertin-Mahieux (2010) Columbia University
tb2332@columbia.edu

This code uses 7digital API and info contained in HDF5 song
file to get a preview URL.

This is part of the Million Song Dataset project from
LabROSA (Columbia University) and The Echo Nest.
"""

import os
import sys
import logging
import requests
from xml.dom import minidom
import numpy as np

# 尝试加载.env文件
try:
    from dotenv import load_dotenv
    # 找到项目根目录并加载.env文件
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_path = os.path.join(project_root, '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        print(f"已从 {env_path} 加载环境变量")
    else:
        print(f"警告: .env文件不存在: {env_path}")
except ImportError:
    print("警告: python-dotenv未安装，无法从.env文件加载变量")

logger = logging.getLogger(__name__)

# 尝试获取7digital API密钥
DIGITAL7_API_KEY = os.environ.get('SEVENDIGITAL_API_KEY') or os.environ.get('DIGITAL7_API_KEY')
DIGITAL7_API_SECRET = os.environ.get('SEVENDIGITAL_API_SECRET')

# 如果仍然没有API密钥，使用硬编码的测试值（仅用于开发）
if not DIGITAL7_API_KEY:
    print("警告: 未找到7digital API密钥，将使用硬编码的测试值")
    # 注意: 这些是无效的测试值
    DIGITAL7_API_KEY = "your_api_key_here"
    DIGITAL7_API_SECRET = "your_api_secret_here"

print(f"7digital API密钥: {DIGITAL7_API_KEY[:5]}{'*' * 10}")
print(f"7digital API密钥密钥: {DIGITAL7_API_SECRET[:3] if DIGITAL7_API_SECRET else None}{'*' * 10 if DIGITAL7_API_SECRET else 'Not Set'}")


def url_call(url):
    """
    向7digital API发送简单请求
    假设我们不会进行密集查询，此功能不够强健
    以XML文档形式返回响应
    
    参数:
        url: API请求URL
        
    返回:
        XML文档对象
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return minidom.parseString(response.text).documentElement
    except Exception as e:
        logger.error(f"7digital API请求失败: {str(e)}")
        return None


def levenshtein(s1, s2):
    """
    计算Levenshtein距离(编辑距离)
    
    参数:
        s1, s2: 要比较的两个字符串
        
    返回:
        编辑距离值
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if not s1:
        return len(s2)
 
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
 
    return previous_row[-1]


def get_closest_track(tracklist, target):
    """
    基于编辑距离找到最接近的曲目
    可能不是精确匹配，应该检查!
    
    参数:
        tracklist: 曲目名称列表
        target: 目标曲目名称
        
    返回:
        最接近的曲目名称
    """
    if not tracklist:
        return None
        
    dists = [levenshtein(x, target) for x in tracklist]
    best = np.argmin(dists)
    return tracklist[best]


def get_trackid_from_text_search(title, artistname=''):
    """
    使用7digital搜索API搜索艺术家+标题
    
    参数:
        title: 歌曲标题
        artistname: 艺术家名称(可选)
        
    返回:
        如有问题返回None，否则返回元组(title, trackid)
    """
    if not DIGITAL7_API_KEY:
        logger.warning("未设置7digital API密钥")
        return None
        
    url = 'https://api.7digital.com/1.2/track/search?'
    url += 'oauth_consumer_key=' + DIGITAL7_API_KEY
    query = title
    if artistname:
        query = artistname + ' ' + query
    
    # 使用requests库的URL编码
    from urllib.parse import quote
    query = quote(query)
    url += '&q=' + query
    
    xmldoc = url_call(url)
    if not xmldoc:
        return None
        
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
        
    resultelem = xmldoc.getElementsByTagName('searchResult')
    if len(resultelem) == 0:
        return None
        
    track = resultelem[0].getElementsByTagName('track')[0]
    tracktitle = track.getElementsByTagName('title')[0].firstChild.data
    trackid = int(track.getAttribute('id'))
    return (tracktitle, trackid)

    
def get_tracks_from_artistid(artistid):
    """
    获取艺术家的所有发行专辑，然后获取每个专辑的曲目
    
    参数:
        artistid: 7digital的艺术家ID
        
    返回:
        <曲目名称> -> <曲目id>的字典，如有问题则返回None
    """
    if not DIGITAL7_API_KEY:
        logger.warning("未设置7digital API密钥")
        return None
        
    url = 'https://api.7digital.com/1.2/artist/releases?'
    url += '&artistid=' + str(artistid)
    url += '&oauth_consumer_key=' + DIGITAL7_API_KEY
    
    xmldoc = url_call(url)
    if not xmldoc:
        return None
        
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
        
    releaseselem = xmldoc.getElementsByTagName('releases')
    if not releaseselem:
        return None
        
    releases = releaseselem[0].getElementsByTagName('release')
    if len(releases) == 0:
        return None
        
    releases_ids = [int(x.getAttribute('id')) for x in releases]
    res = {}
    
    for rid in releases_ids:
        tmpres = get_tracks_from_releaseid(rid)
        if tmpres is not None:
            res.update(tmpres)
            
    return res


def get_tracks_from_releaseid(releaseid):
    """
    获取特定发行专辑的所有曲目
    
    参数:
        releaseid: 7digital的专辑ID
        
    返回:
        <曲目名称> -> <曲目id>的字典，如有问题则返回None
    """
    if not DIGITAL7_API_KEY:
        logger.warning("未设置7digital API密钥")
        return None
        
    url = 'https://api.7digital.com/1.2/release/tracks?'
    url += 'releaseid=' + str(releaseid)
    url += '&oauth_consumer_key=' + DIGITAL7_API_KEY
    
    xmldoc = url_call(url)
    if not xmldoc:
        return None
        
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return None
        
    tracks = xmldoc.getElementsByTagName('track')
    if len(tracks) == 0:
        return None
        
    res = {}
    for t in tracks:
        tracktitle = t.getElementsByTagName('title')[0].firstChild.data
        trackid = int(t.getAttribute('id'))
        res[tracktitle] = trackid
        
    return res
    

def get_preview_from_trackid(trackid):
    """
    获取特定曲目的预览URL
    
    参数:
        trackid: 7digital的曲目ID
        
    返回:
        预览URL，如有问题则返回空字符串
    """
    if not DIGITAL7_API_KEY:
        logger.warning("未设置7digital API密钥")
        return ''
        
    url = 'https://api.7digital.com/1.2/track/preview?redirect=false'
    url += '&trackid=' + str(trackid)
    url += '&oauth_consumer_key=' + DIGITAL7_API_KEY
    
    xmldoc = url_call(url)
    if not xmldoc:
        return ''
        
    status = xmldoc.getAttribute('status')
    if status != 'ok':
        return ''
        
    urlelem = xmldoc.getElementsByTagName('url')
    if not urlelem or not urlelem[0].firstChild:
        return ''
        
    preview = urlelem[0].firstChild.nodeValue
    return preview


def get_preview_url(track_name, artist_name, track_7digitalid=-1, 
                   release_7digitalid=-1, artist_7digitalid=-1):
    """
    尝试通过多种方法获取歌曲预览URL
    
    参数:
        track_name: 歌曲名称
        artist_name: 艺术家名称
        track_7digitalid: 7digital曲目ID (如果已知)
        release_7digitalid: 7digital专辑ID (如果已知)
        artist_7digitalid: 7digital艺术家ID (如果已知)
        
    返回:
        预览URL，如果无法获取则返回空字符串
    """
    if not DIGITAL7_API_KEY:
        logger.warning("未设置7digital API密钥，无法获取预览URL")
        return ''
    
    # 优先使用已有的7digital曲目ID
    if track_7digitalid >= 0:
        preview = get_preview_from_trackid(track_7digitalid)
        if preview:
            logger.info(f"通过track_id获取预览URL成功: {track_name}")
            return preview
        logger.warning(f"通过track_id获取预览URL失败: {track_name}")
    
    # 尝试通过专辑ID
    if release_7digitalid >= 0:
        tracks_name_ids = get_tracks_from_releaseid(release_7digitalid)
        if tracks_name_ids:
            closest_track = get_closest_track(list(tracks_name_ids.keys()), track_name)
            if closest_track:
                if closest_track != track_name:
                    logger.info(f"近似匹配歌曲: {track_name} -> {closest_track}")
                preview = get_preview_from_trackid(tracks_name_ids[closest_track])
                if preview:
                    logger.info(f"通过release_id获取预览URL成功: {track_name}")
                    return preview
        logger.warning(f"通过release_id获取预览URL失败: {track_name}")
    
    # 尝试通过艺术家ID
    if artist_7digitalid >= 0:
        tracks_name_ids = get_tracks_from_artistid(artist_7digitalid)
        if tracks_name_ids:
            closest_track = get_closest_track(list(tracks_name_ids.keys()), track_name)
            if closest_track:
                if closest_track != track_name:
                    logger.info(f"近似匹配歌曲: {track_name} -> {closest_track}")
                preview = get_preview_from_trackid(tracks_name_ids[closest_track])
                if preview:
                    logger.info(f"通过artist_id获取预览URL成功: {track_name}")
                    return preview
        logger.warning(f"通过artist_id获取预览URL失败: {track_name}")
    
    # 最后尝试通过文本搜索
    res = get_trackid_from_text_search(track_name, artistname=artist_name)
    if res:
        closest_track, trackid = res
        if closest_track != track_name:
            logger.info(f"文本搜索近似匹配: {track_name} -> {closest_track}")
        preview = get_preview_from_trackid(trackid)
        if preview:
            logger.info(f"通过文本搜索获取预览URL成功: {track_name}")
            return preview
    
    logger.warning(f"无法为歌曲获取预览URL: {track_name} - {artist_name}")
    return '' 