#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify API 功能演示脚本
展示如何使用Spotify API获取歌曲信息、专辑封面和预览播放等功能
"""

from backend.utils.spotify_integration import SpotifyManager
import json
import os
import sys

def print_json(data):
    """格式化打印JSON数据"""
    print(json.dumps(data, ensure_ascii=False, indent=2))

def main():
    # 创建Spotify管理器
    client_id = os.environ.get('SPOTIFY_CLIENT_ID', '4f1a2f4e1e034050ac432f8ebba72484')
    client_secret = os.environ.get('SPOTIFY_CLIENT_SECRET', '4abd4c31749748c8b89f7807c61a3f11')
    spotify = SpotifyManager(client_id=client_id, client_secret=client_secret)
    
    # 确保连接成功
    if not spotify.sp:
        print(f"连接Spotify API失败: {spotify.last_error}")
        return
    
    print("1. 搜索歌曲演示")
    print("-" * 50)
    song_name = input("请输入歌曲名称: ") or "Shape of You"
    artist_name = input("请输入艺术家名称(可选): ") or "Ed Sheeran"
    
    print(f"\n搜索歌曲: {song_name} - {artist_name}")
    tracks = spotify.search_track(song_name, artist_name, limit=3)
    
    if not tracks:
        print("未找到匹配的歌曲。")
    else:
        print(f"找到 {len(tracks)} 首匹配的歌曲:")
        for i, track in enumerate(tracks):
            print(f"  {i+1}. {track['name']} - {track['artists'][0]['name']}")
            print(f"     预览链接: {track.get('preview_url') or '无预览'}")
            print(f"     Spotify链接: {track['external_urls'].get('spotify')}")
            print()
    
    print("\n2. 获取歌曲详细信息演示")
    print("-" * 50)
    
    if tracks:
        track_id = tracks[0]['id']
        print(f"获取歌曲详情，ID: {track_id}")
        track_info = spotify.get_track_info(track_id=track_id)
        
        print("歌曲详细信息:")
        print(f"  名称: {track_info.get('track_name')}")
        print(f"  艺术家: {', '.join([artist['name'] for artist in track_info.get('artists', [])])}")
        print(f"  专辑: {track_info.get('album_name')}")
        print(f"  发行日期: {track_info.get('release_date')}")
        print(f"  流派: {', '.join(track_info.get('genres', ['未知']))}")
        print(f"  专辑封面: {track_info.get('album_cover') or '无'}")
        print(f"  预览链接: {track_info.get('preview_url') or '无预览'}")
        print(f"  时长: {track_info.get('duration_ms', 0) / 1000:.2f} 秒")
        print()
    
    print("\n3. 获取艺术家信息演示")
    print("-" * 50)
    
    artist_name = input("请输入艺术家名称: ") or "Taylor Swift"
    print(f"\n获取艺术家信息: {artist_name}")
    
    artist_info = spotify.get_artist_info(artist_name=artist_name)
    if not artist_info:
        print(f"未找到艺术家: {artist_name}")
    else:
        print("艺术家详细信息:")
        print(f"  名称: {artist_info.get('name')}")
        print(f"  流派: {', '.join(artist_info.get('genres', ['未知']))}")
        print(f"  粉丝数: {artist_info.get('followers', 0):,}")
        print(f"  热门程度: {artist_info.get('popularity')}/100")
        print(f"  图片: {artist_info.get('image') or '无'}")
        
        if artist_info.get('top_tracks'):
            print("\n  热门歌曲:")
            for i, track in enumerate(artist_info['top_tracks']):
                print(f"    {i+1}. {track['name']} - 专辑: {track['album']}")
                print(f"       预览链接: {track.get('preview_url') or '无预览'}")
        
        if artist_info.get('related_artists'):
            print("\n  相关艺术家:")
            for i, artist in enumerate(artist_info['related_artists']):
                print(f"    {i+1}. {artist['name']}")
        
        print()
    
    print("\n4. 获取新发行专辑演示")
    print("-" * 50)
    
    country = input("请输入国家代码(如US, CN): ") or "US"
    print(f"\n获取 {country} 地区的新发行专辑")
    
    new_releases = spotify.get_new_releases(limit=5, country=country)
    if not new_releases:
        print(f"未找到 {country} 地区的新发行专辑")
    else:
        print(f"最新发行的 {len(new_releases)} 张专辑:")
        for i, album in enumerate(new_releases):
            artists = ", ".join([artist['name'] for artist in album.get('artists', [])])
            print(f"  {i+1}. {album['album_name']} - {artists}")
            print(f"     发行日期: {album.get('release_date') or '未知'}")
            print(f"     封面: {album.get('image') or '无'}")
            print(f"     Spotify链接: {album.get('external_url')}")
            print()
    
    print("\n5. 获取相似歌曲推荐演示")
    print("-" * 50)
    
    if tracks:
        track_id = tracks[0]['id']
        track_name = tracks[0]['name']
        artist_name = tracks[0]['artists'][0]['name']
        
        print(f"获取与 '{track_name}' 相似的歌曲:")
        similar_tracks = spotify.get_similar_tracks(track_id=track_id, limit=5)
        
        if not similar_tracks:
            print("未找到相似歌曲")
        else:
            print(f"发现 {len(similar_tracks)} 首相似歌曲:")
            for i, track in enumerate(similar_tracks):
                artists = ", ".join([artist['name'] for artist in track.get('artists', [])])
                print(f"  {i+1}. {track['track_name']} - {artists}")
                print(f"     专辑: {track.get('album_name')}")
                print(f"     预览链接: {track.get('preview_url') or '无预览'}")
                print(f"     Spotify链接: {track.get('external_url')}")
                print()

if __name__ == "__main__":
    main() 