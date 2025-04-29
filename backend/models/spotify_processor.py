#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spotify数据处理器
用于获取和处理Spotify音乐信息，补充MSD数据集
"""

import os
import time
import json
import logging
import pandas as pd
import numpy as np
import spotipy
import pickle
from spotipy.oauth2 import SpotifyClientCredentials
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

# 设置日志
logger = logging.getLogger('spotify_processor')

class SpotifyProcessor:
    """Spotify数据处理类，用于获取音乐信息并与MSD数据集整合"""
    
    def __init__(self, client_id=None, client_secret=None, cache_dir='spotify_cache', 
                 max_workers=5, batch_size=50, strategy='all', force_process=False):
        """
        初始化Spotify处理器
        
        参数:
            client_id (str): Spotify API Client ID
            client_secret (str): Spotify API Client Secret
            cache_dir (str): Spotify数据缓存目录
            max_workers (int): 并行处理的最大工作线程数
            batch_size (int): API调用批处理大小
            strategy (str): 处理策略 ('all', 'popular', 'diverse')
            force_process (bool): 强制重新处理数据，忽略缓存
        """
        # 从环境变量或参数获取认证信息
        self.client_id = client_id or os.environ.get('SPOTIFY_CLIENT_ID')
        self.client_secret = client_secret or os.environ.get('SPOTIFY_CLIENT_SECRET')
        
        # 添加force_process属性
        self.force_process = force_process
        
        # 验证认证信息
        if not self.client_id or not self.client_secret:
            logger.warning("未提供Spotify API凭证，将无法获取Spotify数据")
            self.sp = None
        else:
            # 初始化Spotify客户端
            credentials = SpotifyClientCredentials(
                client_id=self.client_id,
                client_secret=self.client_secret
            )
            self.sp = spotipy.Spotify(client_credentials_manager=credentials)
            logger.info("Spotify客户端初始化成功")
        
        # 设置缓存目录
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # 读取缓存的映射（MSD歌曲ID到Spotify ID）
        self.mapping_file = self.cache_dir / 'msd_to_spotify.json'
        self.id_mapping = self._load_mapping()
        
        # 读取缓存的特征
        self.features_file = self.cache_dir / 'spotify_features.parquet'
        self.sp_features = self._load_features()
        
        # 新增的缓存文件
        self.cache_file = self.cache_dir / 'spotify_cache.pkl'
        
        # 并行处理参数
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.strategy = strategy
    
    def _load_mapping(self):
        """加载缓存的MSD与Spotify ID映射"""
        if self.mapping_file.exists() and not self.force_process:
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"已加载 {len(mapping)} 条MSD-Spotify ID映射")
                return mapping
            except Exception as e:
                logger.error(f"加载映射文件出错: {e}")
        elif self.force_process and self.mapping_file.exists():
            logger.info("强制重新处理，忽略现有的ID映射缓存")
        
        return {}
    
    def _save_mapping(self):
        """保存MSD与Spotify ID映射到缓存"""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.id_mapping, f, ensure_ascii=False, indent=2)
            logger.info(f"已保存 {len(self.id_mapping)} 条MSD-Spotify ID映射")
        except Exception as e:
            logger.error(f"保存映射文件出错: {e}")
    
    def _load_features(self):
        """加载缓存的Spotify特征"""
        if self.force_process and self.features_file.exists():
            logger.info("由于force_process=True，跳过加载特征缓存")
            return pd.DataFrame()
            
        if self.features_file.exists():
            try:
                features = pd.read_parquet(self.features_file)
                logger.info(f"已加载 {len(features)} 条Spotify特征数据")
                return features
            except Exception as e:
                logger.error(f"加载特征文件出错: {e}")
                
        return pd.DataFrame()
    
    def _save_features(self):
        """保存Spotify特征到缓存"""
        if not self.sp_features.empty:
            try:
                self.sp_features.to_parquet(self.features_file, index=False)
                logger.info(f"已保存 {len(self.sp_features)} 条Spotify特征数据")
            except Exception as e:
                logger.error(f"保存特征文件出错: {e}")
    
    def save_cache(self):
        """保存Spotify缓存到文件"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.id_mapping, f)
        logger.info(f"保存了 {len(self.id_mapping)} 条Spotify映射到缓存")

    def load_cache(self):
        """加载Spotify缓存"""
        if os.path.exists(self.cache_file) and not self.force_process:
            with open(self.cache_file, 'rb') as f:
                self.id_mapping = pickle.load(f)
            logger.info(f"从缓存加载了 {len(self.id_mapping)} 条Spotify映射")
        elif self.force_process and os.path.exists(self.cache_file):
            logger.info("强制重新处理，忽略现有缓存")
                
    def _search_with_backoff(self, query, retry_count=5, initial_backoff=2):
        """使用指数退避策略的Spotify搜索"""
        backoff = initial_backoff
        
        for attempt in range(retry_count):
            try:
                results = self.sp.search(q=query, type='track', limit=1)
                return results
            except Exception as e:
                if "rate limit" in str(e).lower():
                    sleep_time = backoff * (2 ** attempt)
                    logger.info(f"速率限制，等待 {sleep_time}秒后重试...")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"Spotify搜索错误: {e}")
                    break
        
        return None
    
    def search_track(self, title, artist, retries=3, delay=1):
        """
        在Spotify搜索歌曲
        
        参数:
            title (str): 歌曲标题
            artist (str): 艺术家名称
            retries (int): 重试次数
            delay (float): 重试间隔(秒)
            
        返回:
            dict 或 None: Spotify歌曲信息或None(如果未找到)
        """
        if not self.sp:
            return None
        
        # 构建搜索查询
        query = f"track:{title} artist:{artist}"
        
        # 使用指数退避策略
        results = self._search_with_backoff(query, retry_count=retries)
        if not results:
            return None
            
        if results and results['tracks']['items']:
            # 返回第一个匹配结果
            return results['tracks']['items'][0]
        return None
    
    def _search_spotify(self, song):
        """为并行处理准备的搜索方法"""
        title = song.get('title', '')
        artist = song.get('artist_name', '')
        song_id = song.get('song_id', '')
        
        if not title or not artist:
            return None
            
        track = self.search_track(title, artist)
        if track:
            return {'song_id': song_id, 'spotify_id': track.get('id')}
        
        return None
    
    def parallel_spotify_search(self, songs, max_workers=None):
        """并行搜索Spotify歌曲"""
        if not max_workers:
            max_workers = self.max_workers
            
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._search_spotify, song): song for song in songs}
            
            for future in tqdm(concurrent.futures.as_completed(futures), 
                               total=len(futures), desc="并行Spotify搜索"):
                song = futures[future]
                try:
                    result = future.result()
                    if result:
                        song_id = result.get('song_id')
                        spotify_id = result.get('spotify_id')
                        if song_id and spotify_id:
                            results[song_id] = spotify_id
                except Exception as e:
                    logger.warning(f"处理歌曲 {song.get('title', 'unknown')} 时出错: {e}")
                    
        return results
    
    def process_songs_in_batches(self, songs, batch_size=None):
        """批量处理歌曲"""
        if batch_size is None:
            batch_size = self.batch_size
            
        results = {}
        song_batches = [songs[i:i+batch_size] for i in range(0, len(songs), batch_size)]
        
        for batch in tqdm(song_batches, desc="处理Spotify批次"):
            # 处理一批歌曲
            batch_results = self.parallel_spotify_search(batch)
            results.update(batch_results)
            # 添加延迟避免速率限制
            time.sleep(1)
        
        return results
    
    def filter_songs_for_spotify(self, songs, n=10000, strategy=None):
        """智能选择要处理的歌曲"""
        if strategy is None:
            strategy = self.strategy
            
        if strategy == 'all' or len(songs) <= n:
            return songs
            
        if strategy == 'popular':
            # 按流行度排序
            if 'song_hotttnesss' in songs.columns:
                return songs.sort_values('song_hotttnesss', ascending=False).head(n)
            else:
                return songs.head(n)
                
        elif strategy == 'diverse':
            # 确保不同艺术家的歌曲都有代表
            artists = {}
            selected = []
            
            for _, song in songs.iterrows():
                artist = song.get('artist_name', 'unknown')
                if artist not in artists or len(artists[artist]) < 3:
                    if artist not in artists:
                        artists[artist] = []
                    artists[artist].append(song)
                    selected.append(song)
                    
                    if len(selected) >= n:
                        break
            
            return pd.DataFrame(selected)
        
        # 默认返回前n首
        return songs.head(n)
    
    def get_audio_features(self, spotify_ids, retries=3, delay=1):
        """
        获取Spotify音频特征
        
        参数:
            spotify_ids (list): Spotify歌曲ID列表
            retries (int): 重试次数
            delay (float): 重试间隔(秒)
            
        返回:
            list: 音频特征列表
        """
        if not self.sp:
            return []
            
        # 如果force_process为True，或者没有缓存的特征，则获取所有ID的特征
        if not self.force_process and not self.sp_features.empty:
            # 检查哪些ID已经在缓存中
            cached_ids = set(self.sp_features['id']) if 'id' in self.sp_features.columns else set()
            # 只获取不在缓存中的ID
            missing_ids = [sid for sid in spotify_ids if sid not in cached_ids]
            
            if not missing_ids:
                logger.info("所有请求的特征都在缓存中")
                return []
                
            logger.info(f"从 {len(spotify_ids)} 个ID中，{len(missing_ids)} 个需要获取")
            spotify_ids = missing_ids
        else:
            if self.force_process:
                logger.info(f"由于force_process=True，获取所有 {len(spotify_ids)} 个特征")
            else:
                logger.info(f"没有找到有效的缓存特征，调用Spotify API获取 {len(spotify_ids)} 个特征")
            
        # 限制每批次请求数量（Spotify API限制）
        batch_size = 100
        all_features = []
        
        # 分批处理
        for i in range(0, len(spotify_ids), batch_size):
            batch_ids = spotify_ids[i:i+batch_size]
            
            for attempt in range(retries):
                try:
                    batch_features = self.sp.audio_features(batch_ids)
                    # 过滤掉None值
                    batch_features = [f for f in batch_features if f]
                    all_features.extend(batch_features)
                    break
                    
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(f"获取音频特征时出错: {e}, 重试中...")
                        time.sleep(delay)
                    else:
                        logger.error(f"获取音频特征失败: {e}")
                        
            # 避免API限制
            if i + batch_size < len(spotify_ids):
                time.sleep(0.5)
                
        return all_features
    
    def get_track_popularity(self, spotify_ids, retries=3, delay=1):
        """
        获取Spotify歌曲流行度
        
        参数:
            spotify_ids (list): Spotify歌曲ID列表
            retries (int): 重试次数
            delay (float): 重试间隔(秒)
            
        返回:
            dict: 歌曲ID到流行度的映射
        """
        if not self.sp:
            return {}
            
        popularity_map = {}
        
        for spotify_id in spotify_ids:
            for attempt in range(retries):
                try:
                    track = self.sp.track(spotify_id)
                    if track:
                        popularity_map[spotify_id] = track.get('popularity', 0)
                    break
                    
                except Exception as e:
                    if attempt < retries - 1:
                        logger.warning(f"获取歌曲流行度时出错 ({spotify_id}): {e}, 重试中...")
                        time.sleep(delay)
                    else:
                        logger.error(f"获取歌曲流行度失败 ({spotify_id}): {e}")
            
            # 避免API限制
            time.sleep(0.2)
        
        return popularity_map
    
    def enrich_msd_data(self, songs_df, max_songs=None, batch_size=None, save_interval=1000):
        """
        使用Spotify数据丰富MSD数据集
        
        参数:
            songs_df (DataFrame): MSD歌曲数据
            max_songs (int): 处理的最大歌曲数（用于测试）
            batch_size (int): 批处理大小
            save_interval (int): 缓存保存间隔
            
        返回:
            tuple: (带有Spotify特征的MSD数据, Spotify特征DataFrame)
        """
        if not self.sp:
            logger.warning("Spotify客户端未初始化，跳过数据丰富")
            return songs_df, pd.DataFrame()
            
        # 如果强制重新处理，清空特征缓存
        if self.force_process:
            logger.info("强制重新处理，清空特征缓存")
            self.sp_features = pd.DataFrame()
        # 否则加载特征缓存（如果尚未加载）
        elif self.sp_features.empty:
            self.sp_features = self._load_features()
            
        # 筛选需要处理的歌曲
        if max_songs and max_songs < len(songs_df):
            # 使用智能选择策略
            process_songs = self.filter_songs_for_spotify(songs_df, n=max_songs)
        else:
            process_songs = songs_df.copy()
        
        # 检查哪些歌曲需要获取Spotify ID
        songs_to_search = []
        for idx, row in process_songs.iterrows():
            song_id = row['song_id']
            if song_id not in self.id_mapping:
                songs_to_search.append(row.to_dict())
        
        logger.info(f"需要搜索 {len(songs_to_search)} 首歌曲的Spotify ID")
        
        if songs_to_search:
            # 使用并行批处理
            if batch_size is None:
                batch_size = self.batch_size
                
            search_results = self.process_songs_in_batches(songs_to_search, batch_size)
            
            # 更新映射
            if search_results:
                self.id_mapping.update(search_results)
                # 保存映射
                self._save_mapping()
                # 同时更新缓存
                self.save_cache()
        
        # 获取映射的Spotify ID列表
        spotify_ids = []
        msd_to_spotify_map = {}
        
        for song_id in process_songs['song_id']:
            if song_id in self.id_mapping:
                spotify_id = self.id_mapping[song_id]
                spotify_ids.append(spotify_id)
                msd_to_spotify_map[song_id] = spotify_id
        
        logger.info(f"成功映射 {len(spotify_ids)} 首歌曲到Spotify ID")
        
        # 检查哪些Spotify ID已经有缓存的特征
        cached_ids = set()
        if not self.sp_features.empty and 'id' in self.sp_features.columns:
            cached_ids = set(self.sp_features['id'])
        
        # 获取需要查询的ID
        new_ids = [id for id in spotify_ids if id not in cached_ids]
        
        if new_ids:
            logger.info(f"从Spotify获取 {len(new_ids)} 首歌曲的音频特征")
            
            # 获取音频特征
            audio_features = self.get_audio_features(new_ids)
            
            if audio_features:
                # 合并新旧特征
                if self.sp_features.empty:
                    self.sp_features = pd.DataFrame(audio_features)
                else:
                    self.sp_features = pd.concat([self.sp_features, pd.DataFrame(audio_features)], ignore_index=True)
                
                # 保存特征
                self._save_features()
        
        # 如果有特征数据，将其合并到歌曲数据中
        if not self.sp_features.empty:
            logger.info("将Spotify特征合并到MSD数据中")
            
            # 创建MSD ID到Spotify特征的映射
            sp_features_dict = {}
            
            for song_id, spotify_id in msd_to_spotify_map.items():
                sp_data = self.sp_features[self.sp_features['id'] == spotify_id]
                if not sp_data.empty:
                    sp_features_dict[song_id] = sp_data.iloc[0].to_dict()
            
            # 添加Spotify特征到MSD数据
            enriched_df = process_songs.copy()
            
            # 要添加的关键特征
            key_features = [
                'danceability', 'energy', 'key', 'loudness', 'mode', 
                'speechiness', 'acousticness', 'instrumentalness', 
                'liveness', 'valence', 'tempo'
            ]
            
            # 初始化特征列
            for feature in key_features:
                enriched_df[f'sp_{feature}'] = None
            
            # 填充特征数据
            for idx, row in enriched_df.iterrows():
                song_id = row['song_id']
                if song_id in sp_features_dict:
                    sp_data = sp_features_dict[song_id]
                    for feature in key_features:
                        if feature in sp_data:
                            enriched_df.at[idx, f'sp_{feature}'] = sp_data[feature]
            
            logger.info(f"成功为 {len(sp_features_dict)} 首歌曲添加Spotify特征")
            
            # 创建特征DataFrame
            spotify_features_df = self.create_spotify_features(enriched_df)
            return enriched_df, spotify_features_df
        
        return process_songs, pd.DataFrame()
    
    def create_spotify_features(self, songs_df):
        """
        从丰富的歌曲数据中提取Spotify特征
        
        参数:
            songs_df (DataFrame): 包含Spotify特征的歌曲数据
            
        返回:
            DataFrame: Spotify音频特征
        """
        # 检查是否有Spotify特征
        sp_columns = [col for col in songs_df.columns if col.startswith('sp_')]
        if not sp_columns:
            logger.warning("歌曲数据中没有Spotify特征")
            return pd.DataFrame()
        
        # 创建特征DataFrame
        features_df = songs_df[['song_id'] + sp_columns].copy()
        
        # 处理缺失值
        for col in sp_columns:
            features_df[col] = features_df[col].fillna(features_df[col].median())
        
        logger.info(f"创建了 {len(features_df)} 行Spotify特征数据")
        return features_df
    
    def combine_with_msd_features(self, msd_features, spotify_features):
        """
        将Spotify特征与MSD特征合并
        
        参数:
            msd_features (DataFrame): MSD音频特征
            spotify_features (DataFrame): Spotify音频特征
            
        返回:
            DataFrame: 合并后的特征
        """
        if spotify_features.empty:
            logger.warning("没有Spotify特征可合并")
            return msd_features
        
        if msd_features.empty:
            logger.warning("没有MSD特征可合并")
            return spotify_features
        
        # 合并特征
        combined = pd.merge(
            msd_features, 
            spotify_features, 
            on='song_id', 
            how='left'
        )
        
        # 处理缺失值
        sp_columns = [col for col in combined.columns if col.startswith('sp_')]
        for col in sp_columns:
            combined[col] = combined[col].fillna(combined[col].median())
        
        logger.info(f"合并了MSD和Spotify特征，共 {len(combined)} 行")
        return combined

    def load_msd_spotify_mapping(self, cache_dir='processed_data/spotify_cache'):
        """加载或创建MSD和Spotify ID的映射文件"""
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        mapping_file = os.path.join(cache_dir, 'msd_spotify_mapping.csv')
        
        # 如果映射文件存在，直接加载
        if os.path.exists(mapping_file) and not self.force_process:
            print(f"加载MSD-Spotify ID映射从 {mapping_file}")
            mapping_df = pd.read_csv(mapping_file)
            print(f"已加载 {len(mapping_df)} 条MSD-Spotify ID映射")
            return mapping_df
        
        # 如果需要强制处理或文件不存在，重新创建
        print("创建新的MSD-Spotify ID映射")
        # 创建空映射文件
        mapping_df = pd.DataFrame(columns=['song_id', 'spotify_id', 'spotify_name', 'spotify_artist'])
        # 保存空映射
        mapping_df.to_csv(mapping_file, index=False)
        return mapping_df

    def enrich_songs_with_spotify_features(self, songs_df, mapping_df, max_api_calls=1000):
        """使用Spotify API丰富歌曲数据，并保存特征到缓存"""
        cache_dir = 'processed_data/spotify_cache'
        os.makedirs(cache_dir, exist_ok=True)
        features_file = os.path.join(cache_dir, 'spotify_features.parquet')
        
        # 如果特征文件存在且不强制处理，直接加载
        if os.path.exists(features_file) and not self.force_process:
            print(f"加载Spotify音频特征从 {features_file}")
            try:
                features_df = pd.read_parquet(features_file)
                print(f"已加载 {len(features_df)} 首歌曲的Spotify特征")
                return features_df
            except Exception as e:
                print(f"加载Spotify特征缓存失败: {e}")
                print("将重新获取特征")
        elif self.force_process:
            print("强制重新获取Spotify特征，忽略缓存")
        
        # 如果需要重新获取特征
        print("开始从Spotify API获取音频特征...")
        # 获取特征的代码... 