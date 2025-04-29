import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm
import os
import json
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置日志记录器
logger = logging.getLogger(__name__)

# 导入Spotify处理器
from backend.models.spotify_processor import SpotifyProcessor

class MSDDataProcessor:
    """MSD数据处理器"""
    
    def __init__(self, output_dir="processed_data", use_spotify=False, 
                 spotify_client_id=None, spotify_client_secret=None,
                 spotify_batch_size=50, spotify_workers=5, spotify_strategy="all",
                 spotify_cache_file=None, force_process=False):
        """
        初始化MSD数据处理器
        
        参数:
            output_dir (str): 处理后数据的输出目录
            use_spotify (bool): 是否使用Spotify数据
            spotify_client_id (str): Spotify API客户端ID
            spotify_client_secret (str): Spotify API客户端密钥
            spotify_batch_size (int): Spotify API批处理大小
            spotify_workers (int): Spotify API并发工作线程数
            spotify_strategy (str): Spotify处理策略 (all, popular, diverse)
            spotify_cache_file (str): Spotify缓存文件路径
            force_process (bool): 是否强制重新处理数据，忽略缓存
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Spotify相关设置
        self.use_spotify = use_spotify
        self.spotify_batch_size = spotify_batch_size
        self.spotify_workers = spotify_workers
        self.spotify_strategy = spotify_strategy
        self.spotify_features = None  # 用于存储获取的Spotify特征
        self.force_process = force_process  # 添加force_process属性
        
        if use_spotify:
            cache_dir = os.path.join(output_dir, 'spotify_cache')
            if spotify_cache_file:
                cache_dir = os.path.dirname(spotify_cache_file)
                
            self.spotify_processor = SpotifyProcessor(
                client_id=spotify_client_id,
                client_secret=spotify_client_secret,
                cache_dir=cache_dir,
                max_workers=spotify_workers,
                batch_size=spotify_batch_size,
                strategy=spotify_strategy,
                force_process=force_process  # 传递force_process参数
            )
            
            # 如果指定了自定义缓存文件
            if spotify_cache_file:
                self.spotify_processor.cache_file = Path(spotify_cache_file)
                # 尝试加载自定义缓存
                self.spotify_processor.load_cache()
        else:
            self.spotify_processor = None
        
    def load_h5_metadata(self, h5_path):
        """加载MSD的HDF5元数据文件"""
        print("加载MSD元数据...")
        with h5py.File(h5_path, "r") as f:
            try:
                # 尝试直接访问预期路径
                metadata = f['metadata']['songs']
                analysis = f['analysis']['songs']
            except KeyError:
                # 备用方案：搜索正确路径
                print("尝试查找正确的HDF5数据路径...")
                metadata = f.get('metadata', f).get('songs', None)
                analysis = f.get('analysis', f).get('songs', None)
                
            # 提取关键字段 - 修复编码问题
            song_df = pd.DataFrame({
                'song_id': [s.decode('utf-8', errors='replace') for s in metadata['song_id'][:]],
                'title': [s.decode('utf-8', errors='replace') for s in metadata['title'][:]],
                'artist_name': [s.decode('utf-8', errors='replace') for s in metadata['artist_name'][:]],
            })
            
            # 添加可能存在的分析特征
            if analysis is not None:
                if 'duration' in analysis.dtype.names:
                    song_df['duration'] = analysis['duration'][:]
                if 'tempo' in analysis.dtype.names:
                    song_df['tempo'] = analysis['tempo'][:]
                if 'loudness' in analysis.dtype.names:
                    song_df['loudness'] = analysis['loudness'][:]
                if 'key' in analysis.dtype.names:
                    song_df['key'] = analysis['key'][:]
                if 'mode' in analysis.dtype.names:
                    song_df['mode'] = analysis['mode'][:]
                if 'song_hotttnesss' in analysis.dtype.names:
                    song_df['song_hotttnesss'] = analysis['song_hotttnesss'][:]
                if 'danceability' in analysis.dtype.names:
                    song_df['danceability'] = analysis['danceability'][:]
                if 'energy' in analysis.dtype.names:
                    song_df['energy'] = analysis['energy'][:]
        
        # 填充缺失值
        song_df = song_df.fillna({
            'tempo': 120.0,
            'loudness': -20.0,
            'duration': 240.0,
            'key': 0,
            'mode': 0,
            'title': 'Unknown Title',
            'artist_name': 'Unknown Artist',
            'song_hotttnesss': 0.0,
            'danceability': 0.5,
            'energy': 0.5
        })
        
        # 添加派生特征
        song_df['energy_ratio'] = (song_df['loudness'] + 60) / 60
        song_df['tempo_norm'] = song_df['tempo'] / 200.0
            
        print(f"成功加载了 {len(song_df)} 首歌曲元数据")
        return song_df
    
    def load_triplets(self, triplet_path, chunk_size=1e6, limit=None):
        """分块加载用户-歌曲-播放次数三元组数据"""
        print("加载用户播放记录...")
        chunks = []
        processed_chunks = 0
        
        # 使用分块读取减少内存压力
        reader = pd.read_csv(
            triplet_path,
            sep='\t', 
            names=['user_id', 'song_id', 'plays'],
            dtype={'user_id': str, 'song_id': str, 'plays': np.int32},
            chunksize=int(chunk_size),
            on_bad_lines='skip'
        )
        
        for chunk in tqdm(reader, desc="加载三元组数据"):
            # 基本清洗
            chunk = chunk.dropna()
            chunk = chunk[chunk['plays'] > 0]
            
            chunks.append(chunk)
            processed_chunks += 1
            
            # 限制处理的数据量(用于测试)
            if limit and processed_chunks >= limit:
                print(f"达到指定块限制({limit})，停止加载")
                break
                
        triplets = pd.concat(chunks, ignore_index=True)
        print(f"加载了 {len(triplets)} 条播放记录，涉及 {triplets['user_id'].nunique()} 个用户")
        return triplets
    
    def align_data(self, song_df, triplets):
        """确保三元组数据中的歌曲都存在于元数据中"""
        print("对齐数据...")
        valid_songs = set(song_df['song_id'])
        mask = triplets['song_id'].isin(valid_songs)
        filtered = triplets[mask]
        
        print(f"过滤无效歌曲记录: {len(triplets) - len(filtered)}/{len(triplets)}")
        return filtered
    
    def normalize_plays(self, triplets, method='log'):
        """
        将播放次数归一化为评分
        
        参数:
            triplets: 包含用户、歌曲和播放次数的DataFrame
            method: 归一化方法，可选 'log'(对数)、'linear'(线性)、'percentile'(百分比)
            
        返回:
            添加了rating列的DataFrame
        """
        print(f"使用 {method} 方法归一化播放次数...")
        
        if method == 'log':
            # 对数变换+归一化到1-5分
            triplets['rating'] = triplets['plays'].apply(
                lambda x: min(5, max(1, int(np.log2(x + 1) + 1)))
            )
        elif method == 'linear':
            # 基于用户平均播放次数的相对比例
            # 1. 计算每个用户的平均播放次数
            user_avg_plays = triplets.groupby('user_id')['plays'].mean()
            
            # 2. 对每个用户计算相对播放比例并转换为评分
            def get_relative_rating(row):
                user_avg = user_avg_plays[row['user_id']]
                relative = row['plays'] / max(1, user_avg)
                # 将相对比例转换为1-5评分
                if relative <= 0.5: return 1
                elif relative <= 0.8: return 2
                elif relative <= 1.2: return 3
                elif relative <= 2.0: return 4
                else: return 5
            
            triplets['rating'] = triplets.apply(get_relative_rating, axis=1)
        elif method == 'percentile':
            # 基于百分比排名
            # 计算每个用户的播放次数百分比排名
            def percentile_rating(user_group):
                plays = user_group['plays'].values
                percentiles = np.percentile(plays, [20, 40, 60, 80])
                
                ratings = np.ones(len(plays))
                for i, p in enumerate(percentiles):
                    ratings[plays > p] = i + 2
                
                user_group['rating'] = ratings
                return user_group
            
            triplets = triplets.groupby('user_id').apply(percentile_rating)
        else:
            # 默认使用对数方法
            triplets['rating'] = triplets['plays'].apply(
                lambda x: min(5, max(1, int(np.log2(x + 1) + 1)))
            )
        
        # 确保评分是整数1-5
        triplets['rating'] = triplets['rating'].astype(int)
        triplets['rating'] = triplets['rating'].clip(1, 5)
        
        # 打印评分分布
        rating_counts = triplets['rating'].value_counts().sort_index()
        print("评分分布:")
        for rating, count in rating_counts.items():
            print(f"  评分 {rating}: {count} ({count/len(triplets)*100:.1f}%)")
            
        return triplets
    
    def extract_audio_features(self, song_df):
        """
        从歌曲数据中提取音频特征，并返回一个包含特征的数据框
        """
        logger.debug(f"提取音频特征，歌曲数量: {len(song_df)}")
        
        # 确保song_id列为字符串类型
        song_df = song_df.copy()
        song_df['song_id'] = song_df['song_id'].astype(str)
        
        # 检查是否有缓存的Spotify特征
        if self.spotify_features is not None and not self.spotify_features.empty:
            logger.debug(f"使用缓存的Spotify特征，特征数量: {len(self.spotify_features)}")
            # 确保spotify_features中的song_id也是字符串类型
            self.spotify_features['song_id'] = self.spotify_features['song_id'].astype(str)
            
            # 将song_id设为索引以便合并
            song_df_indexed = song_df.set_index('song_id')
            spotify_features_indexed = self.spotify_features.set_index('song_id')
            
            # 只保留song_df中存在的歌曲ID的特征
            valid_song_ids = set(song_df['song_id']) & set(self.spotify_features['song_id'])
            features = spotify_features_indexed.loc[list(valid_song_ids)].reset_index()
            
            if len(features) > 0:
                logger.debug(f"找到 {len(features)} 首歌曲的音频特征")
                # 确保特征为数值类型，除了song_id列
                numeric_cols = features.columns.difference(['song_id'])
                for col in numeric_cols:
                    try:
                        features[col] = pd.to_numeric(features[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"转换列 {col} 为数值型时出错: {e}")
                
                # 填充缺失值，注意这里先复制一份numeric_cols列进行均值计算，避免对song_id进行mean操作
                if not features[numeric_cols].empty:
                    features[numeric_cols] = features[numeric_cols].fillna(features[numeric_cols].mean())
                return features
        
        # 如果没有缓存的特征或没有匹配的特征，则尝试创建模拟特征
        logger.warning("没有找到有效的音频特征，创建模拟特征")
        return self.create_mock_audio_features(song_df)

    def create_mock_audio_features(self, song_df):
        """
        创建模拟的音频特征
        """
        # 确保song_id是字符串类型
        song_df = song_df.copy()
        song_df['song_id'] = song_df['song_id'].astype(str)
        
        # 定义特征名称
        feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                         'liveness', 'loudness', 'speechiness', 'tempo', 'valence']
        
        # 为每首歌曲创建随机特征
        np.random.seed(42)  # 设置随机种子以确保结果可重复
        num_songs = len(song_df)
        
        # 创建模拟特征DataFrame
        mock_features = pd.DataFrame()
        mock_features['song_id'] = song_df['song_id'].values
        
        # 为每个特征生成随机值
        for feature in feature_names:
            if feature == 'loudness':
                # loudness通常是负值，范围约为-60到0
                mock_features[feature] = np.random.uniform(-60, 0, num_songs)
            elif feature == 'tempo':
                # tempo通常在60-200之间
                mock_features[feature] = np.random.uniform(60, 200, num_songs)
            else:
                # 其他特征通常在0-1之间
                mock_features[feature] = np.random.uniform(0, 1, num_songs)
        
        logger.debug(f"创建了 {num_songs} 首歌曲的模拟音频特征")
        return mock_features
    
    def create_user_features(self, triplets):
        """根据用户行为创建用户特征"""
        print("创建用户特征...")
        # 每个用户的播放统计
        user_stats = triplets.groupby('user_id').agg({
            'plays': ['count', 'mean', 'std', 'max', 'sum'],
            'song_id': 'nunique'
        })
        
        user_stats.columns = ['total_interactions', 'avg_plays', 'std_plays', 'max_plays', 'total_plays', 'unique_songs']
        user_stats = user_stats.fillna(0)
        
        # 添加衍生特征
        user_stats['plays_per_song'] = user_stats['total_plays'] / user_stats['unique_songs'].clip(lower=1)
        user_stats['diversity_score'] = user_stats['unique_songs'] / user_stats['total_interactions'].clip(lower=1)
        
        # 添加用户类型
        def get_user_type(row):
            total_plays = row['total_plays']
            if total_plays < 10:
                return 'casual'
            elif total_plays < 100:
                return 'regular'
            else:
                return 'active'
                
        user_stats['user_type'] = user_stats.apply(get_user_type, axis=1)
        
        # 重置索引，确保user_id是一列
        user_stats = user_stats.reset_index()
        
        return user_stats
    
    def enrich_with_spotify(self, songs_df, max_songs=1000):
        """使用Spotify数据丰富歌曲信息"""
        if not self.use_spotify or self.spotify_processor is None:
            print("未启用Spotify数据丰富功能")
            return songs_df
            
        print("使用Spotify API丰富歌曲数据...")
        print(f"Spotify配置: 批处理大小={self.spotify_batch_size}, 并行线程={self.spotify_workers}, 策略={self.spotify_strategy}")
        
        # 使用Spotify处理器获取额外数据
        enriched_songs, spotify_features = self.spotify_processor.enrich_msd_data(
            songs_df=songs_df,
            max_songs=max_songs,
            batch_size=self.spotify_batch_size
        )
        
        # 保存spotify特征用于音频特征提取
        self.spotify_features = spotify_features
        
        print(f"Spotify数据丰富完成，处理了 {len(enriched_songs)} 首歌曲")
        return enriched_songs
    
    def process_msd_data(self, h5_file, triplet_file, output_dir=None, 
                         chunk_limit=None, max_spotify_songs=1000, rating_method='log',
                         force_process=False, spotify_batch_size=None, spotify_workers=None,
                         spotify_strategy=None, user_sample=None, no_filter_inactive_users=False):
        """
        处理MSD数据并生成推荐系统所需的数据格式
        
        参数:
            h5_file: HDF5元数据文件路径
            triplet_file: 三元组数据文件路径
            output_dir: 处理后数据的保存目录 (可选)
            chunk_limit: 限制处理的数据块数量 (用于测试)
            max_spotify_songs: 最多处理的Spotify歌曲数量
            rating_method: 播放次数归一化方法 (log, linear, percentile)
            force_process: 是否强制重新处理数据
            spotify_batch_size: Spotify API批次大小
            spotify_workers: Spotify API工作线程数
            spotify_strategy: Spotify处理策略
            user_sample: 限制使用的用户数量，随机采样指定数量的用户
            no_filter_inactive_users: 不过滤不活跃用户，保留所有用户
            
        返回:
            包含处理后数据的元组 (songs, interactions, audio_features, user_features)
        """
        # 使用提供的或初始化时设置的输出目录
        if output_dir is None:
            output_dir = self.output_dir
            
        # 更新Spotify参数(如果提供的话)
        if spotify_batch_size is not None:
            self.spotify_batch_size = spotify_batch_size
        if spotify_workers is not None:
            self.spotify_workers = spotify_workers
        if spotify_strategy is not None:
            self.spotify_strategy = spotify_strategy
            
        # 如果我们使用Spotify，更新force_process参数
        if self.use_spotify and self.spotify_processor is not None:
            self.spotify_processor.force_process = force_process
            
        # 检查是否已有处理好的数据
        songs_file = os.path.join(output_dir, 'songs.csv')
        interactions_file = os.path.join(output_dir, 'interactions.csv')
        audio_file = os.path.join(output_dir, 'audio_features.csv')
        users_file = os.path.join(output_dir, 'user_features.csv')
        
        # 检查缓存数据是否存在且是否需要强制重新处理
        if os.path.exists(songs_file) and os.path.exists(interactions_file) and not force_process:
            print(f"加载已处理数据从 {output_dir}")
            songs = pd.read_csv(songs_file)
            interactions = pd.read_csv(interactions_file)
            
            if os.path.exists(audio_file):
                audio_features = pd.read_csv(audio_file)
            else:
                audio_features = None
                
            if os.path.exists(users_file):
                user_features = pd.read_csv(users_file)
            else:
                user_features = None
                
            return songs, interactions, audio_features, user_features
        
        # 如果启用了force_process，输出信息
        if force_process:
            print("启用了force_process，将重新处理所有数据，忽略缓存")
        
        # 执行数据处理
        song_df = self.load_h5_metadata(h5_file)
        triplets = self.load_triplets(triplet_file, limit=chunk_limit)

        # ===== 修改：优先使用缓存的Spotify特征 =====
        spotify_song_ids = None
        if self.use_spotify and self.spotify_processor is not None:
            # 检查是否有缓存的Spotify特征
            spotify_cache_dir = os.path.join(output_dir, 'spotify_cache')
            spotify_features_path = os.path.join(spotify_cache_dir, 'spotify_features.parquet')
            
            if os.path.exists(spotify_features_path) and not force_process:
                print(f"直接加载缓存的Spotify特征: {spotify_features_path}")
                try:
                    self.spotify_features = pd.read_parquet(spotify_features_path)
                    spotify_song_ids = set(self.spotify_features['song_id'].astype(str))
                    print(f"使用缓存的Spotify特征，共 {len(spotify_song_ids)} 首歌曲")
                except Exception as e:
                    print(f"加载缓存的Spotify特征失败: {e}, 将重新获取特征")
                    self.spotify_features = None
            elif force_process and os.path.exists(spotify_features_path):
                print("启用了force_process，忽略缓存的Spotify特征")
                self.spotify_features = None
            
            # 如果没有缓存或加载失败，则调用API获取
            if self.spotify_features is None or spotify_song_ids is None:
                print("没有找到有效的缓存特征，调用Spotify API...")
                enriched_songs = self.enrich_with_spotify(song_df, max_songs=max_spotify_songs)
                if self.spotify_features is not None and not self.spotify_features.empty:
                    spotify_song_ids = set(self.spotify_features['song_id'].astype(str))
                    print(f"重新获取Spotify特征成功，共 {len(spotify_song_ids)} 首歌曲")
                else:
                    print("警告：未获取到Spotify特征，将使用全部歌曲")
            
            # 只保留有Spotify特征的歌曲
            if spotify_song_ids:
                print(f"仅保留有Spotify特征的歌曲，共 {len(spotify_song_ids)} 首")
                song_df = song_df[song_df['song_id'].isin(spotify_song_ids)].reset_index(drop=True)
        
        # ===== 只保留与有Spotify特征歌曲有关的交互 =====
        if spotify_song_ids is not None:
            before_filter = len(triplets)
            triplets = triplets[triplets['song_id'].isin(spotify_song_ids)].reset_index(drop=True)
            print(f"仅保留与Spotify特征歌曲有关的交互，共 {len(triplets)} 条 (过滤掉 {before_filter - len(triplets)} 条)")
            
            # 只保留与这些歌曲有交互的用户
            users_with_spotify = set(triplets['user_id'])
            print(f"仅保留与Spotify特征歌曲有交互的用户，共 {len(users_with_spotify)} 名")
            
            # 用户交互数量统计
            user_interactions = triplets.groupby('user_id').size()
            active_users = user_interactions[user_interactions >= 5].index
            print(f"其中活跃用户(>=5次交互)数量: {len(active_users)}")
            
            # 打印交互分布统计
            interactions_stats = user_interactions.describe()
            print(f"每位用户的交互次数统计: 最小={interactions_stats['min']:.0f}, 平均={interactions_stats['mean']:.1f}, 最大={interactions_stats['max']:.0f}")
        
        # 用户采样功能 - 如果指定了采样数量
        if user_sample is not None and user_sample > 0:
            unique_users = triplets['user_id'].unique()
            sample_size = min(user_sample, len(unique_users))
            print(f"采样 {sample_size} 名用户，从原始用户集合 {len(unique_users)} 名用户中")
            # 随机采样用户
            sampled_users = np.random.choice(unique_users, size=sample_size, replace=False)
            triplets = triplets[triplets['user_id'].isin(sampled_users)]
            print(f"采样后的用户数: {triplets['user_id'].nunique()}, 交互数: {len(triplets)}")
        
        # 过滤不活跃用户 - 如果没有禁用过滤
        if not no_filter_inactive_users:
            print("过滤不活跃用户...")
            # 计算用户活跃度
            user_activity = triplets.groupby('user_id').size()
            active_users = user_activity[user_activity >= 5].index  # 至少有5次播放
            triplets = triplets[triplets['user_id'].isin(active_users)]
            print(f"保留活跃用户后的交互数: {len(triplets)}, 用户数: {triplets['user_id'].nunique()}")
        else:
            print("不过滤不活跃用户，保留所有用户。")
        
        # 对齐数据
        triplets = self.align_data(song_df, triplets)
        
        # 归一化播放次数
        triplets = self.normalize_plays(triplets, method=rating_method)
        
        # 提取音频特征
        audio_features = self.extract_audio_features(song_df)
        
        # 创建用户特征
        user_features = self.create_user_features(triplets)
        
        # 保存处理后的数据
        self._save_processed_data(song_df, triplets, audio_features, user_features, output_dir)
        
        return song_df, triplets, audio_features, user_features
        
    def _save_processed_data(self, songs, interactions, audio_features, user_features, output_dir=None):
        """保存处理后的数据"""
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"保存处理后的数据到 {output_dir}...")
        
        # 设置索引列
        songs = songs.set_index('song_id', drop=False)
        audio_features = audio_features.set_index('song_id', drop=False)
        
        # 以Parquet格式保存（高效存储）
        songs.to_parquet(os.path.join(output_dir, 'songs.parquet'), index=False)
        interactions.to_parquet(os.path.join(output_dir, 'interactions.parquet'), index=False)
        audio_features.to_parquet(os.path.join(output_dir, 'audio_features.parquet'), index=False)
        user_features.to_parquet(os.path.join(output_dir, 'user_features.parquet'), index=False)
        
        # 同时保存为CSV格式（便于检查）
        songs.to_csv(os.path.join(output_dir, 'songs.csv'), index=False)
        interactions.to_csv(os.path.join(output_dir, 'interactions.csv'), index=False)
        audio_features.to_csv(os.path.join(output_dir, 'audio_features.csv'), index=False)
        user_features.to_csv(os.path.join(output_dir, 'user_features.csv'), index=False)
        
        print("数据保存完成") 