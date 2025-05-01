#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合音乐推荐系统模型

实现混合推荐策略，结合协同过滤、内容特征、上下文感知和深度学习等多种推荐方法。
"""

import os
import pickle
import numpy as np
import pandas as pd
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import json
import sqlite3
from datetime import datetime

# 尝试导入可选依赖
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    from surprise import SVD, Dataset, Reader
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridMusicRecommender:
    """
    混合音乐推荐系统类

    同时使用多种推荐技术:
    1. 协同过滤 (Collaborative Filtering)
    2. 内容特征 (Content-based)
    3. 上下文感知 (Context-aware)
    4. 深度学习 (Deep Learning)
    """

    def __init__(self, data_dir="data", use_msd=False, weights=None,
                 use_cf=True, use_content=True, use_context=True, use_deep_learning=False):
        """
        初始化混合音乐推荐系统

        参数:
            data_dir: 数据目录
            use_msd: 是否使用MSD数据集
            weights: 各算法权重 {方法名: 权重值}
            use_cf: 是否使用协同过滤
            use_content: 是否使用基于内容的推荐
            use_context: 是否使用上下文感知推荐
            use_deep_learning: 是否使用深度学习模型
        """
        self.data_dir = data_dir
        self.use_msd = use_msd

        # 设置各推荐算法启用状态
        self.use_cf = use_cf
        self.use_content = use_content
        self.use_context = use_context
        self.use_deep_learning = use_deep_learning

        # 默认各算法权重
        self.weights = weights or {
            'collaborative': 0.4,
            'content': 0.25,
            'context': 0.15,
            'deep_learning': 0.2
        }

        # 数据
        self.ratings_df = None
        self.songs_df = None
        self.users_df = None

        # 模型组件
        self.cf_model = None
        self.content_similarity = None
        self.song_indices = None
        self.context_model = None
        self.dl_model = None

        # 特征数据
        self.song_features = None
        self.user_features = None

        # 模型辅助数据
        self.user_id_map = None
        self.song_id_map = None

        # MSD增强数据
        self.msd_data = {}

    def load_data(self, ratings_file=None, songs_file=None, users_file=None):
        """
        加载数据

        参数:
            ratings_file: 评分数据文件路径（可选，默认使用data_dir/ratings.csv）
            songs_file: 歌曲数据文件路径（可选，默认使用data_dir/songs.csv）
            users_file: 用户数据文件路径（可选，默认使用data_dir/users.csv）

        返回:
            成功加载则返回True，否则返回False
        """
        logger.info("加载推荐系统数据...")

        try:
            # 确定文件路径
            ratings_path = ratings_file if ratings_file else os.path.join(self.data_dir, "ratings.csv")
            songs_path = songs_file if songs_file else os.path.join(self.data_dir, "songs.csv")
            users_path = users_file if users_file else os.path.join(self.data_dir, "users.csv")

            # 加载评分数据
            if os.path.exists(ratings_path):
                self.ratings_df = pd.read_csv(ratings_path)
                logger.info(f"加载了 {len(self.ratings_df)} 条评分记录")
            else:
                logger.warning(f"评分数据文件不存在: {ratings_path}")
                return False

            # 加载歌曲数据
            if os.path.exists(songs_path):
                self.songs_df = pd.read_csv(songs_path)
                logger.info(f"加载了 {len(self.songs_df)} 首歌曲信息")
            else:
                logger.warning(f"歌曲数据文件不存在: {songs_path}")
                return False

            # 加载用户数据
            if os.path.exists(users_path):
                self.users_df = pd.read_csv(users_path)
                logger.info(f"加载了 {len(self.users_df)} 名用户信息")
            else:
                logger.warning(f"用户数据文件不存在: {users_path}")
                return False

            # 检查数据有效性
            if len(self.ratings_df) == 0 or len(self.songs_df) == 0 or len(self.users_df) == 0:
                logger.warning("加载的数据集为空")
                return False

            # 创建ID映射
            self.user_id_map = {uid: i for i, uid in enumerate(self.users_df['user_id'].unique())}
            self.song_id_map = {sid: i for i, sid in enumerate(self.songs_df['song_id'].unique())}

            # 处理交互数据中的列名
            # 如果interactions.csv使用了ratings.csv中的列名
            if 'song_id' not in self.ratings_df.columns and 'track_id' in self.ratings_df.columns:
                self.ratings_df = self.ratings_df.rename(columns={'track_id': 'song_id'})

            # 如果评分列名使用的是rating而不是listen_count
            if 'rating' not in self.ratings_df.columns and 'listen_count' in self.ratings_df.columns:
                self.ratings_df = self.ratings_df.rename(columns={'listen_count': 'rating'})

            return True

        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            return False

    def preprocess_data(self):
        """预处理数据"""
        logger.info("预处理推荐系统数据...")

        try:
            # 准备歌曲特征
            self._prepare_song_features()

            # 准备用户特征
            self._prepare_user_features()

            # 确保评分数据中的用户ID和歌曲ID都存在于特征中
            valid_users = set(self.users_df['user_id'])
            valid_songs = set(self.songs_df['song_id'])

            # 过滤无效记录
            self.ratings_df = self.ratings_df[
                self.ratings_df['user_id'].isin(valid_users) &
                self.ratings_df['song_id'].isin(valid_songs)
            ]

            logger.info(f"预处理后有 {len(self.ratings_df)} 条有效评分记录")
            return True

        except Exception as e:
            logger.error(f"预处理数据失败: {str(e)}")
            return False

    def _prepare_song_features(self):
        """准备歌曲特征数据"""
        try:
            # 确保歌曲数据中有必要的字段
            required_fields = ['title', 'artist_name', 'genre']
            missing_fields = [field for field in required_fields if field not in self.songs_df.columns]

            # 检查并处理缺失字段
            if missing_fields:
                logger.warning(f"歌曲数据中缺少字段: {missing_fields}")

                # 如果缺少artist_name但有artist字段，进行重命名
                if 'artist_name' in missing_fields and 'artist' in self.songs_df.columns:
                    self.songs_df = self.songs_df.rename(columns={'artist': 'artist_name'})
                    missing_fields.remove('artist_name')

                # 为其他缺失字段创建默认值
                for field in missing_fields:
                    if field == 'title':
                        self.songs_df['title'] = '未知歌曲'
                    elif field == 'artist_name':
                        self.songs_df['artist_name'] = '未知艺术家'
                    elif field == 'genre':
                        self.songs_df['genre'] = '未知流派'

            # 确保genre列存在，如果不存在，尝试从其他列推断
            if 'genre' not in self.songs_df.columns and 'genre_id' in self.songs_df.columns:
                self.songs_df['genre'] = self.songs_df['genre_id'].astype(str)

            # 文本特征: 标题和艺术家
            self.songs_df['text_features'] = self.songs_df['title'].astype(str) + ' ' + self.songs_df['artist_name'].astype(str)

            # 创建歌曲特征矩阵
            feature_cols = []

            # 1. 类别特征 (genre, language, region) - 独热编码
            for col in ['genre']:
                if col in self.songs_df.columns:
                    # 处理可能的NaN值
                    self.songs_df[col] = self.songs_df[col].fillna('未知')
                    # 将类别列拆分为多个值(如果是列表或以逗号分隔)
                    if isinstance(self.songs_df[col].iloc[0], str) and ',' in self.songs_df[col].iloc[0]:
                        # 将','分隔的字符串拆分为列表
                        genres_series = self.songs_df[col].str.split(',')
                        # 创建所有可能的类别值集合
                        all_categories = set()
                        for genres in genres_series:
                            if isinstance(genres, list):
                                all_categories.update(g.strip() for g in genres)

                        # 为每个类别创建一个列
                        for category in all_categories:
                            if category:  # 确保不是空字符串
                                col_name = f"{col}_{category}"
                                self.songs_df[col_name] = self.songs_df[col].apply(
                                    lambda x: 1 if isinstance(x, str) and category in x.split(',') else 0
                                )
                                feature_cols.append(col_name)
                    else:
                        try:
                            # 常规独热编码
                            dummies = pd.get_dummies(self.songs_df[col], prefix=col)
                            self.songs_df = pd.concat([self.songs_df, dummies], axis=1)
                            feature_cols.extend(dummies.columns)
                        except Exception as e:
                            logger.warning(f"为{col}列创建独热编码时出错: {str(e)}")

            # 2. 数值特征 - 标准化
            num_features = ['year', 'duration', 'tempo', 'energy', 'danceability', 'popularity']
            for col in num_features:
                if col in self.songs_df.columns and self.songs_df[col].notna().any():
                    try:
                        # 填充缺失值
                        self.songs_df[col] = self.songs_df[col].fillna(self.songs_df[col].median())

                        # 标准化
                        scaler = StandardScaler()
                        norm_col = f'{col}_norm'
                        self.songs_df[norm_col] = scaler.fit_transform(
                            self.songs_df[col].values.reshape(-1, 1)
                        )
                        feature_cols.append(norm_col)
                    except Exception as e:
                        logger.warning(f"处理数值特征{col}时出错: {str(e)}")

            # 3. 文本特征 - TF-IDF向量化
            if 'text_features' in self.songs_df.columns:
                try:
                    # 使用TF-IDF获取文本特征
                    tfidf = TfidfVectorizer(max_features=10, stop_words='english')
                    text_features = tfidf.fit_transform(self.songs_df['text_features'].fillna(''))

                    # 转换为DataFrame
                    text_feature_df = pd.DataFrame(
                        text_features.toarray(),
                        columns=[f'text_{i}' for i in range(text_features.shape[1])]
                    )

                    # 将文本特征添加到歌曲数据
                    self.songs_df = pd.concat([self.songs_df, text_feature_df], axis=1)
                    feature_cols.extend(text_feature_df.columns)
                except Exception as e:
                    logger.warning(f"处理文本特征时出错: {str(e)}")

            # 选择特征列
            if feature_cols:
                self.song_features = self.songs_df[feature_cols].values

                # 降维 (如果特征太多)
                if self.song_features.shape[1] > 20 and self.song_features.shape[0] > 20:
                    pca = PCA(n_components=min(20, self.song_features.shape[0], self.song_features.shape[1]))
                    self.song_features = pca.fit_transform(self.song_features)

                logger.info(f"创建了歌曲特征矩阵: {self.song_features.shape}")
            else:
                # 如果没有特征，则创建一个简单的特征矩阵(每首歌一个随机特征向量)
                logger.warning("没有找到有效的歌曲特征，创建随机特征")
                self.song_features = np.random.rand(len(self.songs_df), 10)

        except Exception as e:
            logger.error(f"准备歌曲特征时出错: {str(e)}")
            logger.exception("详细错误信息:")
            # 创建随机特征作为备选
            self.song_features = np.random.rand(len(self.songs_df), 10)

    def _prepare_user_features(self):
        """准备用户特征数据"""
        try:
            # 确保用户数据中有必要的字段
            feature_cols = []

            # 1. 类别特征 (gender, region) - 独热编码
            for col in ['gender', 'region', 'country']:
                if col in self.users_df.columns:
                    try:
                        # 处理缺失值
                        self.users_df[col] = self.users_df[col].fillna('未知')

                        # 独热编码
                        dummies = pd.get_dummies(self.users_df[col], prefix=col)
                        self.users_df = pd.concat([self.users_df, dummies], axis=1)
                        feature_cols.extend(dummies.columns)
                    except Exception as e:
                        logger.warning(f"为{col}列创建独热编码时出错: {str(e)}")

            # 2. 数值特征 (age, registration_time) - 标准化
            num_features = ['age', 'age_group', 'registration_time']
            for col in num_features:
                if col in self.users_df.columns:
                    # 对年龄组进行特殊处理
                    if col == 'age_group' and 'age_group' in self.users_df.columns:
                        try:
                            # 将年龄组映射到数值
                            age_map = {
                                '18-24': 1,
                                '25-34': 2,
                                '35-44': 3,
                                '45-54': 4,
                                '55+': 5
                            }
                            self.users_df['age_num'] = self.users_df['age_group'].map(age_map).fillna(3)

                            # 标准化
                            scaler = StandardScaler()
                            self.users_df['age_norm'] = scaler.fit_transform(
                                self.users_df['age_num'].values.reshape(-1, 1)
                            )
                            feature_cols.append('age_norm')
                        except Exception as e:
                            logger.warning(f"处理年龄组特征时出错: {str(e)}")

                    # 对数值类型数据标准化
                    elif self.users_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]:
                        try:
                            # 填充缺失值
                            self.users_df[col] = self.users_df[col].fillna(self.users_df[col].median())

                            # 标准化
                            scaler = StandardScaler()
                            norm_col = f'{col}_norm'
                            self.users_df[norm_col] = scaler.fit_transform(
                                self.users_df[col].values.reshape(-1, 1)
                            )
                            feature_cols.append(norm_col)
                        except Exception as e:
                            logger.warning(f"处理数值特征{col}时出错: {str(e)}")

            # 选择特征列
            if feature_cols:
                self.user_features = self.users_df[feature_cols].values

                # 降维 (如果特征太多)
                if self.user_features.shape[1] > 20 and self.user_features.shape[0] > 20:
                    pca = PCA(n_components=min(20, self.user_features.shape[0], self.user_features.shape[1]))
                    self.user_features = pca.fit_transform(self.user_features)

                logger.info(f"创建了用户特征矩阵: {self.user_features.shape}")
            else:
                # 如果没有特征，则创建一个简单的特征矩阵
                logger.warning("没有找到有效的用户特征，创建随机特征")
                self.user_features = np.random.rand(len(self.users_df), 5)

        except Exception as e:
            logger.error(f"准备用户特征时出错: {str(e)}")
            logger.exception("详细错误信息:")
            # 创建随机特征作为备选
            self.user_features = np.random.rand(len(self.users_df), 5)

    def pretrain_with_msd(self, msd_dir=None, spotify_data_path=None, chunk_limit=5, rating_style="log"):
        """
        使用MSD数据集预训练模型
        
        参数:
            msd_dir: MSD数据集目录，包含h5文件和triplets文件
            spotify_data_path: Spotify API数据文件路径（可选）
            chunk_limit: 处理的数据块数量限制
            rating_style: 评分转换方式，可选"log"、"linear"、"percentile"
        """
        if not self.use_msd:
            logger.info("未启用MSD预训练")
            return False

        logger.info("使用MSD数据集进行预训练...")

        try:
            # 1. 导入MSD处理模块
            try:
                from backend.data_processor.msd_processor import MSDDataProcessor
            except ImportError:
                logger.error("无法导入MSD数据处理器，请确保安装了所需依赖")
                return False
                
            # 2. 设置MSD目录和文件路径
            if msd_dir is None:
                msd_dir = os.path.join(self.data_dir, "msd")
            
            # 查找h5文件和triplets文件
            h5_file = None
            triplets_file = None
            
            if os.path.isdir(msd_dir):
                # 查找h5文件
                h5_files = [f for f in os.listdir(msd_dir) if f.endswith('.h5')]
                if h5_files:
                    h5_file = os.path.join(msd_dir, h5_files[0])
                
                # 查找triplets文件
                triplet_files = [f for f in os.listdir(msd_dir) if 'triplets' in f.lower() and f.endswith('.txt')]
                if triplet_files:
                    triplets_file = os.path.join(msd_dir, triplet_files[0])
            
            if h5_file is None or triplets_file is None:
                logger.error(f"未找到MSD数据文件（h5和triplets）在目录: {msd_dir}")
                return False
                
            logger.info(f"使用MSD文件: {h5_file} 和 {triplets_file}")
            
            # 3. 初始化MSD数据处理器
            data_processor = MSDDataProcessor(
                h5_path=h5_file,
                triplet_path=triplets_file
            )
            
            # 4. 加载MSD数据
            logger.info("加载MSD数据...")
            data_processor.load_data(chunk_limit=chunk_limit)
            
            # 5. 转换评分格式
            logger.info(f"转换评分数据，使用{rating_style}方法...")
            data_processor.convert_ratings(method=rating_style)
            
            # 6. 加载Spotify数据（如果可用）
            spotify_features = None
            if spotify_data_path is not None and os.path.exists(spotify_data_path):
                try:
                    logger.info(f"加载Spotify特征数据: {spotify_data_path}")
                    import pickle
                    with open(spotify_data_path, 'rb') as f:
                        spotify_features = pickle.load(f)
                    logger.info(f"加载了 {len(spotify_features)} 个Spotify歌曲特征")
                except Exception as e:
                    logger.warning(f"加载Spotify数据失败: {str(e)}")
            
            # 7. 提取用户-歌曲交互数据
            user_song_data = data_processor.get_user_song_data()
            # 提取歌曲特征
            song_features = data_processor.get_song_features()
            
            # 8. 从DeepRecommender导入
            from backend.models.deep_learning import DeepRecommender
            
            # 9. 初始化深度学习模型
            n_users = data_processor.get_user_count()
            n_items = data_processor.get_song_count()
            
            logger.info(f"创建深度学习模型: {n_users} 用户, {n_items} 项目")
            
            # 创建DeepRecommender模型
            self.dl_model = DeepRecommender(
                n_users=n_users,
                n_items=n_items,
                embedding_dim=128,
                item_features=song_features,
                use_mmoe=True,
                spotify_features=spotify_features
            )
            
            # 10. 处理MSD数据为训练格式
            user_indices, item_indices, ratings = self.dl_model.process_msd_data(
                user_song_data=user_song_data,
                song_features=song_features,
                spotify_data=spotify_features
            )
            
            # 11. 训练深度学习模型
            logger.info("训练深度学习模型...")
            self.dl_model.compile_model(learning_rate=0.001)
            history = self.dl_model.fit(
                user_indices=user_indices,
                item_indices=item_indices,
                ratings=ratings,
                epochs=20,
                batch_size=256,
                validation_split=0.1
            )
            
            # 12. 更新ID映射
            self.user_id_map = self.dl_model.user_map
            self.song_id_map = self.dl_model.item_map
            
            # 13. 获取歌曲和用户数据
            if self.songs_df is None or len(self.songs_df) == 0:
                # 创建歌曲数据框
                songs_data = []
                for song_id, idx in self.song_id_map.items():
                    song_info = data_processor.get_song_metadata(song_id)
                    if song_info:
                        songs_data.append(song_info)
                
                self.songs_df = pd.DataFrame(songs_data)
                logger.info(f"从MSD创建了歌曲数据框，包含 {len(self.songs_df)} 首歌曲")
            
            # 创建用户数据框（如果不存在）
            if self.users_df is None or len(self.users_df) == 0:
                user_ids = list(self.user_id_map.keys())
                self.users_df = pd.DataFrame({
                    'user_id': user_ids,
                    'name': [f"User_{uid[:8]}" for uid in user_ids]
                })
                logger.info(f"创建了用户数据框，包含 {len(self.users_df)} 个用户")
            
            # 14. 创建评分数据
            if self.ratings_df is None or len(self.ratings_df) == 0:
                self.ratings_df = user_song_data
                logger.info(f"导入了 {len(self.ratings_df)} 条评分记录")
            
            logger.info("MSD预训练完成")
            return True

        except Exception as e:
            logger.error(f"MSD预训练失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def train_collaborative_filtering(self):
        """训练协同过滤模型"""
        logger.info("训练协同过滤模型...")

        try:
            # 检查是否有足够的评分数据
            if self.ratings_df is None or len(self.ratings_df) == 0:
                logger.warning("没有评分数据，创建简单的协同过滤模型")
                self._create_simple_cf_model()
                return True

            # 尝试使用Surprise库
            try:
                # 准备Surprise数据格式
                from surprise import SVD, Dataset, Reader
                reader = Reader(rating_scale=(1, 5))
                data = Dataset.load_from_df(
                    self.ratings_df[['user_id', 'song_id', 'rating']],
                    reader
                )

                # 训练SVD模型
                self.cf_model = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02)
                trainset = data.build_full_trainset()
                self.cf_model.fit(trainset)

                logger.info("协同过滤模型训练完成")
                return True

            except ImportError:
                # 如果Surprise库不可用，使用简单的矩阵分解
                logger.warning("Surprise库不可用，使用简单的矩阵分解替代")
                self._create_simple_cf_model()
                return True

        except Exception as e:
            logger.error(f"训练协同过滤模型失败: {str(e)}")
            # 创建简单的备用模型
            self._create_simple_cf_model()
            return True

    def _create_simple_cf_model(self):
        """创建一个简单的协同过滤模型替代品"""
        logger.info("创建简单的协同过滤模型...")

        # 使用全局类定义，而不是嵌套类，以便可以pickle
        from backend.models.simple_cf_model import SimpleCF, Prediction

        # 计算用户平均评分
        user_means = {}
        if self.ratings_df is not None and len(self.ratings_df) > 0:
            for user_id, group in self.ratings_df.groupby('user_id'):
                user_means[user_id] = group['rating'].mean()

            # 计算全局平均评分
            global_mean = self.ratings_df['rating'].mean()
        else:
            global_mean = 3.0  # 默认平均评分

        # 创建模型
        self.cf_model = SimpleCF()
        self.cf_model.user_means = user_means
        self.cf_model.global_mean = global_mean

        logger.info("简单协同过滤模型创建完成")

    def train_content_based(self):
        """训练基于内容的推荐模型"""
        try:
            logger.info("训练内容特征模型...")

            # 检查歌曲数据
            if self.songs_df is None or len(self.songs_df) == 0:
                logger.error("没有可用的歌曲数据来训练内容特征模型")
                return False

            # 检查是否有text_features列，如果没有则创建
            if 'text_features' not in self.songs_df.columns:
                # 创建text_features列，综合歌曲的多个文本特征
                text_cols = ['title', 'artist_name', 'description', 'genre']
                available_cols = [col for col in text_cols if col in self.songs_df.columns]

                if not available_cols:
                    # 如果一个文本特征都没有，创建一个简单的描述
                    self.songs_df['text_features'] = "Song " + self.songs_df['song_id'].astype(str)
                else:
                    # 合并可用的文本特征
                    self.songs_df['text_features'] = self.songs_df[available_cols[0]].fillna('')
                    for col in available_cols[1:]:
                        self.songs_df['text_features'] += ' ' + self.songs_df[col].fillna('')

            # 确保文本特征非空
            self.songs_df['text_features'] = self.songs_df['text_features'].fillna('unknown')

            # 提取文本特征 - 为避免"no terms remain"错误，降低min_df
            try:
                tfidf = TfidfVectorizer(
                    analyzer='word',
                    ngram_range=(1, 2),
                    min_df=1,  # 降低min_df设置，避免过滤掉所有词语
                    max_df=0.95,
                    stop_words='english'
                )

                # 使用text_features列提取TF-IDF特征
                tfidf_matrix = tfidf.fit_transform(self.songs_df['text_features'])

                # 计算歌曲间的相似度矩阵
                self.content_similarity = cosine_similarity(tfidf_matrix)

                # 创建歌曲ID到矩阵索引的映射
                self.song_indices = pd.Series(self.songs_df.index, index=self.songs_df['song_id']).drop_duplicates()

                logger.info(f"内容特征模型训练完成，特征矩阵大小: {tfidf_matrix.shape}")
                return True

            except Exception as e:
                logger.error(f"训练内容特征模型失败: {str(e)}")

                # 使用备用方法 - 简单的元数据相似度
                try:
                    logger.info("尝试使用备用方法训练内容特征模型...")

                    # 使用数值特征来计算相似度
                    numeric_features = ['year', 'tempo', 'energy', 'danceability', 'valence', 'acousticness']
                    available_features = [f for f in numeric_features if f in self.songs_df.columns]

                    if len(available_features) > 0:
                        # 提取可用特征
                        features_df = self.songs_df[available_features].fillna(0)

                        # 标准化特征
                        scaler = StandardScaler()
                        scaled_features = scaler.fit_transform(features_df)

                        # 计算相似度矩阵
                        self.content_similarity = cosine_similarity(scaled_features)

                        # 创建歌曲ID到矩阵索引的映射
                        self.song_indices = pd.Series(self.songs_df.index, index=self.songs_df['song_id']).drop_duplicates()

                        logger.info(f"备用内容特征模型训练完成，使用 {len(available_features)} 个数值特征")
                        return True

                    else:
                        # 如果没有可用的数值特征，创建随机相似度
                        logger.warning("无可用特征，使用随机相似度矩阵")
                        random_features = np.random.rand(len(self.songs_df), 5)  # 5个随机特征
                        self.content_similarity = cosine_similarity(random_features)
                        self.song_indices = pd.Series(self.songs_df.index, index=self.songs_df['song_id']).drop_duplicates()
                        return True

                except Exception as e2:
                    logger.error(f"备用内容特征模型训练也失败: {str(e2)}")
                    return False

        except Exception as e:
            logger.error(f"训练内容特征模型出错: {str(e)}")
            return False

    def train_context_aware(self):
        """训练上下文感知推荐模型"""
        try:
            logger.info("训练上下文感知模型...")

            # 检查特征数据
            if self.song_features is None or self.user_features is None:
                logger.warning("没有足够的特征数据来训练上下文感知模型")
                return False

            # 检查评分数据
            if self.ratings_df is None or len(self.ratings_df) == 0:
                logger.warning("没有评分数据来训练上下文感知模型")
                return False

            # 创建训练数据 - 合并用户特征、歌曲特征和评分
            try:
                # 选择最多10000个样本来训练
                sample_size = min(10000, len(self.ratings_df))
                ratings_sample = self.ratings_df.sample(n=sample_size) if len(self.ratings_df) > sample_size else self.ratings_df

                # 确保用户ID和歌曲ID存在于特征数据中
                valid_users = set(self.users_df['user_id'])
                valid_songs = set(self.songs_df['song_id'])

                valid_ratings = ratings_sample[
                    ratings_sample['user_id'].isin(valid_users) &
                    ratings_sample['song_id'].isin(valid_songs)
                ]

                if len(valid_ratings) == 0:
                    logger.warning("没有有效的评分数据来训练上下文模型")
                    # 创建简单的备用模型
                    self._create_default_context_model()
                    return True

                # 创建合并特征 (用户特征 + 歌曲特征)
                X_train = []
                y_train = []

                for _, row in valid_ratings.iterrows():
                    user_id = row['user_id']
                    song_id = row['song_id']
                    rating = row['rating']

                    # 获取用户和歌曲特征
                    if user_id in self.user_features and song_id in self.song_features:
                        user_feat = self.user_features[user_id]
                        song_feat = self.song_features[song_id]

                        # 合并特征
                        combined_feat = np.concatenate([user_feat, song_feat])
                        X_train.append(combined_feat)

                        # 评分转换为喜好（高于平均值为1，否则为0）
                        y_train.append(1 if rating >= 3.0 else 0)

                # 转换为numpy数组
                X_train = np.array(X_train)
                y_train = np.array(y_train)

                if len(X_train) == 0:
                    logger.warning("没有足够的上下文数据进行训练")
                    # 创建简单的备用模型
                    from sklearn.dummy import DummyClassifier
                    self.context_model = DummyClassifier(strategy="most_frequent")
                    self.context_model.fit([[0,0]], [1])  # 使用最简单的数据拟合
                    return True

                # 训练模型
                from sklearn.linear_model import LogisticRegression
                self.context_model = LogisticRegression(C=1.0, solver='liblinear', max_iter=200)
                self.context_model.fit(X_train, y_train)

                # 评估模型
                train_accuracy = self.context_model.score(X_train, y_train)
                logger.info(f"上下文感知模型训练完成，训练准确率: {train_accuracy:.4f}")

                return True

            except Exception as e:
                logger.error(f"训练上下文感知模型失败: {str(e)}")

                # 创建简单的备用模型
                self._create_default_context_model()
                logger.info("已创建备用上下文感知模型")

                return True

        except Exception as e:
            logger.error(f"训练上下文感知模型出错: {str(e)}")
            return False

    def _create_default_context_model(self):
        """创建一个简单的默认上下文感知模型"""
        logger.info("创建默认上下文感知模型...")

        from sklearn.dummy import DummyClassifier

        # 创建一个简单的特征矩阵和标签
        X = np.array([[0, 0], [1, 1]])  # 2个样本，2个特征
        y = np.array([0, 1])  # 二元标签

        # 训练一个简单的模型
        self.context_model = DummyClassifier(strategy="most_frequent")
        self.context_model.fit(X, y)

        logger.info("默认上下文感知模型创建完成")

    def train_deep_learning_models(self):
        """训练深度学习模型"""
        logger.info("训练深度学习模型...")

        # 如果已经从MSD预训练了深度学习模型，则跳过
        if self.dl_model is not None:
            logger.info("深度学习模型已经存在，跳过训练")
            return True

        try:
            # 如果没有足够的数据，跳过深度学习模型训练
            if self.ratings_df is None or len(self.ratings_df) < 100:
                logger.warning("数据量不足，跳过深度学习模型训练")
                return False

            # 检查ID映射是否存在
            if self.user_id_map is None or self.song_id_map is None:
                logger.warning("ID映射未创建，跳过深度学习模型训练")
                return False

            # 准备训练数据
            n_users = len(self.user_id_map)
            n_songs = len(self.song_id_map)

            # 确保有足够的用户和歌曲
            if n_users < 5 or n_songs < 5:
                logger.warning(f"用户数({n_users})或歌曲数({n_songs})不足，跳过深度学习模型训练")
                return False

            # 映射ID到索引 - 确保所有ID都在映射中
            valid_user_mask = self.ratings_df['user_id'].isin(self.user_id_map)
            valid_song_mask = self.ratings_df['song_id'].isin(self.song_id_map)
            valid_ratings = self.ratings_df[valid_user_mask & valid_song_mask]

            if len(valid_ratings) < 50:
                logger.warning(f"有效评分数据不足({len(valid_ratings)}条)，跳过深度学习模型训练")
                return False

            # 导入DeepRecommender
            try:
                from backend.models.deep_learning import DeepRecommender
            except ImportError:
                logger.error("无法导入DeepRecommender模型，请确保相关文件存在")
                return False

            # 准备用户和物品索引映射
            user_map = {uid: idx for idx, uid in enumerate(set(valid_ratings['user_id']))}
            song_map = {sid: idx for idx, sid in enumerate(set(valid_ratings['song_id']))}
            reverse_user_map = {idx: uid for uid, idx in user_map.items()}
            reverse_song_map = {idx: sid for sid, idx in song_map.items()}

            # 准备特征数据
            item_features = None
            if hasattr(self, 'song_features') and self.song_features is not None:
                # 确保特征矩阵形状正确
                n_songs_features = len(song_map)
                feature_dim = next(iter(self.song_features.values())).shape[0] if isinstance(self.song_features, dict) else self.song_features.shape[1]
                
                # 创建特征矩阵
                item_features = np.zeros((n_songs_features, feature_dim))
                
                # 填充特征
                if isinstance(self.song_features, dict):
                    for song_id, idx in song_map.items():
                        if song_id in self.song_features:
                            item_features[idx] = self.song_features[song_id]
                else:
                    # 假设是DataFrame或matrix
                    for song_id, idx in song_map.items():
                        song_idx = self.songs_df[self.songs_df['song_id'] == song_id].index
                        if len(song_idx) > 0:
                            item_features[idx] = self.song_features[song_idx[0]]

            # 准备训练数据
            user_indices = np.array([user_map[uid] for uid in valid_ratings['user_id']])
            song_indices = np.array([song_map[sid] for sid in valid_ratings['song_id']])
            
            # 获取评分并标准化到0-1范围
            ratings = valid_ratings['rating'].values
            max_rating = ratings.max()
            if max_rating > 0:
                ratings = ratings / max_rating
            
            # 创建DeepRecommender实例
            self.dl_model = DeepRecommender(
                n_users=len(user_map),
                n_items=len(song_map),
                embedding_dim=100,  # 较小的嵌入维度
                item_features=item_features,
                use_mmoe=True  # 使用多目标混合专家模型
            )
            
            # 设置ID映射
            self.dl_model.user_map = user_map
            self.dl_model.item_map = song_map
            self.dl_model.reverse_user_map = reverse_user_map
            self.dl_model.reverse_item_map = reverse_song_map
            
            # 编译并训练模型
            self.dl_model.compile_model(learning_rate=0.001)
            
            # 训练模型
            history = self.dl_model.fit(
                user_indices=user_indices,
                item_indices=song_indices,
                ratings=ratings,
                epochs=15,            # 减少轮数
                batch_size=64,        # 较小的批次大小
                validation_split=0.1  # 使用10%的数据作为验证集
            )
            
            logger.info("深度学习模型训练完成")
            return True

        except Exception as e:
            logger.error(f"训练深度学习模型失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_cf_recommendations(self, user_id, top_n=10):
        """使用协同过滤模型获取推荐"""
        if self.cf_model is None:
            logger.warning("协同过滤模型未训练")
            return []

        try:
            # 获取用户未评分的歌曲
            user_rated_songs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['song_id'])
            unrated_songs = set(self.songs_df['song_id']) - user_rated_songs

            # 预测评分
            predictions = []
            for song_id in unrated_songs:
                try:
                    pred = self.cf_model.predict(user_id, song_id)
                    predictions.append((song_id, pred.est))
                except:
                    continue

            # 排序并返回前N个推荐
            predictions.sort(key=lambda x: x[1], reverse=True)
            return predictions[:top_n]

        except Exception as e:
            logger.error(f"协同过滤推荐失败: {str(e)}")
            return []

    def get_content_recommendations(self, song_ids, top_n=10):
        """基于内容特征获取相似歌曲推荐"""
        if self.content_similarity is None or self.song_indices is None:
            logger.warning("内容特征模型未训练")
            return []

        try:
            # 获取所有输入歌曲的相似歌曲
            all_similar = []
            for song_id in song_ids:
                if song_id in self.song_indices:
                    idx = self.song_indices[song_id]
                    song_scores = list(enumerate(self.content_similarity[idx]))
                    song_scores = sorted(song_scores, key=lambda x: x[1], reverse=True)
                    song_scores = song_scores[1:top_n+1]  # 排除自身

                    # 获取相似歌曲ID
                    similar_songs = [(self.songs_df.iloc[i]['song_id'], score)
                                    for i, score in song_scores]
                    all_similar.extend(similar_songs)

            # 合并并排序相似歌曲
            seen = set()
            unique_similar = []
            for song_id, score in all_similar:
                if song_id not in seen and song_id not in song_ids:
                    seen.add(song_id)
                    unique_similar.append((song_id, score))

            unique_similar.sort(key=lambda x: x[1], reverse=True)
            return unique_similar[:top_n]

        except Exception as e:
            logger.error(f"内容特征推荐失败: {str(e)}")
            return []

    def get_context_recommendations(self, user_id, context=None, top_n=10):
        """基于上下文获取推荐"""
        if self.context_model is None:
            logger.warning("上下文感知模型未训练")
            return []

        try:
            # 获取用户特征
            if user_id not in self.user_id_map:
                logger.warning(f"用户 {user_id} 不在训练数据中")
                return []

            user_idx = self.user_id_map[user_id]
            user_feat = self.user_features[user_idx].reshape(1, -1)

            # 获取用户未评分的歌曲
            user_rated_songs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['song_id'])
            unrated_songs = []

            # 处理上下文信息
            context_features = {}
            if context:
                # 提取情绪信息
                if 'emotion' in context:
                    emotion = context['emotion']
                    context_features['emotion'] = emotion

                # 提取活动信息
                if 'activity' in context:
                    activity = context['activity']
                    context_features['activity'] = activity

                # 提取时间信息
                if 'time_of_day' in context:
                    time_of_day = context['time_of_day']
                    context_features['time_of_day'] = time_of_day

                # 提取设备信息
                if 'device' in context:
                    device = context['device']
                    context_features['device'] = device

                logger.info(f"使用上下文特征: {context_features}")

            # 批量处理歌曲，避免内存问题
            batch_size = 1000
            all_predictions = []

            # 获取所有未评分的歌曲ID
            unrated_song_ids = [song_id for song_id in self.song_id_map if song_id not in user_rated_songs]

            # 分批处理
            for i in range(0, len(unrated_song_ids), batch_size):
                batch_song_ids = unrated_song_ids[i:i+batch_size]
                batch_predictions = []

                for song_id in batch_song_ids:
                    song_idx = self.song_id_map[song_id]
                    song_feat = self.song_features[song_idx].reshape(1, -1)

                    # 获取歌曲信息
                    song_info = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0]

                    # 计算上下文匹配分数
                    context_score = 1.0  # 默认分数

                    if context:
                        # 情绪匹配
                        if 'emotion' in context_features and 'mood' in song_info:
                            song_mood = song_info['mood']
                            user_emotion = context_features['emotion']

                            # 情绪匹配度计算
                            if user_emotion == 'happy' and song_mood in ['happy', 'excited']:
                                context_score *= 1.5
                            elif user_emotion == 'sad' and song_mood in ['sad', 'relaxed']:
                                context_score *= 1.5
                            elif user_emotion == 'relaxed' and song_mood in ['relaxed', 'neutral']:
                                context_score *= 1.5
                            elif user_emotion == 'excited' and song_mood in ['excited', 'energetic']:
                                context_score *= 1.5
                            else:
                                context_score *= 0.7

                        # 活动匹配
                        if 'activity' in context_features and 'activity_suitability' in song_info:
                            song_activity = song_info['activity_suitability']
                            user_activity = context_features['activity']

                            # 活动匹配度计算
                            if user_activity == song_activity:
                                context_score *= 1.5
                            elif (user_activity == 'studying' and song_activity in ['relaxing', 'studying']) or \
                                 (user_activity == 'exercising' and song_activity in ['exercising', 'socializing']) or \
                                 (user_activity == 'relaxing' and song_activity in ['relaxing', 'studying']):
                                context_score *= 1.2
                            else:
                                context_score *= 0.8

                        # 时间匹配
                        if 'time_of_day' in context_features and 'time_suitability' in song_info:
                            song_time = song_info['time_suitability']
                            user_time = context_features['time_of_day']

                            # 时间匹配度计算
                            if user_time == song_time or song_time == 'anytime':
                                context_score *= 1.3
                            else:
                                context_score *= 0.9

                    # 合并用户和歌曲特征
                    X = np.hstack([user_feat, song_feat])

                    # 使用模型预测基础概率
                    try:
                        # 检查模型类型
                        if hasattr(self.context_model, 'predict_proba'):
                            proba = self.context_model.predict_proba(X)
                            # 检查概率数组的形状
                            if proba.shape[1] > 1:
                                base_prob = proba[0, 1]  # 正类的概率
                            else:
                                # 只有一个类别的情况
                                base_prob = proba[0, 0]
                        else:
                            # 如果模型不支持概率预测，使用二元预测
                            pred = self.context_model.predict(X)
                            base_prob = float(pred[0])
                    except Exception as e:
                        logger.debug(f"预测概率时出错: {str(e)}")
                        base_prob = 0.5  # 默认概率

                    # 结合上下文分数
                    final_prob = base_prob * context_score

                    # 添加到预测列表
                    batch_predictions.append((song_id, final_prob))

                all_predictions.extend(batch_predictions)

                # 释放内存
                del batch_predictions
                import gc
                gc.collect()

            # 排序并返回前N个推荐
            all_predictions.sort(key=lambda x: x[1], reverse=True)
            top_predictions = all_predictions[:top_n]

            # 为每个推荐添加详细信息
            detailed_recommendations = []
            for song_id, score in top_predictions:
                song_info = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0].to_dict()
                song_info['score'] = score

                # 添加解释
                explanation = self._generate_context_explanation(song_info, context_features)
                song_info['explanation'] = explanation

                detailed_recommendations.append(song_info)

            return detailed_recommendations

        except Exception as e:
            logger.error(f"上下文感知推荐失败: {str(e)}")
            logger.exception("详细错误信息:")
            return []

    def _generate_context_explanation(self, song_info, context_features):
        """生成基于上下文的推荐解释"""
        explanations = []

        # 基于情绪的解释
        if 'emotion' in context_features and 'mood' in song_info:
            user_emotion = context_features['emotion']
            song_mood = song_info.get('mood', '')

            if user_emotion == 'happy' and song_mood in ['happy', 'excited']:
                explanations.append("这首欢快的歌曲适合你当前的愉悦心情")
            elif user_emotion == 'sad' and song_mood in ['sad', 'relaxed']:
                explanations.append("这首情感丰富的歌曲与你当前的心情相符")
            elif user_emotion == 'relaxed' and song_mood in ['relaxed', 'neutral']:
                explanations.append("这首平静的歌曲适合你当前放松的状态")
            elif user_emotion == 'excited' and song_mood in ['excited', 'energetic']:
                explanations.append("这首充满活力的歌曲适合你当前兴奋的心情")

        # 基于活动的解释
        if 'activity' in context_features:
            activity = context_features['activity']

            if activity == 'studying':
                explanations.append("这首平静的歌曲适合学习时听")
            elif activity == 'exercising':
                explanations.append("这首充满活力的歌曲适合运动时听")
            elif activity == 'relaxing':
                explanations.append("这首舒缓的歌曲适合放松时听")
            elif activity == 'socializing':
                explanations.append("这首歌曲适合社交场合")
            elif activity == 'commuting':
                explanations.append("这首歌曲适合通勤时听")

        # 基于时间的解释
        if 'time_of_day' in context_features:
            time_of_day = context_features['time_of_day']

            if time_of_day == 'morning':
                explanations.append("这首歌曲适合早晨听")
            elif time_of_day == 'afternoon':
                explanations.append("这首歌曲适合下午听")
            elif time_of_day == 'evening':
                explanations.append("这首歌曲适合晚上听")
            elif time_of_day == 'night':
                explanations.append("这首歌曲适合夜晚听")

        # 如果没有解释，添加默认解释
        if not explanations:
            explanations.append("这首歌曲符合你当前的上下文需求")

        return " ".join(explanations)

    def get_dl_recommendations(self, user_id, top_n=10):
        """使用深度学习模型获取推荐"""
        if self.dl_model is None:
            logger.warning("深度学习模型未训练")
            return []

        try:
            # 检查模型类型 - 如果是DeepRecommender类实例
            if hasattr(self.dl_model, 'predict') and hasattr(self.dl_model, 'user_map'):
                # 使用DeepRecommender类的预测方法
                
                # 检查用户是否在映射中
                if user_id not in self.dl_model.user_map:
                    logger.warning(f"用户 {user_id} 不在深度学习模型的用户映射中")
                    return []
                
                # 获取用户已评分的歌曲
                if self.ratings_df is not None:
                    user_rated_songs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['song_id'])
                else:
                    user_rated_songs = set()
                
                # 获取所有候选物品ID
                candidate_songs = [song_id for song_id in self.dl_model.item_map.keys() 
                                  if song_id not in user_rated_songs]
                
                # 如果候选歌曲过多，随机采样以提高效率
                if len(candidate_songs) > 1000:
                    import random
                    candidate_songs = random.sample(candidate_songs, 1000)
                
                # 使用DeepRecommender的predict方法进行预测
                predictions = self.dl_model.predict(user_id, candidate_songs)
                
                # 转换为列表格式
                if isinstance(predictions, dict):
                    pred_list = [(song_id, score) for song_id, score in predictions.items()]
                else:
                    # 如果返回的不是字典，尝试其他格式转换
                    pred_list = []
                    for i, song_id in enumerate(candidate_songs):
                        try:
                            # 尝试从predictions中获取分数
                            if isinstance(predictions, (list, np.ndarray)):
                                if len(predictions) > i:
                                    score = float(predictions[i])
                                    pred_list.append((song_id, score))
                            else:
                                # 如果无法处理predictions的格式，跳过
                                continue
                        except:
                            continue
                
                # 排序并返回前N个推荐
                pred_list.sort(key=lambda x: x[1], reverse=True)
                return pred_list[:top_n]
                
            else:
                # 传统方法 - 假设dl_model是Keras模型
                # 检查用户是否在训练数据中
                if user_id not in self.user_id_map:
                    logger.warning(f"用户 {user_id} 不在训练数据中")
                    return []

                user_idx = self.user_id_map[user_id]

                # 获取用户未评分的歌曲
                user_rated_songs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['song_id'])
                unrated_songs = []
                for song_id in self.song_id_map:
                    if song_id not in user_rated_songs:
                        song_idx = self.song_id_map[song_id]
                        unrated_songs.append((song_id, song_idx))

                # 批量预测
                user_indices = np.array([user_idx] * len(unrated_songs))
                song_indices = np.array([idx for _, idx in unrated_songs])

                # 安全检查 - 确保模型有predict方法
                if not hasattr(self.dl_model, 'predict'):
                    logger.error("深度学习模型没有predict方法")
                    return []

                predicted_ratings = self.dl_model.predict(
                    [user_indices, song_indices],
                    batch_size=128,
                    verbose=0
                ).flatten()

                # 将预测结果与歌曲ID配对并排序
                song_ids = [sid for sid, _ in unrated_songs]
                predictions = list(zip(song_ids, predicted_ratings))
                predictions.sort(key=lambda x: x[1], reverse=True)

                return predictions[:top_n]

        except Exception as e:
            logger.error(f"深度学习推荐失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_hybrid_recommendations(self, user_id, context=None, liked_songs=None, top_n=10):
        """获取混合推荐结果"""
        logger.info(f"为用户 {user_id} 生成混合推荐...")

        # 初始化权重和结果
        weights = self.weights
        all_recommendations = {}
        final_scores = {}

        # 1. 协同过滤推荐
        if self.cf_model:
            cf_recs = self.get_cf_recommendations(user_id, top_n=top_n*2)
            all_recommendations['collaborative'] = cf_recs

            # 添加到总分
            for song_id, score in cf_recs:
                if song_id not in final_scores:
                    final_scores[song_id] = 0
                final_scores[song_id] += score * weights['collaborative']

        # 2. 内容特征推荐
        if self.content_similarity is not None and liked_songs:
            content_recs = self.get_content_recommendations(liked_songs, top_n=top_n*2)
            all_recommendations['content'] = content_recs

            # 添加到总分
            for song_id, score in content_recs:
                if song_id not in final_scores:
                    final_scores[song_id] = 0
                final_scores[song_id] += score * weights['content']

        # 3. 上下文感知推荐
        if self.context_model:
            context_recs = self.get_context_recommendations(user_id, context, top_n=top_n*2)
            all_recommendations['context'] = context_recs

            # 添加到总分
            # 检查返回的推荐格式
            if context_recs and isinstance(context_recs[0], dict):
                # 新格式：字典列表
                for rec in context_recs:
                    song_id = rec.get('song_id')
                    score = rec.get('score', 0.0)
                    if song_id not in final_scores:
                        final_scores[song_id] = 0
                    final_scores[song_id] += score * weights['context']
            else:
                # 旧格式：(song_id, score) 元组列表
                for song_id, score in context_recs:
                    if song_id not in final_scores:
                        final_scores[song_id] = 0
                    final_scores[song_id] += score * weights['context']

        # 4. 深度学习推荐
        if self.dl_model:
            dl_recs = self.get_dl_recommendations(user_id, top_n=top_n*2)
            all_recommendations['deep_learning'] = dl_recs

            # 添加到总分
            for song_id, score in dl_recs:
                if song_id not in final_scores:
                    final_scores[song_id] = 0
                final_scores[song_id] += score * weights['deep_learning']

        # 排序并返回前N个混合推荐
        sorted_recs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)

        # 获取前N个推荐的详细信息
        top_recs = []
        for song_id, score in sorted_recs[:top_n]:
            song_info = self.songs_df[self.songs_df['song_id'] == song_id].iloc[0].to_dict()
            song_info['score'] = score
            top_recs.append(song_info)

        logger.info(f"生成了 {len(top_recs)} 条混合推荐")
        return top_recs

    def adjust_weights(self, new_weights):
        """调整各算法的权重"""
        # 验证权重合法性
        if not all(0 <= w <= 1 for w in new_weights.values()):
            logger.error("权重必须在0到1之间")
            return False

        # 归一化权重
        total = sum(new_weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in new_weights.items()}
            logger.info(f"已调整算法权重: {self.weights}")
            return True
        else:
            logger.error("权重总和必须大于0")
            return False

    def save_model(self, model_path):
        """保存模型到文件"""
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f"模型已保存到: {model_path}")
            return True
        except Exception as e:
            logger.error(f"保存模型失败: {str(e)}")
            return False

    @classmethod
    def load_model(cls, model_path):
        """从文件加载模型"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info(f"从文件加载模型: {model_path}")
            return model
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            return None

    def train(self):
        """
        根据配置训练混合推荐模型

        根据初始化时设置的参数，选择性地训练不同的推荐模型组件
        """
        logger.info("开始训练混合推荐模型...")

        # 确保数据已加载和预处理
        if self.ratings_df is None or self.songs_df is None or self.users_df is None:
            logger.error("训练失败：数据未加载")
            return False

        # 预处理数据
        if not self.preprocess_data():
            logger.error("训练失败：数据预处理出错")
            return False

        try:
            # 如果使用MSD数据集，预训练模型
            if self.use_msd:
                self.pretrain_with_msd()

            # 根据配置训练对应的推荐模型
            success = True

            # 训练协同过滤模型
            if self.use_cf:
                logger.info("训练协同过滤模型...")
                if not self.train_collaborative_filtering():
                    logger.warning("协同过滤模型训练失败")
                    success = False

            # 训练基于内容的推荐模型
            if self.use_content:
                logger.info("训练基于内容的推荐模型...")
                if not self.train_content_based():
                    logger.warning("基于内容的推荐模型训练失败")
                    success = False

            # 训练上下文感知推荐模型
            if self.use_context:
                logger.info("训练上下文感知推荐模型...")
                if not self.train_context_aware():
                    logger.warning("上下文感知推荐模型训练失败")
                    success = False

            # 训练深度学习推荐模型
            if self.use_deep_learning:
                logger.info("训练深度学习推荐模型...")
                if not self.train_deep_learning_models():
                    logger.warning("深度学习推荐模型训练失败")
                    success = False

            if success:
                logger.info("混合推荐模型训练完成")
            else:
                logger.warning("部分推荐模型训练失败")

            return success

        except Exception as e:
            logger.error(f"训练混合推荐模型时出错: {str(e)}")
            logger.exception("详细错误信息:")
            return False

    def add_new_rating(self, user_id, song_id, rating, timestamp=None, context=None):
        """
        添加或更新单条评分数据，支持实时调整

        参数:
            user_id: 用户ID
            song_id: 歌曲ID
            rating: 评分 (1-5)
            timestamp: 时间戳 (可选)
            context: 上下文信息 (可选)，如 {'emotion': 'happy', 'activity': 'exercising'}

        返回:
            添加成功返回True，否则返回False
        """
        try:
            # 检查用户和歌曲ID是否有效
            if not self._validate_ids(user_id, song_id):
                logger.warning(f"添加评分失败：无效的用户ID {user_id} 或歌曲ID {song_id}")
                return False

            # 设置默认时间戳
            if timestamp is None:
                timestamp = int(time.time())

            # 创建新评分数据
            new_rating = {
                'user_id': user_id,
                'song_id': song_id,
                'rating': float(rating),
                'timestamp': timestamp
            }

            # 添加上下文信息（如果有）
            if context:
                for key, value in context.items():
                    new_rating[key] = value
                logger.info(f"添加上下文信息: {context}")

            # 检查评分数据是否已存在
            existing_idx = -1
            for i, r in enumerate(self.ratings_df.values):
                if self.ratings_df.iloc[i]['user_id'] == user_id and self.ratings_df.iloc[i]['song_id'] == song_id:
                    existing_idx = i
                    break

            # 更新或添加评分
            if existing_idx >= 0:
                # 对于已存在的评分，我们需要保留原有的列并更新值
                for key, value in new_rating.items():
                    if key in self.ratings_df.columns:
                        self.ratings_df.at[existing_idx, key] = value
                logger.info(f"更新评分：用户 {user_id} 对歌曲 {song_id} 的评分更新为 {rating}")
            else:
                # 确保新评分的列与现有DataFrame匹配
                new_rating_df = pd.DataFrame([new_rating])
                # 添加缺失的列
                for col in self.ratings_df.columns:
                    if col not in new_rating_df.columns:
                        new_rating_df[col] = None
                # 只保留ratings_df中存在的列
                new_rating_df = new_rating_df[[col for col in new_rating_df.columns if col in self.ratings_df.columns]]

                self.ratings_df = pd.concat([self.ratings_df, new_rating_df], ignore_index=True)
                logger.info(f"添加评分：用户 {user_id} 对歌曲 {song_id} 的评分为 {rating}")

            # 如果协同过滤模型已经存在，更新模型
            if self.cf_model:
                self._update_cf_model_for_rating(user_id, song_id, rating)

            # 如果有上下文信息，可以更新上下文感知模型
            if context and hasattr(self, 'train_context_aware'):
                logger.info(f"使用上下文信息更新上下文感知模型")
                self.train_context_aware()

            return True

        except Exception as e:
            logger.error(f"添加评分时出错: {str(e)}")
            logger.exception("详细错误信息:")
            return False

    def _validate_ids(self, user_id, song_id):
        """验证用户ID和歌曲ID是否有效"""
        # 检查用户ID
        if self.users_df is not None and not self.users_df['user_id'].isin([user_id]).any():
            # 用户不存在，自动创建
            new_user = self._create_new_user(user_id)
            if new_user is not None:
                self.users_df = pd.concat([self.users_df, pd.DataFrame([new_user])], ignore_index=True)
                logger.info(f"创建新用户：{user_id}")
            else:
                return False

        # 检查歌曲ID
        if self.songs_df is not None and not self.songs_df['song_id'].isin([song_id]).any():
            logger.warning(f"歌曲 {song_id} 不存在于数据库中")
            return False

        return True

    def _create_new_user(self, user_id):
        """创建新用户
        
        参数:
            user_id: 用户ID
            
        返回:
            创建成功返回True，否则返回False
        """
        try:
            if user_id in self.user_id_map:
                return True
                
            # 初始化空的user_id_map字典如果不存在
            if self.user_id_map is None:
                self.user_id_map = {}
                
            # 添加用户到映射
            next_id = len(self.user_id_map)
            self.user_id_map[user_id] = next_id
            
            # 在用户特征中添加新用户（如果已初始化）
            if self.user_features is not None:
                # 创建新用户特征向量（使用平均值或零向量）
                if len(self.user_features) > 0:
                    avg_features = np.mean(self.user_features, axis=0)
                    new_user_features = avg_features.reshape(1, -1)
                else:
                    # 如果没有现有用户，使用零向量
                    feature_dim = 10  # 默认特征维度
                    new_user_features = np.zeros((1, feature_dim))
                
                # 添加到用户特征矩阵
                if isinstance(self.user_features, np.ndarray):
                    self.user_features = np.vstack([self.user_features, new_user_features])
                else:
                    self.user_features = new_user_features
            
            logger.info(f"创建了新用户: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"创建新用户失败: {str(e)}")
            return False

    def _update_cf_model_for_rating(self, user_id, song_id, rating):
        """为单个新评分更新协同过滤模型"""
        try:
            # 检查是否使用surprise库的模型
            if hasattr(self.cf_model, 'partial_fit'):
                # 模型支持增量学习
                from surprise import Trainset

                # 获取用户和歌曲的内部ID
                raw_uid = user_id
                raw_iid = song_id

                # 使用已有trainset的映射获取内部ID
                if hasattr(self.cf_model, 'trainset'):
                    trainset = self.cf_model.trainset
                    try:
                        uid = trainset.to_inner_uid(raw_uid)
                        iid = trainset.to_inner_iid(raw_iid)
                        self.cf_model.partial_fit(uid, iid, float(rating))
                        logger.info(f"协同过滤模型已针对用户 {user_id} 和歌曲 {song_id} 进行增量更新")
                        return True
                    except ValueError:
                        # ID不在训练集中，需要重新训练
                        pass

            # 如果无法增量更新，记录待更新状态，稍后批量更新
            if not hasattr(self, '_pending_cf_updates'):
                self._pending_cf_updates = []

            self._pending_cf_updates.append((user_id, song_id, float(rating)))

            # 如果累积了足够多的更新，执行批量重训练
            if len(self._pending_cf_updates) >= 10:
                self._batch_update_cf_model()

            return True

        except Exception as e:
            logger.error(f"更新协同过滤模型时出错: {str(e)}")
            return False

    def _batch_update_cf_model(self):
        """批量更新协同过滤模型"""
        try:
            if not hasattr(self, '_pending_cf_updates') or not self._pending_cf_updates:
                return False

            logger.info(f"执行协同过滤模型批量更新，处理 {len(self._pending_cf_updates)} 条评分")

            # 重新训练协同过滤模型
            self.train_collaborative_filtering()

            # 清空待更新列表
            self._pending_cf_updates = []

            return True

        except Exception as e:
            logger.error(f"批量更新协同过滤模型时出错: {str(e)}")
            return False

    def update_cf_model(self, ratings):
        """
        使用新评分批量更新协同过滤模型

        参数:
            ratings: 评分列表，每项为 (user_id, song_id, rating) 元组
        """
        try:
            # 将新评分添加到数据集
            added = 0
            for user_id, song_id, rating in ratings:
                if self.add_new_rating(user_id, song_id, rating):
                    added += 1

            # 如果添加了新评分，重新训练模型
            if added > 0:
                self.train_collaborative_filtering()
                logger.info(f"协同过滤模型已使用 {added} 条新评分更新")

            return added > 0

        except Exception as e:
            logger.error(f"更新协同过滤模型时出错: {str(e)}")
            return False

    def update_content_model(self, user_id, preferences=None):
        """
        更新基于内容的推荐模型

        参数:
            user_id: 用户ID
            preferences: 用户偏好字典，格式如 {'genres': {'pop': 0.8}, 'moods': {'happy': 0.6}}
        """
        try:
            # 检查是否有更新内容模型的必要
            if not self.use_content or self.content_similarity is None:
                return False

            # 如果提供了偏好，更新用户特征
            if preferences and self.user_features is not None:
                # 获取用户的内部ID
                if user_id in self.user_id_map:
                    internal_id = self.user_id_map[user_id]

                    # 更新用户特征向量
                    if 'genres' in preferences:
                        # 提取流派偏好
                        for genre, weight in preferences['genres'].items():
                            genre_col = f"genre_{genre}"
                            if genre_col in self.users_df.columns:
                                col_idx = self.users_df.columns.get_loc(genre_col)
                                if internal_id < len(self.user_features) and col_idx < self.user_features.shape[1]:
                                    self.user_features[internal_id, col_idx] = weight

                    # 更新其他偏好特征
                    # ...

                    logger.info(f"用户 {user_id} 的内容特征已更新")

            return True

        except Exception as e:
            logger.error(f"更新内容模型时出错: {str(e)}")
            return False

    def normalize_user_data(self, user_id):
        """
        标准化用户数据，处理不完整的用户特征

        参数:
            user_id: 用户ID

        返回:
            成功返回True，否则返回False
        """
        try:
            # 检查用户是否存在
            if self.users_df is None or not self.users_df['user_id'].isin([user_id]).any():
                new_user = self._create_new_user(user_id)
                if new_user is not None:
                    self.users_df = pd.concat([self.users_df, pd.DataFrame([new_user])], ignore_index=True)
                    logger.info(f"标准化过程中创建新用户：{user_id}")
                else:
                    return False

            # 获取用户数据
            user_data = self.users_df[self.users_df['user_id'] == user_id].iloc[0]

            # 处理缺失值
            for col in self.users_df.columns:
                if pd.isna(user_data[col]) and col != 'user_id':
                    # 根据列类型设置适当的默认值
                    dtype = self.users_df[col].dtype
                    if pd.api.types.is_numeric_dtype(dtype):
                        # 对于数值列，使用该列的平均值或0
                        mean_val = self.users_df[col].mean()
                        self.users_df.loc[self.users_df['user_id'] == user_id, col] = mean_val if not pd.isna(mean_val) else 0
                    else:
                        # 对于分类列，使用最常见值或空字符串
                        most_common = self.users_df[col].mode().iloc[0] if len(self.users_df[col].mode()) > 0 else ''
                        self.users_df.loc[self.users_df['user_id'] == user_id, col] = most_common if not pd.isna(most_common) else ''

            # 更新用户特征
            self._prepare_user_features()

            logger.info(f"用户 {user_id} 的数据已标准化")
            return True

        except Exception as e:
            logger.error(f"标准化用户数据时出错: {str(e)}")
            return False

    def get_algorithm_weights(self, user_id=None):
        """
        获取当前算法权重

        参数:
            user_id: 用户ID (可选)

        返回:
            权重字典
        """
        # 如果实现了用户个性化的权重系统，可以返回特定用户的权重
        if hasattr(self, 'algorithm_weights') and isinstance(self.algorithm_weights, dict):
            if user_id is not None and user_id in self.algorithm_weights:
                return self.algorithm_weights[user_id]

        # 否则返回全局权重
        return self.weights

    def adjust_weights(self, new_weights, user_id=None):
        """
        调整算法权重

        参数:
            new_weights: 新权重字典
            user_id: 用户ID (可选，用于个性化权重)

        返回:
            调整成功返回True，否则返回False
        """
        # 验证权重合法性
        if not all(0 <= w <= 1 for w in new_weights.values()):
            logger.error("权重必须在0到1之间")
            return False

        # 归一化权重
        total = sum(new_weights.values())
        if total > 0:
            normalized_weights = {k: v/total for k, v in new_weights.items()}

            # 如果提供了用户ID，设置用户个性化权重
            if user_id is not None:
                if not hasattr(self, 'algorithm_weights'):
                    self.algorithm_weights = {}
                self.algorithm_weights[user_id] = normalized_weights
                logger.info(f"已为用户 {user_id} 调整算法权重: {normalized_weights}")
            else:
                # 否则设置全局权重
                self.weights = normalized_weights
                logger.info(f"已调整全局算法权重: {normalized_weights}")

            return True
        else:
            logger.error("权重总和必须大于0")
            return False

    def find_songs_by_artist(self, artist_name, limit=10):
        """
        查找艺术家的歌曲

        参数:
            artist_name: 艺术家名称
            limit: 最大返回数量

        返回:
            歌曲ID列表
        """
        try:
            if self.songs_df is None:
                return []

            # 查找匹配的艺术家
            matches = self.songs_df[
                self.songs_df['artist_name'].str.contains(artist_name, case=False, na=False)
            ]

            if len(matches) == 0:
                return []

            # 返回歌曲ID
            return matches['song_id'].tolist()[:limit]

        except Exception as e:
            logger.error(f"查找艺术家歌曲时出错: {str(e)}")
            return []

    def recommend(self, user_id, context=None, top_n=10, num_recommendations=None, include_rated=False):
        """
        为用户生成推荐，自动处理不完整数据

        参数:
            user_id: 用户ID
            context: 上下文信息 (可选)
            top_n: 推荐数量 (已废弃，保留向后兼容)
            num_recommendations: 推荐数量 (新参数)
            include_rated: 是否包含已评分的歌曲

        返回:
            推荐列表
        """
        # 使用num_recommendations参数(如果提供)，否则使用top_n
        n = num_recommendations if num_recommendations is not None else top_n
        
        # 标准化用户数据
        self.normalize_user_data(user_id)

        # 获取用户喜欢的歌曲
        liked_songs = self._get_user_liked_songs(user_id)

        # 获取推荐
        recommendations = self.get_hybrid_recommendations(
            user_id=user_id,
            context=context,
            liked_songs=liked_songs,
            top_n=n
        )

        # 如果需要排除已评分歌曲
        if not include_rated and self.ratings_df is not None:
            # 获取用户已评分的歌曲
            rated_songs = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['song_id'].tolist())
            # 过滤掉已评分的歌曲
            recommendations = [rec for rec in recommendations if rec['song_id'] not in rated_songs]
            
            # 如果过滤后推荐数量不足，可以获取更多推荐
            if len(recommendations) < n:
                more_recs = self.get_hybrid_recommendations(
                    user_id=user_id,
                    context=context,
                    liked_songs=liked_songs,
                    top_n=n*2  # 获取更多，以确保过滤后数量足够
                )
                # 添加未出现在当前推荐中的歌曲
                current_ids = set(rec['song_id'] for rec in recommendations)
                for rec in more_recs:
                    if rec['song_id'] not in current_ids and rec['song_id'] not in rated_songs:
                        recommendations.append(rec)
                        current_ids.add(rec['song_id'])
                        if len(recommendations) >= n:
                            break

        # 为每个推荐添加预测评分和一个简单的解释
        for rec in recommendations:
            # 预测评分
            pred_rating = self._predict_rating(user_id, rec['song_id'])
            rec['predicted_score'] = pred_rating

            # 生成简单的推荐解释
            rec['explanation'] = self._generate_recommendation_explanation(user_id, rec, context)

        return recommendations[:n]

    def _get_user_liked_songs(self, user_id, rating_threshold=4.0):
        """获取用户喜欢的歌曲ID列表"""
        try:
            if self.ratings_df is None:
                return []

            # 查找用户的高评分歌曲
            user_ratings = self.ratings_df[
                (self.ratings_df['user_id'] == user_id) &
                (self.ratings_df['rating'] >= rating_threshold)
            ]

            if len(user_ratings) == 0:
                return []

            return user_ratings['song_id'].tolist()

        except Exception as e:
            logger.error(f"获取用户喜欢的歌曲时出错: {str(e)}")
            return []

    def _predict_rating(self, user_id, song_id):
        """预测用户对歌曲的评分"""
        try:
            # 使用混合模型预测评分
            # 这里实现一个简单版本，实际应用中可能更复杂

            # 协同过滤评分
            cf_score = 0.0
            if self.cf_model:
                try:
                    cf_score = self.cf_model.predict(user_id, song_id).est
                except:
                    cf_score = 0.0

            # 内容相似度评分
            content_score = 0.0
            liked_songs = self._get_user_liked_songs(user_id)
            if self.content_similarity is not None and liked_songs:
                content_recs = self.get_content_recommendations(liked_songs, top_n=1000)
                for rec_id, score in content_recs:
                    if rec_id == song_id:
                        content_score = score
                        break

            # 使用全局或用户特定的权重
            weights = self.get_algorithm_weights(user_id)

            # 计算加权评分
            weighted_score = (
                cf_score * weights.get('collaborative', 0.4) +
                content_score * weights.get('content', 0.3)
            ) / (weights.get('collaborative', 0.4) + weights.get('content', 0.3))

            # 确保评分在0-1范围内
            return min(max(weighted_score, 0.0), 1.0)

        except Exception as e:
            logger.error(f"预测评分时出错: {str(e)}")
            return 0.5  # 默认中等评分

    def _generate_recommendation_explanation(self, user_id, recommendation, context=None):
        """生成推荐解释"""
        song_id = recommendation['song_id']
        artist_name = recommendation.get('artist_name', '未知艺术家')
        title = recommendation.get('title', '未知歌曲')

        explanations = []

        # 基于协同过滤的解释
        if self.ratings_df is not None:
            # 检查用户是否对同一艺术家的其他歌曲有高评分
            artist_ratings = self.ratings_df[
                (self.ratings_df['user_id'] == user_id) &
                (self.ratings_df['song_id'] != song_id)
            ]

            if len(artist_ratings) > 0:
                for _, row in artist_ratings.iterrows():
                    rated_song_id = row['song_id']
                    if rated_song_id in self.songs_df['song_id'].values:
                        rated_song = self.songs_df[self.songs_df['song_id'] == rated_song_id].iloc[0]
                        if rated_song['artist_name'] == artist_name and row['rating'] >= 4.0:
                            explanations.append(f"你喜欢{artist_name}的其他歌曲")
                            break

        # 基于内容的解释
        if 'genre' in recommendation:
            genre = recommendation['genre']
            explanations.append(f"这是一首{genre}歌曲")

        # 基于上下文的解释
        if context:
            if 'mood' in context and recommendation.get('valence') is not None:
                mood = context['mood']
                if isinstance(mood, dict) and 'primary_emotion' in mood:
                    mood_name = mood['primary_emotion']
                    # 情绪与价值的匹配
                    if mood_name == 'happy' and recommendation.get('valence', 0) > 0.7:
                        explanations.append("这首欢快的歌曲适合你当前的心情")
                    elif mood_name == 'sad' and recommendation.get('valence', 0) < 0.3:
                        explanations.append("这首情感丰富的歌曲与你当前的心情相符")

            if 'activity' in context:
                activity = context['activity']
                if activity == 'studying' and recommendation.get('energy', 0) < 0.5:
                    explanations.append("这首平静的歌曲适合学习时听")
                elif activity == 'exercising' and recommendation.get('energy', 0) > 0.7:
                    explanations.append("这首充满活力的歌曲适合运动时听")

        # 如果没有生成解释，添加一个默认解释
        if not explanations:
            explanations.append(f"这首歌的风格可能符合你的口味")

        return " ".join(explanations)

    def update_user_emotion_vector(self, user_id, emotion, emotion_description=None):
        """
        根据用户当前情绪更新用户情绪向量
        
        参数:
            user_id: 用户ID
            emotion: 情绪标签（如'高兴', '悲伤'等）
            emotion_description: 情绪的详细描述（可选）
        
        返回:
            是否更新成功
        """
        try:
            logger.info(f"更新用户 {user_id} 的情绪向量: {emotion}")
            
            # 情绪权重映射
            emotion_weights = {
                '高兴': 1.0,
                '悲伤': 1.0,
                '愤怒': 1.0,
                '恐惧': 1.0,
                '惊讶': 0.8,
                '期待': 0.8,
                '焦虑': 0.9,
                '平静': 0.7,
                '兴奋': 1.0,
                '无聊': 0.7,
                '疲倦': 0.7,
                '困惑': 0.7,
                '满足': 0.8,
                '喜爱': 0.9,
                '感激': 0.8,
            }
            
            # 默认权重
            weight = emotion_weights.get(emotion, 0.5) 
            
            # 首先尝试获取现有的用户偏好
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'music_recommender.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 检查是否存在标准化偏好表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS normalized_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                preferences TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 获取现有偏好
            cursor.execute('SELECT preferences FROM normalized_preferences WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            
            now = datetime.now().isoformat()
            
            if result:
                preferences = json.loads(result[0])
                
                # 如果不存在moods类别，创建它
                if 'moods' not in preferences:
                    preferences['moods'] = {}
                
                # 更新情绪偏好值
                if emotion in preferences['moods']:
                    # 调整现有情绪值 (增加权重，但最大为1.0)
                    current_value = preferences['moods'][emotion]
                    preferences['moods'][emotion] = min(current_value + weight * 0.2, 1.0)
                else:
                    # 添加新情绪
                    preferences['moods'][emotion] = weight * 0.5
                
                # 保存更新的偏好
                preferences_json = json.dumps(preferences, ensure_ascii=False)
                cursor.execute(
                    'UPDATE normalized_preferences SET preferences = ?, timestamp = ? WHERE user_id = ?',
                    (preferences_json, now, user_id)
                )
            else:
                # 创建新的偏好记录
                preferences = {
                    'genres': {},
                    'moods': {emotion: weight * 0.5},
                    'eras': {},
                    'artists': {},
                    'listening_frequency': 0.5,
                    'discovery_level': 0.5
                }
                
                preferences_json = json.dumps(preferences, ensure_ascii=False)
                cursor.execute(
                    'INSERT INTO normalized_preferences (user_id, preferences, timestamp) VALUES (?, ?, ?)',
                    (user_id, preferences_json, now)
                )
            
            # 记录用户情绪历史到新表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_emotion_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                emotion TEXT NOT NULL,
                description TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            cursor.execute(
                'INSERT INTO user_emotion_history (user_id, emotion, description, timestamp) VALUES (?, ?, ?, ?)',
                (user_id, emotion, emotion_description or '', now)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"已更新用户 {user_id} 的情绪向量并记录历史")
            return True
            
        except Exception as e:
            logger.error(f"更新用户情绪向量失败: {str(e)}")
            return False
    
    def update_user_dialogue_vector(self, user_id, dialogue_data):
        """
        根据用户对话内容更新用户对话向量
        
        参数:
            user_id: 用户ID
            dialogue_data: 对话数据，应包含以下字段:
                - user_message: 用户消息
                - ai_response: AI响应
                - timestamp: 时间戳
                - emotion: 可选，情绪标签
        
        返回:
            是否更新成功
        """
        try:
            logger.info(f"更新用户 {user_id} 的对话向量")
            
            # 提取对话数据
            user_message = dialogue_data.get('user_message', '')
            ai_response = dialogue_data.get('ai_response', '')
            timestamp = dialogue_data.get('timestamp', datetime.now().isoformat())
            emotion = dialogue_data.get('emotion', None)
            
            # 如果有情绪数据，同时更新情绪向量
            if emotion and emotion != "未知":
                self.update_user_emotion_vector(user_id, emotion)
            
            # 从对话内容中提取关键词
            import re
            keywords = []
            
            # 结合用户消息和AI响应
            combined_text = f"{user_message} {ai_response}"
            
            # 尝试使用jieba提取关键词
            try:
                import jieba
                import jieba.analyse
                keywords = jieba.analyse.extract_tags(combined_text, topK=10, withWeight=True)
                logger.info(f"使用jieba提取了 {len(keywords)} 个关键词")
            except ImportError:
                # 如果jieba不可用，使用简单的分词方法
                logger.warning("jieba模块不可用，使用简单分词")
                words = re.findall(r'\w+', combined_text)
                # 统计词频
                word_freq = {}
                for word in words:
                    if len(word) > 1:  # 忽略单字符
                        word_freq[word] = word_freq.get(word, 0) + 1
                
                # 按频率排序选取前10个词
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
                keywords = [(word, freq/max(1, len(words))) for word, freq in sorted_words]
                logger.info(f"使用简单分词提取了 {len(keywords)} 个关键词")
            
            # 创建或更新用户对话历史表
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'music_recommender.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建用户对话历史表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_dialogue_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                user_message TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                keywords TEXT,
                emotion TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 保存对话历史
            cursor.execute(
                'INSERT INTO user_dialogue_history (user_id, user_message, ai_response, keywords, emotion, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                (
                    user_id, 
                    user_message, 
                    ai_response, 
                    json.dumps([{"word": k, "weight": w} for k, w in keywords], ensure_ascii=False),
                    emotion or '',
                    timestamp
                )
            )
            
            # 创建或检查normalized_preferences表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS normalized_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                preferences TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            # 更新用户偏好
            cursor.execute('SELECT preferences FROM normalized_preferences WHERE user_id = ?', (user_id,))
            result = cursor.fetchone()
            
            if result:
                preferences = json.loads(result[0])
            else:
                # 创建新的偏好记录
                preferences = {
                    'genres': {},
                    'moods': {},
                    'eras': {},
                    'artists': {},
                    'listening_frequency': 0.5,
                    'discovery_level': 0.5
                }
                
                # 保存初始偏好
                preferences_json = json.dumps(preferences, ensure_ascii=False)
                cursor.execute(
                    'INSERT INTO normalized_preferences (user_id, preferences, timestamp) VALUES (?, ?, ?)',
                    (user_id, preferences_json, timestamp)
                )
                
                # 重新获取刚插入的记录
                cursor.execute('SELECT preferences FROM normalized_preferences WHERE user_id = ?', (user_id,))
                result = cursor.fetchone()
                if result:
                    preferences = json.loads(result[0])
            
            # 如果已经有偏好记录，更新音乐相关偏好
            if preferences:
                # 检查关键词中是否有音乐相关的词汇
                music_keywords = ['音乐', '歌曲', '歌手', '专辑', '旋律', '节奏', '流行', '摇滚', '古典', '爵士', '电子', '嘻哈', '民谣']
                
                # 提取关键词中的音乐流派信息
                genre_keywords = {
                    '流行': 'Pop', '摇滚': 'Rock', '古典': 'Classical', '爵士': 'Jazz', 
                    '电子': 'Electronic', '嘻哈': 'Hip-Hop', '民谣': 'Folk', '金属': 'Metal',
                    '乡村': 'Country', '蓝调': 'Blues', 'R&B': 'R&B'
                }
                
                for keyword, weight in keywords:
                    # 检查是否是音乐流派关键词
                    if keyword in genre_keywords:
                        genre = genre_keywords[keyword]
                        if 'genres' not in preferences:
                            preferences['genres'] = {}
                        
                        # 更新流派偏好
                        current_value = preferences['genres'].get(genre, 0)
                        preferences['genres'][genre] = min(current_value + weight * 0.1, 1.0)
                
                # 保存更新的偏好
                preferences_json = json.dumps(preferences, ensure_ascii=False)
                cursor.execute(
                    'UPDATE normalized_preferences SET preferences = ?, timestamp = ? WHERE user_id = ?',
                    (preferences_json, timestamp, user_id)
                )
            
            conn.commit()
            conn.close()
            
            logger.info(f"已更新用户 {user_id} 的对话向量并记录历史")
            return True
            
        except Exception as e:
            logger.error(f"更新用户对话向量失败: {str(e)}")
            return False
    
    def update_user_preference(self, user_id, item_id, preference_type='rating', weight=1.0):
        """
        更新用户对特定歌曲的偏好
        
        参数:
            user_id: 用户ID
            item_id: 歌曲ID
            preference_type: 偏好类型 ('rating', 'listening', 'ai_recommendation')
            weight: 偏好权重 (0.0-1.0)
        
        返回:
            是否更新成功
        """
        try:
            logger.info(f"更新用户 {user_id} 对歌曲 {item_id} 的偏好")
            
            # 权重映射
            type_weights = {
                'rating': 1.0,       # 用户主动评分
                'listening': 0.5,    # 听歌行为
                'ai_recommendation': 0.3  # AI推荐的歌曲
            }
            
            # 调整权重
            actual_weight = weight * type_weights.get(preference_type, 0.5)
            
            # 获取歌曲信息
            song_info = None
            track_id = item_id
            
            # 如果是Spotify ID，需要转换格式
            if isinstance(track_id, str) and track_id.startswith('spotify:track:'):
                track_id = track_id.split(':')[-1]
            
            # 创建或更新用户偏好历史表
            db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'music_recommender.db')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建用户歌曲偏好表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_song_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                track_id TEXT NOT NULL,
                preference_type TEXT NOT NULL,
                weight REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            
            now = datetime.now().isoformat()
            
            # 记录偏好
            cursor.execute(
                'INSERT INTO user_song_preferences (user_id, track_id, preference_type, weight, timestamp) VALUES (?, ?, ?, ?, ?)',
                (user_id, item_id, preference_type, actual_weight, now)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"已更新用户 {user_id} 对歌曲 {item_id} 的偏好记录")
            return True
            
        except Exception as e:
            logger.error(f"更新用户歌曲偏好失败: {str(e)}")
            return False
    
    def update_user_vector(self, user_id, track_id, play_data):
        """
        根据用户听歌行为更新用户向量
        
        参数:
            user_id: 用户ID
            track_id: 歌曲ID
            play_data: 播放数据，应包含以下字段:
                - duration: 播放时长（秒）
                - timestamp: 时间戳
                - completed: 是否播放完成
        
        返回:
            是否更新成功
        """
        try:
            logger.info(f"更新用户 {user_id} 的听歌向量")
            
            # 提取播放数据
            duration = play_data.get('duration', 0)
            completed = play_data.get('completed', False)
            
            # 根据播放情况计算权重
            if completed:
                weight = 0.8  # 完整播放
            elif duration > 60:
                weight = 0.5  # 播放超过1分钟
            elif duration > 30:
                weight = 0.3  # 播放超过30秒
            else:
                weight = 0.1  # 短暂播放
            
            # 更新用户偏好
            return self.update_user_preference(
                user_id=user_id,
                item_id=track_id,
                preference_type='listening',
                weight=weight
            )
            
        except Exception as e:
            logger.error(f"更新用户听歌向量失败: {str(e)}")
            return False

# 用于单独测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 创建推荐系统实例
    recommender = HybridMusicRecommender(data_dir='processed_data', use_msd=True)

    # 示例用户数据
    user_id = "test_user"

    # 示例游戏数据
    game_data = [
        {
            'type': 'genre_selection',
            'selections': {'pop': 0.9, 'rock': 0.7, 'classical': 0.3}
        },
        {
            'type': 'mood_selection',
            'mood': 'happy',
            'intensity': 0.8
        }
    ]

    # 示例聊天数据
    chat_data = [
        {
            'role': 'user',
            'content': '我今天心情很好，想听一些欢快的歌曲'
        },
        {
            'role': 'assistant',
            'content': '好的，我会推荐一些欢快的歌曲给您'
        },
        {
            'role': 'user',
            'content': '我很喜欢周杰伦的歌'
        }
    ]

    # 获取推荐
    recommendations = recommender.get_hybrid_recommendations(user_id, game_data, chat_data)

    # 打印推荐结果
    print(f"为用户 {user_id} 生成了 {len(recommendations)} 条推荐:")
    for i, rec in enumerate(recommendations[:5]):  # 只打印前5条
        print(f"{i+1}. 《{rec['track_name']}》 - {rec['artist_name']}")
        print(f"   推荐理由: {rec['explanation']}")

    # 测试代码
    try:
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # 初始化混合推荐系统
        recommender = HybridMusicRecommender(data_dir="../data", use_msd=False)
        
        # 测试用户ID
        test_user_id = "test_user_001"
        
        # 测试更新用户情绪向量
        print("测试更新用户情绪向量...")
        result1 = recommender.update_user_emotion_vector(
            user_id=test_user_id,
            emotion="高兴",
            emotion_description="用户表现得很开心"
        )
        print(f"结果: {'成功' if result1 else '失败'}\n")
        
        # 测试更新用户对话向量
        print("测试更新用户对话向量...")
        result2 = recommender.update_user_dialogue_vector(
            user_id=test_user_id,
            dialogue_data={
                'user_message': '我今天听了一首很好听的流行歌曲',
                'ai_response': '真棒！看起来你很喜欢流行音乐，我可以推荐更多类似的歌曲给你',
                'timestamp': '2023-06-15T12:34:56',
                'emotion': '高兴'
            }
        )
        print(f"结果: {'成功' if result2 else '失败'}\n")
        
        # 测试更新用户歌曲偏好
        print("测试更新用户歌曲偏好...")
        result3 = recommender.update_user_preference(
            user_id=test_user_id,
            item_id='spotify:track:123456',
            preference_type='rating',
            weight=0.8
        )
        print(f"结果: {'成功' if result3 else '失败'}\n")
        
        # 测试更新用户听歌向量
        print("测试更新用户听歌向量...")
        result4 = recommender.update_user_vector(
            user_id=test_user_id,
            track_id='spotify:track:123456',
            play_data={
                'duration': 120,
                'completed': True,
                'timestamp': '2023-06-15T12:34:56'
            }
        )
        print(f"结果: {'成功' if result4 else '失败'}\n")
        
        print("全部测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()