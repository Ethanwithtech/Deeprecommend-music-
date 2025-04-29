#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音乐推荐系统高级算法训练脚本

此脚本用于训练音乐推荐系统中使用的多种推荐算法，包括：
1. SVD++ (奇异值分解++)
2. NCF (神经协同过滤)
3. MLP (多层感知机)
4. 内容特征提取模型

该脚本设计用于在Google Colab中运行，可以使用GPU加速训练过程。
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import pickle
import json
import time
import random
from datetime import datetime
from tqdm.notebook import tqdm
from collections import defaultdict

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('music_recommender_trainer')

# 检查是否在Google Colab环境中运行
try:
    import google.colab
    IN_COLAB = True
    logger.info("检测到在Google Colab环境中运行")
except:
    IN_COLAB = False
    logger.info("在本地环境中运行")

# 安装必要的依赖
if IN_COLAB:
    logger.info("安装必要的依赖...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "surprise", "tensorflow", "keras", "scikit-learn", "librosa", "spotipy"])

# 导入依赖库
try:
    from surprise import SVD, SVDpp, Dataset, Reader
    from surprise.model_selection import cross_validate, train_test_split
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential, load_model, save_model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    logger.info("成功导入所有依赖库")
except ImportError as e:
    logger.error(f"导入依赖库时出错: {str(e)}")
    sys.exit(1)

class MusicRecommenderTrainer:
    """音乐推荐系统训练类
    
    用于训练和评估多种推荐算法
    """
    
    def __init__(self, data_dir='./data', output_dir='./models', use_gpu=True):
        """初始化训练器
        
        Args:
            data_dir: 数据目录
            output_dir: 输出模型目录
            use_gpu: 是否使用GPU加速
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # 创建目录
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 数据存储
        self.user_ratings = None  # 用户评分数据
        self.songs_df = None      # 歌曲元数据
        self.interactions_df = None  # 用户-歌曲交互
        
        # 模型存储
        self.svdpp_model = None   # SVD++模型
        self.ncf_model = None     # 神经协同过滤模型
        self.mlp_model = None     # 多层感知机模型
        self.content_model = None # 内容特征模型
        
        # 编码器
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        # 配置GPU
        if self.use_gpu:
            self._configure_gpu()
    
    def _configure_gpu(self):
        """配置GPU"""
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # 限制TensorFlow只使用第一个GPU
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # 限制内存增长
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info(f"使用GPU: {gpus[0]}")
            else:
                logger.warning("没有找到可用的GPU，将使用CPU")
                self.use_gpu = False
        except Exception as e:
            logger.error(f"配置GPU时出错: {str(e)}")
            self.use_gpu = False
    
    def load_data(self, songs_file=None, interactions_file=None, create_sample=True):
        """加载数据
        
        Args:
            songs_file: 歌曲元数据文件路径
            interactions_file: 用户-歌曲交互文件路径
            create_sample: 是否创建样本数据
        """
        # 设置默认文件路径
        if songs_file is None:
            songs_file = os.path.join(self.data_dir, 'songs.csv')
        if interactions_file is None:
            interactions_file = os.path.join(self.data_dir, 'user_song_interactions.csv')
        
        # 检查是否存在文件
        if not os.path.exists(songs_file) or not os.path.exists(interactions_file):
            if create_sample:
                logger.info("数据文件不存在，创建样本数据...")
                self._create_sample_data(songs_file, interactions_file)
            else:
                logger.error("数据文件不存在且不创建样本数据")
                return False
        
        try:
            # 加载歌曲数据
            self.songs_df = pd.read_csv(songs_file)
            logger.info(f"加载了 {len(self.songs_df)} 首歌曲的元数据")
            
            # 加载交互数据
            self.interactions_df = pd.read_csv(interactions_file)
            logger.info(f"加载了 {len(self.interactions_df)} 条用户-歌曲交互记录")
            
            # 编码用户和物品ID
            self.user_encoder.fit(self.interactions_df['user_id'].unique())
            self.item_encoder.fit(self.interactions_df['song_id'].unique())
            
            # 构建用户评分字典
            self.user_ratings = defaultdict(dict)
            for _, row in self.interactions_df.iterrows():
                self.user_ratings[row['user_id']][row['song_id']] = row['rating']
            
            logger.info(f"构建了 {len(self.user_ratings)} 个用户的评分数据")
            return True
            
        except Exception as e:
            logger.error(f"加载数据时出错: {str(e)}")
            return False
    
    def _create_sample_data(self, songs_file, interactions_file):
        """创建样本数据
        
        Args:
            songs_file: 歌曲元数据输出文件路径
            interactions_file: 用户-歌曲交互输出文件路径
        """
        try:
            # 样本歌曲数据
            sample_songs = []
            for i in range(1, 10001):
                song = {
                    'song_id': f"S{i:05d}",
                    'track_name': f"Song {i}",
                    'artist_name': f"Artist {i % 500 + 1}",
                    'album_name': f"Album {i % 1000 + 1}",
                    'genre': random.choice(['Pop', 'Rock', 'Jazz', 'Classical', 'Hip Hop', 'Electronic', 'Country', 'R&B']),
                    'year': random.randint(1980, 2023),
                    'tempo': random.uniform(60, 180),
                    'loudness': random.uniform(-20, 0),
                    'danceability': random.uniform(0, 1),
                    'energy': random.uniform(0, 1),
                    'acousticness': random.uniform(0, 1),
                    'instrumentalness': random.uniform(0, 1),
                    'valence': random.uniform(0, 1)
                }
                sample_songs.append(song)
            
            # 创建歌曲数据框
            self.songs_df = pd.DataFrame(sample_songs)
            
            # 样本用户-歌曲交互数据
            sample_interactions = []
            num_users = 1000
            for user_id in range(1, num_users + 1):
                # 每个用户评分20-200首歌
                num_ratings = random.randint(20, 200)
                # 随机选择歌曲ID
                song_indices = random.sample(range(len(sample_songs)), num_ratings)
                
                for idx in song_indices:
                    interaction = {
                        'user_id': f"U{user_id:05d}",
                        'song_id': sample_songs[idx]['song_id'],
                        'rating': random.uniform(1, 5),
                        'timestamp': int(datetime.now().timestamp())
                    }
                    sample_interactions.append(interaction)
            
            # 创建交互数据框
            self.interactions_df = pd.DataFrame(sample_interactions)
            
            # 构建用户评分字典
            self.user_ratings = defaultdict(dict)
            for _, row in self.interactions_df.iterrows():
                self.user_ratings[row['user_id']][row['song_id']] = row['rating']
            
            # 保存样本数据
            self.songs_df.to_csv(songs_file, index=False)
            self.interactions_df.to_csv(interactions_file, index=False)
            
            logger.info(f"创建了样本数据: {len(self.songs_df)}首歌曲, {len(self.interactions_df)}条交互记录")
            
            # 编码用户和物品ID
            self.user_encoder.fit(self.interactions_df['user_id'].unique())
            self.item_encoder.fit(self.interactions_df['song_id'].unique())
            
            return True
            
        except Exception as e:
            logger.error(f"创建样本数据时出错: {str(e)}")
            return False
    
    def preprocess_data(self):
        """预处理数据
        
        对数据进行标准化和编码
        """
        if self.interactions_df is None or self.songs_df is None:
            logger.error("请先加载数据")
            return False
        
        try:
            # 标准化歌曲特征
            feature_columns = ['tempo', 'loudness', 'danceability', 'energy', 'acousticness', 'instrumentalness', 'valence']
            feature_columns = [col for col in feature_columns if col in self.songs_df.columns]
            
            if feature_columns:
                scaler = StandardScaler()
                self.songs_df[feature_columns] = scaler.fit_transform(self.songs_df[feature_columns])
                logger.info(f"标准化了歌曲特征: {feature_columns}")
            
            # 为NCF和MLP准备数据
            # 编码用户ID和歌曲ID为整数
            self.interactions_df['user_idx'] = self.user_encoder.transform(self.interactions_df['user_id'])
            self.interactions_df['song_idx'] = self.item_encoder.transform(self.interactions_df['song_id'])
            
            logger.info("数据预处理完成")
            return True
            
        except Exception as e:
            logger.error(f"预处理数据时出错: {str(e)}")
            return False
    
    def train_svdpp(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """训练SVD++模型
        
        Args:
            n_factors: 潜在因子数
            n_epochs: 训练轮数
            lr_all: 学习率
            reg_all: 正则化参数
            
        Returns:
            训练好的SVD++模型
        """
        if self.interactions_df is None:
            logger.error("请先加载数据")
            return None
        
        try:
            logger.info(f"开始训练SVD++模型: n_factors={n_factors}, n_epochs={n_epochs}")
            start_time = time.time()
            
            # 准备Surprise数据集
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(
                self.interactions_df[['user_id', 'song_id', 'rating']], 
                reader
            )
            
            # 划分训练集和测试集
            trainset, testset = train_test_split(data, test_size=0.2)
            
            # 创建并训练SVD++模型
            self.svdpp_model = SVDpp(
                n_factors=n_factors,
                n_epochs=n_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                verbose=True
            )
            
            self.svdpp_model.fit(trainset)
            
            # 评估模型
            test_rmse = np.sqrt(np.mean([(self.svdpp_model.predict(uid, iid, r).est - r) ** 2 for (uid, iid, r) in testset]))
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'svdpp_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(self.svdpp_model, f)
            
            end_time = time.time()
            logger.info(f"SVD++模型训练完成，用时: {end_time - start_time:.2f}秒")
            logger.info(f"测试集RMSE: {test_rmse:.4f}")
            logger.info(f"模型已保存至: {model_path}")
            
            return self.svdpp_model
            
        except Exception as e:
            logger.error(f"训练SVD++模型时出错: {str(e)}")
            return None
    
    def train_ncf(self, embedding_dim=100, layers=[256, 128, 64], lr=0.001, epochs=20, batch_size=256):
        """训练神经协同过滤模型
        
        Args:
            embedding_dim: 嵌入维度
            layers: 神经网络层结构
            lr: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练好的NCF模型
        """
        if self.interactions_df is None:
            logger.error("请先加载数据")
            return None
        
        try:
            logger.info(f"开始训练NCF模型: embedding_dim={embedding_dim}, layers={layers}")
            start_time = time.time()
            
            # 准备数据
            user_input = self.interactions_df['user_idx'].values
            item_input = self.interactions_df['song_idx'].values
            ratings = self.interactions_df['rating'].values / 5.0  # 缩放到0-1
            
            # 用户和物品的数量
            num_users = len(self.user_encoder.classes_)
            num_items = len(self.item_encoder.classes_)
            
            # 划分训练集和测试集
            indices = np.arange(len(self.interactions_df))
            np.random.shuffle(indices)
            
            train_indices = indices[:int(0.8 * len(indices))]
            test_indices = indices[int(0.8 * len(indices)):]
            
            user_input_train = user_input[train_indices]
            item_input_train = item_input[train_indices]
            ratings_train = ratings[train_indices]
            
            user_input_test = user_input[test_indices]
            item_input_test = item_input[test_indices]
            ratings_test = ratings[test_indices]
            
            # 构建NCF模型
            # 用户嵌入
            user_input_layer = Input(shape=(1,), name='user_input')
            user_embedding = Embedding(num_users, embedding_dim, name='user_embedding')(user_input_layer)
            user_vec = Flatten(name='user_flatten')(user_embedding)
            
            # 物品嵌入
            item_input_layer = Input(shape=(1,), name='item_input')
            item_embedding = Embedding(num_items, embedding_dim, name='item_embedding')(item_input_layer)
            item_vec = Flatten(name='item_flatten')(item_embedding)
            
            # 神经CF层
            concat = Concatenate()([user_vec, item_vec])
            
            # 全连接层
            fc_layer = concat
            for i, layer_size in enumerate(layers):
                fc_layer = Dense(layer_size, activation='relu', name=f'layer{i}')(fc_layer)
                fc_layer = Dropout(0.2)(fc_layer)
            
            # 输出层
            output = Dense(1, activation='sigmoid', name='output')(fc_layer)
            
            self.ncf_model = Model(inputs=[user_input_layer, item_input_layer], outputs=output)
            self.ncf_model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mae'])
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint(
                    os.path.join(self.output_dir, 'ncf_model_checkpoint.h5'), 
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # 训练模型
            history = self.ncf_model.fit(
                [user_input_train, item_input_train],
                ratings_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([user_input_test, item_input_test], ratings_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型
            loss, mae = self.ncf_model.evaluate([user_input_test, item_input_test], ratings_test, verbose=0)
            rmse = np.sqrt(loss) * 5  # 转换回原始评分范围
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'ncf_model.h5')
            self.ncf_model.save(model_path)
            
            # 保存模型配置
            model_config = {
                'embedding_dim': embedding_dim,
                'layers': layers,
                'num_users': num_users,
                'num_items': num_items,
                'user_encoder': self.user_encoder.classes_.tolist(),
                'item_encoder': self.item_encoder.classes_.tolist()
            }
            
            with open(os.path.join(self.output_dir, 'ncf_model_config.json'), 'w') as f:
                json.dump(model_config, f)
            
            end_time = time.time()
            logger.info(f"NCF模型训练完成，用时: {end_time - start_time:.2f}秒")
            logger.info(f"测试集RMSE: {rmse:.4f}, MAE: {mae*5:.4f}")
            logger.info(f"模型已保存至: {model_path}")
            
            # 绘制训练历史
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='训练损失')
            plt.plot(history.history['val_loss'], label='验证损失')
            plt.title('NCF模型损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='训练MAE')
            plt.plot(history.history['val_mae'], label='验证MAE')
            plt.title('NCF模型MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'ncf_training_history.png'))
            
            return self.ncf_model
            
        except Exception as e:
            logger.error(f"训练NCF模型时出错: {str(e)}")
            return None
    
    def train_mlp(self, embedding_dim=64, layers=[128, 64, 32], lr=0.001, epochs=20, batch_size=256):
        """训练多层感知机模型
        
        Args:
            embedding_dim: 嵌入维度
            layers: 神经网络层结构
            lr: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            
        Returns:
            训练好的MLP模型
        """
        if self.interactions_df is None:
            logger.error("请先加载数据")
            return None
        
        try:
            logger.info(f"开始训练MLP模型: embedding_dim={embedding_dim}, layers={layers}")
            start_time = time.time()
            
            # 准备数据
            user_input = self.interactions_df['user_idx'].values
            item_input = self.interactions_df['song_idx'].values
            ratings = self.interactions_df['rating'].values / 5.0  # 缩放到0-1
            
            # 用户和物品的数量
            num_users = len(self.user_encoder.classes_)
            num_items = len(self.item_encoder.classes_)
            
            # 划分训练集和测试集
            indices = np.arange(len(self.interactions_df))
            np.random.shuffle(indices)
            
            train_indices = indices[:int(0.8 * len(indices))]
            test_indices = indices[int(0.8 * len(indices)):]
            
            user_input_train = user_input[train_indices]
            item_input_train = item_input[train_indices]
            ratings_train = ratings[train_indices]
            
            user_input_test = user_input[test_indices]
            item_input_test = item_input[test_indices]
            ratings_test = ratings[test_indices]
            
            # 构建MLP模型
            # 用户输入和嵌入
            user_input_layer = Input(shape=(1,), name='user_input')
            user_embedding = Embedding(num_users, embedding_dim, name='user_embedding')(user_input_layer)
            user_vec = Flatten(name='user_flatten')(user_embedding)
            
            # 物品输入和嵌入
            item_input_layer = Input(shape=(1,), name='item_input')
            item_embedding = Embedding(num_items, embedding_dim, name='item_embedding')(item_input_layer)
            item_vec = Flatten(name='item_flatten')(item_embedding)
            
            # 连接用户和物品嵌入
            concat = Concatenate()([user_vec, item_vec])
            
            # 多层感知机层
            mlp_layer = concat
            for i, layer_size in enumerate(layers):
                mlp_layer = Dense(layer_size, activation='relu', name=f'layer{i}')(mlp_layer)
                mlp_layer = Dropout(0.2)(mlp_layer)
            
            # 输出层
            output = Dense(1, activation='sigmoid', name='output')(mlp_layer)
            
            self.mlp_model = Model(inputs=[user_input_layer, item_input_layer], outputs=output)
            self.mlp_model.compile(optimizer=Adam(lr=lr), loss='mse', metrics=['mae'])
            
            # 回调函数
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6),
                ModelCheckpoint(
                    os.path.join(self.output_dir, 'mlp_model_checkpoint.h5'), 
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
            
            # 训练模型
            history = self.mlp_model.fit(
                [user_input_train, item_input_train],
                ratings_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([user_input_test, item_input_test], ratings_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # 评估模型
            loss, mae = self.mlp_model.evaluate([user_input_test, item_input_test], ratings_test, verbose=0)
            rmse = np.sqrt(loss) * 5  # 转换回原始评分范围
            
            # 保存模型
            model_path = os.path.join(self.output_dir, 'mlp_model.h5')
            self.mlp_model.save(model_path)
            
            # 保存模型配置
            model_config = {
                'embedding_dim': embedding_dim,
                'layers': layers,
                'num_users': num_users,
                'num_items': num_items,
                'user_encoder': self.user_encoder.classes_.tolist(),
                'item_encoder': self.item_encoder.classes_.tolist()
            }
            
            with open(os.path.join(self.output_dir, 'mlp_model_config.json'), 'w') as f:
                json.dump(model_config, f)
            
            end_time = time.time()
            logger.info(f"MLP模型训练完成，用时: {end_time - start_time:.2f}秒")
            logger.info(f"测试集RMSE: {rmse:.4f}, MAE: {mae*5:.4f}")
            logger.info(f"模型已保存至: {model_path}")
            
            # 绘制训练历史
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history.history['loss'], label='训练损失')
            plt.plot(history.history['val_loss'], label='验证损失')
            plt.title('MLP模型损失')
            plt.xlabel('Epoch')
            plt.ylabel('损失')
            plt.legend()
            
            plt.subplot(1, 2, 2)
            plt.plot(history.history['mae'], label='训练MAE')
            plt.plot(history.history['val_mae'], label='验证MAE')
            plt.title('MLP模型MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'mlp_training_history.png'))
            
            return self.mlp_model
            
        except Exception as e:
            logger.error(f"训练MLP模型时出错: {str(e)}")
            return None
    
    def train_content_model(self, n_components=50):
        """训练内容特征模型
        
        使用PCA降维提取歌曲内容特征
        
        Args:
            n_components: PCA组件数
            
        Returns:
            训练好的内容特征
        """
        if self.songs_df is None:
            logger.error("请先加载数据")
            return None
        
        try:
            logger.info(f"开始训练内容特征模型: n_components={n_components}")
            start_time = time.time()
            
            # 选择特征列
            feature_columns = ['tempo', 'loudness', 'danceability', 'energy', 'acousticness', 'instrumentalness', 'valence']
            feature_columns = [col for col in feature_columns if col in self.songs_df.columns]
            
            if not feature_columns:
                logger.error("没有找到有效的特征列")
                return None
            
            # 提取特征
            features = self.songs_df[feature_columns].values
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 使用PCA降维
            pca = PCA(n_components=min(n_components, len(feature_columns)))
            song_features = pca.fit_transform(features_scaled)
            
            # 创建歌曲特征字典
            self.content_model = {}
            for i, song_id in enumerate(self.songs_df['song_id']):
                self.content_model[song_id] = song_features[i]
            
            # 保存内容模型
            model_path = os.path.join(self.output_dir, 'content_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'features': self.content_model,
                    'scaler': scaler,
                    'pca': pca,
                    'feature_columns': feature_columns
                }, f)
            
            # 计算歌曲相似度矩阵
            similarity_matrix = cosine_similarity(song_features)
            
            # 保存相似度矩阵
            similarity_path = os.path.join(self.output_dir, 'song_similarity_matrix.npy')
            np.save(similarity_path, similarity_matrix)
            
            # 保存歌曲ID映射
            song_id_map = {i: song_id for i, song_id in enumerate(self.songs_df['song_id'])}
            with open(os.path.join(self.output_dir, 'song_id_map.json'), 'w') as f:
                json.dump(song_id_map, f)
            
            end_time = time.time()
            logger.info(f"内容特征模型训练完成，用时: {end_time - start_time:.2f}秒")
            logger.info(f"解释方差比例: {sum(pca.explained_variance_ratio_):.4f}")
            logger.info(f"模型已保存至: {model_path}")
            
            # 可视化特征PCA结果
            if n_components >= 2:
                plt.figure(figsize=(10, 8))
                plt.scatter(song_features[:, 0], song_features[:, 1], alpha=0.3)
                plt.title('歌曲特征PCA可视化')
                plt.xlabel('主成分1')
                plt.ylabel('主成分2')
                plt.savefig(os.path.join(self.output_dir, 'content_features_pca.png'))
            
            return self.content_model
            
        except Exception as e:
            logger.error(f"训练内容特征模型时出错: {str(e)}")
            return None
    
    def load_models(self):
        """加载训练好的模型"""
        try:
            # 加载SVD++模型
            svdpp_path = os.path.join(self.output_dir, 'svdpp_model.pkl')
            if os.path.exists(svdpp_path):
                with open(svdpp_path, 'rb') as f:
                    self.svdpp_model = pickle.load(f)
                logger.info("加载了SVD++模型")
            
            # 加载NCF模型
            ncf_path = os.path.join(self.output_dir, 'ncf_model.h5')
            ncf_config_path = os.path.join(self.output_dir, 'ncf_model_config.json')
            if os.path.exists(ncf_path) and os.path.exists(ncf_config_path):
                self.ncf_model = load_model(ncf_path)
                logger.info("加载了NCF模型")
            
            # 加载MLP模型
            mlp_path = os.path.join(self.output_dir, 'mlp_model.h5')
            mlp_config_path = os.path.join(self.output_dir, 'mlp_model_config.json')
            if os.path.exists(mlp_path) and os.path.exists(mlp_config_path):
                self.mlp_model = load_model(mlp_path)
                logger.info("加载了MLP模型")
            
            # 加载内容模型
            content_path = os.path.join(self.output_dir, 'content_model.pkl')
            if os.path.exists(content_path):
                with open(content_path, 'rb') as f:
                    content_data = pickle.load(f)
                    self.content_model = content_data['features']
                logger.info("加载了内容特征模型")
            
            return True
            
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            return False
    
    def train_all_models(self):
        """训练所有模型"""
        # 预处理数据
        logger.info("开始数据预处理...")
        self.preprocess_data()
        
        # 训练SVD++模型
        logger.info("开始训练SVD++模型...")
        self.train_svdpp()
        
        # 训练NCF模型
        logger.info("开始训练NCF模型...")
        self.train_ncf()
        
        # 训练MLP模型
        logger.info("开始训练MLP模型...")
        self.train_mlp()
        
        # 训练内容特征模型
        logger.info("开始训练内容特征模型...")
        self.train_content_model()
        
        logger.info("所有模型训练完成")

def download_models_to_local(models_dir, local_dir='../backend/models/trained_models'):
    """将训练好的模型从Colab下载到本地
    
    Args:
        models_dir: Colab中的模型目录
        local_dir: 本地模型保存目录
    """
    if IN_COLAB:
        from google.colab import files
        import shutil
        
        # 压缩模型目录
        shutil.make_archive('trained_models', 'zip', models_dir)
        
        # 下载到本地
        files.download('trained_models.zip')
        
        logger.info("模型已打包为trained_models.zip并准备下载")
        logger.info(f"请解压到本地的{local_dir}目录")
    else:
        logger.info("不在Colab环境中，跳过下载")

def main():
    """主函数"""
    # 设置随机种子
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # 创建训练器
    trainer = MusicRecommenderTrainer(
        data_dir='./data',
        output_dir='./models',
        use_gpu=True
    )
    
    # 加载数据
    logger.info("开始加载数据...")
    if not trainer.load_data(create_sample=True):
        logger.error("加载数据失败，退出")
        return
    
    # 训练所有模型
    trainer.train_all_models()
    
    # 准备下载模型
    download_models_to_local('./models')

if __name__ == "__main__":
    main() 