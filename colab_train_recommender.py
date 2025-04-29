#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度推荐音乐系统 - 训练脚本
此脚本用于在Google Colab上训练混合推荐系统
包含SVD++、神经协同过滤(NCF)、多层感知机(MLP)和基于用户的协同过滤方法
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import SVD, SVDpp, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split as surprise_split
import matplotlib.pyplot as plt
import os
import pickle
import time
import logging
import subprocess
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# 检查是否在Colab环境中运行
try:
    import google.colab
    IN_COLAB = True
    print("在Google Colab环境中运行")
except:
    IN_COLAB = False
    print("在本地环境中运行")

# 如果在Colab中，安装必要的库
if IN_COLAB:
    print("安装必要的库...")
    subprocess.check_call(['pip', 'install', 'surprise'])
    subprocess.check_call(['pip', 'install', 'scikit-learn'])
    subprocess.check_call(['pip', 'install', 'tensorflow'])
    subprocess.check_call(['pip', 'install', 'pandas'])
    subprocess.check_call(['pip', 'install', 'numpy'])
    subprocess.check_call(['pip', 'install', 'matplotlib'])

# 设置随机种子以确保结果可复现
np.random.seed(42)
tf.random.set_seed(42)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Google Drive挂载（在Colab环境中使用）
MOUNT_DRIVE = True  # 设置为True以挂载Google Drive

if MOUNT_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/gdrive')
        MODEL_PATH = '/content/gdrive/MyDrive/music_recommender_models/'
        DATA_PATH = '/content/gdrive/MyDrive/music_data/'
        # 创建模型保存目录
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(DATA_PATH, exist_ok=True)
        logger.info(f"Google Drive已挂载，模型将保存至: {MODEL_PATH}")
    except ImportError:
        logger.warning("未检测到Google Colab环境，使用本地路径")
        MODEL_PATH = './models/'
        DATA_PATH = './data/'
        os.makedirs(MODEL_PATH, exist_ok=True)
        os.makedirs(DATA_PATH, exist_ok=True)
else:
    MODEL_PATH = './models/'
    DATA_PATH = './data/'
    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(DATA_PATH, exist_ok=True)

#-----------------------------------------------------------------------------
# 数据加载与预处理
#-----------------------------------------------------------------------------

class DataLoader:
    """数据加载和预处理类"""
    
    def __init__(self, ratings_path=None, songs_path=None, sample_size=None):
        """
        初始化数据加载器
        
        参数:
            ratings_path: 评分数据路径
            songs_path: 歌曲数据路径
            sample_size: 样本大小，用于测试时减少数据量
        """
        self.ratings_path = ratings_path
        self.songs_path = songs_path
        self.sample_size = sample_size
        self.ratings_df = None
        self.songs_df = None
        self.user_ids = None
        self.song_ids = None
        self.user_to_idx = None
        self.song_to_idx = None
        self.n_users = 0
        self.n_songs = 0
    
    def load_demo_data(self):
        """加载示例数据，当没有提供实际数据路径时使用"""
        print("加载示例数据...")
        
        # 创建模拟用户数据
        n_users = 1000
        n_songs = 3000
        n_ratings = 50000
        
        # 生成随机用户ID
        user_ids = [f"user_{i}" for i in range(1, n_users+1)]
        # 生成随机歌曲ID
        song_ids = [f"song_{i}" for i in range(1, n_songs+1)]
        
        # 生成随机评分数据
        users = np.random.choice(user_ids, n_ratings)
        songs = np.random.choice(song_ids, n_ratings)
        ratings = np.random.randint(1, 6, n_ratings)
        timestamps = np.random.randint(1000000000, 1600000000, n_ratings)
        
        # 创建评分DataFrame
        self.ratings_df = pd.DataFrame({
            'userId': users,
            'songId': songs,
            'rating': ratings,
            'timestamp': timestamps
        })
        
        # 生成歌曲元数据
        genres = ['流行', '摇滚', '电子', '古典', '爵士', '嘻哈', '民谣', '乡村']
        titles = []
        artists = []
        song_genres = []
        
        for i in range(1, n_songs+1):
            titles.append(f"Song Title {i}")
            artists.append(f"Artist {np.random.randint(1, 200)}")
            song_genres.append(np.random.choice(genres))
        
        # 创建歌曲DataFrame
        self.songs_df = pd.DataFrame({
            'songId': song_ids,
            'title': titles,
            'artist': artists,
            'genre': song_genres
        })
        
        # 为歌曲添加文本描述，用于基于内容的推荐
        self.songs_df['description'] = self.songs_df.apply(
            lambda row: f"{row['title']} by {row['artist']} is a {row['genre']} song.", axis=1
        )
        
        self._process_ids()
    
    def load_data(self):
        """从指定路径加载数据"""
        if self.ratings_path and os.path.exists(self.ratings_path):
            print(f"从 {self.ratings_path} 加载评分数据...")
            self.ratings_df = pd.read_csv(self.ratings_path)
            
            # 如果需要，取样本数据以加快处理速度
            if self.sample_size and self.sample_size < len(self.ratings_df):
                self.ratings_df = self.ratings_df.sample(n=self.sample_size, random_state=42)
        
        if self.songs_path and os.path.exists(self.songs_path):
            print(f"从 {self.songs_path} 加载歌曲数据...")
            self.songs_df = pd.read_csv(self.songs_path)
        
        if self.ratings_df is None or self.songs_df is None:
            print("未找到数据文件，加载示例数据...")
            self.load_demo_data()
        else:
            self._process_ids()
    
    def _process_ids(self):
        """处理用户ID和歌曲ID，创建映射字典"""
        self.user_ids = self.ratings_df['userId'].unique()
        self.song_ids = self.ratings_df['songId'].unique()
        
        self.n_users = len(self.user_ids)
        self.n_songs = len(self.song_ids)
        
        print(f"数据集包含 {self.n_users} 个用户和 {self.n_songs} 首歌曲")
        print(f"共 {len(self.ratings_df)} 条评分记录")
        
        # 创建ID到索引的映射字典
        self.user_to_idx = {user: i for i, user in enumerate(self.user_ids)}
        self.song_to_idx = {song: i for i, song in enumerate(self.song_ids)}
    
    def prepare_surprise_data(self):
        """为Surprise库准备数据"""
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['userId', 'songId', 'rating']], reader)
        return data
    
    def prepare_neural_cf_data(self):
        """为神经协同过滤模型准备数据"""
        # 将用户ID和歌曲ID转换为索引
        user_indices = self.ratings_df['userId'].map(self.user_to_idx).values
        song_indices = self.ratings_df['songId'].map(self.song_to_idx).values
        ratings = self.ratings_df['rating'].values
        
        # 分割训练集和测试集
        train_indices, test_indices = train_test_split(
            np.arange(len(self.ratings_df)), test_size=0.2, random_state=42
        )
        
        X_train = [user_indices[train_indices], song_indices[train_indices]]
        y_train = ratings[train_indices]
        
        X_test = [user_indices[test_indices], song_indices[test_indices]]
        y_test = ratings[test_indices]
        
        return (X_train, y_train), (X_test, y_test)
    
    def prepare_content_data(self):
        """为基于内容的推荐准备数据"""
        # 确保歌曲DataFrame有描述列
        if 'description' not in self.songs_df.columns:
            self.songs_df['description'] = self.songs_df.apply(
                lambda row: f"{row['title']} by {row['artist']}", axis=1
            )
        
        # 创建TF-IDF向量
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.songs_df['description'])
        
        return tfidf_matrix

#-----------------------------------------------------------------------------
# 推荐算法模型
#-----------------------------------------------------------------------------

class SVDppModel:
    """SVD++算法模型"""
    
    def __init__(self, n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """初始化SVD++模型参数"""
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.model = None
    
    def train(self, data):
        """训练SVD++模型"""
        print("训练SVD++模型...")
        # 分割训练集和测试集
        trainset, testset = surprise_split(data, test_size=0.2, random_state=42)
        
        # 创建并训练模型
        self.model = SVDpp(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=True
        )
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.model.fit(trainset)
        
        # 计算训练时间
        train_time = time.time() - start_time
        print(f"SVD++模型训练完成，耗时 {train_time:.2f} 秒")
        
        # 在测试集上评估模型
        test_predictions = self.model.test(testset)
        rmse = accuracy.rmse(test_predictions)
        mae = accuracy.mae(test_predictions)
        
        print(f"SVD++模型测试结果: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return rmse, mae
    
    def save_model(self, path):
        """保存训练好的模型"""
        if self.model:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"SVD++模型已保存到 {path}")
    
    def load_model(self, path):
        """加载已训练的模型"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"从 {path} 加载SVD++模型")
            return True
        return False
    
    def recommend(self, user_id, n=10, item_pool=None):
        """为用户推荐歌曲"""
        if not self.model:
            print("请先训练模型或加载已训练的模型")
            return []
        
        if item_pool is None:
            # 使用所有歌曲作为推荐池
            trainset = self.model.trainset
            item_pool = trainset.all_items()
        
        # 获取用户的内部ID
        try:
            inner_user_id = trainset.to_inner_uid(user_id)
        except ValueError:
            print(f"用户 {user_id} 不在训练数据中")
            return []
        
        # 预测用户对所有歌曲的评分
        predictions = []
        for item_id in item_pool:
            # 转换为原始歌曲ID
            raw_item_id = trainset.to_raw_iid(item_id)
            # 预测评分
            predicted_rating = self.model.predict(user_id, raw_item_id).est
            predictions.append((raw_item_id, predicted_rating))
        
        # 按预测评分排序并返回前n个推荐
        recommendations = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
        
        return recommendations


class NeuralCFModel:
    """神经协同过滤模型"""
    
    def __init__(self, n_users, n_items, embedding_size=50, layers=[64, 32, 16, 8]):
        """初始化神经协同过滤模型"""
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_size = embedding_size
        self.layers = layers
        self.model = None
    
    def build_model(self):
        """构建神经协同过滤模型"""
        # 用户输入和嵌入
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        user_embedding = Embedding(
            input_dim=self.n_users + 1,
            output_dim=self.embedding_size,
            name='user_embedding',
            embeddings_regularizer=l2(0.01)
        )(user_input)
        user_vector = Flatten(name='user_flatten')(user_embedding)
        
        # 歌曲输入和嵌入
        item_input = Input(shape=(1,), dtype='int32', name='item_input')
        item_embedding = Embedding(
            input_dim=self.n_items + 1,
            output_dim=self.embedding_size,
            name='item_embedding',
            embeddings_regularizer=l2(0.01)
        )(item_input)
        item_vector = Flatten(name='item_flatten')(item_embedding)
        
        # 连接用户和歌曲向量
        concat = Concatenate()([user_vector, item_vector])
        
        # 添加全连接层
        for i, layer_size in enumerate(self.layers):
            if i == 0:
                x = Dense(layer_size, activation='relu', name=f'layer{i}')(concat)
            else:
                x = Dense(layer_size, activation='relu', name=f'layer{i}')(x)
            x = Dropout(0.2)(x)
        
        # 输出层
        output = Dense(1, activation='linear', name='prediction')(x)
        
        # 编译模型
        model = Model(inputs=[user_input, item_input], outputs=output)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def train(self, train_data, test_data, epochs=20, batch_size=256):
        """训练神经协同过滤模型"""
        if self.model is None:
            self.build_model()
        
        print("训练神经协同过滤模型...")
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test, y_test)
        )
        
        # 计算训练时间
        train_time = time.time() - start_time
        print(f"神经协同过滤模型训练完成，耗时 {train_time:.2f} 秒")
        
        # 评估模型
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        rmse = np.sqrt(loss)
        
        print(f"神经协同过滤模型测试结果: RMSE = {rmse:.4f}, MAE = {mae:.4f}")
        
        return history, rmse, mae
    
    def save_model(self, path):
        """保存训练好的模型"""
        if self.model:
            self.model.save(path)
            print(f"神经协同过滤模型已保存到 {path}")
    
    def load_model(self, path):
        """加载已训练的模型"""
        if os.path.exists(path):
            self.model = load_model(path)
            print(f"从 {path} 加载神经协同过滤模型")
            return True
        return False
    
    def recommend(self, user_id, user_to_idx, song_to_idx, idx_to_song, n=10):
        """为用户推荐歌曲"""
        if not self.model:
            print("请先训练模型或加载已训练的模型")
            return []
        
        # 获取用户索引
        try:
            user_idx = user_to_idx[user_id]
        except KeyError:
            print(f"用户 {user_id} 不在训练数据中")
            return []
        
        # 为所有歌曲生成预测
        song_indices = list(song_to_idx.values())
        user_input = np.array([user_idx] * len(song_indices))
        item_input = np.array(song_indices)
        
        # 预测评分
        predictions = self.model.predict([user_input, item_input], verbose=0).flatten()
        
        # 将预测结果与歌曲ID配对
        song_ids = [idx_to_song[idx] for idx in song_indices]
        prediction_pairs = list(zip(song_ids, predictions))
        
        # 按预测评分排序并返回前n个推荐
        recommendations = sorted(prediction_pairs, key=lambda x: x[1], reverse=True)[:n]
        
        return recommendations


class ContentBasedRecommender:
    """基于内容的推荐模型"""
    
    def __init__(self):
        """初始化基于内容的推荐模型"""
        self.tfidf_matrix = None
        self.song_indices = None
        self.songs_df = None
    
    def fit(self, tfidf_matrix, songs_df):
        """训练基于内容的推荐模型"""
        self.tfidf_matrix = tfidf_matrix
        self.songs_df = songs_df
        # 创建歌曲索引映射
        self.song_indices = {song: i for i, song in enumerate(songs_df['songId'])}
        
        print("基于内容的推荐模型准备完成")
    
    def compute_similarity(self):
        """计算歌曲之间的相似度"""
        if self.tfidf_matrix is None:
            print("请先训练模型")
            return None
        
        # 计算余弦相似度
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        return cosine_sim
    
    def recommend_similar_songs(self, song_id, cosine_sim, n=10):
        """推荐与给定歌曲相似的歌曲"""
        # 获取歌曲索引
        try:
            idx = self.song_indices[song_id]
        except KeyError:
            print(f"歌曲 {song_id} 不在数据集中")
            return []
        
        # 获取相似度分数
        sim_scores = list(enumerate(cosine_sim[idx]))
        
        # 按相似度排序
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # 排除自身
        sim_scores = sim_scores[1:n+1]
        
        # 获取歌曲索引
        song_indices = [i[0] for i in sim_scores]
        
        # 返回相似歌曲ID和相似度
        similar_songs = [(self.songs_df.iloc[i]['songId'], sim_scores[j][1]) 
                         for j, i in enumerate(song_indices)]
        
        return similar_songs
    
    def recommend_for_user(self, user_ratings, cosine_sim, n=10):
        """基于用户评分历史推荐歌曲"""
        # 用户评分格式: [(song_id, rating), ...]
        
        if not user_ratings:
            print("用户没有评分历史")
            return []
        
        # 只考虑高评分(>=4)的歌曲
        liked_songs = [song_id for song_id, rating in user_ratings if rating >= 4]
        
        if not liked_songs:
            print("用户没有高评分的歌曲")
            return []
        
        # 为每首喜欢的歌曲获取相似歌曲
        similar_songs = []
        for song_id in liked_songs:
            similar = self.recommend_similar_songs(song_id, cosine_sim, n=5)
            similar_songs.extend(similar)
        
        # 去除用户已经评分的歌曲
        rated_songs = [song_id for song_id, _ in user_ratings]
        similar_songs = [s for s in similar_songs if s[0] not in rated_songs]
        
        # 按相似度排序
        similar_songs = sorted(similar_songs, key=lambda x: x[1], reverse=True)
        
        # 去除重复并返回前n个推荐
        seen = set()
        recommendations = []
        for song_id, similarity in similar_songs:
            if song_id not in seen:
                recommendations.append((song_id, similarity))
                seen.add(song_id)
                if len(recommendations) >= n:
                    break
        
        return recommendations


class UserBasedCF:
    """基于用户的协同过滤模型"""
    
    def __init__(self, k=30):
        """
        初始化基于用户的协同过滤模型
        
        参数:
            k: 用于推荐的最近邻数量
        """
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None
        self.ratings_df = None
        self.user_to_idx = None
        self.idx_to_user = None
        self.item_to_idx = None
        self.idx_to_item = None
    
    def fit(self, ratings_df):
        """训练基于用户的协同过滤模型"""
        self.ratings_df = ratings_df
        
        # 创建用户和物品的索引映射
        unique_users = ratings_df['userId'].unique()
        unique_items = ratings_df['songId'].unique()
        
        self.user_to_idx = {user: i for i, user in enumerate(unique_users)}
        self.idx_to_user = {i: user for i, user in enumerate(unique_users)}
        self.item_to_idx = {item: i for i, item in enumerate(unique_items)}
        self.idx_to_item = {i: item for i, item in enumerate(unique_items)}
        
        # 创建用户-物品评分矩阵
        n_users = len(unique_users)
        n_items = len(unique_items)
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in ratings_df.iterrows():
            user_idx = self.user_to_idx[row['userId']]
            item_idx = self.item_to_idx[row['songId']]
            self.user_item_matrix[user_idx, item_idx] = row['rating']
        
        # 计算用户相似度
        self._compute_user_similarity()
        
        print("基于用户的协同过滤模型训练完成")
    
    def _compute_user_similarity(self):
        """计算用户之间的相似度"""
        # 使用余弦相似度
        self.user_similarity = cosine_similarity(self.user_item_matrix)
    
    def recommend(self, user_id, n=10):
        """为用户推荐歌曲"""
        if user_id not in self.user_to_idx:
            print(f"用户 {user_id} 不在训练数据中")
            return []
        
        user_idx = self.user_to_idx[user_id]
        
        # 获取用户已评分的物品
        user_ratings = self.user_item_matrix[user_idx]
        rated_items = np.where(user_ratings > 0)[0]
        rated_items_set = set(rated_items)
        
        # 获取相似用户
        similar_users = np.argsort(self.user_similarity[user_idx])[::-1][1:self.k+1]
        
        # 计算预测评分
        recommendations = {}
        for item_idx in range(self.user_item_matrix.shape[1]):
            if item_idx in rated_items_set:
                continue  # 跳过已经评分的物品
            
            prediction = 0
            total_similarity = 0
            
            for similar_user_idx in similar_users:
                # 如果相似用户评价过这个物品
                if self.user_item_matrix[similar_user_idx, item_idx] > 0:
                    similarity = self.user_similarity[user_idx, similar_user_idx]
                    rating = self.user_item_matrix[similar_user_idx, item_idx]
                    
                    prediction += similarity * rating
                    total_similarity += similarity
            
            if total_similarity > 0:
                prediction /= total_similarity
                recommendations[self.idx_to_item[item_idx]] = prediction
        
        # 按预测评分排序并返回前n个推荐
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n]
        
        return sorted_recommendations

#-----------------------------------------------------------------------------
# 混合推荐系统
#-----------------------------------------------------------------------------

class HybridRecommender:
    """混合推荐系统，结合多种推荐算法"""
    
    def __init__(self, data_loader):
        """
        初始化混合推荐系统
        
        参数:
            data_loader: 数据加载器实例
        """
        self.data_loader = data_loader
        self.svdpp_model = None
        self.ncf_model = None
        self.content_model = None
        self.user_cf_model = None
        
        # 权重配置
        self.weights = {
            'svdpp': 0.3,    # SVD++权重
            'ncf': 0.2,      # 神经协同过滤权重
            'content': 0.3,  # 基于内容权重
            'user_cf': 0.2   # 基于用户的协同过滤权重
        }
    
    def train_all_models(self, save_path='models'):
        """训练所有推荐模型"""
        # 确保存储路径存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        print("\n=== 开始训练所有推荐模型 ===\n")
        
        # 1. 训练SVD++模型
        surprise_data = self.data_loader.prepare_surprise_data()
        self.svdpp_model = SVDppModel()
        self.svdpp_model.train(surprise_data)
        self.svdpp_model.save_model(os.path.join(save_path, 'svdpp_model.pkl'))
        
        # 2. 训练神经协同过滤模型
        (X_train, y_train), (X_test, y_test) = self.data_loader.prepare_neural_cf_data()
        self.ncf_model = NeuralCFModel(
            n_users=self.data_loader.n_users,
            n_items=self.data_loader.n_songs
        )
        self.ncf_model.train((X_train, y_train), (X_test, y_test), epochs=10)
        self.ncf_model.save_model(os.path.join(save_path, 'ncf_model.h5'))
        
        # 3. 训练基于内容的推荐模型
        tfidf_matrix = self.data_loader.prepare_content_data()
        self.content_model = ContentBasedRecommender()
        self.content_model.fit(tfidf_matrix, self.data_loader.songs_df)
        
        # 4. 训练基于用户的协同过滤模型
        self.user_cf_model = UserBasedCF()
        self.user_cf_model.fit(self.data_loader.ratings_df)
        
        print("\n=== 所有推荐模型训练完成 ===\n")
    
    def load_models(self, load_path='models'):
        """加载已训练的模型"""
        print("\n=== 尝试加载已训练的模型 ===\n")
        
        # 加载SVD++模型
        svdpp_path = os.path.join(load_path, 'svdpp_model.pkl')
        self.svdpp_model = SVDppModel()
        svdpp_loaded = self.svdpp_model.load_model(svdpp_path)
        
        # 加载神经协同过滤模型
        ncf_path = os.path.join(load_path, 'ncf_model.h5')
        self.ncf_model = NeuralCFModel(
            n_users=self.data_loader.n_users,
            n_items=self.data_loader.n_songs
        )
        ncf_loaded = self.ncf_model.load_model(ncf_path)
        
        # 初始化基于内容的推荐模型
        tfidf_matrix = self.data_loader.prepare_content_data()
        self.content_model = ContentBasedRecommender()
        self.content_model.fit(tfidf_matrix, self.data_loader.songs_df)
        
        # 初始化基于用户的协同过滤模型
        self.user_cf_model = UserBasedCF()
        self.user_cf_model.fit(self.data_loader.ratings_df)
        
        # 如果任何模型加载失败，则训练所有模型
        if not (svdpp_loaded and ncf_loaded):
            print("一些模型未能加载，将重新训练所有模型")
            self.train_all_models(load_path)
        else:
            print("所有模型加载成功")
    
    def set_weights(self, weights):
        """
        设置混合推荐的权重
        
        参数:
            weights: 字典，包含每个模型的权重
        """
        self.weights = weights
        # 确保权重总和为1
        total = sum(weights.values())
        if total != 1.0:
            for key in weights:
                self.weights[key] = weights[key] / total
    
    def recommend(self, user_id, n=10):
        """
        为用户生成混合推荐
        
        参数:
            user_id: 用户ID
            n: 推荐数量
            
        返回:
            推荐歌曲列表
        """
        # 1. 获取SVD++推荐
        svdpp_recs = []
        if self.svdpp_model and self.svdpp_model.model:
            svdpp_recs = self.svdpp_model.recommend(user_id, n=n*2)
        
        # 2. 获取神经协同过滤推荐
        ncf_recs = []
        if self.ncf_model and self.ncf_model.model:
            idx_to_song = {idx: song_id for song_id, idx in self.data_loader.song_to_idx.items()}
            ncf_recs = self.ncf_model.recommend(
                user_id, 
                self.data_loader.user_to_idx,
                self.data_loader.song_to_idx,
                idx_to_song,
                n=n*2
            )
        
        # 3. 获取基于内容的推荐
        content_recs = []
        if self.content_model:
            # 获取用户评分历史
            user_ratings = []
            if user_id in self.data_loader.ratings_df['userId'].values:
                user_df = self.data_loader.ratings_df[self.data_loader.ratings_df['userId'] == user_id]
                user_ratings = [(row['songId'], row['rating']) for _, row in user_df.iterrows()]
            
            # 如果用户有评分历史，基于内容推荐
            if user_ratings:
                cosine_sim = self.content_model.compute_similarity()
                content_recs = self.content_model.recommend_for_user(user_ratings, cosine_sim, n=n*2)
        
        # 4. 获取基于用户的协同过滤推荐
        user_cf_recs = []
        if self.user_cf_model:
            user_cf_recs = self.user_cf_model.recommend(user_id, n=n*2)
        
        # 5. 融合所有推荐结果
        final_recs = self._fusion_strategy(svdpp_recs, ncf_recs, content_recs, user_cf_recs, n)
        
        return final_recs
    
    def _fusion_strategy(self, svdpp_recs, ncf_recs, content_recs, user_cf_recs, n):
        """
        融合策略，将各种推荐结果组合成最终推荐
        
        参数:
            svdpp_recs: SVD++推荐结果
            ncf_recs: 神经协同过滤推荐结果
            content_recs: 基于内容推荐结果
            user_cf_recs: 基于用户协同过滤推荐结果
            n: 最终推荐数量
            
        返回:
            最终推荐列表
        """
        # 初始化歌曲得分字典
        song_scores = {}
        
        # 处理SVD++推荐
        for song_id, score in svdpp_recs:
            if song_id not in song_scores:
                song_scores[song_id] = 0
            # 归一化评分并加权
            song_scores[song_id] += (score / 5.0) * self.weights['svdpp']
        
        # 处理NCF推荐
        for song_id, score in ncf_recs:
            if song_id not in song_scores:
                song_scores[song_id] = 0
            # 归一化评分并加权
            song_scores[song_id] += (score / 5.0) * self.weights['ncf']
        
        # 处理基于内容推荐
        for song_id, score in content_recs:
            if song_id not in song_scores:
                song_scores[song_id] = 0
            # 相似度已经是0-1范围，直接加权
            song_scores[song_id] += score * self.weights['content']
        
        # 处理基于用户协同过滤推荐
        for song_id, score in user_cf_recs:
            if song_id not in song_scores:
                song_scores[song_id] = 0
            # 归一化评分并加权
            song_scores[song_id] += (score / 5.0) * self.weights['user_cf']
        
        # 按得分排序
        sorted_songs = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前n个推荐
        return sorted_songs[:n]
    
    def recommend_by_emotion(self, user_id, emotion, n=10):
        """
        基于情感状态推荐音乐
        
        参数:
            user_id: 用户ID
            emotion: 情感状态（快乐、悲伤、放松、激动等）
            n: 推荐数量
            
        返回:
            情感相关的推荐歌曲列表
        """
        # 情感到音乐特征的映射
        emotion_features = {
            '快乐': ['流行', '舞曲', '欢快'],
            '悲伤': ['民谣', '抒情', '钢琴'],
            '放松': ['轻音乐', '古典', '爵士'],
            '激动': ['摇滚', '电子', '嘻哈'],
            '专注': ['器乐', '古典', '电子'],
            '浪漫': ['爵士', '民谣', '流行']
        }
        
        # 如果情感不在预定义列表中，使用默认推荐
        if emotion not in emotion_features:
            print(f"未找到情感 '{emotion}' 的特征映射，使用默认推荐")
            return self.recommend(user_id, n)
        
        # 获取情感相关的特征
        features = emotion_features[emotion]
        
        # 获取包含这些特征的歌曲
        matching_songs = []
        for _, row in self.data_loader.songs_df.iterrows():
            # 检查歌曲的描述或流派是否包含情感特征
            song_text = (str(row['genre']) + ' ' + str(row.get('description', ''))).lower()
            matches = [feature.lower() in song_text for feature in features]
            if any(matches):
                matching_songs.append(row['songId'])
        
        # 如果没有匹配的歌曲，使用默认推荐
        if not matching_songs:
            print(f"没有找到匹配情感 '{emotion}' 的歌曲，使用默认推荐")
            return self.recommend(user_id, n)
        
        # 基于匹配的歌曲进行个性化推荐
        # 使用基于内容的推荐模型找到相似歌曲
        cosine_sim = self.content_model.compute_similarity()
        
        # 为每首匹配的歌曲获取相似歌曲
        similar_songs = []
        for song_id in matching_songs[:10]:  # 只使用前10首匹配歌曲
            if song_id in self.content_model.song_indices:
                similar = self.content_model.recommend_similar_songs(song_id, cosine_sim, n=3)
                similar_songs.extend(similar)
        
        # 如果用户有评分历史，考虑用户偏好
        user_ratings = []
        if user_id in self.data_loader.ratings_df['userId'].values:
            user_df = self.data_loader.ratings_df[self.data_loader.ratings_df['userId'] == user_id]
            user_ratings = [(row['songId'], row['rating']) for _, row in user_df.iterrows()]
        
        # 混合结果：情感匹配的歌曲 + 用户偏好
        emotion_recs = self._blend_with_preferences(similar_songs, user_ratings, n)
        
        return emotion_recs
    
    def _blend_with_preferences(self, emotion_matches, user_ratings, n):
        """
        将情感匹配的歌曲与用户偏好混合
        
        参数:
            emotion_matches: 情感匹配的歌曲列表
            user_ratings: 用户评分历史
            n: 推荐数量
            
        返回:
            混合后的推荐列表
        """
        # 如果用户没有评分历史，直接返回情感匹配的歌曲
        if not user_ratings:
            return sorted(emotion_matches, key=lambda x: x[1], reverse=True)[:n]
        
        # 获取用户评分过的歌曲ID
        rated_songs = [song_id for song_id, _ in user_ratings]
        
        # 计算用户对每首歌曲的平均评分
        avg_rating = sum(rating for _, rating in user_ratings) / len(user_ratings)
        
        # 调整情感匹配的歌曲得分
        adjusted_matches = []
        for song_id, similarity in emotion_matches:
            # 排除用户已评分的歌曲
            if song_id in rated_songs:
                continue
            
            # 保持原始相似度得分
            adjusted_score = similarity
            
            # 如果有特定用户偏好，可以在这里加入更复杂的逻辑
            
            adjusted_matches.append((song_id, adjusted_score))
        
        # 按调整后的得分排序并返回前n个
        return sorted(adjusted_matches, key=lambda x: x[1], reverse=True)[:n]
    
    def process_user_critique(self, user_id, recommendations, critique, n=10):
        """
        处理用户反馈，调整推荐结果
        
        参数:
            user_id: 用户ID
            recommendations: 当前推荐列表
            critique: 用户反馈（如"更多摇滚"、"少一点电子乐"）
            n: 返回的推荐数量
            
        返回:
            调整后的推荐列表和解释
        """
        # 解析用户反馈
        critique_type, direction = self._parse_critique(critique)
        
        if not critique_type:
            return recommendations, f"无法理解反馈 '{critique}'，保持原推荐列表。"
        
        # 调整推荐结果
        adjusted_recs = self._adjust_recommendations(
            user_id, recommendations, critique_type, direction, n
        )
        
        # 生成解释
        explanation = self._generate_explanation(adjusted_recs, critique_type, direction)
        
        return adjusted_recs, explanation
    
    def _parse_critique(self, critique):
        """
        解析用户反馈
        
        参数:
            critique: 用户反馈字符串
            
        返回:
            (critique_type, direction) 元组
        """
        # 支持的音乐特征类型
        feature_types = ['摇滚', '流行', '电子', '古典', '爵士', '嘻哈', 
                         '民谣', '乡村', '抒情', '欢快', '舞曲', '轻音乐']
        
        # 方向词
        more_words = ['更多', '多一点', '增加', '喜欢']
        less_words = ['更少', '少一点', '减少', '不喜欢']
        
        critique = critique.lower()
        
        # 查找特征类型
        found_type = None
        for feature in feature_types:
            if feature.lower() in critique:
                found_type = feature
                break
        
        if not found_type:
            return None, None
        
        # 判断方向
        direction = 0  # 默认中性
        for word in more_words:
            if word in critique:
                direction = 1  # 更多
                break
        
        for word in less_words:
            if word in critique:
                direction = -1  # 更少
                break
        
        return found_type, direction
    
    def _adjust_recommendations(self, user_id, recommendations, critique_type, direction, n):
        """
        根据用户反馈调整推荐结果
        
        参数:
            user_id: 用户ID
            recommendations: 当前推荐列表
            critique_type: 反馈类型
            direction: 反馈方向
            n: 返回的推荐数量
            
        返回:
            调整后的推荐列表
        """
        # 如果未找到反馈类型或方向为中性，返回原推荐列表
        if not critique_type or direction == 0:
            return recommendations
        
        # 获取符合条件的歌曲
        matching_songs = []
        for _, row in self.data_loader.songs_df.iterrows():
            song_text = (str(row['genre']) + ' ' + str(row.get('description', ''))).lower()
            if critique_type.lower() in song_text:
                matching_songs.append(row['songId'])
        
        # 如果方向为"更多"，增加符合条件的歌曲
        adjusted_recs = []
        if direction > 0:
            # 1. 保留原推荐中符合条件的歌曲
            for song_id, score in recommendations:
                song_info = self.data_loader.songs_df[self.data_loader.songs_df['songId'] == song_id]
                if not song_info.empty:
                    song_text = (str(song_info['genre'].iloc[0]) + ' ' + 
                               str(song_info.get('description', '').iloc[0])).lower()
                    
                    if critique_type.lower() in song_text:
                        # 增加匹配歌曲的得分
                        adjusted_recs.append((song_id, score * 1.5))
                    else:
                        adjusted_recs.append((song_id, score))
            
            # 2. 添加更多符合条件的歌曲
            # 排除已经在推荐列表中的歌曲
            existing_songs = [song_id for song_id, _ in recommendations]
            new_matches = [song_id for song_id in matching_songs if song_id not in existing_songs]
            
            # 对于新添加的歌曲，给予适当的初始得分
            if new_matches:
                # 获取当前推荐中的最低得分作为基准
                min_score = min([score for _, score in recommendations]) if recommendations else 0.5
                base_score = min_score * 1.2  # 稍高于最低分
                
                # 使用基于内容的推荐为新歌曲计算得分
                cosine_sim = self.content_model.compute_similarity()
                for song_id in new_matches[:20]:  # 最多考虑20首新歌曲
                    if song_id in self.content_model.song_indices:
                        # 计算与用户喜欢的歌曲的平均相似度
                        sim_score = base_score
                        adjusted_recs.append((song_id, sim_score))
        
        # 如果方向为"更少"，减少符合条件的歌曲
        elif direction < 0:
            for song_id, score in recommendations:
                song_info = self.data_loader.songs_df[self.data_loader.songs_df['songId'] == song_id]
                if not song_info.empty:
                    song_text = (str(song_info['genre'].iloc[0]) + ' ' + 
                               str(song_info.get('description', '').iloc[0])).lower()
                    
                    if critique_type.lower() in song_text:
                        # 大幅降低匹配歌曲的得分
                        adjusted_recs.append((song_id, score * 0.3))
                    else:
                        # 适当提高不匹配歌曲的得分
                        adjusted_recs.append((song_id, score * 1.2))
        
        # 排序并返回前n个调整后的推荐
        return sorted(adjusted_recs, key=lambda x: x[1], reverse=True)[:n]
    
    def _generate_explanation(self, recommendations, critique_type, direction):
        """
        为调整后的推荐生成解释
        
        参数:
            recommendations: 调整后的推荐列表
            critique_type: 反馈类型
            direction: 反馈方向
            
        返回:
            解释文本
        """
        if direction > 0:
            return f"已调整推荐，增加了更多{critique_type}类型的音乐。"
        elif direction < 0:
            return f"已调整推荐，减少了{critique_type}类型的音乐。"
        else:
            return "已根据您的反馈调整了推荐列表。"

#-----------------------------------------------------------------------------
# 模型评估和可视化
#-----------------------------------------------------------------------------

def evaluate_recommendations(recommendations, test_data):
    """
    评估推荐结果
    
    参数:
        recommendations: 推荐结果列表，格式为[(song_id, score), ...]
        test_data: 测试数据，包含用户实际评分
        
    返回:
        评估指标
    """
    # 从推荐结果中提取歌曲ID
    recommended_songs = [song_id for song_id, _ in recommendations]
    
    # 从测试数据中获取用户实际喜欢的歌曲（评分>=4）
    actual_liked = set(test_data[test_data['rating'] >= 4]['songId'].values)
    
    # 计算准确率
    hits = len(set(recommended_songs) & actual_liked)
    precision = hits / len(recommended_songs) if recommended_songs else 0
    
    # 计算召回率
    recall = hits / len(actual_liked) if actual_liked else 0
    
    # 计算F1分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算命中率
    hit_rate = 1 if hits > 0 else 0
    
    # 计算NDCG
    ndcg = compute_ndcg(recommendations, actual_liked)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'hit_rate': hit_rate,
        'ndcg': ndcg
    }

def compute_ndcg(recommendations, actual_liked, k=10):
    """
    计算NDCG@k
    
    参数:
        recommendations: 推荐结果列表，格式为[(song_id, score), ...]
        actual_liked: 用户实际喜欢的歌曲集合
        k: 推荐列表长度
        
    返回:
        NDCG@k值
    """
    # 获取推荐列表中的前k个歌曲
    recommended_songs = [song_id for song_id, _ in recommendations[:k]]
    
    # 创建相关性列表（1表示相关，0表示不相关）
    relevance = [1 if song_id in actual_liked else 0 for song_id in recommended_songs]
    
    # 计算DCG
    dcg = 0
    for i, rel in enumerate(relevance):
        dcg += rel / np.log2(i + 2)  # i+2因为i从0开始，log底数为2
    
    # 计算理想DCG
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = 0
    for i, rel in enumerate(ideal_relevance):
        idcg += rel / np.log2(i + 2)
    
    # 计算NDCG
    ndcg = dcg / idcg if idcg > 0 else 0
    
    return ndcg

def plot_metrics(metrics, title):
    """
    绘制评估指标图表
    
    参数:
        metrics: 评估指标字典
        title: 图表标题
    """
    plt.figure(figsize=(10, 6))
    
    names = list(metrics.keys())
    values = list(metrics.values())
    
    plt.bar(names, values, color='skyblue')
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel('评估指标')
    plt.ylabel('分数')
    
    # 在柱状图上添加具体数值
    for i, v in enumerate(values):
        plt.text(i, v + 0.05, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.show()

def compare_models(evaluation_results):
    """
    比较不同推荐模型的性能
    
    参数:
        evaluation_results: 不同模型的评估结果字典
    """
    metrics = ['precision', 'recall', 'f1', 'ndcg']
    models = list(evaluation_results.keys())
    
    # 为每个指标创建柱状图
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        values = [evaluation_results[model][metric] for model in models]
        
        plt.bar(models, values, color=['blue', 'green', 'orange', 'red', 'purple'])
        plt.ylim(0, 1)
        plt.title(f'不同模型的{metric}比较')
        plt.xlabel('模型')
        plt.ylabel(f'{metric}分数')
        
        # 在柱状图上添加具体数值
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.show()

#-----------------------------------------------------------------------------
# 主函数
#-----------------------------------------------------------------------------

def main():
    """主函数"""
    print("\n===== 深度推荐音乐系统 - 训练脚本 =====\n")
    
    # 处理命令行参数
    import argparse
    parser = argparse.ArgumentParser(description='训练混合推荐系统')
    parser.add_argument('--ratings', type=str, help='评分数据路径')
    parser.add_argument('--songs', type=str, help='歌曲数据路径')
    parser.add_argument('--sample', type=int, default=None, help='样本大小')
    parser.add_argument('--models', type=str, default='models', help='模型保存路径')
    parser.add_argument('--load', action='store_true', help='加载已有模型')
    parser.add_argument('--eval', action='store_true', help='评估模型')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    args = parser.parse_args()
    
    # 创建数据加载器
    data_loader = DataLoader(
        ratings_path=args.ratings,
        songs_path=args.songs,
        sample_size=args.sample
    )
    
    # 加载数据
    data_loader.load_data()
    
    # 创建混合推荐系统
    recommender = HybridRecommender(data_loader)
    
    # 如果指定加载已有模型
    if args.load:
        recommender.load_models(args.models)
    else:
    # 训练所有模型
        recommender.train_all_models(args.models)
    
    # 如果指定评估模型
    if args.eval:
        evaluate_models(recommender, data_loader)
    
    # 如果指定运行演示
    if args.demo:
        run_demo(recommender, data_loader)
    
    print("\n===== 训练脚本执行完毕 =====\n")

def evaluate_models(recommender, data_loader):
    """评估所有推荐模型"""
    print("\n=== 评估所有推荐模型 ===\n")
    
    # 分割训练集和测试集
    train_df, test_df = train_test_split(
        data_loader.ratings_df, test_size=0.2, random_state=42
    )
    
    # 重新设置数据加载器的评分数据为训练集
    temp_df = data_loader.ratings_df.copy()
    data_loader.ratings_df = train_df
    
    # 获取测试用户
    test_users = test_df['userId'].unique()
    
    # 评估结果
    evaluation_results = {}
    
    # 1. 评估SVD++模型
    print("评估SVD++模型...")
    svdpp_metrics_total = {'precision': 0, 'recall': 0, 'f1': 0, 'hit_rate': 0, 'ndcg': 0}
    user_count = 0
    
    for user_id in test_users[:20]:  # 为节省时间，只评估前20个用户
        # 获取用户在测试集中的评分
        user_test = test_df[test_df['userId'] == user_id]
        
        if len(user_test) == 0:
            continue
            
        # 获取SVD++推荐
        svdpp_recs = recommender.svdpp_model.recommend(user_id, n=10)
        
        if not svdpp_recs:
            continue
        
        # 评估推荐结果
        metrics = evaluate_recommendations(svdpp_recs, user_test)
        
        # 累计指标
        for k in svdpp_metrics_total:
            svdpp_metrics_total[k] += metrics[k]
        
        user_count += 1
    
    # 计算平均值
    if user_count > 0:
        for k in svdpp_metrics_total:
            svdpp_metrics_total[k] /= user_count
    
    evaluation_results['SVD++'] = svdpp_metrics_total
    
    # 2. 评估神经协同过滤模型
    print("评估神经协同过滤模型...")
    ncf_metrics_total = {'precision': 0, 'recall': 0, 'f1': 0, 'hit_rate': 0, 'ndcg': 0}
    user_count = 0
    
    for user_id in test_users[:20]:
        # 获取用户在测试集中的评分
        user_test = test_df[test_df['userId'] == user_id]
        
        if len(user_test) == 0:
            continue
            
        # 获取NCF推荐
        idx_to_song = {idx: song_id for song_id, idx in data_loader.song_to_idx.items()}
        ncf_recs = recommender.ncf_model.recommend(
            user_id, 
            data_loader.user_to_idx,
            data_loader.song_to_idx,
            idx_to_song,
            n=10
        )
        
        if not ncf_recs:
            continue
        
        # 评估推荐结果
        metrics = evaluate_recommendations(ncf_recs, user_test)
        
        # 累计指标
        for k in ncf_metrics_total:
            ncf_metrics_total[k] += metrics[k]
        
        user_count += 1
    
    # 计算平均值
    if user_count > 0:
        for k in ncf_metrics_total:
            ncf_metrics_total[k] /= user_count
    
    evaluation_results['NCF'] = ncf_metrics_total
    
    # 3. 评估混合推荐模型
    print("评估混合推荐模型...")
    hybrid_metrics_total = {'precision': 0, 'recall': 0, 'f1': 0, 'hit_rate': 0, 'ndcg': 0}
    user_count = 0
    
    for user_id in test_users[:20]:
        # 获取用户在测试集中的评分
        user_test = test_df[test_df['userId'] == user_id]
        
        if len(user_test) == 0:
            continue
            
        # 获取混合推荐
        hybrid_recs = recommender.recommend(user_id, n=10)
        
        if not hybrid_recs:
            continue
        
        # 评估推荐结果
        metrics = evaluate_recommendations(hybrid_recs, user_test)
        
        # 累计指标
        for k in hybrid_metrics_total:
            hybrid_metrics_total[k] += metrics[k]
        
        user_count += 1
    
    # 计算平均值
    if user_count > 0:
        for k in hybrid_metrics_total:
            hybrid_metrics_total[k] /= user_count
    
    evaluation_results['Hybrid'] = hybrid_metrics_total
    
    # 恢复数据加载器的评分数据
    data_loader.ratings_df = temp_df
    
    # 比较不同模型的性能
    compare_models(evaluation_results)
    
    # 打印评估结果
    print("\n=== 评估结果 ===\n")
    for model, metrics in evaluation_results.items():
        print(f"{model}模型:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()
    
    return evaluation_results

def run_demo(recommender, data_loader):
    """运行演示"""
    print("\n=== 运行推荐系统演示 ===\n")
    
    # 随机选择一个用户
    user_id = np.random.choice(data_loader.user_ids)
    print(f"为用户 {user_id} 生成推荐...")
    
    # 获取用户评分历史
    user_ratings = data_loader.ratings_df[data_loader.ratings_df['userId'] == user_id]
    print(f"用户共有 {len(user_ratings)} 条评分记录")
    
    # 展示用户部分评分历史
    if len(user_ratings) > 0:
        print("\n用户评分历史(部分):")
        for i, (_, row) in enumerate(user_ratings.sample(min(5, len(user_ratings))).iterrows()):
            song_info = data_loader.songs_df[data_loader.songs_df['songId'] == row['songId']]
            if not song_info.empty:
                song_title = song_info['title'].iloc[0]
                song_artist = song_info['artist'].iloc[0]
                print(f"  {i+1}. {song_title} - {song_artist}: {row['rating']}分")
    
    # 1. SVD++推荐
    print("\nSVD++推荐结果:")
    svdpp_recs = recommender.svdpp_model.recommend(user_id, n=5)
    for i, (song_id, score) in enumerate(svdpp_recs):
        song_info = data_loader.songs_df[data_loader.songs_df['songId'] == song_id]
        if not song_info.empty:
            song_title = song_info['title'].iloc[0]
            song_artist = song_info['artist'].iloc[0]
            print(f"  {i+1}. {song_title} - {song_artist}: {score:.2f}分")
    
    # 2. 神经协同过滤推荐
    print("\n神经协同过滤推荐结果:")
    idx_to_song = {idx: song_id for song_id, idx in data_loader.song_to_idx.items()}
    ncf_recs = recommender.ncf_model.recommend(
        user_id, 
        data_loader.user_to_idx,
        data_loader.song_to_idx,
        idx_to_song,
        n=5
    )
    for i, (song_id, score) in enumerate(ncf_recs):
        song_info = data_loader.songs_df[data_loader.songs_df['songId'] == song_id]
        if not song_info.empty:
            song_title = song_info['title'].iloc[0]
            song_artist = song_info['artist'].iloc[0]
            print(f"  {i+1}. {song_title} - {song_artist}: {score:.2f}分")
    
    # 3. 混合推荐
    print("\n混合推荐结果:")
    hybrid_recs = recommender.recommend(user_id, n=10)
    for i, (song_id, score) in enumerate(hybrid_recs):
        song_info = data_loader.songs_df[data_loader.songs_df['songId'] == song_id]
        if not song_info.empty:
            song_title = song_info['title'].iloc[0]
            song_artist = song_info['artist'].iloc[0]
            print(f"  {i+1}. {song_title} - {song_artist}: {score:.2f}分")
    
    # 4. 情感推荐示例
    print("\n情感推荐示例(快乐):")
    emotion_recs = recommender.recommend_by_emotion(user_id, '快乐', n=5)
    for i, (song_id, score) in enumerate(emotion_recs):
        song_info = data_loader.songs_df[data_loader.songs_df['songId'] == song_id]
        if not song_info.empty:
            song_title = song_info['title'].iloc[0]
            song_artist = song_info['artist'].iloc[0]
            print(f"  {i+1}. {song_title} - {song_artist}: {score:.2f}分")
    
    # 5. 用户反馈调整示例
    print("\n基于用户反馈调整推荐:")
    adjusted_recs, explanation = recommender.process_user_critique(
        user_id, hybrid_recs, "更多摇滚", n=5
    )
    print(f"  {explanation}")
    for i, (song_id, score) in enumerate(adjusted_recs):
        song_info = data_loader.songs_df[data_loader.songs_df['songId'] == song_id]
        if not song_info.empty:
            song_title = song_info['title'].iloc[0]
            song_artist = song_info['artist'].iloc[0]
            print(f"  {i+1}. {song_title} - {song_artist}: {score:.2f}分")

# Google Colab示例代码
def colab_example():
    """在Google Colab中运行的示例代码"""
    if not IN_COLAB:
        print("此函数仅在Google Colab环境中有效")
        return
    
    # 示例代码
    print("""
# 1. 加载必要的库
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from surprise import SVD, SVDpp, Dataset, Reader, accuracy
import matplotlib.pyplot as plt
import os
import pickle
import time

# 2. 运行训练脚本
# 使用示例数据
!python colab_train_recommender.py --demo

# 3. 使用自己的数据
# 上传评分数据和歌曲数据
from google.colab import files
uploaded = files.upload()  # 上传ratings.csv和songs.csv

# 训练模型
!python colab_train_recommender.py --ratings ratings.csv --songs songs.csv --models my_models --eval --demo

# 4. 调整混合权重
# 在Python代码中修改权重
recommender.set_weights({
    'svdpp': 0.3,
    'ncf': 0.3,
    'content': 0.2,
    'user_cf': 0.2
})

# 5. 情感推荐示例
emotion_recs = recommender.recommend_by_emotion(user_id, '悲伤', n=10)

# 6. 用户反馈处理示例
initial_recs = recommender.recommend(user_id, n=10)
adjusted_recs, explanation = recommender.process_user_critique(
    user_id, initial_recs, "更多古典", n=10
)
    """)

# 运行主函数
if __name__ == "__main__":
    main() 