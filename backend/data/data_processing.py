import pandas as pd
import numpy as np
import h5py
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('data_processing')

# 数据路径
LASTFM_DATA_PATH = 'data/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
USER_PROFILE_PATH = 'data/lastfm-dataset-1K/userid-profile.tsv'
PROCESSED_DATA_DIR = 'processed_data'

# 创建处理数据的目录
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def load_and_process_lastfm_data(sample_size=100000):
    """加载并处理Last.fm数据集"""
    print("正在加载Last.fm数据...")
    
    # 加载用户播放记录
    cols = ['user_id', 'timestamp', 'artist_id', 'artist_name', 'track_id', 'track_name']
    df = pd.read_csv(LASTFM_DATA_PATH, sep='\t', names=cols, nrows=sample_size)
    
    # 加载用户资料
    user_profiles = pd.read_csv(USER_PROFILE_PATH, sep='\t', 
                                names=['user_id', 'gender', 'age', 'country', 'signup_date'])
    
    # 数据清洗
    df = df.dropna(subset=['user_id', 'track_id', 'artist_name', 'track_name'])
    
    # 创建用户-歌曲矩阵（用户对每首歌的播放次数）
    user_song_plays = df.groupby(['user_id', 'track_id']).size().reset_index(name='plays')
    
    # 将播放次数归一化为1-5评分
    min_plays = user_song_plays['plays'].min()
    max_plays = user_song_plays['plays'].max()
    user_song_plays['rating'] = 1 + 4 * (user_song_plays['plays'] - min_plays) / (max_plays - min_plays)
    
    # 创建歌曲元数据表
    songs_metadata = df[['track_id', 'track_name', 'artist_id', 'artist_name']].drop_duplicates()
    
    # 保存处理后的数据
    user_song_plays.to_pickle(f'{PROCESSED_DATA_DIR}/user_song_plays.pkl')
    songs_metadata.to_pickle(f'{PROCESSED_DATA_DIR}/songs_metadata.pkl')
    user_profiles.to_pickle(f'{PROCESSED_DATA_DIR}/user_profiles.pkl')
    
    print(f"数据处理完成，用户数: {user_song_plays['user_id'].nunique()}, 歌曲数: {songs_metadata.shape[0]}")
    
    return user_song_plays, songs_metadata, user_profiles

def create_content_features(songs_metadata):
    """为基于内容的推荐创建特征向量"""
    print("正在生成歌曲内容特征...")
    
    # 将艺术家名称和歌曲名称组合为文本特征
    songs_metadata['text_features'] = songs_metadata['artist_name'] + ' ' + songs_metadata['track_name']
    
    # 使用TF-IDF向量化文本特征
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(songs_metadata['text_features'])
    
    # 计算歌曲间的相似度矩阵
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # 保存特征
    with open(f'{PROCESSED_DATA_DIR}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    
    np.save(f'{PROCESSED_DATA_DIR}/song_similarity_matrix.npy', sim_matrix)
    
    return tfidf, sim_matrix

def process_data(data_dir='data', output_dir='processed_data', use_msd=False, sample_size=None):
    """主数据处理函数
    
    参数:
        data_dir: 原始数据目录
        output_dir: 处理后数据输出目录
        use_msd: 是否使用百万歌曲数据集(MSD)
        sample_size: 采样大小，如果为None则处理全部数据
    """
    logger.info(f"开始处理数据，使用MSD: {use_msd}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    if use_msd:
        # 处理百万歌曲数据集
        logger.info("处理百万歌曲数据集(MSD)")
        metadata_file = os.path.join(data_dir, 'msd_summary_file.h5')
        triplets_file = os.path.join(data_dir, 'train_triplets.txt')
        
        if not os.path.exists(metadata_file):
            logger.error(f"元数据文件不存在: {metadata_file}")
            return False
        
        if not os.path.exists(triplets_file):
            logger.error(f"用户播放记录文件不存在: {triplets_file}")
            return False
        
        process_msd_dataset(triplets_file, metadata_file, output_dir, sample_size)
    else:
        # 处理原始数据集（非MSD）- 保留原有的处理逻辑
        logger.info("处理原始数据集（非MSD）")
        # 这里保留您原来的数据处理逻辑
        # process_original_dataset(data_dir, output_dir, sample_size)
        pass
    
    logger.info("数据处理完成")
    return True

def process_msd_dataset(triplets_file, metadata_file, output_dir='processed_data', sample_size=None):
    """处理百万歌曲数据集
    
    参数:
        triplets_file: 用户-歌曲-播放次数记录文件
        metadata_file: H5格式的元数据文件
        output_dir: 输出目录
        sample_size: 采样大小
    """
    logger.info("开始处理MSD数据集")
    
    # 1. 处理元数据文件
    logger.info("处理MSD元数据")
    meta_data = process_msd_metadata(metadata_file, sample_size)
    meta_data.to_pickle(os.path.join(output_dir, "songs_metadata.pkl"))
    logger.info(f"元数据处理完成，共 {len(meta_data)} 首歌曲")
    
    # 2. 处理用户播放记录
    logger.info("处理用户播放记录")
    user_song_plays = process_msd_triplets(triplets_file, meta_data, sample_size)
    user_song_plays.to_pickle(os.path.join(output_dir, "user_song_plays.pkl"))
    logger.info(f"用户播放记录处理完成，共 {len(user_song_plays)} 条记录")
    
    # 3. 计算歌曲相似度矩阵
    logger.info("计算歌曲相似度矩阵")
    song_sim_matrix = compute_song_similarity(meta_data)
    np.save(os.path.join(output_dir, "song_similarity_matrix.npy"), song_sim_matrix)
    logger.info("歌曲相似度矩阵计算完成")
    
    # 4. 创建ID映射
    create_id_mappings(meta_data, user_song_plays, output_dir)
    
    return True

def process_msd_metadata(metadata_file, sample_size=None):
    """处理H5格式的元数据文件
    
    参数:
        metadata_file: H5格式的元数据文件路径
        sample_size: 采样大小，如果为None则处理全部数据
    
    返回:
        DataFrame 包含歌曲元数据
    """
    logger.info(f"从文件加载元数据: {metadata_file}")
    
    with h5py.File(metadata_file, "r") as f:
        # 根据MSD文件结构获取元数据
        # 注意：实际的结构可能需要根据文件内容调整
        a_group_key = list(f.keys())[0]  # 通常是'metadata'或'analysis'
        group = f[a_group_key]
        meta_data = group[list(group.keys())[0]][()]
    
    # 转换为DataFrame
    meta_data = pd.DataFrame(meta_data)
    
    # 解码字符串类型列
    for column in meta_data.select_dtypes(['object']):
        try:
            meta_data[column] = meta_data[column].str.decode('utf-8')
        except:
            logger.warning(f"列 {column} 解码失败，保持原状")
    
    # 仅保留需要的列
    keep_columns = ['song_id', 'title', 'artist_name', 'artist_id', 'year', 'duration']
    keep_columns = [col for col in keep_columns if col in meta_data.columns]
    meta_data = meta_data[keep_columns]
    
    # 采样
    if sample_size and len(meta_data) > sample_size:
        logger.info(f"对元数据进行采样: {sample_size}")
        meta_data = meta_data.sample(sample_size, random_state=42)
    
    return meta_data

def process_msd_triplets(triplets_file, meta_data, sample_size=None):
    """处理用户播放记录文件
    
    参数:
        triplets_file: 用户-歌曲-播放次数记录文件路径
        meta_data: 包含有效歌曲ID的DataFrame
        sample_size: 采样大小，如果为None则处理全部数据
    
    返回:
        DataFrame 包含用户-歌曲-评分数据
    """
    logger.info(f"从文件加载用户播放记录: {triplets_file}")
    
    # 获取有效歌曲ID
    valid_song_ids = set(meta_data['song_id'])
    logger.info(f"元数据中共有 {len(valid_song_ids)} 个有效歌曲ID")
    
    if sample_size:
        # 采样处理
        logger.info(f"对用户播放记录进行采样: {sample_size}")
        user_song_plays = pd.read_csv(
            triplets_file, 
            sep='\t', 
            names=['user_id', 'song_id', 'plays'],
            nrows=sample_size*10  # 多读取一些，因为后面会过滤
        )
        
        # 过滤只保留元数据中存在的歌曲
        user_song_plays = user_song_plays[user_song_plays['song_id'].isin(valid_song_ids)]
        
        if len(user_song_plays) > sample_size:
            user_song_plays = user_song_plays.sample(sample_size, random_state=42)
    else:
        # 分块处理完整数据
        logger.info("分块处理完整用户播放记录")
        chunks = pd.read_csv(
            triplets_file, 
            sep='\t', 
            names=['user_id', 'song_id', 'plays'],
            chunksize=1000000  # 每次处理100万行
        )
        
        all_data = []
        for i, chunk in enumerate(chunks):
            logger.info(f"处理第 {i+1} 个数据块")
            filtered_chunk = chunk[chunk['song_id'].isin(valid_song_ids)]
            all_data.append(filtered_chunk)
            
            # 如果已经达到采样大小，就不再继续处理
            if sample_size and sum(len(df) for df in all_data) >= sample_size:
                break
        
        user_song_plays = pd.concat(all_data)
        
        if sample_size and len(user_song_plays) > sample_size:
            user_song_plays = user_song_plays.sample(sample_size, random_state=42)
    
    # 将播放次数转换为评分(1-5)
    logger.info("将播放次数转换为评分")
    user_song_plays['rating'] = pd.qcut(
        user_song_plays['plays'], 
        q=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
        labels=[1, 2, 3, 4, 5]
    ).astype(int)
    
    return user_song_plays

def compute_song_similarity(meta_data):
    """计算歌曲之间的相似度矩阵
    
    参数:
        meta_data: 包含歌曲元数据的DataFrame
    
    返回:
        numpy数组 歌曲相似度矩阵
    """
    logger.info("计算歌曲相似度")
    
    # 组合特征
    meta_data['text_features'] = meta_data['artist_name'] + ' ' + meta_data['title']
    if 'year' in meta_data.columns:
        meta_data['text_features'] += ' ' + meta_data['year'].astype(str)
    
    # TF-IDF向量化
    logger.info("使用TF-IDF向量化")
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(meta_data['text_features'])
    
    # 计算余弦相似度
    logger.info("计算余弦相似度")
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    return sim_matrix

def create_id_mappings(meta_data, user_song_plays, output_dir):
    """创建ID映射并保存
    
    参数:
        meta_data: 包含歌曲元数据的DataFrame
        user_song_plays: 包含用户播放记录的DataFrame
        output_dir: 输出目录
    """
    logger.info("创建ID映射")
    
    # 用户ID映射
    unique_users = user_song_plays['user_id'].unique()
    user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    # 歌曲ID映射
    unique_songs = meta_data['song_id'].unique()
    song_mapping = {song_id: idx for idx, song_id in enumerate(unique_songs)}
    
    # 反向映射
    reverse_user_mapping = {idx: user_id for user_id, idx in user_mapping.items()}
    reverse_song_mapping = {idx: song_id for song_id, idx in song_mapping.items()}
    
    # 保存映射
    mappings = {
        'user_mapping': user_mapping,
        'song_mapping': song_mapping,
        'reverse_user_mapping': reverse_user_mapping,
        'reverse_song_mapping': reverse_song_mapping
    }
    
    with open(os.path.join(output_dir, 'id_mappings.pkl'), 'wb') as f:
        pickle.dump(mappings, f)
    
    logger.info(f"ID映射已保存，用户数: {len(user_mapping)}，歌曲数: {len(song_mapping)}")

# 命令行入口
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='处理音乐推荐系统数据')
    parser.add_argument('--data_dir', type=str, default='data', help='原始数据目录')
    parser.add_argument('--output_dir', type=str, default='processed_data', help='处理后数据输出目录')
    parser.add_argument('--use_msd', action='store_true', help='是否使用百万歌曲数据集')
    parser.add_argument('--sample', type=int, default=None, help='采样大小，为None则处理全部数据')
    
    args = parser.parse_args()
    
    process_data(args.data_dir, args.output_dir, args.use_msd, args.sample) 