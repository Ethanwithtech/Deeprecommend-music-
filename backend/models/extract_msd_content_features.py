#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从MSD数据集提取内容特征

提取Million Song Dataset中的音频特征，用于内容特征推荐
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def extract_audio_features(h5_file, song_id=None):
    """
    从h5文件中提取音频特征
    
    参数:
        h5_file: h5文件路径
        song_id: 指定歌曲ID (可选)
        
    返回:
        音频特征字典
    """
    try:
        with h5py.File(h5_file, 'r') as f:
            # 检查文件结构
            if 'analysis' not in f:
                logger.warning(f"文件 {h5_file} 中没有analysis组")
                return {}
            
            # 获取歌曲ID
            if song_id is None and 'metadata' in f and 'songs' in f['metadata']:
                try:
                    song_id_raw = f['metadata']['songs'][0]['song_id']
                    if hasattr(song_id_raw, 'decode'):
                        song_id = song_id_raw.decode('utf-8')
                    else:
                        song_id = str(song_id_raw)
                except:
                    song_id = "unknown"
            
            # 提取基本音频特征
            features = {'song_id': song_id}
            
            # 1. 节奏特征
            try:
                if 'beats_start' in f['analysis']:
                    beats = f['analysis']['beats_start'][0]
                    if len(beats) > 1:
                        # 计算节拍间隔
                        beat_intervals = np.diff(beats)
                        features['beat_intervals_mean'] = float(np.mean(beat_intervals))
                        features['beat_intervals_std'] = float(np.std(beat_intervals))
                        features['beat_regularity'] = float(features['beat_intervals_std'] / features['beat_intervals_mean'])
                
                if 'tempo' in f['analysis']['songs'][0]:
                    features['tempo'] = float(f['analysis']['songs'][0]['tempo'])
            except Exception as e:
                logger.debug(f"提取节奏特征时出错: {str(e)}")
            
            # 2. 音高特征
            try:
                if 'segments_pitches' in f['analysis']:
                    pitches = f['analysis']['segments_pitches'][0]
                    if len(pitches) > 0:
                        # 计算音高特征统计量
                        pitch_mean = np.mean(pitches, axis=0)
                        pitch_std = np.std(pitches, axis=0)
                        
                        # 添加12个音高特征
                        for i in range(12):
                            features[f'pitch_{i}_mean'] = float(pitch_mean[i])
                            features[f'pitch_{i}_std'] = float(pitch_std[i])
                        
                        # 计算主要音高
                        features['dominant_pitch'] = int(np.argmax(pitch_mean))
                        
                        # 计算音高熵 (表示音高分布的均匀性)
                        normalized_pitch = pitch_mean / np.sum(pitch_mean)
                        entropy = -np.sum(normalized_pitch * np.log2(normalized_pitch + 1e-10))
                        features['pitch_entropy'] = float(entropy)
            except Exception as e:
                logger.debug(f"提取音高特征时出错: {str(e)}")
            
            # 3. 音色特征
            try:
                if 'segments_timbre' in f['analysis']:
                    timbre = f['analysis']['segments_timbre'][0]
                    if len(timbre) > 0:
                        # 计算音色特征统计量
                        timbre_mean = np.mean(timbre, axis=0)
                        timbre_std = np.std(timbre, axis=0)
                        
                        # 添加12个音色特征
                        for i in range(min(12, len(timbre_mean))):
                            features[f'timbre_{i}_mean'] = float(timbre_mean[i])
                            features[f'timbre_{i}_std'] = float(timbre_std[i])
                        
                        # 音色特征解释
                        # timbre_0: 响度
                        # timbre_1: 亮度
                        # timbre_2: 陡峭度
                        features['loudness'] = float(timbre_mean[0]) if len(timbre_mean) > 0 else 0
                        features['brightness'] = float(timbre_mean[1]) if len(timbre_mean) > 1 else 0
                        features['attack'] = float(timbre_mean[2]) if len(timbre_mean) > 2 else 0
            except Exception as e:
                logger.debug(f"提取音色特征时出错: {str(e)}")
            
            # 4. 结构特征
            try:
                if 'sections_start' in f['analysis']:
                    sections = f['analysis']['sections_start'][0]
                    features['num_sections'] = len(sections)
                    
                    if len(sections) > 1:
                        # 计算段落长度
                        section_lengths = np.diff(sections)
                        features['section_length_mean'] = float(np.mean(section_lengths))
                        features['section_length_std'] = float(np.std(section_lengths))
            except Exception as e:
                logger.debug(f"提取结构特征时出错: {str(e)}")
            
            # 5. 响度特征
            try:
                if 'segments_loudness_max' in f['analysis']:
                    loudness_max = f['analysis']['segments_loudness_max'][0]
                    if len(loudness_max) > 0:
                        features['loudness_max_mean'] = float(np.mean(loudness_max))
                        features['loudness_max_std'] = float(np.std(loudness_max))
                        features['loudness_range'] = float(np.max(loudness_max) - np.min(loudness_max))
                
                if 'loudness' in f['analysis']['songs'][0]:
                    features['overall_loudness'] = float(f['analysis']['songs'][0]['loudness'])
            except Exception as e:
                logger.debug(f"提取响度特征时出错: {str(e)}")
            
            # 6. 调性特征
            try:
                if 'key' in f['analysis']['songs'][0]:
                    features['key'] = int(f['analysis']['songs'][0]['key'])
                if 'mode' in f['analysis']['songs'][0]:
                    features['mode'] = int(f['analysis']['songs'][0]['mode'])
            except Exception as e:
                logger.debug(f"提取调性特征时出错: {str(e)}")
            
            # 7. 高级特征
            try:
                # 计算能量特征 (基于响度和音色)
                if 'loudness_max_mean' in features and 'timbre_0_mean' in features:
                    # 归一化响度 (通常是负值，越大表示越响)
                    norm_loudness = (features['loudness_max_mean'] + 60) / 60  # 假设响度范围在-60到0之间
                    norm_loudness = max(0, min(1, norm_loudness))  # 限制在0-1范围内
                    
                    # 结合音色特征计算能量
                    features['energy'] = float(norm_loudness * (1 + abs(features['timbre_0_mean']) / 100))
                
                # 计算舞蹈性特征 (基于节奏规律性和能量)
                if 'beat_regularity' in features and 'energy' in features:
                    # 节奏规律性越高，舞蹈性越强
                    rhythm_factor = 1 - min(1, features['beat_regularity'])
                    features['danceability'] = float(rhythm_factor * features['energy'])
                
                # 计算情感价值 (valence) - 基于音高和音色特征
                if 'pitch_entropy' in features and 'mode' in features:
                    # 大调通常更积极，小调更消极
                    mode_factor = 0.7 if features['mode'] == 1 else 0.3
                    
                    # 音高熵高表示更复杂的和声，可能与情感复杂性相关
                    entropy_norm = min(1, features['pitch_entropy'] / 3.5)  # 归一化熵值
                    
                    # 结合计算情感价值
                    features['valence'] = float(mode_factor * (1 - entropy_norm) + 0.5 * entropy_norm)
                
                # 计算原声性 (acousticness)
                if 'timbre_1_mean' in features and 'timbre_2_mean' in features:
                    # 亮度低、陡峭度低的音色通常更原声
                    brightness_factor = max(0, 1 - abs(features['timbre_1_mean']) / 100)
                    attack_factor = max(0, 1 - abs(features['timbre_2_mean']) / 100)
                    features['acousticness'] = float((brightness_factor + attack_factor) / 2)
            except Exception as e:
                logger.debug(f"计算高级特征时出错: {str(e)}")
            
            return features
    
    except Exception as e:
        logger.error(f"处理文件 {h5_file} 时出错: {str(e)}")
        return {}

def process_msd_directory(msd_dir, output_file, max_files=None):
    """
    处理MSD目录中的所有h5文件，提取音频特征
    
    参数:
        msd_dir: MSD数据目录
        output_file: 输出文件路径
        max_files: 最大处理文件数
    """
    logger.info(f"处理MSD目录: {msd_dir}")
    
    # 查找所有h5文件
    h5_files = []
    for root, dirs, files in os.walk(msd_dir):
        for file in files:
            if file.endswith('.h5'):
                h5_files.append(os.path.join(root, file))
    
    logger.info(f"找到 {len(h5_files)} 个h5文件")
    
    # 限制处理文件数
    if max_files and len(h5_files) > max_files:
        h5_files = h5_files[:max_files]
        logger.info(f"限制处理 {max_files} 个文件")
    
    # 提取特征
    all_features = []
    for i, h5_file in enumerate(h5_files):
        if i % 100 == 0:
            logger.info(f"处理文件 {i+1}/{len(h5_files)}: {h5_file}")
        
        features = extract_audio_features(h5_file)
        if features:
            all_features.append(features)
    
    # 转换为DataFrame
    features_df = pd.DataFrame(all_features)
    logger.info(f"提取了 {len(features_df)} 首歌曲的特征")
    
    # 保存特征
    features_df.to_csv(output_file, index=False)
    logger.info(f"特征已保存到: {output_file}")
    
    return features_df

def normalize_features(features_df, output_file=None):
    """
    标准化特征
    
    参数:
        features_df: 特征DataFrame
        output_file: 输出文件路径 (可选)
    
    返回:
        标准化后的特征DataFrame
    """
    logger.info("标准化特征...")
    
    # 分离ID列和特征列
    id_col = features_df['song_id']
    feature_cols = features_df.drop(columns=['song_id'])
    
    # 找出数值列
    numeric_cols = feature_cols.select_dtypes(include=[np.number]).columns
    
    # 标准化数值特征
    scaler = StandardScaler()
    normalized_features = pd.DataFrame(
        scaler.fit_transform(feature_cols[numeric_cols]),
        columns=numeric_cols
    )
    
    # 重新添加ID列
    normalized_features.insert(0, 'song_id', id_col)
    
    # 保存标准化特征
    if output_file:
        normalized_features.to_csv(output_file, index=False)
        logger.info(f"标准化特征已保存到: {output_file}")
    
    return normalized_features

def reduce_dimensions(features_df, n_components=20, output_file=None):
    """
    降维特征
    
    参数:
        features_df: 特征DataFrame
        n_components: 降维后的维度
        output_file: 输出文件路径 (可选)
    
    返回:
        降维后的特征DataFrame
    """
    logger.info(f"降维特征到 {n_components} 维...")
    
    # 分离ID列和特征列
    id_col = features_df['song_id']
    feature_cols = features_df.drop(columns=['song_id'])
    
    # 找出数值列
    numeric_cols = feature_cols.select_dtypes(include=[np.number]).columns
    
    # 降维
    pca = PCA(n_components=min(n_components, len(numeric_cols), len(feature_cols)))
    reduced_features = pd.DataFrame(
        pca.fit_transform(feature_cols[numeric_cols]),
        columns=[f'pca_{i}' for i in range(pca.n_components_)]
    )
    
    # 输出解释方差
    explained_variance = pca.explained_variance_ratio_
    logger.info(f"解释方差: {explained_variance}")
    logger.info(f"累计解释方差: {np.sum(explained_variance):.4f}")
    
    # 重新添加ID列
    reduced_features.insert(0, 'song_id', id_col)
    
    # 保存降维特征
    if output_file:
        reduced_features.to_csv(output_file, index=False)
        logger.info(f"降维特征已保存到: {output_file}")
    
    return reduced_features

def merge_with_song_metadata(features_df, metadata_file, output_file=None):
    """
    将特征与歌曲元数据合并
    
    参数:
        features_df: 特征DataFrame
        metadata_file: 元数据文件路径
        output_file: 输出文件路径 (可选)
    
    返回:
        合并后的DataFrame
    """
    logger.info(f"合并特征与元数据: {metadata_file}")
    
    # 加载元数据
    metadata_df = pd.read_csv(metadata_file)
    logger.info(f"加载了 {len(metadata_df)} 首歌曲的元数据")
    
    # 合并数据
    merged_df = pd.merge(metadata_df, features_df, on='song_id', how='left')
    logger.info(f"合并后有 {len(merged_df)} 首歌曲")
    
    # 保存合并数据
    if output_file:
        merged_df.to_csv(output_file, index=False)
        logger.info(f"合并数据已保存到: {output_file}")
    
    return merged_df

def add_context_features(songs_df, output_file=None):
    """
    添加上下文特征
    
    参数:
        songs_df: 歌曲DataFrame
        output_file: 输出文件路径 (可选)
    
    返回:
        添加上下文特征后的DataFrame
    """
    logger.info("添加上下文特征...")
    
    # 复制DataFrame以避免修改原始数据
    result_df = songs_df.copy()
    
    # 添加情绪标签
    if 'energy' in result_df.columns and 'valence' in result_df.columns:
        # 创建情绪标签
        def assign_mood(row):
            energy = row.get('energy', 0.5)
            valence = row.get('valence', 0.5)
            
            if energy > 0.8 and valence > 0.8:
                return 'excited'
            elif energy > 0.8 and valence < 0.4:
                return 'angry'
            elif energy < 0.4 and valence > 0.8:
                return 'relaxed'
            elif energy < 0.4 and valence < 0.4:
                return 'sad'
            elif energy > 0.6:
                return 'energetic'
            elif valence > 0.6:
                return 'happy'
            else:
                return 'neutral'
        
        result_df['mood'] = result_df.apply(assign_mood, axis=1)
    
    # 添加活动适合度
    if 'tempo' in result_df.columns and 'energy' in result_df.columns:
        # 创建活动适合度标签
        def assign_activity(row):
            tempo = row.get('tempo', 120)
            energy = row.get('energy', 0.5)
            
            if tempo > 120 and energy > 0.7:
                return 'exercising'
            elif tempo < 80 and energy < 0.4:
                return 'relaxing'
            elif 80 <= tempo <= 110 and energy < 0.5:
                return 'studying'
            elif tempo > 100 and energy > 0.6:
                return 'socializing'
            elif 90 <= tempo <= 120:
                return 'working'
            else:
                return 'commuting'
        
        result_df['activity_suitability'] = result_df.apply(assign_activity, axis=1)
    
    # 添加时间适合度
    if 'tempo' in result_df.columns and 'energy' in result_df.columns:
        # 创建时间适合度标签
        def assign_time(row):
            tempo = row.get('tempo', 120)
            energy = row.get('energy', 0.5)
            
            if tempo > 120 and energy > 0.7:
                return 'morning'
            elif 90 <= tempo <= 120 and energy > 0.5:
                return 'afternoon'
            elif tempo > 100 and energy > 0.6:
                return 'evening'
            elif tempo < 90 or energy < 0.4:
                return 'night'
            else:
                return 'anytime'
        
        result_df['time_suitability'] = result_df.apply(assign_time, axis=1)
    
    # 保存结果
    if output_file:
        result_df.to_csv(output_file, index=False)
        logger.info(f"添加上下文特征后的数据已保存到: {output_file}")
    
    return result_df

if __name__ == "__main__":
    # 获取命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="从MSD数据集提取内容特征")
    parser.add_argument("--msd_dir", help="MSD数据目录", required=True)
    parser.add_argument("--output_dir", help="输出目录", default="processed_data")
    parser.add_argument("--max_files", help="最大处理文件数", type=int, default=1000)
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理MSD目录
    features_file = os.path.join(args.output_dir, "audio_features.csv")
    features_df = process_msd_directory(args.msd_dir, features_file, args.max_files)
    
    # 标准化特征
    norm_file = os.path.join(args.output_dir, "audio_features_normalized.csv")
    norm_features_df = normalize_features(features_df, norm_file)
    
    # 降维特征
    reduced_file = os.path.join(args.output_dir, "audio_features_reduced.csv")
    reduced_features_df = reduce_dimensions(norm_features_df, 20, reduced_file)
    
    # 合并元数据
    metadata_file = os.path.join(args.output_dir, "songs.csv")
    if os.path.exists(metadata_file):
        merged_file = os.path.join(args.output_dir, "songs_with_features.csv")
        merged_df = merge_with_song_metadata(reduced_features_df, metadata_file, merged_file)
        
        # 添加上下文特征
        context_file = os.path.join(args.output_dir, "songs_with_context.csv")
        context_df = add_context_features(merged_df, context_file)
    
    logger.info("处理完成")
