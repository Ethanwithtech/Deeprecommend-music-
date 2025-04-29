#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MSD推荐系统API接口

该模块提供基于Million Song Dataset训练的模型的API接口。
"""

import os
import logging
import numpy as np
from flask import Blueprint, request, jsonify
from scipy.spatial.distance import cosine
import pickle

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('msd_recommendation_api')

# 创建Blueprint
msd_api = Blueprint('msd_api', __name__)

# 全局变量
MODEL_PATH = os.environ.get('MSD_MODEL_PATH', 'models/msd_model.pkl')
model_data = None

def load_model():
    """加载MSD推荐模型"""
    global model_data
    
    if model_data is not None:
        return True
    
    try:
        logger.info(f"加载MSD模型: {MODEL_PATH}")
        with open(MODEL_PATH, 'rb') as f:
            model_data = pickle.load(f)
        
        logger.info("模型加载成功")
        return True
    except Exception as e:
        logger.error(f"加载模型时出错: {e}")
        return False

def find_similar_songs(song_id, n=10):
    """
    根据歌曲ID找到相似歌曲
    
    参数:
        song_id: 目标歌曲ID
        n: 返回的相似歌曲数量
        
    返回:
        相似歌曲列表，每个元素包含歌曲ID、相似度和元数据
    """
    if not load_model():
        return []
    
    # 检查歌曲是否在我们的数据中
    if song_id not in model_data['song_factors']:
        logger.warning(f"歌曲ID不存在: {song_id}")
        return []
    
    # 获取目标歌曲的因子
    target_factors = model_data['song_factors'][song_id]
    
    # 计算与所有其他歌曲的相似度
    similarities = []
    for sid, factors in model_data['song_factors'].items():
        if sid != song_id:  # 排除自身
            # 计算余弦相似度 (1 - 余弦距离)
            similarity = 1 - cosine(target_factors, factors)
            similarities.append((sid, similarity))
    
    # 按相似度排序并取前n个
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_similar = similarities[:n]
    
    # 组装结果
    result = []
    for sid, similarity in top_similar:
        song_info = {
            'song_id': sid,
            'similarity': similarity
        }
        
        # 添加元数据（如果有）
        if sid in model_data['song_metadata']:
            song_info['metadata'] = model_data['song_metadata'][sid]
        
        result.append(song_info)
    
    return result

def get_user_recommendations(user_id, n=10, include_metadata=True):
    """
    为用户生成推荐
    
    参数:
        user_id: 用户ID
        n: 推荐数量
        include_metadata: 是否包含歌曲元数据
        
    返回:
        推荐歌曲列表
    """
    if not load_model():
        return []
    
    # 检查用户是否在我们的数据中
    if user_id not in model_data['user_history']:
        logger.warning(f"用户ID不存在: {user_id}")
        return []
    
    # 获取用户历史
    user_songs = model_data['user_history'][user_id]
    
    # 用户已经听过的歌曲
    user_song_set = set(user_songs)
    
    # 为用户听过的每首歌曲找到相似歌曲
    candidates = {}
    for song_id in user_songs:
        similar_songs = find_similar_songs(song_id, n=50)  # 获取更多候选项
        
        for similar in similar_songs:
            sid = similar['song_id']
            # 排除用户已经听过的歌曲
            if sid not in user_song_set:
                # 如果这首歌已经是候选项，增加它的分数
                if sid in candidates:
                    candidates[sid]['score'] += similar['similarity']
                else:
                    candidates[sid] = {
                        'song_id': sid,
                        'score': similar['similarity']
                    }
                    
                    # 添加元数据（如果请求且可用）
                    if include_metadata and sid in model_data['song_metadata']:
                        candidates[sid]['metadata'] = model_data['song_metadata'][sid]
    
    # 将候选项转换为列表并按分数排序
    recommendations = list(candidates.values())
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    
    return recommendations[:n]

# API端点
@msd_api.route('/similar_songs/<song_id>', methods=['GET'])
def api_similar_songs(song_id):
    """获取相似歌曲API"""
    try:
        # 获取参数
        n = request.args.get('n', default=10, type=int)
        
        # 获取相似歌曲
        similar_songs = find_similar_songs(song_id, n)
        
        return jsonify({
            'status': 'success',
            'song_id': song_id,
            'similar_songs': similar_songs
        })
    except Exception as e:
        logger.error(f"处理相似歌曲请求时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@msd_api.route('/recommendations/<user_id>', methods=['GET'])
def api_user_recommendations(user_id):
    """获取用户推荐API"""
    try:
        # 获取参数
        n = request.args.get('n', default=10, type=int)
        include_metadata = request.args.get('include_metadata', default='true', type=str).lower() == 'true'
        
        # 获取推荐
        recommendations = get_user_recommendations(user_id, n, include_metadata)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'recommendations': recommendations
        })
    except Exception as e:
        logger.error(f"处理用户推荐请求时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@msd_api.route('/metadata/<song_id>', methods=['GET'])
def api_song_metadata(song_id):
    """获取歌曲元数据API"""
    try:
        if not load_model():
            return jsonify({
                'status': 'error',
                'message': '无法加载模型'
            }), 500
        
        if song_id not in model_data['song_metadata']:
            return jsonify({
                'status': 'error',
                'message': f'找不到歌曲ID: {song_id}'
            }), 404
        
        metadata = model_data['song_metadata'][song_id]
        
        return jsonify({
            'status': 'success',
            'song_id': song_id,
            'metadata': metadata
        })
    except Exception as e:
        logger.error(f"处理歌曲元数据请求时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@msd_api.route('/model_info', methods=['GET'])
def api_model_info():
    """获取模型信息API"""
    try:
        if not load_model():
            return jsonify({
                'status': 'error',
                'message': '无法加载模型'
            }), 500
        
        # 准备模型基本信息
        info = {
            'song_count': len(model_data['song_factors']),
            'user_count': len(model_data['user_history']),
            'metadata_count': len(model_data['song_metadata'])
        }
        
        return jsonify({
            'status': 'success',
            'model_info': info
        })
    except Exception as e:
        logger.error(f"处理模型信息请求时出错: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500 