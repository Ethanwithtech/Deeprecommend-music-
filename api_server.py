#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
API服务器，用于处理前端请求并提供音乐推荐
"""

import os
import sys
import json
import logging
import time
from flask import Flask, request, jsonify

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 引入推荐系统及相关类
# 从backend导入推荐引擎
from backend.models.recommendation_engine import MusicRecommender
from backend.models.hybrid_music_recommender import HybridMusicRecommender

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RecommenderAPI')

# 创建Flask应用
app = Flask(__name__)

# 全局推荐器实例
recommender = None

def load_recommender():
    """加载推荐模型"""
    global recommender
    
    model_path = 'models/trained/hybrid_recommender_10k.pkl'
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return False
    
    try:
        # 使用HybridMusicRecommender直接加载模型
        recommender = HybridMusicRecommender()
        success = recommender.load_model(model_path)
        if success:
            logger.info(f"成功加载模型: {model_path}")
            return True
        else:
            logger.error(f"加载模型失败: {model_path}")
            return False
    except Exception as e:
        logger.error(f"加载模型出错: {str(e)}")
        logger.error("尝试创建新的推荐器实例...")
        
        # 如果加载失败，创建一个新的实例
        try:
            # 尝试使用简化的推荐器
            recommender = MusicRecommender()
            logger.info("创建了新的推荐器实例")
            return True
        except Exception as e2:
            logger.error(f"创建推荐器实例失败: {str(e2)}")
            return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'model_loaded': recommender is not None
    })

@app.route('/api/user_vector', methods=['POST'])
def update_user_vector():
    """处理用户向量数据"""
    if not recommender:
        return jsonify({
            'status': 'error',
            'message': '推荐模型未加载'
        }), 500
    
    try:
        data = request.get_json()
        
        # 验证请求数据
        if not data or 'user_id' not in data or 'user_vector' not in data:
            return jsonify({
                'status': 'error',
                'message': '请求格式错误，需要user_id和user_vector字段'
            }), 400
        
        user_id = data['user_id']
        user_vector = data['user_vector']
        
        # 处理用户向量
        try:
            # 首先尝试使用process_user_vector方法
            success = recommender.process_user_vector(user_id, user_vector)
        except AttributeError:
            # 如果方法不存在，尝试其他可能的方法
            try:
                success = recommender.update_user_preference(user_id, user_vector)
            except AttributeError:
                # 如果还是找不到方法，则返回错误
                return jsonify({
                    'status': 'error',
                    'message': '当前推荐器不支持更新用户向量'
                }), 500
        
        if success:
            return jsonify({
                'status': 'success',
                'message': f'成功更新用户 {user_id} 的向量数据'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'处理用户向量失败'
            }), 500
    
    except Exception as e:
        logger.error(f"处理用户向量出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'处理用户向量时出现错误: {str(e)}'
        }), 500

@app.route('/api/recommend', methods=['GET'])
def get_recommendations():
    """获取推荐"""
    if not recommender:
        return jsonify({
            'status': 'error',
            'message': '推荐模型未加载'
        }), 500
    
    try:
        user_id = request.args.get('user_id')
        context = request.args.get('context', None)
        top_n = int(request.args.get('top_n', 10))
        
        # 如果有user_vector参数，解析它
        user_vector = None
        if 'user_vector' in request.args:
            try:
                user_vector = json.loads(request.args.get('user_vector'))
            except:
                return jsonify({
                    'status': 'error',
                    'message': 'user_vector参数格式错误，应为JSON数组'
                }), 400
        
        # 验证请求参数
        if not user_id:
            return jsonify({
                'status': 'error',
                'message': '缺少必要参数: user_id'
            }), 400
        
        # 生成推荐
        try:
            # 尝试使用推荐方法
            recommendations = recommender.recommend(
                user_id, 
                context=context, 
                top_n=top_n,
                user_vector=user_vector
            )
        except TypeError:
            # 如果参数不匹配，尝试不同的参数组合
            try:
                recommendations = recommender.recommend(user_id, top_n=top_n)
            except Exception as e:
                logger.error(f"生成推荐出错: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'生成推荐时出现错误: {str(e)}'
                }), 500
        
        # 格式化推荐结果
        formatted_recs = []
        for rec in recommendations:
            # 处理不同格式的推荐结果
            if isinstance(rec, tuple):
                song_id, score = rec
                formatted_recs.append({
                    'song_id': song_id,
                    'title': '未知歌曲',
                    'artist': '未知艺术家',
                    'score': score
                })
            elif isinstance(rec, dict):
                formatted_recs.append({
                    'song_id': rec.get('song_id', ''),
                    'title': rec.get('title', rec.get('track_name', '未知歌曲')),
                    'artist': rec.get('artist_name', rec.get('artist', '未知艺术家')),
                    'score': rec.get('score', 0)
                })
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'context': context,
            'recommendations': formatted_recs
        })
    
    except Exception as e:
        logger.error(f"生成推荐出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'生成推荐时出现错误: {str(e)}'
        }), 500

@app.route('/api/contexts', methods=['GET'])
def get_contexts():
    """获取可用的上下文类型"""
    if not recommender:
        return jsonify({
            'status': 'error',
            'message': '推荐模型未加载或上下文模型不可用'
        }), 500
    
    try:
        # 尝试获取上下文信息
        try:
            contexts = list(recommender.context_model.context_factors.keys())
        except AttributeError:
            contexts = ['morning', 'afternoon', 'evening', 'night', 'work', 'study', 'workout', 'relax']
        
        return jsonify({
            'status': 'success',
            'contexts': contexts
        })
    
    except Exception as e:
        logger.error(f"获取上下文类型出错: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取上下文类型时出现错误: {str(e)}'
        }), 500

def main():
    """主函数"""
    # 加载推荐模型
    if load_recommender():
        logger.info("成功加载推荐模型")
    else:
        logger.warning("无法加载推荐模型，将使用简化版推荐")
    
    # 运行Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main() 