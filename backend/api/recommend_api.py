#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合推荐系统API服务
提供用户消息处理、推荐生成和用户反馈处理的RESTful接口
"""

from flask import Flask, request, jsonify
import os
import logging
import json
import argparse
from pathlib import Path
import sys

# 添加项目根目录到系统路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.models.ai_music_agent import MusicRecommenderAgent

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 全局变量
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/hybrid_model.pkl')

# 初始化代理（延迟加载）
agent = None

def get_agent():
    """获取或创建推荐代理实例"""
    global agent
    if agent is None:
        logger.info(f"初始化推荐代理，使用模型路径: {MODEL_PATH}")
        agent = MusicRecommenderAgent(
            load_pretrained=os.path.exists(MODEL_PATH),
            pretrained_model_path=MODEL_PATH
        )
    return agent

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'ok', 'message': '推荐系统API服务正常运行'})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """处理用户消息并生成推荐"""
    data = request.json
    if not data or 'user_id' not in data or 'message' not in data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    user_id = data['user_id']
    message = data['message']
    
    logger.info(f"用户 {user_id} 请求推荐: {message}")
    
    try:
        # 获取推荐
        result = get_agent().process_message(user_id, message)
        return jsonify(result)
    except Exception as e:
        logger.error(f"推荐生成错误: {str(e)}")
        return jsonify({'error': f'推荐生成失败: {str(e)}'}), 500

@app.route('/api/feedback', methods=['POST'])
def feedback():
    """处理用户对歌曲的反馈"""
    data = request.json
    if not data or 'user_id' not in data or 'song_id' not in data or 'rating' not in data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    user_id = data['user_id']
    song_id = data['song_id']
    rating = float(data['rating'])
    
    logger.info(f"用户 {user_id} 对歌曲 {song_id} 的评分: {rating}")
    
    try:
        # 处理反馈
        result = get_agent().handle_new_user_feedback(user_id, song_id, rating)
        return jsonify(result)
    except Exception as e:
        logger.error(f"处理反馈错误: {str(e)}")
        return jsonify({'error': f'处理反馈失败: {str(e)}'}), 500

@app.route('/api/mood_recommend', methods=['POST'])
def mood_recommend():
    """基于情绪的推荐"""
    data = request.json
    if not data or 'user_id' not in data or 'mood' not in data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    user_id = data['user_id']
    mood = data['mood']
    top_n = data.get('top_n', 5)
    
    logger.info(f"用户 {user_id} 请求基于情绪 {mood} 的推荐")
    
    try:
        # 获取基于情绪的推荐
        recommendations = get_agent().get_mood_based_recommendations(user_id, mood, top_n)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"情绪推荐错误: {str(e)}")
        return jsonify({'error': f'情绪推荐失败: {str(e)}'}), 500

@app.route('/api/activity_recommend', methods=['POST'])
def activity_recommend():
    """基于活动场景的推荐"""
    data = request.json
    if not data or 'user_id' not in data or 'activity' not in data:
        return jsonify({'error': '缺少必要参数'}), 400
    
    user_id = data['user_id']
    activity = data['activity']
    top_n = data.get('top_n', 5)
    
    logger.info(f"用户 {user_id} 请求基于活动 {activity} 的推荐")
    
    try:
        # 获取基于活动的推荐
        recommendations = get_agent().get_activity_based_recommendations(user_id, activity, top_n)
        return jsonify({'recommendations': recommendations})
    except Exception as e:
        logger.error(f"活动推荐错误: {str(e)}")
        return jsonify({'error': f'活动推荐失败: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """训练或重新训练模型"""
    data = request.json
    output_path = data.get('output_path', MODEL_PATH)
    
    logger.info(f"请求训练模型，输出路径: {output_path}")
    
    try:
        # 导入预训练模块
        from backend.models.pretrainer import create_sample_data, train_and_save_model
        
        # 创建示例数据并训练模型
        data = create_sample_data("processed_data")
        model = train_and_save_model(data, output_path)
        
        # 重新加载模型
        global agent
        agent = None  # 清除现有代理
        
        return jsonify({'status': 'success', 'message': f'模型已成功训练并保存到 {output_path}'})
    except Exception as e:
        logger.error(f"模型训练错误: {str(e)}")
        return jsonify({'error': f'模型训练失败: {str(e)}'}), 500

def run_server(host='0.0.0.0', port=5000, debug=False):
    """运行API服务器"""
    # 确保模型目录存在
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='启动推荐系统API服务器')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器主机地址')
    parser.add_argument('--port', type=int, default=5000, help='服务器监听端口')
    parser.add_argument('--model-path', type=str, default='models/hybrid_model.pkl', help='模型路径')
    parser.add_argument('--debug', action='store_true', help='是否启用调试模式')
    args = parser.parse_args()
    
    # 设置模型路径
    MODEL_PATH = args.model_path
    
    print(f"启动API服务器，监听地址: {args.host}:{args.port}")
    run_server(args.host, args.port, args.debug) 