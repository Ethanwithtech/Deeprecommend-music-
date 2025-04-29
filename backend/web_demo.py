import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import logging
import random
import time
import threading
from sklearn.model_selection import train_test_split

# 导入我们的模型
from backend.models.hybrid_recommender import HybridRecommender

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')
CORS(app)  # 启用跨域资源共享

# 加载模型和数据 - 设置默认路径指向预训练模型
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/trained/hybrid_recommender.pkl')
DATA_PATH = os.environ.get('DATA_PATH', 'models/trained/processed_data')

# 全局变量
recommender = None
songs = None
users = None
interactions = None
audio_features = None
user_features = None
training_in_progress = False

def load_model_and_data():
    """加载模型和数据"""
    global recommender, songs, users, interactions, audio_features, user_features
    
    try:
        # 检查模型目录是否存在
        model_dir = os.path.dirname(MODEL_PATH)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.warning(f"模型目录不存在，已创建: {model_dir}")
            
        # 检查数据目录是否存在
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH, exist_ok=True)
            logger.warning(f"数据目录不存在，已创建: {DATA_PATH}")
            
        # 加载模型
        if os.path.exists(MODEL_PATH):
            logger.info(f"加载模型: {MODEL_PATH}")
            recommender = HybridRecommender()
            recommender.load_model(MODEL_PATH)
            logger.info("模型加载成功")
            
            # 检查是否包含深度学习模型
            if recommender.has_deep_model:
                logger.info("检测到深度学习模型已加载")
            else:
                logger.info("未检测到深度学习模型")
        else:
            logger.warning(f"模型文件不存在: {MODEL_PATH}")
            recommender = None
        
        # 加载歌曲数据
        song_path = os.path.join(DATA_PATH, 'songs.csv')
        if os.path.exists(song_path):
            logger.info(f"加载歌曲数据: {song_path}")
            songs = pd.read_csv(song_path)
            logger.info(f"加载了 {len(songs)} 首歌曲")
        else:
            logger.warning(f"歌曲数据文件不存在: {song_path}")
            songs = None
        
        # 加载用户交互数据
        interaction_path = os.path.join(DATA_PATH, 'interactions.csv')
        if os.path.exists(interaction_path):
            logger.info(f"加载用户交互数据: {interaction_path}")
            interactions = pd.read_csv(interaction_path)
            users = interactions['user_id'].unique()
            logger.info(f"加载了 {len(users)} 个用户")
        else:
            logger.warning(f"用户交互数据文件不存在: {interaction_path}")
            interactions = None
            users = None
            
        # 加载音频特征和用户特征（如果存在）
        audio_path = os.path.join(DATA_PATH, 'audio_features.csv')
        if os.path.exists(audio_path):
            logger.info(f"加载音频特征: {audio_path}")
            audio_features = pd.read_csv(audio_path)
            logger.info(f"加载了 {len(audio_features)} 条音频特征")
        else:
            logger.warning(f"音频特征文件不存在: {audio_path}")
            audio_features = None
            
        user_features_path = os.path.join(DATA_PATH, 'user_features.csv')
        if os.path.exists(user_features_path):
            logger.info(f"加载用户特征: {user_features_path}")
            user_features = pd.read_csv(user_features_path)
            logger.info(f"加载了 {len(user_features)} 条用户特征")
        else:
            logger.warning(f"用户特征文件不存在: {user_features_path}")
            user_features = None
            
    except Exception as e:
        logger.error(f"加载模型或数据时出错: {str(e)}")
        recommender, songs, users, interactions, audio_features, user_features = None, None, None, None, None, None

# 训练深度学习模型的后台任务
def train_deep_model_task():
    """后台训练深度学习模型的任务"""
    global recommender, interactions, audio_features, songs, user_features, training_in_progress
    
    logger.info("开始训练深度学习模型")
    training_in_progress = True
    
    try:
        # 分割数据集
        train_data, _ = train_test_split(interactions, test_size=0.2, random_state=42)
        
        # 训练模型
        start_time = time.time()
        recommender.train(
            interactions=train_data,
            audio_features=audio_features,
            songs=songs,
            user_features=user_features,
            train_deep_model=True  # 启用深度学习模型
        )
        elapsed = time.time() - start_time
        
        # 保存模型
        recommender.save_model(MODEL_PATH)
        
        logger.info(f"深度学习模型训练完成，耗时 {elapsed:.2f} 秒")
    except Exception as e:
        logger.error(f"训练深度学习模型时出错: {str(e)}")
    finally:
        training_in_progress = False

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/api/users')
def get_users():
    """获取用户列表"""
    if users is None:
        return jsonify({"error": "未加载用户数据"}), 404
    
    # 限制返回的用户数量
    max_users = min(100, len(users))
    user_sample = list(users[:max_users])
    
    return jsonify({
        "users": user_sample,
        "total": len(users)
    })

@app.route('/api/songs')
def get_songs():
    """获取歌曲列表"""
    if songs is None:
        return jsonify({"error": "未加载歌曲数据"}), 404
    
    # 限制返回的歌曲数量
    max_songs = min(100, len(songs))
    songs_sample = songs.head(max_songs).to_dict('records')
    
    return jsonify({
        "songs": songs_sample,
        "total": len(songs)
    })

@app.route('/api/recommend')
def get_recommendations():
    """获取推荐"""
    if recommender is None:
        return jsonify({"error": "推荐模型未加载"}), 404
    
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "缺少用户ID参数"}), 400
    
    top_n = request.args.get('top_n', 10, type=int)
    
    try:
        # 获取推荐
        recommendations = recommender.recommend(user_id, top_n=top_n)
        
        # 歌曲详情现在直接由recommender返回，不需要额外处理
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"生成推荐时出错: {str(e)}")
        return jsonify({"error": f"生成推荐失败: {str(e)}"}), 500

@app.route('/api/song/<song_id>')
def get_song_details(song_id):
    """获取歌曲详情"""
    if songs is None:
        return jsonify({"error": "未加载歌曲数据"}), 404
    
    song_info = songs[songs['song_id'] == song_id]
    if song_info.empty:
        return jsonify({"error": "未找到该歌曲"}), 404
    
    return jsonify(song_info.iloc[0].to_dict())

@app.route('/api/create_user', methods=['POST'])
def create_user():
    """创建新用户（冷启动）"""
    if recommender is None:
        return jsonify({"error": "推荐模型未加载"}), 404
    
    # 获取用户偏好
    data = request.get_json()
    if not data:
        return jsonify({"error": "缺少用户偏好数据"}), 400
    
    # 生成新用户ID
    new_user_id = f"U{random.randint(1000000, 9999999)}"
    
    # 提取用户偏好
    preferences = {}
    
    # 处理音乐风格
    if 'genres' in data and data['genres']:
        preferences['genres'] = data['genres']
        
    # 处理节奏和能量偏好    
    if 'tempo' in data and isinstance(data['tempo'], (int, float)):
        preferences['tempo'] = int(data['tempo'])
        
    if 'energy' in data and isinstance(data['energy'], (int, float)):
        preferences['energy'] = int(data['energy'])
    
    # 获取推荐数量    
    top_n = data.get('top_n', 10)
    
    # 获取推荐
    try:
        # 使用冷启动推荐
        recommendations = recommender._cold_start_recommend(
            user_preferences=preferences,
            top_n=top_n
        )
        
        return jsonify({
            "user_id": new_user_id,
            "recommendations": recommendations
        })
        
    except Exception as e:
        logger.error(f"生成冷启动推荐时出错: {str(e)}")
        return jsonify({"error": f"生成推荐失败: {str(e)}"}), 500

@app.route('/api/update_weights', methods=['POST'])
def update_user_weights():
    """更新用户模型权重"""
    if recommender is None:
        return jsonify({"error": "推荐模型未加载"}), 404
    
    data = request.get_json()
    if not data or 'user_id' not in data:
        return jsonify({"error": "缺少用户ID"}), 400
    
    user_id = data['user_id']
    weights = {}
    
    # 收集权重
    if 'cf_weight' in data:
        weights['cf_weight'] = float(data['cf_weight'])
    if 'content_weight' in data:
        weights['content_weight'] = float(data['content_weight'])
    if 'context_weight' in data:
        weights['context_weight'] = float(data['context_weight'])
    if 'deep_weight' in data:
        weights['deep_weight'] = float(data['deep_weight'])
    
    try:
        # 更新权重
        recommender.update_weights(user_id, **weights)
        
        # 获取新的推荐
        top_n = data.get('top_n', 10)
        recommendations = recommender.recommend(user_id, top_n=top_n)
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations,
            "message": "权重更新成功"
        })
    
    except Exception as e:
        logger.error(f"更新用户权重时出错: {str(e)}")
        return jsonify({"error": f"更新用户权重失败: {str(e)}"}), 500

@app.route('/api/train_deep_model', methods=['POST'])
def train_deep_model():
    """启动深度学习模型训练"""
    global training_in_progress
    
    if recommender is None:
        return jsonify({"error": "推荐模型未加载"}), 404
        
    if interactions is None or songs is None:
        return jsonify({"error": "缺少训练所需数据"}), 404
        
    if training_in_progress:
        return jsonify({"error": "已有训练任务在进行中"}), 409
    
    try:
        # 启动训练线程
        train_thread = threading.Thread(target=train_deep_model_task)
        train_thread.daemon = True  # 设置为守护线程，主线程结束时会自动退出
        train_thread.start()
        
        return jsonify({
            "message": "深度学习模型训练已启动",
            "status": "pending"
        })
    except Exception as e:
        logger.error(f"启动深度学习模型训练时出错: {str(e)}")
        return jsonify({"error": f"启动训练失败: {str(e)}"}), 500

@app.route('/api/training_status')
def get_training_status():
    """获取训练状态"""
    return jsonify({
        "training_in_progress": training_in_progress,
        "has_deep_model": recommender.has_deep_model if recommender is not None else False
    })

@app.route('/api/status')
def get_status():
    """获取系统状态"""
    return jsonify({
        "model_loaded": recommender is not None,
        "songs_loaded": songs is not None,
        "users_loaded": users is not None,
        "total_songs": len(songs) if songs is not None else 0,
        "total_users": len(users) if users is not None else 0,
        "has_deep_model": recommender.has_deep_model if recommender is not None else False,
        "training_in_progress": training_in_progress
    })

if __name__ == '__main__':
    # 加载模型和数据
    load_model_and_data()
    
    # 启动应用
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"启动Web应用服务器，端口: {port}, 调试模式: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug) 