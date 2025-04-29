from flask import Flask, request, jsonify, render_template, send_from_directory, render_template_string
from flask_cors import CORS
import os
import json
import pandas as pd
import numpy as np
from recommendation_engine import MusicRecommender
import sqlite3
from datetime import datetime
from ai_music_agent import MusicRecommenderAgent
from spotify_integration import SpotifyManager
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # 启用跨域支持

# 配置选项
DATA_DIR = 'processed_data'
USE_MSD = True  # 启用百万歌曲数据集
FORCE_RETRAIN = False  # 是否强制重新训练模型
SAMPLE_SIZE = None  # 设置为整数值可以限制加载的数据量，用于开发测试
SPOTIFY_CLIENT_ID = os.environ.get('SPOTIFY_CLIENT_ID', '4f1a2f4e1e034050ac432f8ebba72484')
SPOTIFY_CLIENT_SECRET = os.environ.get('SPOTIFY_CLIENT_SECRET', '4abd4c31749748c8b89f7807c61a3f11')

# 初始化数据库
def init_db():
    """初始化SQLite数据库"""
    logger.info("初始化数据库...")
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 创建用户表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    ''')
    
    # 创建用户评分表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_ratings (
        user_id TEXT,
        track_id TEXT,
        rating INTEGER,
        timestamp TEXT,
        PRIMARY KEY (user_id, track_id)
    )
    ''')
    
    # 创建用户反馈表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        track_id TEXT,
        feedback TEXT,
        timestamp TEXT
    )
    ''')
    
    # 创建聊天历史表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        message TEXT,
        response TEXT,
        timestamp TEXT
    )
    ''')
    
    # 创建用户音乐偏好表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_preferences (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        preferences TEXT,
        timestamp TEXT
    )
    ''')
    
    # 添加开发者账号
    cursor.execute('SELECT id FROM users WHERE username = ?', ('test',))
    if not cursor.fetchone():
        now = datetime.now().isoformat()
        cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?)', 
                      ('test', 'test', 'test123', now))
        logger.info("已创建开发者账号: test (密码: test123)")
    
    conn.commit()
    conn.close()
    logger.info("数据库初始化完成")

# 初始化数据库
init_db()

# 初始化推荐引擎
logger.info(f"初始化推荐引擎 (USE_MSD={USE_MSD}, FORCE_RETRAIN={FORCE_RETRAIN})...")
recommender = MusicRecommender(data_dir=DATA_DIR, use_msd=USE_MSD, force_retrain=FORCE_RETRAIN, sample_size=SAMPLE_SIZE)
logger.info("推荐引擎初始化完成")

# 初始化AI代理
logger.info("初始化AI音乐代理...")
ai_agent = MusicRecommenderAgent(data_dir=DATA_DIR, use_msd=USE_MSD)
logger.info("AI音乐代理初始化完成")

# 初始化Spotify管理器
logger.info("初始化Spotify管理器...")
spotify_manager = SpotifyManager(client_id=SPOTIFY_CLIENT_ID, client_secret=SPOTIFY_CLIENT_SECRET)
logger.info("Spotify管理器初始化完成")

@app.route('/')
def home():
    """提供前端页面"""
    # 从文件中读取模板内容
    with open('templates/index.html', 'r', encoding='utf-8') as file:
        template_content = file.read()
    
    # 在Jinja2处理前，将Vue.js的双大括号替换为安全的形式
    # 这里我们使用{$ $}暂时替代Vue.js的{{ }}，渲染后再替换回来
    processed_content = template_content.replace('{{', '{$').replace('}}', '$}')
    
    # 使用Jinja2渲染模板
    rendered_content = render_template_string(processed_content)
    
    # 渲染后，将临时语法替换回Vue.js的双大括号
    final_content = rendered_content.replace('{$', '{{').replace('$}', '}}')
    
    return final_content

@app.route('/static/<path:path>')
def send_static(path):
    """提供静态文件"""
    return send_from_directory('static', path)

@app.route('/api/user/register', methods=['POST'])
def register_user():
    """注册新用户"""
    data = request.json
    username = data.get('username')
    password = data.get('password')
    email = data.get('email', '')
    
    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    
    if not password:
        return jsonify({'error': '密码不能为空'}), 400
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 检查用户是否已存在
    cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
    existing_user = cursor.fetchone()
    
    if existing_user:
        conn.close()
        return jsonify({'error': '用户名已存在'}), 400
    
    # 生成用户ID（使用用户名或生成随机ID）
    user_id = username
    
    # 添加新用户
    now = datetime.now().isoformat()
    cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?)', 
                  (user_id, username, password, now))
    conn.commit()
    conn.close()
    
    logger.info(f"用户注册成功: {username} (ID: {user_id})")
    return jsonify({
        'message': '用户注册成功', 
        'user_id': user_id,
        'username': username
    })

@app.route('/api/user/login', methods=['POST'])
def login_user():
    """用户登录"""
    data = request.json
    user_id = data.get('user_id')
    username = data.get('username')
    password = data.get('password', '')
    email = data.get('email', '')
    
    if not username:
        return jsonify({'error': '用户名不能为空'}), 400
    
    # 特殊处理开发者账号
    is_developer = (username.lower() == 'test' and email == 'a@a.com')
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 检查用户是否存在
    cursor.execute('SELECT id, password FROM users WHERE username = ?', (username,))
    existing_user = cursor.fetchone()
    
    if not existing_user:
        # 如果用户不存在，则创建新用户
        if not user_id:
            user_id = username  # 使用用户名作为ID
            
        now = datetime.now().isoformat()
        cursor.execute('INSERT INTO users VALUES (?, ?, ?, ?)', 
                      (user_id, username, password, now))
        conn.commit()
        logger.info(f"新用户创建成功: {username} (ID: {user_id})")
        
        # 获取用户的评分数据
        cursor.execute('SELECT track_id, rating FROM user_ratings WHERE user_id = ?', (user_id,))
        ratings = cursor.fetchall()
    else:
        # 验证密码（对于非开发者账号）
        user_id, stored_password = existing_user
        
        if not is_developer and password != stored_password:
            conn.close()
            logger.warning(f"用户 {username} 登录失败: 密码不正确")
            return jsonify({'error': '密码不正确'}), 401
        
        # 更新登录时间
        now = datetime.now().isoformat()
        cursor.execute('UPDATE users SET created_at = ? WHERE id = ?', 
                      (now, user_id))
        conn.commit()
        logger.info(f"用户 {username} 登录成功 (ID: {user_id})")
    
    conn.close()
    
    return jsonify({
        'message': '登录成功',
        'user_id': user_id,
        'username': username,
        'is_developer': is_developer,
        'ratings_count': len(ratings)
    })

@app.route('/api/songs/sample', methods=['GET'])
def get_sample_songs():
    """获取样本歌曲用于初始评分"""
    # 从推荐引擎获取热门歌曲
    popular_songs = recommender.get_popular_songs(top_n=20)
    
    return jsonify({
        'songs': popular_songs
    })

@app.route('/api/sample_songs')
def sample_songs():
    """获取样本歌曲用于前端展示"""
    # 从推荐引擎获取热门歌曲
    popular_songs = recommender.get_popular_songs(top_n=20)
    
    return jsonify(popular_songs)

@app.route('/api/user_ratings/<user_id>')
def user_ratings(user_id):
    """获取用户的评分记录"""
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 获取用户的评分数据
    cursor.execute('SELECT track_id, rating FROM user_ratings WHERE user_id = ?', (user_id,))
    ratings = cursor.fetchall()
    
    conn.close()
    
    # 转换为{track_id: rating}格式
    ratings_dict = {track_id: rating for track_id, rating in ratings}
    
    return jsonify(ratings_dict)

@app.route('/api/rate_song', methods=['POST'])
def rate_song_api():
    """API：用户对歌曲进行评分"""
    data = request.json
    user_id = data.get('user_id')
    track_id = data.get('track_id')
    rating = data.get('rating')
    
    if not user_id or not track_id or not rating:
        return jsonify({'error': '缺少必要参数'}), 400
    
    if not isinstance(rating, int) or rating < 1 or rating > 5:
        return jsonify({'error': '评分必须是1-5之间的整数'}), 400
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    cursor.execute('''
    INSERT OR REPLACE INTO user_ratings (user_id, track_id, rating, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (user_id, track_id, rating, now))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': '评分成功'})

@app.route('/api/recommendations/<user_id>')
def user_recommendations(user_id):
    """获取用户推荐歌曲，并用Spotify数据丰富结果"""
    logger.info(f"为用户 {user_id} 生成推荐")
    try:
        # 从推荐引擎获取推荐
        recommendations = recommender.get_hybrid_recommendations(user_id, top_n=10)
        
        # 如果没有足够的推荐（如新用户），补充热门歌曲
        if len(recommendations) < 10:
            logger.info(f"用户 {user_id} 推荐数量不足，补充热门歌曲")
            popular_songs = recommender.get_popular_songs(top_n=10-len(recommendations))
            recommendations.extend(popular_songs)
        
        # 用Spotify数据丰富推荐结果
        enhanced_recommendations = []
        for rec in recommendations:
            enhanced_rec = spotify_manager.enrich_recommendation(rec)
            enhanced_recommendations.append(enhanced_rec)
        
        logger.info(f"成功为用户 {user_id} 生成 {len(enhanced_recommendations)} 条推荐")
        return jsonify(enhanced_recommendations)
    except Exception as e:
        logger.error(f"生成推荐时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '生成推荐时出错'}), 500

@app.route('/api/feedback', methods=['POST'])
def submit_feedback_api():
    """API：提交用户对推荐的反馈"""
    data = request.json
    user_id = data.get('user_id')
    track_id = data.get('track_id')
    feedback_type = data.get('feedback_type')  # 'like' 或 'dislike'
    
    if not user_id or not track_id or not feedback_type:
        return jsonify({'error': '缺少必要参数'}), 400
    
    if feedback_type not in ['like', 'dislike']:
        return jsonify({'error': '反馈类型无效'}), 400
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    
    # 记录反馈
    cursor.execute('''
    INSERT INTO user_feedback (user_id, track_id, feedback, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (user_id, track_id, feedback_type, now))
    
    # 如果是喜欢，添加一个高评分；如果是不喜欢，添加一个低评分
    rating = 5 if feedback_type == 'like' else 1
    cursor.execute('''
    INSERT OR REPLACE INTO user_ratings (user_id, track_id, rating, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (user_id, track_id, rating, now))
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': '反馈提交成功'})

@app.route('/api/evaluation/stats', methods=['GET'])
def get_evaluation_stats():
    """获取评估统计数据（用于管理员）"""
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 获取用户数
    cursor.execute('SELECT COUNT(*) FROM users')
    users_count = cursor.fetchone()[0]
    
    # 获取评分数
    cursor.execute('SELECT COUNT(*) FROM user_ratings')
    ratings_count = cursor.fetchone()[0]
    
    # 获取反馈数
    cursor.execute('SELECT COUNT(*) FROM user_feedback')
    feedback_count = cursor.fetchone()[0]
    
    # 获取平均评分
    cursor.execute('SELECT AVG(rating) FROM user_ratings')
    avg_rating = cursor.fetchone()[0]
    
    # 获取喜欢/不喜欢比例
    cursor.execute('''
    SELECT feedback, COUNT(*) FROM user_feedback
    GROUP BY feedback
    ''')
    feedback_stats = dict(cursor.fetchall())
    
    conn.close()
    
    return jsonify({
        'users_count': users_count,
        'ratings_count': ratings_count,
        'feedback_count': feedback_count,
        'avg_rating': avg_rating,
        'feedback_stats': feedback_stats
    })

@app.route('/api/evaluation', methods=['POST'])
def submit_evaluation_api():
    """API：提交用户满意度问卷"""
    data = request.json
    user_id = data.get('user_id')
    responses = data.get('responses')
    comment = data.get('comment', '')
    
    if not user_id or not responses:
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 保存评估结果到文件
    evaluation_dir = 'evaluation_results'
    os.makedirs(evaluation_dir, exist_ok=True)
    
    with open(f'{evaluation_dir}/evaluation_{user_id}.json', 'w') as f:
        json.dump({
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'responses': responses,
            'comment': comment
        }, f)
    
    return jsonify({'message': '评估提交成功'})

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API：与AI代理对话"""
    data = request.json
    user_id = data.get('user_id')
    message = data.get('message')
    
    if not user_id or not message:
        return jsonify({'error': '缺少必要参数'}), 400
    
    # 获取AI代理回复
    response = ai_agent.process_message(user_id, message)
    
    # 保存对话历史
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    now = datetime.now().isoformat()
    cursor.execute('''
    INSERT INTO chat_history (user_id, message, response, timestamp)
    VALUES (?, ?, ?, ?)
    ''', (user_id, message, response, now))
    
    conn.commit()
    conn.close()
    
    return jsonify({'response': response})

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """获取用户的聊天历史"""
    user_id = request.args.get('user_id')
    
    if not user_id:
        return jsonify({'error': '用户ID不能为空'}), 400
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT message, response, timestamp FROM chat_history
    WHERE user_id = ?
    ORDER BY timestamp ASC
    ''', (user_id,))
    
    history = cursor.fetchall()
    conn.close()
    
    formatted_history = []
    for msg, resp, ts in history:
        formatted_history.append({
            'user_message': msg,
            'ai_response': resp,
            'timestamp': ts
        })
    
    return jsonify({
        'history': formatted_history
    })

@app.route('/api/songs/by_artist/<artist_name>')
def get_songs_by_artist(artist_name):
    """获取特定艺术家的歌曲，并用Spotify数据丰富结果"""
    logger.info(f"获取艺术家 '{artist_name}' 的歌曲")
    try:
        # 从本地推荐引擎获取艺术家歌曲
        songs = recommender.get_recommendations_by_artist(artist_name, top_n=10)
        
        # 如果本地数据不足，补充从Spotify获取的信息
        if len(songs) < 5:
            try:
                # 获取艺术家信息
                artist_info = spotify_manager.get_artist_info(artist_name=artist_name)
                if artist_info and 'top_tracks' in artist_info:
                    # 转换为与本地推荐相同的格式
                    for track in artist_info['top_tracks']:
                        if len(songs) >= 10:
                            break
                        songs.append({
                            'track_id': track['id'],
                            'track_name': track['name'],
                            'artist_name': artist_name,
                            'album_cover': track.get('image'),
                            'preview_url': track.get('preview_url'),
                            'score': float(artist_info.get('popularity', 50)) / 100,
                            'explanation': f"{artist_name}在Spotify上的热门歌曲"
                        })
            except Exception as e:
                logger.warning(f"从Spotify获取艺术家歌曲时出错: {e}")
        
        # 用Spotify数据丰富结果
        enhanced_songs = []
        for song in songs:
            enhanced_song = spotify_manager.enrich_recommendation(song)
            enhanced_songs.append(enhanced_song)
        
        return jsonify(enhanced_songs)
    except Exception as e:
        logger.error(f"获取艺术家歌曲时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取艺术家歌曲时出错'}), 500

@app.route('/api/similar_songs/<track_id>')
def get_similar_songs(track_id):
    """获取与给定歌曲相似的歌曲，并用Spotify数据丰富结果"""
    logger.info(f"获取与歌曲 {track_id} 相似的歌曲")
    try:
        # 从本地推荐引擎获取相似歌曲
        similar_songs = recommender.get_similar_songs(track_id, top_n=5)
        
        # 获取歌曲信息用于Spotify查询
        song_info = None
        for song in similar_songs:
            if song.get('track_id') == track_id:
                song_info = song
                break
        
        if not song_info and len(similar_songs) == 0:
            # 尝试从数据库获取歌曲信息
            try:
                conn = sqlite3.connect('music_recommender.db')
                cursor = conn.cursor()
                cursor.execute('SELECT track_id, track_name, artist_name FROM songs WHERE track_id = ?', (track_id,))
                result = cursor.fetchone()
                if result:
                    song_info = {'track_id': result[0], 'track_name': result[1], 'artist_name': result[2]}
                conn.close()
            except Exception as e:
                logger.warning(f"从数据库获取歌曲信息时出错: {e}")
        
        # 如果本地相似歌曲不足，补充从Spotify获取的相似歌曲
        if len(similar_songs) < 3 and song_info:
            try:
                spotify_similar = spotify_manager.get_similar_tracks(
                    track_name=song_info.get('track_name'),
                    artist_name=song_info.get('artist_name'),
                    limit=5-len(similar_songs)
                )
                similar_songs.extend(spotify_similar)
            except Exception as e:
                logger.warning(f"从Spotify获取相似歌曲时出错: {e}")
        
        # 用Spotify数据丰富结果
        enhanced_songs = []
        for song in similar_songs:
            enhanced_song = spotify_manager.enrich_recommendation(song)
            enhanced_songs.append(enhanced_song)
        
        return jsonify(enhanced_songs)
    except Exception as e:
        logger.error(f"获取相似歌曲时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取相似歌曲时出错'}), 500

@app.route('/api/spotify/track_info', methods=['GET'])
def get_spotify_track_info():
    """获取Spotify歌曲详细信息"""
    track_name = request.args.get('track_name')
    artist_name = request.args.get('artist_name')
    track_id = request.args.get('track_id')
    
    if not track_id and not track_name:
        return jsonify({'error': '必须提供track_id或track_name'}), 400
    
    try:
        track_info = spotify_manager.get_track_info(track_id=track_id, track_name=track_name, artist_name=artist_name)
        if track_info:
            return jsonify(track_info)
        return jsonify({'error': '未找到歌曲信息'}), 404
    except Exception as e:
        logger.error(f"获取Spotify歌曲信息时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取Spotify歌曲信息时出错'}), 500

@app.route('/api/spotify/artist_info', methods=['GET'])
def get_spotify_artist_info():
    """获取Spotify艺术家详细信息"""
    artist_name = request.args.get('artist_name')
    artist_id = request.args.get('artist_id')
    
    if not artist_id and not artist_name:
        return jsonify({'error': '必须提供artist_id或artist_name'}), 400
    
    try:
        artist_info = spotify_manager.get_artist_info(artist_id=artist_id, artist_name=artist_name)
        if artist_info:
            return jsonify(artist_info)
        return jsonify({'error': '未找到艺术家信息'}), 404
    except Exception as e:
        logger.error(f"获取Spotify艺术家信息时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取Spotify艺术家信息时出错'}), 500

@app.route('/api/spotify/new_releases', methods=['GET'])
def get_spotify_new_releases():
    """获取Spotify新发行专辑"""
    limit = request.args.get('limit', 10, type=int)
    country = request.args.get('country', 'US')
    
    try:
        new_releases = spotify_manager.get_new_releases(limit=limit, country=country)
        return jsonify(new_releases)
    except Exception as e:
        logger.error(f"获取Spotify新发行专辑时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取Spotify新发行专辑时出错'}), 500

@app.route('/api/spotify/similar_tracks', methods=['GET'])
def get_spotify_similar_tracks():
    """获取Spotify相似歌曲推荐"""
    track_name = request.args.get('track_name')
    artist_name = request.args.get('artist_name')
    track_id = request.args.get('track_id')
    limit = request.args.get('limit', 5, type=int)
    
    if not track_id and not track_name:
        return jsonify({'error': '必须提供track_id或track_name'}), 400
    
    try:
        similar_tracks = spotify_manager.get_similar_tracks(
            track_id=track_id, 
            track_name=track_name, 
            artist_name=artist_name, 
            limit=limit
        )
        return jsonify(similar_tracks)
    except Exception as e:
        logger.error(f"获取Spotify相似歌曲时出错: {str(e)}", exc_info=True)
        return jsonify({'error': '获取Spotify相似歌曲时出错'}), 500

@app.route('/questionnaire')
def questionnaire():
    """提供问卷调查页面"""
    return render_template('questionnaire.html')

@app.route('/api/user/preferences', methods=['POST'])
def save_user_preferences():
    """保存用户音乐偏好"""
    data = request.json
    user_id = data.get('userId')
    preferences = data.get('preferences')
    
    if not user_id:
        return jsonify({'error': '用户ID不能为空'}), 400
    
    if not preferences:
        return jsonify({'error': '偏好数据不能为空'}), 400
    
    # 将偏好转换为JSON字符串
    preferences_json = json.dumps(preferences, ensure_ascii=False)
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 检查是否存在该用户的偏好数据
    cursor.execute('SELECT id FROM user_preferences WHERE user_id = ?', (user_id,))
    existing_preferences = cursor.fetchone()
    
    now = datetime.now().isoformat()
    
    if existing_preferences:
        # 更新现有偏好
        cursor.execute(
            'UPDATE user_preferences SET preferences = ?, timestamp = ? WHERE user_id = ?',
            (preferences_json, now, user_id)
        )
        logger.info(f"已更新用户 {user_id} 的音乐偏好")
    else:
        # 添加新偏好
        cursor.execute(
            'INSERT INTO user_preferences (user_id, preferences, timestamp) VALUES (?, ?, ?)',
            (user_id, preferences_json, now)
        )
        logger.info(f"已添加用户 {user_id} 的音乐偏好")
    
    conn.commit()
    conn.close()
    
    # 将偏好数据传递给推荐引擎进行处理
    if recommender:
        recommender.process_user_preferences(user_id, preferences, source_type='questionnaire')
        logger.info(f"已将用户 {user_id} 的问卷偏好传递给推荐引擎处理")
    
    return jsonify({
        'message': '用户偏好保存成功',
        'user_id': user_id
    })

@app.route('/api/user/preferences/<user_id>', methods=['GET'])
def get_user_preferences(user_id):
    """获取用户音乐偏好"""
    if not user_id:
        return jsonify({'error': '用户ID不能为空'}), 400
    
    conn = sqlite3.connect('music_recommender.db')
    cursor = conn.cursor()
    
    # 获取用户偏好
    cursor.execute('SELECT preferences FROM user_preferences WHERE user_id = ?', (user_id,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        return jsonify({'error': '未找到该用户的偏好数据'}), 404
    
    # 解析JSON字符串
    preferences = json.loads(result[0])
    
    return jsonify({
        'user_id': user_id,
        'preferences': preferences
    })

@app.route('/api/game/preferences', methods=['POST'])
def save_game_preferences():
    """保存游戏收集的用户音乐偏好数据"""
    data = request.json
    user_id = data.get('userId')
    preferences = data.get('preferences')
    
    if not user_id:
        return jsonify({'error': '用户ID不能为空'}), 400
    
    if not preferences:
        return jsonify({'error': '偏好数据不能为空'}), 400
    
    try:
        # 将偏好转换为JSON字符串
        preferences_json = json.dumps(preferences, ensure_ascii=False)
        
        conn = sqlite3.connect('music_recommender.db')
        cursor = conn.cursor()
        
        # 检查是否存在游戏偏好表
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS game_preferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            preferences TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # 检查是否存在该用户的偏好数据
        cursor.execute('SELECT id FROM game_preferences WHERE user_id = ?', (user_id,))
        existing_preferences = cursor.fetchone()
        
        now = datetime.now().isoformat()
        
        if existing_preferences:
            # 更新现有偏好
            cursor.execute(
                'UPDATE game_preferences SET preferences = ?, timestamp = ? WHERE user_id = ?',
                (preferences_json, now, user_id)
            )
            logger.info(f"已更新用户 {user_id} 的游戏偏好")
        else:
            # 添加新偏好
            cursor.execute(
                'INSERT INTO game_preferences (user_id, preferences, timestamp) VALUES (?, ?, ?)',
                (user_id, preferences_json, now)
            )
            logger.info(f"已添加用户 {user_id} 的游戏偏好")
        
        conn.commit()
        conn.close()
        
        # 将偏好数据传递给推荐引擎进行处理
        if recommender:
            recommender.process_user_preferences(user_id, preferences, source_type='game')
            logger.info(f"已将用户 {user_id} 的游戏偏好传递给推荐引擎处理")
        
        return jsonify({
            'message': '游戏偏好保存成功',
            'user_id': user_id
        })
        
    except Exception as e:
        logger.error(f"保存游戏偏好时出错: {str(e)}")
        return jsonify({'error': f'保存游戏偏好失败: {str(e)}'}), 500

if __name__ == '__main__':
    logger.info("启动音乐推荐系统应用...")
    app.run(debug=True) 