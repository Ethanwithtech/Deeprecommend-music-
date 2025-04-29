import requests
import json
import sqlite3
import re
import os
import logging
from backend.models.recommendation_engine import MusicRecommender
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from backend.utils.ai_service import AIService
from backend.models.emotion_analyzer import EmotionAnalyzer
import time
import copy
from datetime import datetime

# 导入新的HybridRecommender类
from backend.models.hybrid_recommender import HybridRecommender

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MusicRecommenderAgent:
    """
    混合推荐系统的AI代理接口，整合聊天、游戏、问卷等多种用户数据源
    支持实时调整推荐策略
    """
    
    def __init__(self, data_dir="data", use_msd=False, load_pretrained=False, pretrained_model_path=None):
        """
        初始化音乐推荐代理
        
        参数:
            data_dir: 数据目录
            use_msd: 是否使用MSD数据集
            load_pretrained: 是否加载预训练模型
            pretrained_model_path: 预训练模型路径
        """
        self.data_dir = data_dir
        self.use_msd = use_msd
        
        # 初始化推荐系统
        if load_pretrained and pretrained_model_path and os.path.exists(pretrained_model_path):
            # 加载预训练模型
            logger.info(f"加载预训练模型: {pretrained_model_path}")
            self.hybrid_recommender = HybridRecommender()
            self.hybrid_recommender.load_model(pretrained_model_path)
        else:
            # 创建新模型
            logger.info("创建新的混合推荐模型")
            self.hybrid_recommender = HybridRecommender()
            
        # 用户情感状态缓存
        self.user_emotions = {}
        
        # 用户数据存储 - 不同的数据来源会对应不同的用户数据结构
        self.user_data = {
            # 'user_id': {
            #     'chat_data': [],      # 聊天数据
            #     'game_data': [],      # 游戏交互数据
            #     'questionnaire': {},  # 问卷数据
            #     'mood_data': [],      # 情绪数据
            #     'activity_data': [],  # 活动数据
            #     'liked_songs': set(), # 喜欢的歌曲
            #     'disliked_songs': set(), # 不喜欢的歌曲
            #     'ratings': {},        # 评分数据
            #     'last_update_time': 0 # 上次更新时间
            # }
        }
        
        # 用户模型版本控制 - 用于跟踪模型更新，实现增量学习
        self.user_model_versions = {}
        
        # 信息完整度阈值
        self.completeness_threshold = 0.5  # 认为用户数据"足够完整"的阈值
        
        # 情感分析模块
        try:
            from backend.models.emotion_analyzer import EmotionAnalyzer
            self.emotion_analyzer = EmotionAnalyzer()
        except ImportError:
            logger.warning("找不到情感分析模块，将禁用情感分析功能")
            self.emotion_analyzer = None
    
    def process_message(self, user_id, message):
        """
        处理用户消息，提取情感并生成推荐
        
        参数:
            user_id: 用户ID
            message: 用户消息
            
        返回:
            包含推荐和回复的字典
        """
        # 分析消息情感
        emotion = self._analyze_emotion(message)
            
        # 缓存用户情感
        self.user_emotions[user_id] = emotion
        
        # 基于情感生成推荐
        context = {'emotion': emotion}
        try:
            recommendations = self.hybrid_recommender.recommend(user_id, top_n=5, context=context)
        except:
            # 如果用户不存在或模型未训练，返回空推荐
            recommendations = []
        
        # 构建响应
        response = {
            'emotion': emotion,
            'recommendations': recommendations,
            'message': self._generate_response(emotion, recommendations)
        }
        
        return response
    
    def process_game_data(self, user_id, game_data):
        """
        处理游戏交互数据
        
        参数:
            user_id: 用户ID
            game_data: 游戏数据
            
        返回:
            处理状态
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 添加游戏数据
        self.user_data[user_id]['game_data'].append({
            'data': game_data,
            'timestamp': int(time.time())
        })
        
        # 从游戏数据中提取偏好
        preferences = self._extract_preferences_from_game(game_data)
        if preferences:
            self._update_user_preferences(user_id, preferences)
        
        # 更新用户模型
        self._update_user_model(user_id)
        
        return {'status': 'success'}
    
    def process_questionnaire(self, user_id, questionnaire_data):
        """
        处理问卷数据
        
        参数:
            user_id: 用户ID
            questionnaire_data: 问卷数据
            
        返回:
            处理状态
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 更新问卷数据(覆盖方式)
        self.user_data[user_id]['questionnaire'] = {
            'data': questionnaire_data,
            'timestamp': int(time.time())
        }
        
        # 从问卷数据中提取偏好
        preferences = self._extract_preferences_from_questionnaire(questionnaire_data)
        if preferences:
            self._update_user_preferences(user_id, preferences)
        
        # 更新用户模型
        self._update_user_model(user_id)
        
        return {'status': 'success'}
    
    def handle_new_user_feedback(self, user_id, song_id, rating):
        """
        处理用户反馈，动态调整推荐
        
        参数:
            user_id: 用户ID
            song_id: 歌曲ID
            rating: 评分(1-5)
            
        返回:
            处理状态
        """
        # 调整算法权重
        try:
            if rating >= 4:  # 高评分
                # 增加内容推荐权重
                weights = {'cf': 0.5, 'content': 0.4, 'context': 0.1}
                self.hybrid_recommender.update_weights(user_id, weights)
            elif rating <= 2:  # 低评分
                # 增加协同过滤权重
                weights = {'cf': 0.7, 'content': 0.2, 'context': 0.1}
                self.hybrid_recommender.update_weights(user_id, weights)
                
            logger.info(f"已更新用户 {user_id} 的算法权重")
            return {'status': 'success', 'message': '反馈已处理'}
            
        except Exception as e:
            logger.error(f"处理用户反馈时出错: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def get_mood_based_recommendations(self, user_id, mood, top_n=5):
        """
        基于情绪获取推荐
        
        参数:
            user_id: 用户ID
            mood: 情绪标签
            top_n: 推荐数量
            
        返回:
            推荐列表
        """
        context = {'emotion': mood}
        try:
            return self.hybrid_recommender.recommend(user_id, top_n=top_n, context=context)
        except:
            return []
    
    def get_activity_based_recommendations(self, user_id, activity, top_n=5):
        """
        基于活动场景获取推荐
        
        参数:
            user_id: 用户ID
            activity: 活动场景
            top_n: 推荐数量
            
        返回:
            推荐列表
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 记录活动数据
        self.user_data[user_id]['activity_data'].append({
            'activity': activity,
            'timestamp': int(time.time())
        })
        
        # 准备活动上下文
        context = {'activity': activity}
        
        # 获取推荐
        recommendations = self.hybrid_recommender.recommend(user_id, top_n=top_n, context=context)
        
        return recommendations
    
    def get_artist_based_recommendations(self, user_id, artist_name, top_n=5):
        """
        基于艺术家获取推荐
        
        参数:
            user_id: 用户ID
            artist_name: 艺术家名称
            top_n: 推荐数量
            
        返回:
            推荐列表
        """
        # 查找该艺术家的歌曲
        artist_songs = self.hybrid_recommender.find_songs_by_artist(artist_name)
        
        if not artist_songs:
            return []
        
        # 使用内容推荐获取相似歌曲
        recommendations = self.hybrid_recommender.get_content_recommendations(
            song_ids=artist_songs[:3],  # 使用该艺术家的前3首歌
            top_n=top_n
        )
        
        return recommendations
    
    def save_user_data(self, output_dir="user_data"):
        """
        保存用户数据
        
        参数:
            output_dir: 输出目录
            
        返回:
            保存状态
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存用户数据
            for user_id, data in self.user_data.items():
                # 将集合转换为列表以便序列化
                serializable_data = copy.deepcopy(data)
                serializable_data['liked_songs'] = list(data['liked_songs'])
                serializable_data['disliked_songs'] = list(data['disliked_songs'])
                
                with open(os.path.join(output_dir, f"{user_id}.json"), 'w', encoding='utf-8') as f:
                    json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"用户数据已保存到: {output_dir}")
            return {'status': 'success'}
        
        except Exception as e:
            logger.error(f"保存用户数据失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def load_user_data(self, input_dir="user_data"):
        """
        加载用户数据
        
        参数:
            input_dir: 输入目录
            
        返回:
            加载状态
        """
        try:
            if not os.path.exists(input_dir):
                logger.warning(f"用户数据目录不存在: {input_dir}")
                return {'status': 'error', 'message': 'Directory not found'}
            
            # 清空当前用户数据
            self.user_data = {}
            
            # 加载用户数据
            for file_name in os.listdir(input_dir):
                if not file_name.endswith('.json'):
                    continue
                
                user_id = file_name[:-5]  # 去除.json后缀
                
                with open(os.path.join(input_dir, file_name), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 将列表转换回集合
                data['liked_songs'] = set(data.get('liked_songs', []))
                data['disliked_songs'] = set(data.get('disliked_songs', []))
                
                self.user_data[user_id] = data
            
            logger.info(f"已加载 {len(self.user_data)} 名用户的数据")
            return {'status': 'success', 'count': len(self.user_data)}
        
        except Exception as e:
            logger.error(f"加载用户数据失败: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _initialize_user_data(self, user_id):
        """初始化用户数据结构"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'chat_data': [],         # 聊天数据
                'game_data': [],         # 游戏交互数据
                'questionnaire': {},     # 问卷数据
                'mood_data': [],         # 情绪数据
                'activity_data': [],     # 活动数据
                'liked_songs': set(),    # 喜欢的歌曲
                'disliked_songs': set(), # 不喜欢的歌曲
                'ratings': {},           # 评分数据
                'preferences': {},       # 提取的偏好
                'last_update_time': int(time.time()) # 上次更新时间
            }
            self.user_model_versions[user_id] = 0
    
    def _calculate_user_data_completeness(self, user_id):
        """
        计算用户数据的完整度
        
        参数:
            user_id: 用户ID
            
        返回:
            完整度得分(0-1)
        """
        # 默认用户不存在
        if user_id not in self.user_data:
            return 0.0
        
        user_data = self.user_data[user_id]
        
        # 计算各数据源的完整度得分
        scores = []
        
        # 聊天数据完整度
        if len(user_data['chat_data']) > 0:
            chat_score = min(1.0, len(user_data['chat_data']) / 10.0)  # 10条消息为满分
            scores.append(chat_score)
        
        # 游戏数据完整度
        if len(user_data['game_data']) > 0:
            game_score = min(1.0, len(user_data['game_data']) / 3.0)  # 3次游戏互动为满分
            scores.append(game_score)
        
        # 问卷数据完整度
        if user_data['questionnaire']:
            scores.append(1.0)  # 有问卷数据为满分
        
        # 评分数据完整度
        if len(user_data['ratings']) > 0:
            rating_score = min(1.0, len(user_data['ratings']) / 5.0)  # 5个评分为满分
            scores.append(rating_score)
        
        # 情绪数据完整度
        if len(user_data['mood_data']) > 0:
            mood_score = min(1.0, len(user_data['mood_data']) / 3.0)  # 3条情绪记录为满分
            scores.append(mood_score)
        
        # 活动数据完整度
        if len(user_data['activity_data']) > 0:
            activity_score = min(1.0, len(user_data['activity_data']) / 3.0)  # 3条活动记录为满分
            scores.append(activity_score)
        
        # 计算平均完整度得分，如果没有任何数据源则为0
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)
    
    def _adjust_algorithm_weights_for_incomplete_data(self, user_id, completeness):
        """
        根据数据完整度动态调整推荐算法权重
        
        参数:
            user_id: 用户ID
            completeness: 数据完整度 (0-1)
        """
        user_data = self.user_data[user_id]
        weights = {}
        
        # 根据不同数据源情况调整权重
        
        # 情况1: 用户有评分数据 - 提高协同过滤权重
        if len(user_data['ratings']) > 0:
            weights['cf'] = 0.4 + (0.2 * (1 - completeness))
            weights['content'] = 0.3
            weights['context'] = 0.1 + (0.1 * completeness)  # 提高完整度会增加上下文权重
        
        # 情况2: 用户有问卷或游戏数据，但评分少 - 提高内容和上下文权重
        elif user_data['questionnaire'] or len(user_data['game_data']) > 0:
            weights['cf'] = 0.2
            weights['content'] = 0.4 + (0.1 * (1 - completeness))
            weights['context'] = 0.3 + (0.1 * (1 - completeness))
        
        # 情况3: 只有聊天数据 - 提高上下文权重
        elif len(user_data['chat_data']) > 0:
            weights['cf'] = 0.1
            weights['content'] = 0.3
            weights['context'] = 0.5 + (0.1 * (1 - completeness))
        
        # 默认情况: 平衡权重略偏向内容
        else:
            weights['cf'] = 0.2
            weights['content'] = 0.4
            weights['context'] = 0.3
        
        # 应用权重调整
        self.hybrid_recommender.update_weights(user_id, weights)
    
    def _extract_preferences_from_message(self, message):
        """从聊天消息中提取用户偏好"""
        preferences = {}
        
        # 提取音乐类型/流派偏好
        genre_keywords = {
            'pop': ['流行', '流行音乐', 'pop', '当代流行'],
            'rock': ['摇滚', '摇滚乐', 'rock', '重金属', '摇滚音乐'],
            'jazz': ['爵士', '爵士乐', 'jazz', '蓝调', '布鲁斯'],
            'classical': ['古典', '古典音乐', 'classical', '交响乐', '室内乐'],
            'electronic': ['电子', '电子音乐', 'electronic', 'EDM', '舞曲'],
            'hiphop': ['嘻哈', '饶舌', 'hip-hop', 'rap', '说唱'],
            'folk': ['民谣', '民歌', 'folk', '传统民谣'],
            'r&b': ['节奏布鲁斯', 'R&B', '蓝调', '灵魂乐'],
            'country': ['乡村', '乡村音乐', 'country']
        }
        
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    if 'genres' not in preferences:
                        preferences['genres'] = {}
                    preferences['genres'][genre] = preferences['genres'].get(genre, 0) + 1
        
        # 提取情绪偏好
        mood_keywords = {
            'happy': ['开心', '快乐', '高兴', '欢快', '愉悦', '激励'],
            'sad': ['悲伤', '难过', '伤心', '忧郁', '失落'],
            'energetic': ['有活力', '精力充沛', '振奋', '活跃', '激烈'],
            'calm': ['平静', '放松', '安静', '舒缓', '轻松'],
            'angry': ['愤怒', '生气', '狂躁', '不爽'],
            'romantic': ['浪漫', '感性', '温情', '爱情']
        }
        
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    if 'moods' not in preferences:
                        preferences['moods'] = {}
                    preferences['moods'][mood] = preferences['moods'].get(mood, 0) + 1
        
        # 提取活动场景偏好
        activity_keywords = {
            'studying': ['学习', '工作', '读书', '专注', '考试'],
            'exercising': ['运动', '锻炼', '健身', '跑步', '健身房'],
            'relaxing': ['放松', '休息', '睡觉', '冥想', '休闲'],
            'partying': ['聚会', '派对', '社交', '朋友聚会'],
            'traveling': ['旅行', '旅游', '开车', '通勤', '出行'],
            'cooking': ['做饭', '烹饪', '厨房', '烘焙']
        }
        
        for activity, keywords in activity_keywords.items():
            for keyword in keywords:
                if keyword in message:
                    if 'activities' not in preferences:
                        preferences['activities'] = {}
                    preferences['activities'][activity] = preferences['activities'].get(activity, 0) + 1
        
        # 提取艺术家名称
        # 简化处理，实际应用中可能需要更复杂的命名实体识别
        artist_patterns = [
            r'喜欢(.*?)的歌',
            r'想听(.*?)的',
            r'我爱(.*?)的音乐',
            r'(.*?)的歌很好听'
        ]
        
        for pattern in artist_patterns:
            matches = re.findall(pattern, message)
            if matches:
                if 'artists' not in preferences:
                    preferences['artists'] = []
                preferences['artists'].extend(matches)
        
        return preferences
    
    def _extract_preferences_from_game(self, game_data):
        """从游戏数据中提取用户偏好"""
        preferences = {}
        
        # 游戏类型判断
        if isinstance(game_data, dict) and 'type' in game_data:
            game_type = game_data['type']
            
            # 流派选择游戏
            if game_type == 'genre_selection' and 'selections' in game_data:
                preferences['genres'] = game_data['selections']
            
            # 情绪选择游戏
            elif game_type == 'mood_selection' and 'mood' in game_data:
                preferences['moods'] = {game_data['mood']: game_data.get('intensity', 1.0)}
            
            # 艺术家选择游戏
            elif game_type == 'artist_selection' and 'artists' in game_data:
                preferences['artists'] = game_data['artists']
            
            # 活动场景选择游戏
            elif game_type == 'activity_selection' and 'activity' in game_data:
                preferences['activities'] = {game_data['activity']: game_data.get('relevance', 1.0)}
        
        return preferences
    
    def _extract_preferences_from_questionnaire(self, questionnaire_data):
        """从问卷数据中提取用户偏好"""
        preferences = {}
        
        # 直接映射问卷中的偏好字段
        # 注意，实际问卷结构可能会有所不同
        
        # 流派偏好
        if 'favorite_genres' in questionnaire_data:
            genres = questionnaire_data['favorite_genres']
            if isinstance(genres, list):
                preferences['genres'] = {genre: 1.0 for genre in genres}
            elif isinstance(genres, dict):
                preferences['genres'] = genres
        
        # 情绪偏好
        if 'mood_preferences' in questionnaire_data:
            preferences['moods'] = questionnaire_data['mood_preferences']
        
        # 艺术家偏好
        if 'favorite_artists' in questionnaire_data:
            preferences['artists'] = questionnaire_data['favorite_artists']
        
        # 活动场景偏好
        if 'activity_preferences' in questionnaire_data:
            preferences['activities'] = questionnaire_data['activity_preferences']
        
        return preferences
    
    def _update_user_preferences(self, user_id, new_preferences):
        """更新用户偏好"""
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 获取当前偏好
        current_prefs = self.user_data[user_id].get('preferences', {})
        
        # 合并各类偏好
        for pref_type, values in new_preferences.items():
            if pref_type not in current_prefs:
                current_prefs[pref_type] = {}
            
            # 根据不同的偏好类型采用不同的合并策略
            if pref_type in ['genres', 'moods', 'activities'] and isinstance(values, dict):
                # 数值型偏好，累加权重
                for key, value in values.items():
                    current_prefs[pref_type][key] = current_prefs[pref_type].get(key, 0) + value
            
            elif pref_type == 'artists' and isinstance(values, list):
                # 列表型偏好，去重合并
                if 'artists' not in current_prefs:
                    current_prefs['artists'] = []
                
                # 合并列表并去重
                artists_set = set(current_prefs['artists']) | set(values)
                current_prefs['artists'] = list(artists_set)
        
        # 保存更新后的偏好
        self.user_data[user_id]['preferences'] = current_prefs
    
    def _get_current_context(self, user_id):
        """获取用户当前上下文"""
        if user_id not in self.user_data:
            return {}
        
        context = {}
        user_data = self.user_data[user_id]
        
        # 添加最近情绪
        if user_data['mood_data']:
            latest_mood = user_data['mood_data'][-1]
            context['mood'] = {
                'primary_emotion': latest_mood['mood'],
                'intensity': latest_mood.get('intensity', 0.5)
            }
        
        # 添加最近活动
        if user_data['activity_data']:
            latest_activity = user_data['activity_data'][-1]
            context['activity'] = latest_activity['activity']
        
        # 添加用户偏好
        if 'preferences' in user_data:
            context['preferences'] = user_data['preferences']
        
        # 添加时间上下文
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            context['time_of_day'] = 'morning'
        elif 12 <= current_hour < 18:
            context['time_of_day'] = 'afternoon'
        elif 18 <= current_hour < 22:
            context['time_of_day'] = 'evening'
        else:
            context['time_of_day'] = 'night'
        
        return context
    
    def _generate_response(self, emotion, recommendations):
        """根据情感和推荐生成回复"""
        if emotion == 'happy':
            return f"看起来您心情不错！这里有一些欢快的歌曲推荐给您。"
        elif emotion == 'sad':
            return f"感觉您有点低落，这些歌曲可能会让您感觉好些。"
        elif emotion == 'calm':
            return f"为您的平静时光准备了这些轻松的曲目。"
        else:
            return f"这是一些您可能喜欢的歌曲。"
    
    def _update_user_model(self, user_id, real_time=False):
        """更新用户推荐模型"""
        # 增加版本号
        if user_id in self.user_model_versions:
            self.user_model_versions[user_id] += 1
        else:
            self.user_model_versions[user_id] = 1
        
        # 获取用户完整度
        completeness = self._calculate_user_data_completeness(user_id)
        
        # 根据完整度调整推荐策略
        self._adjust_algorithm_weights_for_incomplete_data(user_id, completeness)
        
        # 如果是实时更新，并且有足够的数据，执行增量学习
        if real_time and completeness > 0.3:
            # 收集用户的评分数据
            ratings = []
            for song_id, rating_info in self.user_data[user_id]['ratings'].items():
                ratings.append((user_id, song_id, rating_info['rating']))
            
            # 增量更新协同过滤模型
            if hasattr(self.hybrid_recommender, 'update_cf_model') and len(ratings) > 0:
                self.hybrid_recommender.update_cf_model(ratings)
            
            # 更新基于内容的模型
            if hasattr(self.hybrid_recommender, 'update_content_model') and self.user_data[user_id]['preferences']:
                self.hybrid_recommender.update_content_model(user_id, self.user_data[user_id]['preferences'])
        
        logger.info(f"用户 {user_id} 模型已更新到版本 {self.user_model_versions[user_id]}")
        return self.user_model_versions[user_id]
    
    def _analyze_emotion(self, message):
        """分析消息情感"""
        # 简单示例，实际应使用情感分析模型
        positive_words = ['happy', 'excited', 'good', 'joy', 'happy', '开心', '高兴', '快乐']
        negative_words = ['sad', 'depressed', 'bad', 'angry', 'unhappy', '悲伤', '伤心', '难过']
        calm_words = ['calm', 'peaceful', 'relaxed', 'quiet', '平静', '放松', '安宁']
        
        message = message.lower()
        
        if any(word in message for word in positive_words):
            return 'happy'
        elif any(word in message for word in negative_words):
            return 'sad'
        elif any(word in message for word in calm_words):
            return 'calm'
        else:
            return 'neutral'

# API接口，可集成到Flask应用中
def handle_agent_request(user_id, message):
    """
    处理API请求
    
    参数:
        user_id: 用户ID
        message: 用户消息
        
    返回:
        推荐和回复
    """
    # 初始化代理
    agent = MusicRecommenderAgent(load_pretrained=True, pretrained_model_path="models/hybrid_model.pkl")
    
    # 处理消息
    result = agent.process_message(user_id, message)
    return result

if __name__ == "__main__":
    # 测试代码
    agent = MusicRecommenderAgent()
    
    test_messages = [
        "你好，我是新用户",
        "我喜欢周杰伦的歌",
        "能推荐一些歌曲给我吗？",
        "我给《七里香》打4分",
        "我不喜欢摇滚乐",
        "谢谢你的推荐"
    ]
    
    for msg in test_messages:
        print(f"\n用户: {msg}")
        response = agent.process_message("test_user_123", msg)
        print(f"AI: {response}") 