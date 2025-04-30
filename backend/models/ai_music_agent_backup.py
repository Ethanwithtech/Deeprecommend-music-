"""
音乐推荐代理，用于处理用户请求，生成个性化推荐
"""
import sys
import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from backend.utils.ai_service import AIService

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 导入本项目模块
try:
    from backend.models.recommendation_engine import MusicRecommender
except ImportError:
    logger.warning("无法导入MusicRecommender，使用简化版推荐引擎")
    class MusicRecommender:
        def __init__(self, *args, **kwargs):
            pass
        def recommend(self, user_id, top_n=5, context=None):
            return []

class MusicRecommenderAgent:
    """音乐推荐代理类，处理用户交互和个性化推荐"""
    
    def __init__(self, data_dir="data", use_msd=False, recommender=None):
        """初始化音乐推荐代理
        
        参数:
            data_dir: 数据目录
            use_msd: 是否使用Million Song Dataset
            recommender: 可选参数，直接传入推荐引擎实例
        """
        # 初始化AI服务
        self.ai_service = AIService()
        
        # 初始化用户数据存储
        self.user_data = {}
        
        # 追踪用户对话次数
        self.conversation_counts = {}
        
        # 用户模型版本跟踪
        self.user_model_versions = {}
        
        # 用户情绪缓存
        self.user_emotions = {}
        
        # 初始化推荐引擎
        if recommender:
            # 使用传入的推荐引擎
            self.hybrid_recommender = recommender
            logger.info("使用传入的推荐引擎实例")
        else:
            # 创建新的推荐引擎实例
            try:
                from backend.models.hybrid_recommender import HybridRecommender
                self.hybrid_recommender = HybridRecommender(data_dir=data_dir, use_msd=use_msd)
                logger.info("成功创建HybridRecommender实例")
            except ImportError:
                # 如果无法导入HybridRecommender，使用基础推荐引擎
                from backend.models.recommendation_engine import MusicRecommender
                self.hybrid_recommender = MusicRecommender(data_dir=data_dir)
                logger.info("无法导入HybridRecommender，使用MusicRecommender替代")
        
        # 加载音乐元数据
        self._load_metadata()
    
    def process_message(self, user_id, message):
        """
        处理用户消息
        
        参数:
            user_id: 用户ID
            message: 用户消息
            
        返回:
            回复消息和推荐
        """
        # 初始化用户数据
        self._initialize_user_data(user_id)
        
        # 更新对话计数
        if user_id not in self.conversation_counts:
            self.conversation_counts[user_id] = 0
        self.conversation_counts[user_id] += 1
        conversation_count = self.conversation_counts[user_id]
        
        # 记录消息
        self.user_data[user_id]['messages'].append({
            'text': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # 提取用户偏好
        preferences = self._extract_preferences_from_message(message)
        if preferences:
            self._update_user_preferences(user_id, preferences)
        
        # 使用AI服务分析情感
        try:
            emotion_data = self.ai_service.analyze_emotion(message)
            emotion = emotion_data.get('emotion', 'neutral')
            emotion_intensity = emotion_data.get('intensity', 0.5)
            emotion_description = emotion_data.get('description', '')
        except Exception as e:
            # 降级到简单情感分析
            emotion = self._analyze_emotion(message)
            emotion_intensity = 0.5
            emotion_description = ''
        
        # 记录情感
        if emotion != 'neutral':
            self.user_data[user_id]['mood_data'].append({
                'mood': emotion,
                'timestamp': datetime.now().isoformat(),
                'intensity': emotion_intensity,
                'description': emotion_description
            })
        
        # 如果消息包含请求推荐的内容，生成推荐
        if any(keyword in message.lower() for keyword in ['推荐', '建议', '喜欢什么', '听什么', 'recommend']):
            # 获取上下文
            context = {'emotion': emotion}
            try:
                recommendations = self.hybrid_recommender.recommend(user_id, top_n=5, context=context)
            except Exception as e:
                # 如果用户不存在或模型未训练，返回空推荐
                recommendations = []
            
            # 获取安慰消息（如果需要）
            comfort_message = ""
            if emotion in ['sad', 'angry', 'anxious', 'stressed', 'depressed']:
                try:
                    comfort_message = self.ai_service.get_comfort_message(
                        emotion, emotion_intensity, emotion_description
                    )
                    comfort_message = f"{comfort_message}\n\n"
                except:
                    comfort_message = ""
            
            # 构建响应
            response = {
                'emotion': emotion,
                'recommendations': recommendations,
                'message': f"{comfort_message}{self._generate_response(emotion, recommendations)}"
            }
            
            return response
        
        # 处理一般对话
        user_history = self.user_data[user_id]['messages']
        user_preferences = self.user_data[user_id].get('preferences', {})
        
        # 生成主动问题
        try:
            proactive_question = self.ai_service.generate_proactive_question(
                user_history, 
                user_preferences, 
                current_emotion=emotion,
                conversation_count=conversation_count
            )
        except Exception as e:
            logger.error(f"生成主动问题失败: {e}")
            proactive_question = "你能告诉我更多关于你喜欢的音乐吗？"
        
        # 默认回复
        return {
            'emotion': emotion,
            'recommendations': [],
            'message': f"我了解您的偏好了。{proactive_question}"
        }
    
    def process_game_data(self, user_id, game_data):
        """
        处理游戏交互数据
        
        参数:
            user_id: 用户ID
            game_data: 游戏数据
            
        返回:
            处理结果
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 提取游戏中的偏好数据
        preferences = self._extract_preferences_from_game(game_data)
        
        # 更新用户偏好
        self._update_user_preferences(user_id, preferences)
        
        # 更新用户模型
        self._update_user_model(user_id)
        
        # 返回处理结果
        return {
            'status': 'success',
            'message': '游戏数据处理成功',
            'user_id': user_id,
            'preferences': preferences
        }
    
    def process_questionnaire(self, user_id, questionnaire_data):
        """
        处理问卷数据
        
        参数:
            user_id: 用户ID
            questionnaire_data: 问卷数据
            
        返回:
            处理结果
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 提取问卷中的偏好数据
        preferences = self._extract_preferences_from_questionnaire(questionnaire_data)
        
        # 更新用户偏好
        self._update_user_preferences(user_id, preferences)
        
        # 更新用户模型
        self._update_user_model(user_id)
        
        # 返回处理结果
        return {
            'status': 'success',
            'message': '问卷数据处理成功',
            'user_id': user_id,
            'preferences': preferences
        }
    
    def handle_new_user_feedback(self, user_id, song_id, rating):
        """
        处理新的用户反馈
        
        参数:
            user_id: 用户ID
            song_id: 歌曲ID
            rating: 评分
            
        返回:
            处理结果
        """
        # 确保用户数据存在
        self._initialize_user_data(user_id)
        
        # 更新用户评分
        self.user_data[user_id]['ratings'][song_id] = {
            'rating': rating,
            'timestamp': datetime.now().isoformat()
        }
        
        # 更新用户模型 (实时更新)
        self._update_user_model(user_id, real_time=True)
        
        # 返回处理结果
        return {
            'status': 'success',
            'message': '用户反馈处理成功',
            'user_id': user_id,
            'song_id': song_id,
            'rating': rating
        }
    
    def get_mood_based_recommendations(self, user_id, mood, top_n=5):
        """
        获取基于情绪的推荐
        
        参数:
            user_id: 用户ID
            mood: 情绪
            top_n: 推荐数量
            
        返回:
            推荐结果
        """
        # 获取上下文信息
        context = self._get_current_context(user_id)
        context['mood'] = mood
        
        # 使用情绪上下文生成推荐
        try:
            recommendations = self.hybrid_recommender.recommend(user_id, top_n=top_n, context=context)
            return recommendations
        except Exception as e:
            logger.error(f"情绪推荐出错: {str(e)}")
            return []
    
    def get_activity_based_recommendations(self, user_id, activity, top_n=5):
        """
        获取基于活动的推荐
        
        参数:
            user_id: 用户ID
            activity: 活动类型
            top_n: 推荐数量
            
        返回:
            推荐结果
        """
        # 记录活动信息
        self._initialize_user_data(user_id)
        self.user_data[user_id]['activity_data'].append({
            'activity': activity,
            'timestamp': datetime.now().isoformat()
        })
        
        # 获取上下文信息
        context = self._get_current_context(user_id)
        context['activity'] = activity
        
        # 使用活动上下文生成推荐
        try:
            recommendations = self.hybrid_recommender.recommend(user_id, top_n=top_n, context=context)
            # 对推荐结果进行后处理，例如根据活动类型调整顺序
            if activity == 'workout':
                # 对于锻炼，优先推荐节奏感强的歌曲
                pass
            elif activity == 'study':
                # 对于学习，优先推荐平静的歌曲
                pass
            
            return recommendations
        except Exception as e:
            logger.error(f"活动推荐出错: {str(e)}")
            return []
    
    def get_artist_based_recommendations(self, user_id, artist_name, top_n=5):
        """
        获取基于艺术家的推荐
        
        参数:
            user_id: 用户ID
            artist_name: 艺术家名称
            top_n: 推荐数量
            
        返回:
            推荐结果
        """
        # 更新用户的艺术家偏好
        self._initialize_user_data(user_id)
        if 'artists' not in self.user_data[user_id]['preferences']:
            self.user_data[user_id]['preferences']['artists'] = []
        
        # 添加艺术家到用户偏好
        artists = self.user_data[user_id]['preferences']['artists']
        if artist_name not in artists:
            artists.append(artist_name)
        
        # 获取基于艺术家的推荐
        try:
            # 如果可能，使用专用的基于艺术家的推荐方法
            if hasattr(self.hybrid_recommender, 'recommend_by_artist'):
                return self.hybrid_recommender.recommend_by_artist(artist_name, top_n)
            
            # 否则使用一般推荐，带有艺术家上下文
            context = {'artist': artist_name}
            return self.hybrid_recommender.recommend(user_id, top_n=top_n, context=context)
        except Exception as e:
            logger.error(f"艺术家推荐出错: {str(e)}")
            return []
    
    def save_user_data(self, output_dir="user_data"):
        """
        保存用户数据到磁盘
        
        参数:
            output_dir: 输出目录
        
        返回:
            成功标志
        """
        try:
            # 确保目录存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # 保存每个用户的数据
            for user_id, data in self.user_data.items():
                user_file = os.path.join(output_dir, f"user_{user_id}.json")
                with open(user_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            
            # 保存用户模型版本信息
            version_file = os.path.join(output_dir, "model_versions.json")
            with open(version_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_model_versions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"用户数据已保存到 {output_dir}")
            return True
        except Exception as e:
            logger.error(f"保存用户数据失败: {str(e)}")
            return False
    
    def load_user_data(self, input_dir="user_data"):
        """
        从磁盘加载用户数据
        
        参数:
            input_dir: 输入目录
            
        返回:
            成功标志
        """
        try:
            # 如果目录不存在，返回
            if not os.path.exists(input_dir):
                logger.info(f"用户数据目录不存在: {input_dir}")
                return False
            
            # 加载所有用户文件
            loaded = False
            for filename in os.listdir(input_dir):
                if filename.startswith("user_") and filename.endswith(".json"):
                    user_id = filename[5:-5]  # 提取用户ID
                    user_file = os.path.join(input_dir, filename)
                    
                    try:
                        with open(user_file, 'r', encoding='utf-8') as f:
                            self.user_data[user_id] = json.load(f)
                        loaded = True
                    except Exception as e:
                        logger.error(f"加载用户数据文件失败 {user_file}: {str(e)}")
            
            # 加载模型版本信息
            version_file = os.path.join(input_dir, "model_versions.json")
            if os.path.exists(version_file):
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        self.user_model_versions = json.load(f)
                except Exception as e:
                    logger.error(f"加载模型版本信息失败: {str(e)}")
            
            if loaded:
                logger.info(f"成功加载用户数据从 {input_dir}")
            else:
                logger.info(f"目录中没有找到用户数据: {input_dir}")
            
            return loaded
        except Exception as e:
            logger.error(f"加载用户数据失败: {str(e)}")
            return False
    
    def _initialize_user_data(self, user_id):
        """初始化用户数据"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                'messages': [],
                'ratings': {},
                'preferences': {},
                'mood_data': [],
                'activity_data': [],
                'created_at': datetime.now().isoformat()
            }
    
    def _calculate_user_data_completeness(self, user_id):
        """
        计算用户数据完整度
        
        参数:
            user_id: 用户ID
            
        返回:
            完整度分数 (0.0-1.0)
        """
        if user_id not in self.user_data:
            return 0.0
        
        # 初始得分
        score = 0.0
        max_score = 4.0  # 总分
        
        user_data = self.user_data[user_id]
        
        # 1. 评分数据
        rating_count = len(user_data['ratings'])
        if rating_count > 0:
            # 根据评分数量给分，最高1分
            rating_score = min(1.0, rating_count / 10.0)
            score += rating_score
        
        # 2. 偏好数据
        if user_data['preferences']:
            # 根据不同类型的偏好给分
            pref_score = 0.0
            if 'genres' in user_data['preferences'] and user_data['preferences']['genres']:
                pref_score += 0.5
            if 'moods' in user_data['preferences'] and user_data['preferences']['moods']:
                pref_score += 0.25
            if 'artists' in user_data['preferences'] and user_data['preferences']['artists']:
                pref_score += 0.25
            
            score += min(1.0, pref_score)
        
        # 3. 情感数据
        if user_data['mood_data']:
            # 根据情感记录数量给分
            mood_score = min(1.0, len(user_data['mood_data']) / 5.0)
            score += mood_score
        
        # 4. 交互数据
        message_count = len(user_data['messages'])
        if message_count > 0:
            # 根据消息数量给分
            interaction_score = min(1.0, message_count / 10.0)
            score += interaction_score
        
        # 返回归一化分数
        return score / max_score
    
    def _adjust_algorithm_weights_for_incomplete_data(self, user_id, completeness):
        """
        根据数据完整度调整算法权重
        
        参数:
            user_id: 用户ID
            completeness: 数据完整度得分
        """
        if not hasattr(self.hybrid_recommender, 'set_weights'):
            return
        
        # 基于数据完整度调整权重
        if completeness < 0.2:
            # 数据非常有限，主要使用基于内容的推荐
            weights = {
                'content': 0.6,
                'cf': 0.2,
                'context': 0.1,
                'deep': 0.1
            }
        elif completeness < 0.5:
            # 数据较少，减少协同过滤权重
            weights = {
                'content': 0.4,
                'cf': 0.3,
                'context': 0.2,
                'deep': 0.1
            }
        else:
            # 数据丰富，使用平衡权重
            weights = {
                'content': 0.3,
                'cf': 0.4,
                'context': 0.2,
                'deep': 0.1
            }
        
        # 设置权重
        self.hybrid_recommender.set_weights(weights)
    
    def _extract_preferences_from_message(self, message):
        """从消息中提取用户偏好"""
        preferences = {}
        
        # 解析消息，提取偏好信息
        # 以下是简单示例，实际系统应使用更复杂的NLP技术
        
        # 1. 提取流派偏好
        genre_keywords = {
            'pop': ['流行', '流行音乐', 'pop', '大众'],
            'rock': ['摇滚', '摇滚乐', 'rock'],
            'classical': ['古典', '古典音乐', '交响乐', 'classical'],
            'jazz': ['爵士', '爵士乐', 'jazz'],
            'electronic': ['电子', '电子音乐', 'EDM', 'electronic'],
            'folk': ['民谣', '民歌', 'folk'],
            'hip-hop': ['嘻哈', '说唱', 'hip-hop', 'rap'],
            'r&b': ['R&B', 'rnb', '节奏蓝调']
        }
        
        message_lower = message.lower()
        
        genres = {}
        for genre, keywords in genre_keywords.items():
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    genres[genre] = 1.0
                    break
        
        if genres:
            preferences['genres'] = genres
        
        # 2. 提取情绪偏好
        mood_keywords = {
            'happy': ['开心', '快乐', '欢快', '高兴', 'happy', 'joyful'],
            'sad': ['伤心', '悲伤', '难过', 'sad', 'depressed'],
            'calm': ['平静', '放松', '安静', 'calm', 'relaxed', 'peaceful'],
            'energetic': ['活力', '动感', '兴奋', 'energetic', 'excited']
        }
        
        moods = {}
        for mood, keywords in mood_keywords.items():
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    moods[mood] = 1.0
                    break
        
        if moods:
            preferences['moods'] = moods
        
        # 3. 提取艺术家偏好
        # 需要更复杂的NER来准确提取艺术家名称
        # 简化示例:
        artist_pattern = r'我喜欢(.*?)的歌' # 简单模式
        
        # 4. 提取活动偏好
        activity_keywords = {
            'workout': ['健身', '运动', '跑步', 'workout', 'running'],
            'study': ['学习', '工作', '阅读', 'study', 'working'],
            'relax': ['休息', '睡觉', '放松', 'relax', 'sleeping'],
            'party': ['聚会', '派对', '社交', 'party', 'social']
        }
        
        activities = {}
        for activity, keywords in activity_keywords.items():
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    activities[activity] = 1.0
                    break
        
        if activities:
            preferences['activities'] = activities
        
        return preferences
    
    def _extract_preferences_from_game(self, game_data):
        """从游戏数据中提取用户偏好"""
        preferences = {}
        
        # 根据游戏类型提取不同偏好
        game_type = game_data.get('game_type', '')
        
        if game_type == 'genre_collector' and 'collected_genres' in game_data:
            # 流派收集游戏
            genres = {}
            for genre_info in game_data['collected_genres']:
                genre_id = genre_info.get('id', '')
                count = genre_info.get('count', 1)
                if genre_id:
                    genres[genre_id] = float(count)
            
            if genres:
                preferences['genres'] = genres
        
        elif game_type == 'mood_matcher' and 'matched_moods' in game_data:
            # 情绪匹配游戏
            moods = {}
            for mood_info in game_data['matched_moods']:
                mood_id = mood_info.get('id', '')
                score = mood_info.get('score', 1.0)
                if mood_id:
                    moods[mood_id] = float(score)
            
            if moods:
                preferences['moods'] = moods
        
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
        try:
            # 使用AI服务生成更个性化的回复
            context = {
                "emotion": emotion,
                "recommendations_count": len(recommendations),
                "has_recommendations": len(recommendations) > 0
            }
            
            # 根据不同情绪和是否有推荐设置消息模板
            if not recommendations or len(recommendations) == 0:
                # 没有推荐时，专注于心理支持和引导用户
                if emotion == 'happy':
                    return "我能感受到你现在心情不错！音乐可以帮助我们保持这种积极的情绪。你平时喜欢听什么类型的音乐呢？是流行、摇滚还是古典音乐？"
                elif emotion == 'sad':
                    return "我感觉你现在的情绪有些低落。音乐可以成为我们心灵的慰藉。愿意分享一下你最近经历了什么吗？也许我可以推荐一些能够共鸣或者提升心情的歌曲。"
                elif emotion == 'calm':
                    return "你现在似乎很平静。在这种状态下，一些与你当前心境相符的音乐可以帮助你保持内心的平和。你平时在平静的时候喜欢听什么类型的音乐呢？"
                elif emotion == 'anxious' or emotion == 'stressed':
                    return "我注意到你可能有些焦虑或压力。音乐疗法研究表明，某些类型的音乐可以帮助缓解这种情绪。你愿意分享一下你现在面临的情况吗？或者告诉我你通常在压力大时会听什么音乐？"
                elif emotion == 'energetic' or emotion == 'excited':
                    return "你现在的能量似乎很充沛！这是享受音乐的绝佳时刻。你喜欢什么类型的活力四射的音乐？也许是流行、摇滚还是电子音乐？"
                else:
                    return "作为你的音乐伙伴，我很想了解更多关于你的音乐品味。你平时喜欢听什么风格的音乐？有没有特别喜欢的艺术家或歌曲？"
            else:
                # 有推荐时，结合推荐和情感支持
                if emotion == 'happy':
                    return f"感受到你愉快的心情真好！我为你精选了{len(recommendations)}首可以延续这种快乐的歌曲。希望你喜欢这些推荐，如果有任何感想，随时告诉我哦！"
                elif emotion == 'sad':
                    return f"我能理解现在的你可能需要一些情感上的共鸣。这里有{len(recommendations)}首歌曲，它们或许能陪伴你度过这段时光。记住，音乐有时能表达我们难以言说的情感。"
                elif emotion == 'calm':
                    return f"在这平静的时刻，我为你挑选了{len(recommendations)}首适合当下氛围的音乐。这些歌曲可以帮助你保持内心的宁静。希望你能喜欢。"
                elif emotion == 'anxious' or emotion == 'stressed':
                    return f"在压力和焦虑的时候，合适的音乐可以成为一剂良药。我为你准备了{len(recommendations)}首可能帮助缓解紧张情绪的歌曲。慢慢聆听，让音乐带走一些负担。"
                elif emotion == 'energetic' or emotion == 'excited':
                    return f"你的活力令人感染！这里有{len(recommendations)}首充满动感的歌曲，希望能与你的能量相匹配。享受这些音乐，让美好的心情继续！"
                else:
                    return f"我为你精心挑选了{len(recommendations)}首歌曲。它们是基于我对你偏好的理解推荐的。希望你会喜欢，也欢迎你分享听后的感受。"
        except Exception as e:
            logger.error(f"生成回复出错: {e}")
            # 出错时的默认响应
            if not recommendations or len(recommendations) == 0:
                return "我很想了解更多关于你的音乐品味。你平时喜欢听什么类型的音乐呢？"
            else:
                return f"这里有一些我认为你可能会喜欢的歌曲。希望你能喜欢这些推荐！"
    
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
    
    def _load_metadata(self):
        """加载音乐元数据"""
        try:
            # 尝试从推荐引擎获取元数据
            self.metadata = getattr(self.hybrid_recommender, 'metadata', {})
            if not self.metadata:
                logger.warning("推荐引擎中没有可用的元数据")
                self.metadata = {}
            else:
                logger.info(f"成功加载元数据，共 {len(self.metadata)} 条记录")
        except Exception as e:
            logger.error(f"加载元数据时出错: {e}")
            self.metadata = {}

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
    agent = MusicRecommenderAgent()
    
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