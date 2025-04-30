"""
AI服务模块 - 支持HKBU GenAI Platform API
"""

import os
import requests
import logging
import json
from backend.models.emotion_analyzer import EmotionAnalyzer
from enum import Enum
from retry import retry

# 配置日志
logger = logging.getLogger(__name__)

class AIProvider(Enum):
    HKBU = "hkbu"

class AIService:
    """
    AI服务类，使用HKBU GenAI Platform API
    """
    
    def __init__(self, api_key=None, provider=AIProvider.HKBU):
        """
        初始化AI服务类
        
        参数:
            api_key: API密钥，默认使用环境变量或固定密钥
            provider: AI提供商，默认为HKBU
        """
        self.api_key = api_key or os.environ.get("HKBU_API_KEY", "06fd2422-8207-4a5b-8aaa-434415ed3a2b")
        self.provider = provider
        self.api_url = "https://genai.hkbu.edu.hk/general/rest/deployments"
        self.model_name = "gpt-4-o"  # 修正模型名称，添加连字符
        self.api_version = "2024-10-21"  # 更新API版本为最新的2024-10-21
        
        logger.info(f"AI服务初始化完成，使用提供商: {self.provider}, 模型: {self.model_name}")
    
    def _call_ai_service(self, messages):
        """
        调用AI服务API
        
        参数:
            messages: 消息列表，包含role和content
            
        返回:
            API返回的JSON响应
        """
        try:
            # 构建API请求URL - 使用模型文档中的固定URL格式
            url = f"{self.api_url}/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # 构建请求头和数据
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            # 按API文档要求构建请求负载
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 800
            }
            
            logger.info(f"发送请求到AI服务: {url}")
            logger.debug(f"请求负载头部: {str(payload)[:100]}...")
            
            # 发送请求并记录详细日志
            response = requests.post(url, headers=headers, json=payload)
            logger.debug(f"API响应状态码: {response.status_code}")
            
            # 如果返回错误，记录详细信息
            if response.status_code != 200:
                logger.error(f"API错误 {response.status_code}: {response.text}")
                return None
                
            # 如果成功，解析并返回JSON响应
            response_json = response.json()
            logger.debug(f"API响应: {str(response_json)[:200]}...")
            return response_json
                
        except Exception as e:
            logger.error(f"调用AI服务失败: {str(e)}")
            return None
    
    @retry(tries=2, delay=1, backoff=2, logger=logger)
    def analyze_emotion(self, text):
        """
        分析文本的情绪
        
        参数:
            text: 要分析的文本
            
        返回:
            包含情绪类型、强度、描述和音乐建议的JSON对象
        """
        logger.info(f"分析情绪: {text[:50]}...")
        
        # 构建系统提示，要求AI分析情绪并返回格式化的结果
        system_prompt = """
        你是一个专业的音乐情感分析师，擅长理解人类复杂的情感状态，并将其与音乐情绪关联。
        请分析以下文本中表达的情绪，并以JSON格式返回结果。
        
        情绪类别应从以下选择最合适的一项（但不限于）:
        - happy (开心): 包括愉悦、兴高采烈、满足、幸福
        - sad (悲伤): 包括忧郁、失落、痛苦、悲痛、思念
        - angry (愤怒): 包括恼火、不满、暴躁、气愤
        - anxious (焦虑): 包括担忧、紧张、不安、恐惧
        - nostalgic (怀旧): 包括回忆、思念过去、温柔的忧伤
        - calm (平静): 包括安宁、放松、平和、满足
        - excited (兴奋): 包括期待、激动、热情
        - lonely (孤独): 包括寂寞、渴望陪伴、孤立感
        - hopeful (有希望): 包括乐观、期待、憧憬
        - tired (疲惫): 包括疲劳、精疲力竭、倦怠
        
        情感强度从0.1到1.0，根据表达的强度给出评分。
        
        同时，为该情绪提供音乐建议，如节奏、风格、乐器等。
        
        请返回以下JSON格式:
        {
            "emotion": "情绪类别",
            "intensity": 强度值,
            "description": "对情绪的简短描述",
            "music_suggestion": "根据情绪的音乐建议",
            "comfort_message": "一段安慰或共鸣的短语，针对该情绪"
        }
        
        注意：确保返回的是有效的JSON格式，不要添加任何其他文本。
        """
        
        try:
            # 调用AI服务分析情绪
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            response = self._call_ai_service(messages)
            
            if response and 'choices' in response:
                # 提取AI回复
                content = response['choices'][0]['message']['content']
                logger.debug(f"情绪分析原始响应: {content[:200]}...")
                
                # 尝试解析JSON，处理可能的格式问题
                try:
                    # 尝试直接解析
                    result = json.loads(content)
                except json.JSONDecodeError:
                    # 如果直接解析失败，尝试找到JSON部分
                    try:
                        # 查找最可能是JSON的部分 (在 { 和 } 之间)
                        import re
                        json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            result = json.loads(json_str)
                        else:
                            raise ValueError("无法在响应中找到有效的JSON")
                    except Exception as e:
                        logger.error(f"JSON提取失败: {e}")
                        return self._get_default_emotion_analysis(text)
                
                # 验证必要字段是否存在
                if "emotion" not in result:
                    result["emotion"] = "neutral"  # 默认情绪
                if "intensity" not in result:
                    result["intensity"] = 0.5      # 默认强度
                if "description" not in result:
                    result["description"] = "平静的情绪状态"  # 默认描述
                if "music_suggestion" not in result:
                    result["music_suggestion"] = "可以尝试聆听轻松的流行音乐或轻柔的古典乐"  # 默认音乐建议
                if "comfort_message" not in result:
                    result["comfort_message"] = "音乐可以陪伴我们度过各种情绪"  # 默认安慰消息
                return result
            
            # 如果API响应没有预期的格式，返回默认分析
            logger.warning("AI服务返回的响应格式不符合预期")
            return self._get_default_emotion_analysis(text)
        
        except Exception as e:
            logger.error(f"情绪分析失败: {e}")
            return self._get_default_emotion_analysis(text)
            
    def _get_default_emotion_analysis(self, text):
        """根据简单文本分析，返回默认的情绪分析"""
        # 简单的关键词情绪分析
        text_lower = text.lower()
        
        # 基本情绪关键词映射
        emotions = {
            'happy': ['开心', '高兴', '快乐', '兴奋', '幸福', '愉快', 'happy', 'joy', 'excited', 'delighted'],
            'sad': ['难过', '伤心', '悲伤', '悲痛', '哭', '痛苦', '失落', 'sad', 'depressed', 'unhappy', 'sorrow'],
            'angry': ['生气', '愤怒', '恼火', '烦', '讨厌', 'angry', 'mad', 'annoyed', 'irritated'],
            'anxious': ['焦虑', '担心', '紧张', '害怕', '忧虑', 'anxious', 'nervous', 'worried', 'afraid'],
            'tired': ['累', '疲倦', '疲惫', '困', 'tired', 'exhausted', 'sleepy'],
            'calm': ['平静', '放松', '安宁', '舒适', 'calm', 'relaxed', 'peaceful'],
            'lonely': ['孤独', '寂寞', '想念', 'lonely', 'alone', 'miss'],
            'hopeful': ['希望', '期待', '梦想', '向往', 'hope', 'wish', 'dream', 'anticipate'],
            'nostalgic': ['怀念', '怀旧', '回忆', '过去', 'nostalgic', 'memory', 'remember']
        }
        
        # 默认情绪为中性
        detected_emotion = "neutral"
        max_matches = 0
        
        # 检查文本中最匹配的情绪关键词
        for emotion, keywords in emotions.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_emotion = emotion
        
        # 音乐建议映射
        music_suggestions = {
            'happy': "欢快、节奏感强的流行音乐或舞曲，例如现代流行乐或拉丁音乐",
            'sad': "抒情的钢琴曲、慢节奏的民谣或情感丰富的古典音乐",
            'angry': "节奏强劲的摇滚乐或金属音乐，可以帮助释放情绪",
            'anxious': "轻柔的环境音乐、自然声音或轻度节奏的冥想音乐",
            'tired': "舒缓的轻音乐、轻柔的爵士乐或平静的古典音乐",
            'calm': "轻柔的器乐、自然声音或慢节奏的钢琴曲",
            'lonely': "温暖的民谣、情感丰富的流行歌曲或富有共鸣的爵士乐",
            'hopeful': "充满正能量的流行歌曲或振奋人心的管弦乐曲",
            'nostalgic': "复古的音乐、经典老歌或带有历史感的音乐作品",
            'neutral': "各种风格的音乐都可尝试，可以根据当前的活动或场景选择"
        }
        
        # 情绪描述映射
        descriptions = {
            'happy': "感到快乐、积极和满足的状态",
            'sad': "感到悲伤、失落或内心低沉的状态",
            'angry': "感到愤怒、烦躁或不满的情绪状态",
            'anxious': "感到担忧、不安或紧张的状态",
            'tired': "感到疲惫、缺乏能量或需要休息",
            'calm': "感到平静、平衡和内心安宁",
            'lonely': "感到孤独、渴望陪伴或联系",
            'hopeful': "对未来充满期待和积极态度",
            'nostalgic': "沉浸在回忆中，对过去有着温和的情感连接",
            'neutral': "平和的情绪状态，没有明显的正面或负面倾向"
        }
        
        # 生成安慰消息
        comfort_messages = {
            'happy': "很高兴看到您的好心情！音乐可以帮助延续这种积极的情绪。",
            'sad': "每个人都有情绪低落的时候，这是完全自然的。音乐可以成为情感的出口，或许能带给您一些慰藉。",
            'angry': "感到愤怒是正常的情绪反应，适当的音乐可以帮助您释放这种情绪，找回平静。",
            'anxious': "焦虑是我们面对不确定性时的自然反应。一些平静的音乐可能有助于缓解这种感觉。",
            'tired': "休息是恢复能量的重要方式。轻柔的音乐可以创造放松的氛围，帮助您得到更好的休息。",
            'calm': "平静的状态是很珍贵的，音乐可以帮助维持这种内心的平衡。",
            'lonely': "音乐有时能成为我们的良伴，在孤独时刻带来温暖和理解。",
            'hopeful': "您的积极态度令人鼓舞！合适的音乐可以进一步提升这种积极的能量。",
            'nostalgic': "回忆有时会带来特别的情感体验，音乐有神奇的力量将我们带回那些珍贵的时刻。",
            'neutral': "音乐可以根据您的需求引导情绪走向不同的方向，无论是放松、振奋还是集中注意力。"
        }
        
        # 强度估计
        if max_matches > 2:
            intensity = 0.8
        elif max_matches > 0:
            intensity = 0.6
        else:
            intensity = 0.4
        
        # 返回结果
        return {
            "emotion": detected_emotion,
            "intensity": intensity,
            "description": descriptions.get(detected_emotion, "普通的情绪状态"),
            "music_suggestion": music_suggestions.get(detected_emotion, "各种风格的音乐都可尝试"),
            "comfort_message": comfort_messages.get(detected_emotion, "音乐可以陪伴我们度过各种情绪")
        }
    
    @retry(tries=2, delay=1, backoff=2, logger=logger)
    def get_comfort_message(self, emotion, intensity, description):
        """
        根据情绪生成安慰消息
        
        参数:
            emotion: 情绪类型
            intensity: 情绪强度
            description: 情绪描述
            
        返回:
            安慰消息
        """
        logger.info(f"生成安慰消息: {emotion}")
        
        # 构建系统提示
        system_prompt = f"""
        你是一位富有同理心的心理咨询师和音乐治疗师，擅长通过关怀和音乐建议来安慰人们。
        请为一位处于以下情绪状态的用户生成一条温暖、真诚的安慰消息:
        
        情绪类型: {emotion}
        情绪强度: {intensity} (0-1之间，1为最强)
        情绪描述: {description}
        
        要求:
        1. 消息应该表达理解和同理心
        2. 不要使用过于专业的心理学术语
        3. 语气要温暖、友好，像好朋友一样
        4. 提到音乐可以如何帮助缓解或增强这种情绪
        5. 长度控制在2-3句话之内
        6. 使用"您"而不是"你"，保持尊重
        7. 消息应该自然流畅，不要过于刻板或公式化
        """
        
        try:
            # 调用AI服务生成安慰消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"为一位感到{emotion}的用户生成安慰消息"}
            ]
            response = self._call_ai_service(messages)
            
            if response and 'choices' in response:
                # 提取AI回复
                content = response['choices'][0]['message']['content']
                return content.strip()
            
            # 如果API调用失败，返回基本安慰消息
            return self._get_default_comfort_message(emotion)
        
        except Exception as e:
            logger.error(f"生成安慰消息失败: {e}")
            return self._get_default_comfort_message(emotion)
    
    def _get_default_comfort_message(self, emotion):
        """根据情绪类型返回默认安慰消息"""
        if emotion in ['sad', 'depressed', 'melancholy']:
            return "我能感受到您的悲伤，这是完全自然的情绪体验。音乐有时能成为我们情感的出口，或许能带给您一些慰藉和理解。"
        elif emotion in ['anxious', 'worried', 'stressed']:
            return "焦虑的感觉确实不舒服，但这是我们身体的自然反应。有些音乐有助于调节我们的神经系统，或许能帮助您找到一些平静。"
        elif emotion in ['angry', 'frustrated', 'irritated']:
            return "感到愤怒是完全正常的，这表明您关心的事物受到了挑战。适当的音乐可以成为情绪宣泄的安全方式，帮助您逐渐找回平衡。"
        elif emotion in ['happy', 'joyful', 'excited']:
            return "很高兴看到您的好心情！适合的音乐能帮助我们保持并放大这种积极情绪，让这美好的感受持续更长时间。"
        elif emotion in ['tired', 'exhausted', 'fatigued']:
            return "疲惫是我们需要休息的信号。某些音乐能帮助我们放松身心，而有些则能温和地提供新的能量，希望能帮助您恢复活力。"
        else:
            return "无论您现在感受如何，音乐都能成为您情感旅程中的良伴。希望通过合适的旋律，能为您带来一些舒适或共鸣。"
    
    @retry(tries=2, delay=1, backoff=2, logger=logger)
    def generate_response(self, system_prompt, user_message):
        """
        生成AI响应
        
        参数:
            system_prompt: 系统提示
            user_message: 用户消息
            
        返回:
            AI生成的响应
        """
        try:
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            response = self._call_ai_service(messages)
            
            if response and 'choices' in response:
                # 提取响应内容
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            
            return "抱歉，我现在无法处理您的请求。请稍后再试。"
                
        except Exception as e:
            logger.error(f"生成AI响应失败: {str(e)}")
            return "抱歉，我现在无法处理您的请求。请稍后再试。"
            
    def send_message(self, messages, system=None):
        """
        发送完整对话消息并获取响应
        
        参数:
            messages: 对话消息列表，每个消息包含role和content
            system: 可选的系统提示
            
        返回:
            AI生成的响应
        """
        try:
            # 准备请求消息
            request_messages = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(messages)
            
            response = self._call_ai_service(request_messages)
            
            if response and 'choices' in response:
                # 提取响应内容
                content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                return content
            
            return "抱歉，AI服务暂时不可用。不过您仍然可以浏览和评价音乐，系统将根据您的喜好提供个性化推荐。如需情感分析服务，请稍后再试。"
                
        except Exception as e:
            logger.error(f"发送消息失败: {str(e)}")
            return "抱歉，AI服务暂时不可用。不过您仍然可以浏览和评价音乐，系统将根据您的喜好提供个性化推荐。如需情感分析服务，请稍后再试。"

    @retry(tries=2, delay=1, backoff=2, logger=logger)
    def generate_proactive_question(self, user_history, user_preferences, current_emotion=None, conversation_count=0):
        """
        生成主动引导用户的问题，增强心理咨询师角色特性
        
        参数:
            user_history: 用户的对话历史
            user_preferences: 已知的用户偏好
            current_emotion: 当前检测到的情绪
            conversation_count: 与用户的对话次数
            
        返回:
            主动引导的问题
        """
        logger.info(f"生成主动引导问题: 情绪={current_emotion}, 对话次数={conversation_count}")
        
        # 转换用户历史为字符串摘要
        history_summary = "这是用户的第一次对话。" if not user_history else f"用户曾提到: {', '.join([h['text'][:30] for h in user_history[-3:]])}..."
        
        # 提取已知的偏好摘要
        known_genres = user_preferences.get('genres', {})
        known_artists = user_preferences.get('artists', [])
        known_moods = user_preferences.get('moods', {})
        
        preference_summary = "我们还不了解用户的音乐偏好。"
        if known_genres or known_artists or known_moods:
            preference_parts = []
            if known_genres:
                top_genres = sorted(known_genres.items(), key=lambda x: x[1], reverse=True)[:2]
                preference_parts.append(f"喜欢的流派: {', '.join([g[0] for g in top_genres])}")
            if known_artists:
                preference_parts.append(f"喜欢的艺术家: {', '.join(known_artists[:2])}")
            if known_moods:
                top_moods = sorted(known_moods.items(), key=lambda x: x[1], reverse=True)[:2]
                preference_parts.append(f"常见情绪: {', '.join([m[0] for m in top_moods])}")
            
            preference_summary = f"用户偏好: {'; '.join(preference_parts)}。"
        
        # 构建系统提示
        system_prompt = f"""
        你是一位专业的音乐心理咨询师，擅长通过引导性问题帮助用户探索他们的音乐偏好和情感需求。
        请根据以下信息，生成一个主动引导的问题:
        
        用户历史: {history_summary}
        用户偏好: {preference_summary}
        当前情绪: {current_emotion or '未知'}
        对话次数: {conversation_count}
        
        要求:
        1. 如果这是前几次对话(1-3次)，问题应该友好且基础，询问用户的一般音乐偏好
        2. 如果已经有5次以上对话，问题应该更深入，探索用户的情感与音乐的连接
        3. 如果知道用户当前情绪，问题应该表现出同理心，并与该情绪相关
        4. 问题应该开放式，避免是/否问题，鼓励用户分享更多
        5. 体现出专业音乐心理咨询师的身份，适当引入音乐心理学概念
        6. 措辞应该温暖、专业且有同理心，像真正的咨询师一样
        7. 问题长度控制在1-2句话，简洁有力
        
        只返回问题本身，不要有其他格式或引言。
        """
        
        try:
            # 调用AI服务生成主动问题
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请生成一个适合当前情况的主动引导问题"}
            ]
            response = self._call_ai_service(messages)
            
            if response and 'choices' in response:
                # 提取问题
                proactive_question = response.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                
                # 如果生成的内容为空，使用默认问题
                if not proactive_question:
                    raise ValueError("生成的问题为空")
                    
                return proactive_question
            
            # 使用默认问题
            raise Exception("生成主动问题的API调用失败")
                
        except Exception as e:
            logger.error(f"生成主动问题失败: {e}")
            
            # 返回默认问题
            default_questions = [
                "你能告诉我更多关于你喜欢的音乐类型吗？这可以帮助我了解你的音乐品味。",
                "最近有什么歌曲特别打动你或让你印象深刻吗？",
                "你通常在什么情况或心情下会特别想听音乐？",
                "是否有某种类型的音乐能够特别影响你的情绪？",
                "你是更喜欢探索新音乐，还是倾向于听熟悉的歌曲？这两种方式对我们的情绪有不同的影响。"
            ]
            
            import random
            return random.choice(default_questions) 