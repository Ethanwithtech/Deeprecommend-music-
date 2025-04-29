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
        self.model_name = "gpt-4-o-mini"  # 使用HKBU提供的模型
        self.api_version = "2024-05-01-preview"
        
        logger.info(f"AI服务初始化完成，使用提供商: {self.provider}, 模型: {self.model_name}")
    
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
        你是一个专业的情绪分析模型。请分析以下文本中表达的情绪，并以JSON格式返回结果。
        返回的JSON必须包含以下字段:
        - emotion: 情绪类型 (如: happy, sad, angry, excited, anxious, neutral等)
        - intensity: 情绪强度 (0.0到1.0之间的数值，0表示无，1表示极强)
        - description: 对情绪状态的简短描述
        - music_suggestion: 适合这种情绪的音乐类型建议
        
        只返回JSON对象，不要有其他文字。
        """
        
        # 调用HKBU API进行情绪分析
        try:
            # 构建API请求URL
            url = f"{self.api_url}/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ]
            
            # 构建请求头和数据
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取情绪分析结果
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            
            # 清理可能的非JSON前缀或后缀
            if content.strip().startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.strip().startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            try:
                emotion_data = json.loads(content)
                return emotion_data
            except json.JSONDecodeError:
                logger.error(f"无法解析情绪分析JSON结果: {content}")
                raise ValueError("情绪分析结果格式错误")
                
        except Exception as e:
            logger.error(f"情绪分析失败: {e}")
            # 返回默认情绪
            return {
                "emotion": "neutral",
                "intensity": 0.5,
                "description": "无法确定情绪状态",
                "music_suggestion": "流行音乐"
            }
    
    @retry(tries=2, delay=1, backoff=2, logger=logger)
    def get_comfort_message(self, emotion, intensity, description):
        """
        生成针对特定情绪的安慰消息
        
        参数:
            emotion: 情绪类型
            intensity: 情绪强度
            description: 情绪描述
            
        返回:
            安慰消息
        """
        logger.info(f"生成安慰消息: 情绪={emotion}, 强度={intensity}")
        
        # 构建系统提示，要求AI生成安慰消息
        system_prompt = f"""
        你是一个富有同理心的音乐推荐助手。用户当前情绪是"{emotion}"，强度为{intensity}，
        描述为"{description}"。请以20-50字的篇幅生成一段温暖的安慰消息，表达对用户情绪的理解，
        并简要提及音乐如何能够帮助他们。保持语气友好、真诚，不要过于夸张或做作。
        消息应该直接面向用户，不要包含引号或其他格式。
        """
        
        # 调用HKBU API生成安慰消息
        try:
            # 构建API请求URL
            url = f"{self.api_url}/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "请生成一条安慰消息"}
            ]
            
            # 构建请求头和数据
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取安慰消息
            comfort_message = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return comfort_message
            
        except Exception as e:
            logger.error(f"生成安慰消息失败: {e}")
            
            # 返回默认安慰消息
            default_messages = {
                "happy": "很高兴看到你心情不错！音乐能让这种愉悦感持续更久，我为你准备了一些适合的歌曲。",
                "sad": "我能感受到你的情绪有些低落。音乐有时能成为心灵的慰藉，让我为你找些能够共鸣的歌曲。",
                "angry": "看起来你有些烦躁。音乐可以帮助释放情绪，我找了一些适合的歌曲，希望能帮你平静下来。",
                "anxious": "我理解你现在可能感到有些焦虑。一些舒缓的音乐也许能帮助你放松心情，要试试吗？",
                "excited": "你的热情真是感染人！让我们用一些充满活力的音乐来配合你的兴奋心情吧！",
                "neutral": "无论你想要什么样的音乐体验，我都能帮你找到合适的歌曲，让我们开始探索吧。"
            }
            
            return default_messages.get(emotion, "我理解你现在的心情。音乐有神奇的力量，让我为你推荐一些可能适合你当前情绪的歌曲。")

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
            # 构建API请求URL
            url = f"{self.api_url}/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # 构建请求消息
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            
            # 构建请求头和数据
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取响应内容
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
                
        except Exception as e:
            logger.error(f"生成AI响应失败: {e}")
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
            # 构建API请求URL
            url = f"{self.api_url}/{self.model_name}/chat/completions?api-version={self.api_version}"
            
            # 准备请求消息
            request_messages = []
            if system:
                request_messages.append({"role": "system", "content": system})
            request_messages.extend(messages)
            
            # 构建请求头和数据
            headers = {
                "Content-Type": "application/json",
                "api-key": self.api_key
            }
            
            payload = {
                "messages": request_messages,
                "temperature": 0.7,
                "max_tokens": 2000
            }
            
            # 发送请求
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            
            # 提取响应内容
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            return content
                
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return "抱歉，AI服务暂时不可用。不过您仍然可以浏览和评价音乐，系统将根据您的喜好提供个性化推荐。如需情感分析服务，请稍后再试。" 