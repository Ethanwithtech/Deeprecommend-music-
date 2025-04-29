import logging
import os
import json
import requests

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EmotionAnalyzer:
    """
    情绪分析器，用于分析用户消息中的情绪，并提供适合该情绪的音乐推荐
    """
    
    def __init__(self, api_key=None):
        """
        初始化情绪分析器
        
        参数:
            api_key: 可选的API密钥，用于调用外部情绪分析服务
        """
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.provider = 'ANTHROPIC' if self.api_key else 'INTERNAL'
        
        # 情绪类别和对应的音乐类型映射
        self.emotion_music_map = {
            'happy': ['流行', '舞曲', '轻快的摇滚', '电子音乐'],
            'sad': ['抒情', '民谣', '古典', '爵士', '蓝调'],
            'angry': ['重金属', '硬摇滚', '说唱', '朋克'],
            'anxious': ['轻音乐', '纯音乐', '环境音乐', '自然声音'],
            'neutral': ['流行', '轻摇滚', '爵士', '电子'],
            'nostalgic': ['复古', '经典', '民谣', '60-90年代流行'],
            'excited': ['电子舞曲', '快节奏流行', '摇滚', 'K-Pop'],
            'calm': ['纯音乐', '古典', '轻爵士', '环境音乐'],
            'bored': ['新兴流派', '实验音乐', '世界音乐', '融合爵士'],
            'lonely': ['抒情', '民谣', '温暖的声音', '治愈系音乐']
        }
        
        logger.info(f"情绪分析器初始化完成，使用提供商: {self.provider}")
    
    def analyze_emotion(self, message):
        """
        分析用户消息中的情绪状态
        
        参数:
            message: 用户消息
            
        返回:
            情绪分析结果: {
                'emotion': 主要情绪，如'happy', 'sad', 'angry', 'anxious', 'neutral'等,
                'intensity': 情绪强度 (0-1),
                'description': 情绪描述,
                'music_suggestion': 音乐类型建议
            }
        """
        logger.info(f"分析用户情绪: {message[:50]}...")
        
        # 如果没有API密钥，使用内部简单规则分析
        if self.provider == 'INTERNAL':
            return self._internal_emotion_analysis(message)
            
        # 使用Claude API进行分析
        system_prompt = """
        你是一个专业的情绪分析助手。请分析用户消息中表达的情绪状态，并提供适合该情绪的音乐类型建议。
        你需要返回一个JSON格式的结果，包含以下字段：
        - emotion: 主要情绪类别 (happy, sad, angry, anxious, calm, excited, nostalgic, neutral等)
        - intensity: 情绪强度 (0到1之间的数值，1表示最强烈)
        - description: 简短描述用户的情绪状态
        - music_suggestion: 适合该情绪的音乐类型或风格建议
        
        仅返回JSON，不要有其他文字。
        """
        
        try:
            # 准备请求
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": "claude-3-haiku-20240307",
                "max_tokens": 300,
                "temperature": 0.3,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": message}
                ]
            }
            
            # 发送请求
            response = requests.post(self.api_url, headers=headers, json=data)
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            content = result.get('content', [{}])[0].get('text', '{}')
            
            # 清理可能的非JSON前缀或后缀
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0].strip()
            elif '```' in content:
                content = content.split('```')[1].split('```')[0].strip()
            
            # 解析JSON
            emotion_data = json.loads(content)
            logger.info(f"情绪分析结果: {emotion_data['emotion']}, 强度: {emotion_data['intensity']}")
            return emotion_data
            
        except Exception as e:
            logger.error(f"情绪分析API调用失败: {e}")
            # 如果API调用失败，回退到内部分析
            return self._internal_emotion_analysis(message)
    
    def _internal_emotion_analysis(self, message):
        """
        使用简单规则分析情绪（当外部API不可用时的后备方案）
        
        参数:
            message: 用户消息
            
        返回:
            情绪分析结果
        """
        message = message.lower()
        
        # 情绪关键词
        emotion_keywords = {
            'happy': ['开心', '高兴', '快乐', '喜悦', '兴奋', '快活', '愉快', '欢乐', '幸福', '笑'],
            'sad': ['难过', '伤心', '悲伤', '痛苦', '悲痛', '哭', '忧郁', '忧伤', '难受', '哀伤'],
            'angry': ['生气', '愤怒', '恼火', '恨', '怒火', '烦躁', '不满', '讨厌', '烦人', '气愤'],
            'anxious': ['紧张', '焦虑', '担心', '害怕', '恐惧', '忧虑', '不安', '慌张', '压力', '惶恐'],
            'nostalgic': ['怀念', '回忆', '思念', '过去', '从前', '童年', '旧时', '昔日', '怀旧', '记忆'],
            'calm': ['平静', '安宁', '安详', '放松', '祥和', '宁静', '舒适', '安心', '轻松', '心静'],
            'excited': ['激动', '热情', '振奋', '亢奋', '活力', '冲动', '热烈', '洋溢', '活跃', '奔放'],
            'bored': ['无聊', '厌烦', '乏味', '枯燥', '烦闷', '单调', '索然', '腻烦', '懒散', '兴致缺缺'],
            'lonely': ['孤独', '寂寞', '独处', '寂寥', '空虚', '单独', '冷清', '孤寂', '孤单', '清冷']
        }
        
        # 计算匹配的关键词数量
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in message)
            if score > 0:
                emotion_scores[emotion] = score
        
        # 如果没有匹配的情绪关键词，默认为neutral
        if not emotion_scores:
            return {
                'emotion': 'neutral',
                'intensity': 0.5,
                'description': '情绪较为平静，没有明显的情绪波动',
                'music_suggestion': '适合一般聆听的流行音乐或轻音乐'
            }
        
        # 确定主要情绪
        primary_emotion = max(emotion_scores, key=emotion_scores.get)
        
        # 计算情绪强度 (0-1)
        intensity = min(emotion_scores[primary_emotion] / 3, 1.0)
        
        # 为情绪生成描述
        descriptions = {
            'happy': '您似乎心情很好，流露出快乐和满足的情绪',
            'sad': '您似乎感到有些低落或悲伤',
            'angry': '您似乎感到有些愤怒或不满',
            'anxious': '您似乎感到紧张或担忧',
            'nostalgic': '您似乎在回忆过去，有些怀旧情绪',
            'calm': '您似乎感到平静和安宁',
            'excited': '您似乎感到兴奋和热情',
            'bored': '您似乎感到有些无聊或乏味',
            'lonely': '您似乎感到有些孤独或寂寞'
        }
        description = descriptions.get(primary_emotion, '您的情绪状态')
        
        # 为情绪推荐音乐类型
        music_options = self.emotion_music_map.get(primary_emotion, ['流行音乐'])
        import random
        music_suggestion = random.choice(music_options)
        
        return {
            'emotion': primary_emotion,
            'intensity': intensity,
            'description': description,
            'music_suggestion': music_suggestion
        }
    
    def get_comfort_message(self, emotion, intensity, description):
        """
        根据情绪类型和强度生成安慰话语
        
        参数:
            emotion: 情绪类型
            intensity: 情绪强度 (0-1)
            description: 情绪描述
            
        返回:
            安慰话语
        """
        if emotion in ['sad', 'depressed', 'upset', 'disappointed', 'heartbroken', 'melancholy']:
            if intensity > 0.7:
                return f"我理解您现在可能感到很沮丧。{description}的感觉确实很难受，请记住这些情绪都是暂时的，您不必独自面对。"
            else:
                return f"看起来您有些低落。{description}是很常见的情绪，有时候通过音乐发泄一下情感是很好的方式。"
                
        elif emotion in ['anxious', 'stressed', 'worried', 'nervous', 'overwhelmed']:
            if intensity > 0.7:
                return f"我能感受到您现在很焦虑。{description}时，音乐可以帮助我们平静下来，调整呼吸，找回平静。"
            else:
                return f"您似乎有些紧张。{description}的时候，适当放松一下，听些舒缓的音乐或许会有所帮助。"
                
        elif emotion in ['angry', 'frustrated', 'irritated', 'annoyed']:
            if intensity > 0.7:
                return f"您现在似乎很生气。{description}的感觉确实令人难受，但请记住，找到释放情绪的健康方式很重要。"
            else:
                return f"您看起来有些烦躁。{description}是很正常的情绪反应，有时候通过音乐可以帮助缓解这种感觉。"
                
        elif emotion in ['happy', 'excited', 'elated', 'joyful', 'cheerful']:
            return f"您现在看起来心情很好！{description}真是美妙的感觉，让我们用欢快的音乐来延续这种好心情。"
            
        elif emotion in ['nostalgic', 'sentimental', 'longing']:
            return f"您似乎有些怀旧。{description}让我们回忆过去的美好时光，音乐常常能唤起那些珍贵的记忆。"
            
        elif emotion in ['bored', 'tired', 'exhausted', 'fatigued']:
            return f"您可能感到有些疲惫或无聊。{description}时，一些振奋精神的音乐或许能帮助您重新找回活力。"
            
        elif emotion in ['lonely', 'isolated', 'abandoned']:
            return f"您似乎感到有些孤独。{description}是我们都会经历的情绪，音乐有时候能成为心灵的陪伴。"
            
        else:  # neutral or other emotions
            return f"谢谢您分享您的心情。音乐是表达和调节情绪的好方法，让我为您选择一些适合现在心情的歌曲。"

# 示例使用
if __name__ == "__main__":
    analyzer = EmotionAnalyzer()
    
    # 测试不同情绪的消息
    test_messages = [
        "今天真是个好日子，我心情特别好！",
        "我感到很难过，最近发生了很多不开心的事情",
        "我很生气，为什么事情总是不顺利",
        "我有点担心未来会怎样，有些焦虑",
        "我很想念以前的朋友和时光"
    ]
    
    for msg in test_messages:
        result = analyzer.analyze_emotion(msg)
        print(f"\n消息: {msg}")
        print(f"分析结果: {result['emotion']}, 强度: {result['intensity']}")
        print(f"描述: {result['description']}")
        print(f"音乐建议: {result['music_suggestion']}")
        comfort = analyzer.get_comfort_message(result['emotion'], result['intensity'], result['description'])
        print(f"安慰话语: {comfort}") 