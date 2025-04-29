/**
 * 情感检测模块
 * 用于分析用户的情感状态并用于音乐推荐
 */

class EmotionDetector {
    constructor() {
        // 预定义的情感状态映射
        this.emotions = {
            "高兴": { valence: 0.8, energy: 0.7 },
            "平静": { valence: 0.6, energy: 0.3 },
            "悲伤": { valence: 0.2, energy: 0.3 },
            "愤怒": { valence: 0.3, energy: 0.8 },
            "兴奋": { valence: 0.7, energy: 0.9 },
            "放松": { valence: 0.6, energy: 0.2 },
            "焦虑": { valence: 0.4, energy: 0.7 }
        };
        
        // 简单的关键词匹配规则（前端简易实现）
        this.keywords = {
            "高兴": ["开心", "快乐", "欢乐", "高兴", "愉快", "喜悦", "好心情"],
            "平静": ["平静", "安宁", "祥和", "舒适", "安心", "平和"],
            "悲伤": ["难过", "伤心", "悲伤", "痛苦", "低落", "郁闷", "失落", "沮丧"],
            "愤怒": ["生气", "愤怒", "气愤", "恼火", "烦躁", "暴躁", "发怒"],
            "兴奋": ["兴奋", "激动", "振奋", "热血", "活力", "精神"],
            "放松": ["放松", "惬意", "休闲", "慵懒", "轻松", "自在"],
            "焦虑": ["焦虑", "紧张", "担心", "忧虑", "不安", "恐惧"]
        };
    }
    
    // 前端简单的情感分析（关键词匹配）
    analyzeLocally(text) {
        if (!text || text.trim() === '') {
            return { emotion: "平静", valence: 0.5, energy: 0.5 };
        }
        
        let matches = {};
        for (let emotion in this.keywords) {
            matches[emotion] = 0;
            this.keywords[emotion].forEach(keyword => {
                if (text.includes(keyword)) {
                    matches[emotion] += 1;
                }
            });
        }
        
        // 找出匹配度最高的情感
        let maxEmotion = "平静"; // 默认是平静
        let maxCount = 0;
        
        for (let emotion in matches) {
            if (matches[emotion] > maxCount) {
                maxCount = matches[emotion];
                maxEmotion = emotion;
            }
        }
        
        // 如果没有匹配，返回中性
        if (maxCount === 0) {
            return { emotion: "平静", valence: 0.5, energy: 0.5 };
        }
        
        return {
            emotion: maxEmotion,
            valence: this.emotions[maxEmotion].valence,
            energy: this.emotions[maxEmotion].energy
        };
    }
    
    // 使用后端API分析情感
    async detectFromText(text) {
        try {
            // 如果有后端API，使用以下代码
            // const response = await axios.post('/api/detect_emotion', { text });
            // return response.data;
            
            // 在没有后端API的情况下，使用前端简单的关键词匹配
            return this.analyzeLocally(text);
        } catch (error) {
            console.error('情感检测失败:', error);
            // 失败时返回中性情绪
            return { emotion: "平静", valence: 0.5, energy: 0.5 };
        }
    }
    
    // 根据音乐风格猜测情感
    mapGenreToEmotion(genre) {
        const genreToEmotion = {
            "流行": { emotion: "高兴", valence: 0.7, energy: 0.6 },
            "摇滚": { emotion: "愤怒", valence: 0.4, energy: 0.8 },
            "电子": { emotion: "兴奋", valence: 0.6, energy: 0.9 },
            "嘻哈": { emotion: "兴奋", valence: 0.6, energy: 0.7 },
            "古典": { emotion: "平静", valence: 0.7, energy: 0.3 },
            "爵士": { emotion: "放松", valence: 0.6, energy: 0.4 },
            "蓝调": { emotion: "悲伤", valence: 0.3, energy: 0.4 },
            "民谣": { emotion: "平静", valence: 0.5, energy: 0.3 }
        };
        
        return genreToEmotion[genre] || { emotion: "平静", valence: 0.5, energy: 0.5 };
    }
    
    /**
     * 根据情绪生成推荐理由
     * @param {string} emotion - 检测到的情绪
     * @returns {Object} - 中英文推荐理由
     */
    generateRecommendationReason(emotion) {
        const reasons = {
            '开心': {
                zh: `检测到您当前心情愉快，为您推荐符合这种情绪的欢快音乐。`,
                en: `Detected your happy mood, recommending upbeat music that matches this emotion.`
            },
            '悲伤': {
                zh: `根据您的忧伤情绪，为您推荐能引起共鸣或提升心情的音乐。`,
                en: `Based on your sad mood, recommending music that resonates with or uplifts your emotions.`
            },
            '愤怒': {
                zh: `检测到您有些愤怒，为您推荐能帮助释放情绪的有力音乐。`,
                en: `Detected your anger, recommending powerful music that helps release emotions.`
            },
            '紧张': {
                zh: `感觉到您有些紧张，为您推荐能帮助放松的轻柔音乐。`,
                en: `Sensing your anxiety, recommending gentle music to help you relax.`
            },
            '放松': {
                zh: `您看起来很放松，为您推荐能维持这种平静氛围的音乐。`,
                en: `You seem relaxed, recommending music to maintain this peaceful atmosphere.`
            },
            '疲倦': {
                zh: `检测到您可能有些疲倦，为您推荐能提升精神的节奏音乐。`,
                en: `Detected your tiredness, recommending rhythmic music to boost your energy.`
            },
            '兴奋': {
                zh: `您看起来很兴奋，为您推荐能配合这种活力的动感音乐。`,
                en: `You seem excited, recommending dynamic music to match this energy.`
            },
            '无聊': {
                zh: `检测到您可能有些无聊，为您推荐能带来新鲜感的有趣音乐。`,
                en: `Detected your boredom, recommending interesting music to bring fresh experiences.`
            },
            '怀旧': {
                zh: `感觉到您的怀旧情绪，为您推荐能唤起美好回忆的经典音乐。`,
                en: `Sensing your nostalgia, recommending classic music to evoke good memories.`
            }
        };
        
        // 检查是否有对应的预设推荐理由
        for (const key in reasons) {
            if (emotion.includes(key)) {
                return reasons[key];
            }
        }
        
        // 默认推荐理由
        return {
            zh: `根据您的心情"${emotion}"，为您推荐可能符合的音乐。`,
            en: `Based on your mood "${emotion}", recommending music that might match.`
        };
    }
} 