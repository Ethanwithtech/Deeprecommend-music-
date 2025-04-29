"""
简单的协同过滤模型实现，用于替代Surprise库
"""
import random
import numpy as np

class Prediction:
    """预测结果类"""
    def __init__(self, uid, iid, est):
        self.uid = uid
        self.iid = iid
        self.est = est

class SimpleCF:
    """简单的协同过滤模型"""
    def __init__(self):
        self.user_means = {}
        self.global_mean = 3.0  # 默认平均评分
        
    def predict(self, user_id, song_id):
        """预测用户对歌曲的评分"""
        # 如果用户有平均评分，使用它，否则使用全局平均值
        user_mean = self.user_means.get(user_id, self.global_mean)
        
        # 添加一些随机性，使推荐更多样化
        random_factor = random.uniform(-0.5, 0.5)
        
        # 预测评分 = 用户平均评分 + 随机因子
        predicted_rating = min(5.0, max(1.0, user_mean + random_factor))
        
        return Prediction(user_id, song_id, predicted_rating)
