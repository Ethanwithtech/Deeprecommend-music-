"""
测试上下文感知推荐功能
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import logging

# 确保backend模块可以被导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """测试上下文感知推荐功能"""
    # 加载模型
    model_path = "models/msd_model.pkl"

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.info("成功加载模型")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        return

    # 创建模拟用户和交互数据
    logger.info("创建模拟用户和交互数据...")

    # 创建模拟用户
    user_id = "TEST_USER_001"

    # 创建模拟歌曲
    songs = [
        {"song_id": f"TEST_SONG_{i:03d}", "title": f"Test Song {i}",
         "artist_name": f"Test Artist {i//5}", "genre": "pop" if i % 3 == 0 else "rock" if i % 3 == 1 else "electronic"}
        for i in range(100)
    ]
    songs_df = pd.DataFrame(songs)

    # 创建模拟交互
    interactions = []
    for i in range(20):
        song_id = f"TEST_SONG_{i:03d}"
        interactions.append({
            "user_id": user_id,
            "song_id": song_id,
            "rating": np.random.randint(3, 6),  # 3-5的随机评分
            "timestamp": int(pd.Timestamp.now().timestamp()) - np.random.randint(0, 60*60*24*30),  # 过去30天内
            "emotion": np.random.choice(["happy", "sad", "relaxed", "excited"]),
            "activity": np.random.choice(["studying", "working", "exercising", "relaxing"]),
            "listening_time": np.random.choice(["morning", "afternoon", "evening", "night"]),
            "device_type": np.random.choice(["mobile", "desktop", "tablet", "speaker"])
        })
    interactions_df = pd.DataFrame(interactions)

    # 添加数据到模型
    logger.info("添加数据到模型...")
    for _, row in interactions_df.iterrows():
        model.add_new_rating(row["user_id"], row["song_id"], row["rating"],
                        context={"emotion": row["emotion"],
                                "activity": row["activity"],
                                "listening_time": row["listening_time"],
                                "device_type": row["device_type"]})

    # 测试不同上下文下的推荐
    contexts = [
        {"emotion": "happy", "activity": "exercising"},
        {"emotion": "sad", "activity": "relaxing"},
        {"emotion": "relaxed", "activity": "studying"},
        {"emotion": "excited", "activity": "socializing"}
    ]

    for i, context in enumerate(contexts):
        logger.info(f"\n测试上下文 {i+1}: {context}")
        recommendations = model.recommend(user_id, context=context, top_n=5)

        logger.info(f"上下文: {context}")
        logger.info("推荐结果:")
        for j, rec in enumerate(recommendations):
            song_id = rec.get('song_id', 'unknown')
            score = rec.get('predicted_score', 0.0)
            title = rec.get('title', 'Unknown Title')
            artist = rec.get('artist_name', 'Unknown Artist')
            explanation = rec.get('explanation', 'No explanation available')
            logger.info(f"{j+1}. {title} by {artist} (ID: {song_id}, 分数: {score:.2f}) - {explanation}")

    # 保存更新后的模型
    logger.info("保存更新后的模型...")
    model.save_model(model_path)
    logger.info("测试完成")

if __name__ == "__main__":
    main()
