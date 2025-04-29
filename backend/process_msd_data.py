import argparse
from models.msd_processor import MSDDataProcessor
from models.hybrid_recommender import HybridRecommender
from sklearn.model_selection import train_test_split
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="处理MSD数据并训练混合推荐模型")
    parser.add_argument("--h5_file", type=str, default=r"C:\Users\dyc06\Desktop\Deeprecommend-music-\msd_summary_file.h5", help="MSD的h5文件路径")
    parser.add_argument("--triplets_file", type=str, default=r"C:\Users\dyc06\Desktop\Deeprecommend-music-\train_triplets.txt\train_triplets.txt", help="MSD的triplets文件路径")
    parser.add_argument("--output_dir", type=str, default="processed_data", help="处理后数据的输出目录")
    parser.add_argument("--model_path", type=str, default="models/hybrid_model.pkl", help="模型保存路径")
    parser.add_argument("--chunk_limit", type=int, default=None, help="处理的数据块数量限制(用于测试)")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # 1. 处理MSD数据
    processor = MSDDataProcessor(output_dir=args.output_dir)
    songs, interactions, audio_features, user_features = processor.process_msd_data(
        args.h5_file, args.triplets_file, chunk_limit=args.chunk_limit
    )
    
    # 确保数据类型正确
    if 'song_id' in interactions.columns:
        interactions['song_id'] = interactions['song_id'].astype(str)
    if 'user_id' in interactions.columns:
        interactions['user_id'] = interactions['user_id'].astype(str)
    if 'song_id' in songs.columns:
        songs['song_id'] = songs['song_id'].astype(str)
    
    # 仅转换评分列为数值类型
    if 'rating' in interactions.columns:
        interactions['rating'] = pd.to_numeric(interactions['rating'], errors='coerce')
        interactions.dropna(subset=['rating'], inplace=True)
    
    # 创建数据字典
    data = {'songs': songs, 'interactions': interactions, 
            'audio_features': audio_features, 'user_features': user_features}
    
    # 2. 分割训练集和测试集
    train_data, test_data = train_test_split(interactions, test_size=0.2, random_state=42)
    print(f"训练集: {len(train_data)} 条记录, 测试集: {len(test_data)} 条记录")
    
    # 3. 训练混合推荐模型
    model = HybridRecommender()
    model.train(
        interactions=train_data,
        audio_features=audio_features,
        songs=songs,
        user_features=user_features
    )
    
    # 4. 保存模型
    model.save_model(args.model_path)
    
    print(f"MSD数据处理和模型训练完成! 模型已保存到 {args.model_path}")

if __name__ == "__main__":
    main() 