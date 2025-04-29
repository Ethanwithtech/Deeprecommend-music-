#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合音乐推荐系统训练脚本

这个脚本实现了混合推荐系统的训练流程，包括：
1. 处理MSD数据集
2. 集成Spotify API获取音乐特征
3. 训练深度学习模型
4. 训练混合推荐系统
5. 保存训练好的模型
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import pickle
import time

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('hybrid_recommender_train.log')
    ]
)
logger = logging.getLogger("HybridRecommenderTrainer")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练混合音乐推荐系统')
    
    # 数据集参数
    parser.add_argument('--msd_dir', type=str, default='C:/Users/dyc06/Desktop/Deeprecommend-music-',
                        help='MSD数据集根目录，包含h5文件和triplets文件')
    parser.add_argument('--h5_file', type=str, default='msd_summary_file.h5',
                        help='H5文件名，如果提供，会覆盖自动查找')
    parser.add_argument('--triplet_file', type=str, default='train_triplets.txt',
                        help='Triplet文件名，如果提供，会覆盖自动查找')
    parser.add_argument('--chunk_limit', type=int, default=1,
                        help='处理的数据块数上限(每块约100万条记录)')
    parser.add_argument('--sample_size', type=int, default=10000,
                        help='要训练的数据样本数量，如果指定，将随机抽取这么多条记录进行训练')
    
    # Spotify API参数
    parser.add_argument('--spotify_data', type=str, default=None,
                        help='Spotify API特征数据文件路径(如果有)')
    parser.add_argument('--no_spotify', action='store_true',
                        help='不使用Spotify API数据')
    
    # 训练参数
    parser.add_argument('--rating_style', type=str, default='log', 
                        choices=['log', 'linear', 'percentile'],
                        help='评分转换方式')
    parser.add_argument('--epochs', type=int, default=20,
                        help='深度学习模型训练轮数')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='深度学习模型训练批次大小')
    parser.add_argument('--skip_deep', action='store_true',
                        help='跳过深度学习模型训练')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='C:/Users/dyc06/Desktop/Deeprecommend-music-/models/trained',
                        help='输出目录')
    parser.add_argument('--model_name', type=str, default='hybrid_recommender_10k.pkl',
                        help='保存的模型文件名')
    
    # 其他参数
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式')
    parser.add_argument('--clean', action='store_true',
                        help='删除多余临时文件')
    
    return parser.parse_args()

def find_msd_files(msd_dir, h5_file, triplet_file):
    """查找MSD数据文件"""
    logger.info(f"在目录 {msd_dir} 中查找MSD数据文件...")
    
    h5_path = None
    triplet_path = None
    
    # 如果直接提供了文件名，直接使用
    if h5_file and os.path.exists(os.path.join(msd_dir, h5_file)):
        h5_path = os.path.join(msd_dir, h5_file)
    
    if triplet_file and os.path.exists(os.path.join(msd_dir, triplet_file)):
        triplet_path = os.path.join(msd_dir, triplet_file)
    
    # 如果没有直接提供，尝试查找
    if not h5_path and os.path.exists(msd_dir):
        h5_files = [f for f in os.listdir(msd_dir) if f.endswith('.h5')]
        if h5_files:
            h5_path = os.path.join(msd_dir, h5_files[0])
    
    if not triplet_path and os.path.exists(msd_dir):
        triplet_files = [f for f in os.listdir(msd_dir) if 'triplet' in f.lower() and f.endswith('.txt')]
        if triplet_files:
            triplet_path = os.path.join(msd_dir, triplet_files[0])
    
    return h5_path, triplet_path

def load_spotify_data(spotify_file):
    """加载Spotify API特征数据"""
    if not spotify_file or not os.path.exists(spotify_file):
        logger.warning(f"Spotify数据文件不存在: {spotify_file}")
        return None
    
    try:
        with open(spotify_file, 'rb') as f:
            spotify_data = pickle.load(f)
        logger.info(f"加载了 {len(spotify_data)} 个Spotify歌曲特征")
        return spotify_data
    except Exception as e:
        logger.error(f"加载Spotify数据失败: {str(e)}")
        return None

def train_hybrid_recommender(args):
    """训练混合推荐系统"""
    logger.info("开始训练混合音乐推荐系统...")
    
    # 查找MSD数据文件
    h5_path, triplet_path = find_msd_files(
        args.msd_dir, 
        args.h5_file, 
        args.triplet_file
    )
    
    if not h5_path or not triplet_path:
        logger.error("找不到必要的MSD数据文件，无法继续")
        return False
    
    logger.info(f"使用MSD文件: {h5_path} 和 {triplet_path}")
    
    # 加载Spotify特征数据（如果有）
    spotify_data = None
    if not args.no_spotify and args.spotify_data:
        spotify_data = load_spotify_data(args.spotify_data)
    
    # 创建输出目录
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"已创建或确认输出目录: {args.output_dir}")
    except Exception as e:
        logger.warning(f"创建输出目录失败: {str(e)}，将尝试使用当前目录")
        args.output_dir = "."
    
    try:
        # 导入必要的模块
        try:
            from backend.models.hybrid_music_recommender import HybridMusicRecommender
            
            # 应用sample_size补丁
            if args.sample_size:
                try:
                    from backend.sample_data_patch import apply_patch
                    apply_patch()
                    logger.info("已加载数据抽样补丁")
                except ImportError:
                    logger.warning("无法导入数据抽样补丁，将使用全部数据")
                    
        except ImportError:
            logger.error("无法导入HybridMusicRecommender模块，请确保安装了必要的依赖")
            return False
        
        # 初始化混合推荐系统
        start_time = time.time()
        recommender = HybridMusicRecommender(
            data_dir=args.msd_dir,
            use_msd=True,
            use_deep_learning=not args.skip_deep
        )
        
        # 使用MSD数据预训练模型
        logger.info("开始使用MSD数据预训练...")
        recommender.pretrain_with_msd(
            msd_dir=args.msd_dir,
            spotify_data_path=args.spotify_data,
            chunk_limit=args.chunk_limit,
            rating_style=args.rating_style,
            sample_size=args.sample_size
        )
        
        # 训练各个子模型
        logger.info("训练协同过滤模型...")
        recommender.train_collaborative_filtering()
        
        logger.info("训练基于内容的推荐模型...")
        recommender.train_content_based()
        
        logger.info("训练上下文感知模型...")
        recommender.train_context_aware()
        
        # 训练完整混合模型
        logger.info("训练完整混合模型...")
        recommender.train()
        
        # 保存模型
        model_path = os.path.join(args.output_dir, args.model_name)
        recommender.save_model(model_path)
        
        training_time = time.time() - start_time
        logger.info(f"混合推荐系统训练完成，耗时: {training_time:.2f}秒")
        logger.info(f"模型已保存到: {model_path}")
        
        # 测试模型
        logger.info("测试混合推荐系统...")
        test_recommender(recommender)
        
        return True
        
    except Exception as e:
        logger.error(f"训练混合推荐系统失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_recommender(recommender):
    """测试混合推荐系统"""
    try:
        # 确保数据已加载
        if not hasattr(recommender, 'ratings_df') or recommender.ratings_df is None:
            logger.warning("没有评分数据，无法测试推荐系统")
            return
        
        # 选择一个随机用户进行测试
        user_ids = recommender.ratings_df['user_id'].unique()
        if len(user_ids) == 0:
            logger.warning("没有有效用户，无法测试推荐系统")
            return
        
        test_user = np.random.choice(user_ids)
        logger.info(f"为测试用户 {test_user} 生成推荐...")
        
        # 生成混合推荐
        recommendations = recommender.recommend(test_user, top_n=5)
        
        # 展示推荐结果
        logger.info("测试推荐结果:")
        for i, (song_id, score, explanation) in enumerate(recommendations, 1):
            logger.info(f"{i}. {song_id}: {score:.4f} - {explanation}")
        
        # 测试情绪推荐
        moods = ["happy", "sad", "calm", "excited"]
        for mood in moods:
            logger.info(f"测试情绪 '{mood}' 的推荐:")
            mood_recs = recommender.get_hybrid_recommendations(
                test_user, 
                context={"mood": mood}, 
                top_n=3
            )
            for i, rec in enumerate(mood_recs, 1):
                if len(rec) >= 3:  # 确保rec至少有3个元素
                    song_id, score, explanation = rec[:3]
                    logger.info(f"{i}. {song_id}: {score:.4f} - {explanation}")
                else:
                    logger.info(f"{i}. {rec}")
        
        logger.info("推荐系统测试完成")
        
    except Exception as e:
        logger.error(f"测试推荐系统失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

def main():
    """主函数"""
    args = parse_args()
    
    # 设置日志级别
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 显示训练配置
    logger.info("混合推荐系统训练配置:")
    logger.info(f"- MSD目录: {args.msd_dir}")
    logger.info(f"- 数据块限制: {args.chunk_limit}")
    if args.sample_size:
        logger.info(f"- 训练样本大小: {args.sample_size}")
    logger.info(f"- 评分转换方式: {args.rating_style}")
    if args.no_spotify:
        logger.info("- Spotify API: 不使用")
    else:
        logger.info(f"- Spotify数据: {args.spotify_data if args.spotify_data else '无'}")
    logger.info(f"- 深度学习: {'跳过' if args.skip_deep else '启用'}")
    logger.info(f"- 输出目录: {args.output_dir}")
    
    # 训练模型
    success = train_hybrid_recommender(args)
    
    if success:
        logger.info("混合音乐推荐系统训练成功")
    else:
        logger.error("混合音乐推荐系统训练失败")
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 