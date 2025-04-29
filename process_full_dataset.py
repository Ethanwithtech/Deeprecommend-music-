"""
重新处理百万歌曲数据集，加载更多歌曲数据

此脚本会调用data_processing.py，重新处理MSD数据集
并且不设置采样限制（或使用更大的采样数），以加载更多歌曲
"""

import os
import sys
import logging
import argparse
from backend.data.data_processing import process_data

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('process_full_dataset')

def main():
    parser = argparse.ArgumentParser(description='处理百万歌曲数据集')
    parser.add_argument('--sample_size', type=int, default=100000, 
                       help='采样大小，设置为较大的值(如100000)或0表示处理全部数据')
    parser.add_argument('--data_dir', type=str, default='.', 
                       help='数据目录，包含msd_summary_file.h5和train_triplets.txt文件')
    parser.add_argument('--output_dir', type=str, default='processed_data', 
                       help='输出目录，处理后的数据将保存在这里')
    parser.add_argument('--force', action='store_true',
                       help='强制重新处理，即使输出文件已存在')
    
    args = parser.parse_args()
    
    # 检查数据文件是否存在
    metadata_file = os.path.join(args.data_dir, 'msd_summary_file.h5')
    triplets_file = os.path.join(args.data_dir, 'train_triplets.txt')
    
    if not os.path.exists(metadata_file):
        logger.error(f"元数据文件不存在: {metadata_file}")
        return False
    
    if not os.path.exists(triplets_file):
        logger.error(f"用户播放记录文件不存在: {triplets_file}")
        return False
    
    # 检查输出目录
    if os.path.exists(args.output_dir) and not args.force:
        # 检查是否已经处理过数据
        songs_metadata_file = os.path.join(args.output_dir, 'songs_metadata.pkl')
        if os.path.exists(songs_metadata_file):
            import pandas as pd
            try:
                songs_metadata = pd.read_pickle(songs_metadata_file)
                logger.info(f"已存在处理好的数据，包含 {len(songs_metadata)} 首歌曲")
                logger.info("如果要重新处理，请使用 --force 参数")
                return True
            except:
                logger.warning("无法读取现有数据文件，将重新处理")
    
    # 设置环境变量
    os.environ['SAMPLE_SIZE'] = str(args.sample_size)
    
    # 处理数据
    logger.info(f"开始处理数据，采样大小: {args.sample_size if args.sample_size > 0 else '全部数据'}")
    success = process_data(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        use_msd=True,
        sample_size=args.sample_size if args.sample_size > 0 else None
    )
    
    if success:
        logger.info("数据处理完成!")
        # 重启应用程序
        logger.info("请重启应用程序以加载新处理的数据")
        logger.info("可以使用 python start_clean.py 命令重启")
    else:
        logger.error("数据处理失败!")
    
    return success

if __name__ == "__main__":
    main() 