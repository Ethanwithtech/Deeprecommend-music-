#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
高效混合推荐模型训练启动器
使用固定的MSD数据路径和Spotify API启动训练流程，实现高效混合核心推荐算法
"""

import os
import sys
import time
import subprocess
import argparse
import shutil
import logging
import platform
import signal
from pathlib import Path

# 固定的数据文件路径
MSD_PATH = "C:/Users/dyc06/Desktop/Deeprecommend-music-"
H5_FILE = "C:/Users/dyc06/Desktop/Deeprecommend-music-/msd_summary_file.h5"
TRIPLET_FILE = "C:/Users/dyc06/Desktop/Deeprecommend-music-/train_triplets.txt/train_triplets.txt"

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", f"msd_training_{time.strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("msd_trainer")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练Million Song Dataset推荐系统')
    
    # 数据文件参数
    parser.add_argument('--h5_file', type=str, default=H5_FILE,
                        help='MSD元数据文件路径')
    parser.add_argument('--triplet_file', type=str, default=TRIPLET_FILE,
                        help='MSD三元组数据文件路径')
    
    # 处理控制参数
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='处理后数据保存目录')
    parser.add_argument('--force_process', action='store_true',
                        help='强制重新处理数据，忽略已存在的处理文件')
    parser.add_argument('--chunk_limit', type=int, default=None,
                        help='限制处理的数据块数量(用于测试)')
    parser.add_argument('--max_interactions', type=int, default=None,
                        help='限制训练使用的最大交互记录数量')
    
    # Spotify参数
    parser.add_argument('--no_spotify', action='store_true',
                        help='不使用Spotify API丰富数据')
    parser.add_argument('--spotify_max_songs', type=int, default=1000,
                        help='使用Spotify处理的最大歌曲数')
    parser.add_argument('--spotify_batch_size', type=int, default=50,
                        help='Spotify API批处理大小')
    parser.add_argument('--spotify_workers', type=int, default=5,
                        help='Spotify并行处理线程数')
    parser.add_argument('--spotify_strategy', type=str, default='popular', 
                        choices=['all', 'popular', 'diverse'],
                        help='Spotify处理策略(all=全部, popular=热门优先, diverse=多样性优先)')
    parser.add_argument('--spotify_cache_file', type=str, default=None,
                        help='Spotify缓存文件路径')
    
    # 模型控制参数
    parser.add_argument('--skip_deep', action='store_true',
                        help='跳过深度学习模型训练')
    parser.add_argument('--skip_hybrid', action='store_true',
                        help='跳过混合模型训练')
    parser.add_argument('--epochs', type=int, default=5,
                        help='深度学习模型训练轮数')
    
    # 新增参数
    parser.add_argument('--log_level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="日志级别 (默认: INFO)")
    
    # 添加用户采样和过滤控制参数
    parser.add_argument('--user_sample', type=int, default=None,
                        help='限制使用的用户数量，随机采样指定数量的用户')
    parser.add_argument('--no_filter_inactive_users', action='store_true',
                        help='不过滤不活跃用户，保留所有用户')
    
    return parser.parse_args()

def build_command(args):
    """构建训练命令"""
    # MSD数据路径 - 直接使用传入的参数，这已经是完整路径
    h5_path = args.h5_file  # 已经是完整路径
    triplet_path = args.triplet_file  # 已经是完整路径
    
    # 构建命令列表
    cmd = [sys.executable, 'backend/train_msd_with_deep.py']
    
    # 添加数据路径参数
    cmd.extend(['--h5_file', h5_path])
    cmd.extend(['--triplet_file', triplet_path])
    
    # 添加输出目录参数
    if hasattr(args, 'output_dir'):
        cmd.extend(['--output_dir', args.output_dir])
    
    # 添加处理参数
    if args.force_process:
        cmd.append('--force_process')
    
    # 添加分块限制参数
    if hasattr(args, 'chunk_limit') and args.chunk_limit:
        cmd.extend(['--chunk_limit', str(args.chunk_limit)])
    
    # 添加交互数限制参数
    if hasattr(args, 'max_interactions') and args.max_interactions:
        cmd.extend(['--max_interactions', str(args.max_interactions)])
    
    # 添加评分方法参数
    cmd.extend(['--rating_method', 'log'])
    
    # 始终使用Spotify API进行增强
    cmd.append('--use_spotify')
    
    # 添加Spotify凭证 - 直接使用默认值
    cmd.extend(['--spotify_client_id', 'bdfa10b0a8bf49a3a413ba67d2ff1706'])
    cmd.extend(['--spotify_client_secret', 'b8e97ad8e96043b4b0d768d3e3c568b4'])
    
    # 添加训练参数
    cmd.extend(['--epochs', str(args.epochs)])
    cmd.extend(['--batch_size', '256'])
    cmd.extend(['--learning_rate', '0.001'])
    cmd.extend(['--embedding_dim', '64'])
    
    # 内存优化 - 通过环境变量添加TensorFlow内存优化
    if hasattr(args, 'optimize_memory') and args.optimize_memory:
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
        os.environ['TF_MEMORY_ALLOCATION'] = 'greedy'
    
    # 添加Spotify参数
    if hasattr(args, 'no_spotify') and args.no_spotify:
        cmd.append('--no_spotify')
    else:
        # 只有当不禁用Spotify时，才添加这些参数
        if hasattr(args, 'spotify_max_songs'):
            cmd.extend(['--spotify_max_songs', str(args.spotify_max_songs)])
        if hasattr(args, 'spotify_batch_size'):
            cmd.extend(['--spotify_batch_size', str(args.spotify_batch_size)])
        if hasattr(args, 'spotify_workers'):
            cmd.extend(['--spotify_workers', str(args.spotify_workers)])
        if hasattr(args, 'spotify_strategy'):
            cmd.extend(['--spotify_strategy', args.spotify_strategy])
        if hasattr(args, 'spotify_cache_file') and args.spotify_cache_file:
            cmd.extend(['--spotify_cache_file', args.spotify_cache_file])
    
    # 添加模型控制参数
    if hasattr(args, 'skip_deep') and args.skip_deep:
        cmd.append('--skip_deep')
    
    if hasattr(args, 'skip_hybrid') and args.skip_hybrid:
        cmd.append('--skip_hybrid')
    
    # 添加日志级别参数
    if hasattr(args, 'log_level') and args.log_level:
        cmd.extend(['--log_level', args.log_level])
    
    # 添加用户采样参数
    if hasattr(args, 'user_sample') and args.user_sample:
        cmd.extend(['--user_sample', str(args.user_sample)])
    
    # 添加不过滤不活跃用户参数
    if hasattr(args, 'no_filter_inactive_users') and args.no_filter_inactive_users:
        cmd.append('--no_filter_inactive_users')
    
    return cmd

def clean_temp_files():
    """清理多余的临时文件"""
    logger.info("清理多余文件和文件夹...")
    
    # 要删除的目录列表
    dirs_to_remove = ['msd_processed', 'msd_processed_data', '__pycache__', 'backend/__pycache__']
    
    # 临时缓存文件(.cache等)
    files_to_remove = ['.cache', '*.pyc', 'backend/*.pyc', 'backend/models/*.pyc']
    
    # 处理目录
    for dir_pattern in dirs_to_remove:
        for dir_path in Path('.').glob(dir_pattern):
            if dir_path.exists():
                try:
                    if dir_path.is_dir():
                        shutil.rmtree(dir_path)
                        logger.info(f"✓ 已删除目录: {dir_path}")
                except Exception as e:
                    logger.error(f"× 无法删除目录 {dir_path}: {e}")
    
    # 处理文件
    for file_pattern in files_to_remove:
        for file_path in Path('.').glob(file_pattern):
            if file_path.exists() and file_path.is_file():
                try:
                    file_path.unlink()
                    logger.info(f"✓ 已删除文件: {file_path}")
                except Exception as e:
                    logger.error(f"× 无法删除文件 {file_path}: {e}")
    
    logger.info("清理完成！")

def display_system_info():
    """显示系统信息"""
    logger.info("=== 系统信息 ===")
    logger.info(f"操作系统: {platform.platform()}")
    logger.info(f"Python版本: {platform.python_version()}")
    
    # 检查GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"检测到GPU: {len(gpus)}个")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i+1}: {gpu}")
        else:
            logger.info("未检测到GPU，将使用CPU训练（较慢）")
    except ImportError:
        logger.info("TensorFlow未安装或无法导入")
    except Exception as e:
        logger.error(f"检查GPU时出错: {e}")
    
    # 检查内存
    try:
        import psutil
        vm = psutil.virtual_memory()
        logger.info(f"系统内存: 总计={vm.total/1024**3:.1f}GB, 可用={vm.available/1024**3:.1f}GB")
    except ImportError:
        logger.info("无法检查系统内存信息(psutil未安装)")
    
    logger.info("===============")

def handle_keyboard_interrupt(process):
    """处理键盘中断"""
    def signal_handler(sig, frame):
        if process and process.poll() is None:
            logger.warning("检测到用户中断，正在优雅地终止训练...")
            if platform.system() == 'Windows':
                # Windows使用taskkill终止进程树
                subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
            else:
                # Linux/Mac使用进程组终止
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            logger.info("训练已终止")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    # 基本依赖
    dependencies = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('tensorflow', 'tensorflow'),
        ('h5py', 'h5py'),
        ('sklearn', 'scikit-learn')
    ]
    
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
        except ImportError:
            missing_deps.append(package_name)
    
    if missing_deps:
        logger.warning(f"缺少以下依赖库: {', '.join(missing_deps)}")
        logger.warning("请使用pip安装这些依赖: pip install " + " ".join(missing_deps))
        response = input("是否继续尝试运行? (y/n): ")
        return response.lower() == 'y'
    
    return True

def main():
    """主函数"""
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 解析命令行参数
    args = parse_args()
    
    # 如果是清理模式，只执行清理
    if hasattr(args, 'clean') and args.clean:
        clean_temp_files()
        return
    
    # 显示系统信息
    display_system_info()
    
    # 确认训练配置
    logger.info(f"训练配置:")
    logger.info(f"- 数据路径: {args.h5_file}")
    logger.info(f"- 完整训练 (无限制)")
    logger.info(f"- 包含深度学习模型: 是")
    logger.info(f"- 训练轮数: {args.epochs}")
    logger.info(f"- 批次大小: 256")
    logger.info(f"- 学习率: 0.001")
    logger.info(f"- 嵌入维度: 64")
    
    # 构建命令
    cmd = build_command(args)
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 启动训练进程
    process = None
    try:
        # 设置键盘中断处理
        process = subprocess.Popen(cmd)
        handle_keyboard_interrupt(process)
        
        # 等待进程完成
        process.wait()
        
        # 检查执行结果
        if process.returncode != 0:
            logger.error(f"训练失败，返回代码: {process.returncode}")
            sys.exit(process.returncode)
        
    except Exception as e:
        logger.error(f"训练执行出错: {e}")
        if process and process.poll() is None:
            process.terminate()
        sys.exit(1)
    
    # 计算训练时间
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"训练完成！总耗时: {int(hours)}小时 {int(minutes)}分钟 {seconds:.1f}秒")
    logger.info(f"训练结果:")
    logger.info(f"- 混合推荐模型: models/hybrid_model.pkl")
    logger.info(f"- 深度学习模型: models/deep_model/")
    logger.info(f"- 处理后数据: processed_data/")
    logger.info(f"- 训练日志: logs/")

if __name__ == "__main__":
    main() 