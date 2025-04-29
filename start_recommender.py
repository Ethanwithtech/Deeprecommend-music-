#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Million Song Dataset 推荐系统启动器
此脚本会启动预训练模型的Web界面
"""

import os
import sys
import argparse
import subprocess
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='启动MSD推荐系统')
    parser.add_argument('--model_path', type=str, default='models/trained/hybrid_recommender.pkl',
                        help='预训练模型的路径')
    parser.add_argument('--data_path', type=str, default='models/trained/processed_data',
                        help='处理好的数据目录')
    parser.add_argument('--port', type=int, default=5000,
                        help='Web服务器端口')
    parser.add_argument('--debug', action='store_true',
                        help='是否启用调试模式')
    return parser.parse_args()

def check_model_exists(model_path):
    """检查模型文件是否存在"""
    if not os.path.exists(model_path):
        logger.error(f"错误: 模型文件不存在: {model_path}")
        logger.info("请先运行预训练脚本: python backend/process_msd_data.py")
        return False
    return True

def check_data_exists(data_path):
    """检查数据目录是否存在"""
    if not os.path.exists(data_path):
        logger.error(f"错误: 数据目录不存在: {data_path}")
        logger.info("请先运行预训练脚本: python backend/process_msd_data.py")
        return False
    
    # 检查必要的数据文件
    required_files = ['songs.csv', 'interactions.csv', 'audio_features.csv']
    for file in required_files:
        file_path = os.path.join(data_path, file)
        if not os.path.exists(file_path):
            logger.error(f"错误: 数据文件不存在: {file_path}")
            return False
    
    return True

def start_server(args):
    """启动Web服务器"""
    # 设置环境变量
    env = os.environ.copy()
    env['MODEL_PATH'] = args.model_path
    env['DATA_PATH'] = args.data_path
    env['PORT'] = str(args.port)
    env['FLASK_DEBUG'] = 'true' if args.debug else 'false'
    
    # 构建启动命令
    cmd = [sys.executable, 'backend/web_demo.py']
    
    try:
        logger.info("启动MSD推荐系统服务器...")
        logger.info(f"模型路径: {args.model_path}")
        logger.info(f"数据路径: {args.data_path}")
        logger.info(f"服务器端口: {args.port}")
        logger.info(f"调试模式: {'开启' if args.debug else '关闭'}")
        
        process = subprocess.Popen(cmd, env=env)
        
        logger.info(f"服务器启动成功！请访问: http://localhost:{args.port}")
        logger.info("按Ctrl+C停止服务器")
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在停止服务器...")
        if process:
            process.terminate()
        logger.info("服务器已停止")
    except Exception as e:
        logger.error(f"启动服务器时出错: {str(e)}")
        if process:
            process.terminate()

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 检查模型和数据
    if not check_model_exists(args.model_path) or not check_data_exists(args.data_path):
        logger.info("提示: 如果你尚未处理MSD数据，请参考README_MSD_TRAINING.md文件获取详细指导")
        return 1
    
    # 启动服务器
    start_server(args)
    
    return 0

if __name__ == '__main__':
    sys.exit(main()) 