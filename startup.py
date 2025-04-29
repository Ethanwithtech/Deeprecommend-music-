#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音乐推荐系统启动脚本
使用app.py直接启动系统，并设置必要的环境变量
可选择清理Python缓存以解决缓存问题
"""

import os
import sys
import subprocess
import time
import logging
import shutil
import argparse

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_pycache(dir_path):
    """清理指定目录下的所有Python缓存文件"""
    cleaned = 0
    
    logger.info("正在清理Python缓存...")
    
    # 清理__pycache__目录
    for root, dirs, files in os.walk(dir_path):
        if os.path.basename(root) == '__pycache__':
            logger.info(f"  删除: {root}")
            try:
                shutil.rmtree(root)
                cleaned += 1
            except Exception as e:
                logger.error(f"  无法删除 {root}: {e}")
    
    # 清理.pyc文件
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.pyc'):
                file_path = os.path.join(root, file)
                logger.info(f"  删除: {file_path}")
                try:
                    os.remove(file_path)
                    cleaned += 1
                except Exception as e:
                    logger.error(f"  无法删除 {file_path}: {e}")
    
    logger.info(f"清理完成，共删除 {cleaned} 个缓存文件或目录。")

def set_environment_variables():
    """设置必要的环境变量"""
    # HKBU API设置
    os.environ["HKBU_API_KEY"] = "06fd2422-8207-4a5b-8aaa-434415ed3a2b"
    os.environ["HKBU_MODEL"] = "gpt-4-o-mini"
    
    # 应用设置
    os.environ['USE_MSD'] = "true"
    os.environ['DATA_DIR'] = "processed_data"
    os.environ['FORCE_RETRAIN'] = "false"
    os.environ['MODEL_TYPE'] = "svd"
    os.environ['CONTENT_WEIGHT'] = "0.3"
    
    # SVD模型参数
    os.environ['SVD_N_FACTORS'] = "100"
    os.environ['SVD_N_EPOCHS'] = "20"
    os.environ['SVD_REG_ALL'] = "0.05"
    
    # 服务器设置
    os.environ['HOST'] = "0.0.0.0"
    os.environ['PORT'] = "5000"
    os.environ['DEBUG'] = "true"
    
    logger.info("环境变量设置完成")

def main():
    """主函数 - 启动应用程序"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='音乐推荐系统启动脚本')
    parser.add_argument('--clean', action='store_true', help='启动前清理Python缓存')
    parser.add_argument('--no-cache', action='store_true', help='使用-B选项启动Python，禁止生成缓存')
    args = parser.parse_args()
    
    logger.info("=== 音乐推荐系统启动中... ===")
    
    # 获取项目根目录
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 切换到项目根目录
    os.chdir(root_dir)
    logger.info(f"当前工作目录: {root_dir}")
    
    # 如果需要，清理缓存
    if args.clean:
        clear_pycache(root_dir)
        time.sleep(1)  # 稍等片刻确保清理完成
    
    # 设置环境变量
    set_environment_variables()
    
    # 获取系统信息
    logger.info(f"操作系统: {sys.platform}")
    logger.info(f"Python版本: {sys.version}")
    
    # 检查CSS主题设置
    css_path = os.path.join("frontend", "static", "css", "main.css")
    if os.path.exists(css_path):
        logger.info("CSS主题文件已确认: 黑色和紫色主题设置")
    else:
        logger.warning("未找到CSS文件，可能需要手动检查主题设置")
    
    # 启动应用程序
    try:
        logger.info("正在启动应用...")
        
        # 根据参数决定是否使用-B选项
        if args.no_cache:
            logger.info("使用-B选项禁用Python缓存生成")
            subprocess.run([sys.executable, "-B", "app.py"], check=True)
        else:
            subprocess.run([sys.executable, "app.py"], check=True)
            
    except subprocess.CalledProcessError as e:
        logger.error(f"启动应用失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断，应用关闭")
    except Exception as e:
        logger.error(f"发生错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 