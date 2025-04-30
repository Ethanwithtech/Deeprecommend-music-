"""
音乐推荐系统主应用程序入口点
这个文件导入重组后的模块，并启动Flask应用程序
"""

import sys
import os
import logging
import webbrowser
import threading
import time
from flask import Flask, render_template, request, jsonify, redirect, url_for

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 避免使用.env文件，直接设置环境变量
logger.info("设置环境变量...")
# 设置默认环境变量
os.environ.setdefault('USE_MSD', 'true')  # 默认使用百万歌曲数据集
os.environ.setdefault('DATA_DIR', 'processed_data')  # 默认数据目录
os.environ.setdefault('FORCE_RETRAIN', 'false')  # 默认不强制重训
os.environ.setdefault('MODEL_TYPE', 'svd')  # 默认使用SVD模型
os.environ.setdefault('SVD_N_FACTORS', '100')  # SVD模型参数: 特征数量
os.environ.setdefault('SVD_N_EPOCHS', '20')  # SVD模型参数: 训练轮数
os.environ.setdefault('SVD_REG_ALL', '0.05')  # SVD模型参数: 正则化参数
os.environ.setdefault('CONTENT_WEIGHT', '0.3')  # 混合推荐中内容推荐的权重
os.environ.setdefault('TOP_N', '10')  # 推荐结果数量
os.environ.setdefault('DEBUG', 'true')  # 调试模式
os.environ.setdefault('PORT', '5000')  # 端口
os.environ.setdefault('HOST', '0.0.0.0')  # 主机

# 手动设置Spotify API凭据（从.env中获取）
try:
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                os.environ[key] = value
    logger.info("已从.env加载Spotify API凭据")
except Exception as e:
    logger.warning(f"加载.env文件出错: {str(e)}")
    # 尝试设置默认Spotify凭据
    if 'SPOTIFY_CLIENT_ID' not in os.environ:
        os.environ['SPOTIFY_CLIENT_ID'] = 'bdfa10b0a8bf49a3a413ba67d2ff1706'
        os.environ['SPOTIFY_CLIENT_SECRET'] = 'b8e97ad8e96043b4b0d768d3e3c568b4'
        logger.info("已使用默认Spotify API凭据")

# 将项目根目录添加到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 设置工作目录
os.chdir(root_dir)
logger.info(f"设置工作目录为: {root_dir}")

# 检查模型文件是否存在
model_path = os.path.join(root_dir, 'models', 'trained', 'hybrid_recommender_10k.pkl')
if os.path.exists(model_path):
    logger.info(f"找到预训练模型: {model_path}")
else:
    logger.warning(f"未找到预训练模型: {model_path}，将使用基本推荐器")

# 添加调试信息
logger.info(f"Python解释器路径: {sys.executable}")
logger.info(f"Python搜索路径: {sys.path}")

# 导入后端API模块
try:
    from backend.api.app import app
    logger.info("成功导入后端API模块")
except ImportError as e:
    logger.error(f"导入后端API模块失败: {str(e)}")
    sys.exit(1)

# 验证静态文件夹和模板文件夹路径
static_folder = os.path.join(root_dir, 'frontend', 'static')
template_folder = os.path.join(root_dir, 'frontend', 'templates')

if os.path.exists(static_folder):
    logger.info(f"静态文件夹存在: {static_folder}")
else:
    logger.error(f"静态文件夹不存在: {static_folder}")

if os.path.exists(template_folder):
    logger.info(f"模板文件夹存在: {template_folder}")
else:
    logger.error(f"模板文件夹不存在: {template_folder}")

# 定义浏览器自动打开函数
def open_browser():
    """在应用启动后自动打开浏览器"""
    time.sleep(1)  # 等待服务器启动
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    url = f"http://localhost:{port}"
    
    logger.info(f"自动打开浏览器访问: {url}")
    webbrowser.open(url)

# 添加缓存控制相关的代码
@app.after_request
def add_header(response):
    """
    添加缓存控制头，防止304缓存问题
    """
    # 对静态资源设置缓存控制
    if request.path.startswith('/static/'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    
    # 对问卷页面特别处理
    if request.path == '/questionnaire' or '/questionnaire?' in request.path:
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0, post-check=0, pre-check=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        # 添加随机数以避免缓存
        if not request.args.get('refresh'):
            return redirect('/questionnaire?refresh=' + str(int(time.time())))
    
    return response

# 添加问卷路由强制刷新
@app.route('/refresh_questionnaire')
def refresh_questionnaire():
    """
    强制刷新问卷页面，删除缓存
    """
    return redirect('/questionnaire?refresh=' + str(int(time.time())))

if __name__ == "__main__":
    # 打印环境配置情况
    logger.info(f"Spotify API: {'已配置' if os.environ.get('SPOTIFY_CLIENT_ID') else '未配置'}")
    logger.info(f"HKBU API: {'已配置' if os.environ.get('HKBU_API_KEY') else '已使用默认配置'}")
    logger.info(f"数据目录: {os.environ.get('DATA_DIR')}")
    logger.info(f"使用MSD: {os.environ.get('USE_MSD')}")
    logger.info(f"强制重训: {os.environ.get('FORCE_RETRAIN')}")
    logger.info(f"模型类型: {os.environ.get('MODEL_TYPE')}")
    logger.info(f"混合权重: {os.environ.get('CONTENT_WEIGHT')}")
    
    # 确保HKBU API配置
    os.environ.setdefault('HKBU_API_KEY', '06fd2422-8207-4a5b-8aaa-434415ed3a2b')
    
    # 启动浏览器线程
    threading.Timer(2.5, open_browser).start()  # 增加延迟时间，等待服务器完全启动
    
    # 启动Flask应用程序
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'true').lower() == 'true'
    
    logger.info(f"启动服务器: {host}:{port}, 调试模式: {debug}")
    # 禁用.env自动加载，避免编码问题
    os.environ['FLASK_SKIP_DOTENV'] = '1'
    
    # 使用线程池处理请求
    from werkzeug.serving import run_simple
    run_simple(host, port, app, use_reloader=False, use_debugger=debug, threaded=True, processes=1) 