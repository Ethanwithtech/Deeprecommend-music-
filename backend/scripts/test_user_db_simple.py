#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简化的用户数据库测试脚本
"""

import os
import sys
import sqlite3
import logging
import uuid
from datetime import datetime

# 添加父级目录到路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入用户管理器
from models.user_manager import UserManager

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 数据库路径
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'music_recommender.db')
        logger.info(f"数据库路径: {db_path}")
        
        # 初始化用户管理器
        user_manager = UserManager(db_path=db_path)
        
        # 打印register_user方法的参数信息
        import inspect
        sig = inspect.signature(UserManager.register_user)
        logger.info(f"UserManager.register_user参数: {sig}")
        
        # 注册普通用户
        test_username = f"test_user_{uuid.uuid4().hex[:8]}"
        test_password = "password123"
        
        logger.info(f"注册普通用户: {test_username}")
        user_id = user_manager.register_user(test_username, test_password)
        logger.info(f"注册结果: {user_id}")
        
        if not user_id:
            logger.error("注册普通用户失败")
            return 1
        
        # 获取用户信息
        logger.info(f"获取用户信息: {user_id}")
        user_info = user_manager.get_user_by_id(user_id)
        logger.info(f"用户信息: {user_info}")
        
        if not user_info:
            logger.error("获取用户信息失败")
            return 1
        
        logger.info("测试成功完成")
        return 0
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 