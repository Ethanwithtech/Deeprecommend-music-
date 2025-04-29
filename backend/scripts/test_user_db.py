#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试用户数据库功能的脚本
运行方式: python test_user_db.py
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

def test_user_manager():
    """测试用户管理器功能"""
    logger.info("======== 开始测试用户管理器 ========")
    
    # 数据库路径
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                         'music_recommender.db')
    logger.info(f"数据库路径: {db_path}")
    
    # 初始化用户管理器
    user_manager = UserManager(db_path=db_path)
    
    # 测试1: 注册普通用户
    test_username1 = f"test_user_{uuid.uuid4().hex[:8]}"
    test_password1 = "password123"
    
    logger.info(f"测试1: 注册普通用户 {test_username1}")
    user_id1 = user_manager.register_user(test_username1, test_password1)
    logger.info(f"注册结果: {user_id1}")
    
    if not user_id1:
        logger.error("注册普通用户失败")
        return False
    
    # 测试2: 注册开发者用户
    test_username2 = f"test_dev_{uuid.uuid4().hex[:8]}"
    test_password2 = "devpassword123"
    
    logger.info(f"测试2: 注册开发者用户 {test_username2}")
    user_id2 = user_manager.register_user(test_username2, test_password2, is_developer=1)
    logger.info(f"注册结果: {user_id2}")
    
    if not user_id2:
        logger.error("注册开发者用户失败")
        return False
    
    # 测试3: 获取用户信息
    logger.info(f"测试3: 获取普通用户信息 {user_id1}")
    user_info1 = user_manager.get_user_by_id(user_id1)
    logger.info(f"用户信息: {user_info1}")
    
    if not user_info1:
        logger.error("获取普通用户信息失败")
        return False
    
    # 测试4: 获取开发者用户信息
    logger.info(f"测试4: 获取开发者用户信息 {user_id2}")
    user_info2 = user_manager.get_user_by_id(user_id2)
    logger.info(f"用户信息: {user_info2}")
    
    if not user_info2:
        logger.error("获取开发者用户信息失败")
        return False
    
    # 测试5: 检查用户开发者状态
    logger.info(f"测试5: 检查普通用户开发者状态 {user_id1}")
    is_dev1 = user_manager.is_developer(user_id1)
    logger.info(f"开发者状态: {is_dev1}")
    
    if is_dev1:
        logger.error("普通用户开发者状态检查错误")
        return False
    
    # 测试6: 检查开发者用户状态
    logger.info(f"测试6: 检查开发者用户状态 {user_id2}")
    is_dev2 = user_manager.is_developer(user_id2)
    logger.info(f"开发者状态: {is_dev2}")
    
    if not is_dev2:
        logger.error("开发者用户状态检查错误")
        return False
    
    # 测试7: 修改用户开发者状态
    logger.info(f"测试7: 将普通用户 {user_id1} 设置为开发者")
    result = user_manager.set_developer_status(user_id1, True)
    logger.info(f"设置结果: {result}")
    
    if not result:
        logger.error("修改用户开发者状态失败")
        return False
    
    # 测试8: 再次检查普通用户是否已变为开发者
    logger.info(f"测试8: 再次检查用户 {user_id1} 开发者状态")
    is_dev1_after = user_manager.is_developer(user_id1)
    logger.info(f"开发者状态: {is_dev1_after}")
    
    if not is_dev1_after:
        logger.error("设置用户开发者状态后检查失败")
        return False
    
    # 测试9: 获取所有用户
    logger.info("测试9: 获取所有用户")
    all_users = user_manager.get_all_users()
    logger.info(f"用户数量: {len(all_users)}")
    
    # 检查我们的测试用户是否在列表中
    test_users_found = 0
    for user in all_users:
        if user['id'] in [user_id1, user_id2]:
            test_users_found += 1
    
    logger.info(f"在所有用户中找到的测试用户数量: {test_users_found}")
    
    if test_users_found != 2:
        logger.error("未能在用户列表中找到所有测试用户")
        return False
    
    logger.info("======== 用户管理器测试完成 ========")
    return True

def main():
    """主函数"""
    try:
        success = test_user_manager()
        
        if success:
            logger.info("所有测试通过")
            return 0
        else:
            logger.error("测试失败")
            return 1
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 