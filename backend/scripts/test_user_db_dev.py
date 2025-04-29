#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试用户管理器开发者功能
"""

import os
import sys
import logging
import uuid
import importlib

# 添加父级目录到路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    try:
        # 强制重新加载user_manager模块
        import models.user_manager
        importlib.reload(models.user_manager)
        from models.user_manager import UserManager
        
        # 打印register_user方法的参数信息
        import inspect
        sig = inspect.signature(UserManager.register_user)
        logger.info(f"UserManager.register_user参数: {sig}")
        
        # 数据库路径
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             'music_recommender.db')
        logger.info(f"数据库路径: {db_path}")
        
        # 初始化用户管理器
        user_manager = UserManager(db_path=db_path)
        
        # 注册开发者用户
        test_username = f"test_dev_{uuid.uuid4().hex[:8]}"
        test_password = "devpassword123"
        
        logger.info(f"注册开发者用户: {test_username}")
        user_id = user_manager.register_user(test_username, test_password, is_developer=1)
        logger.info(f"注册结果: {user_id}")
        
        if not user_id:
            logger.error("注册开发者用户失败")
            return 1
        
        # 检查开发者状态
        logger.info(f"检查开发者状态: {user_id}")
        is_dev = user_manager.is_developer(user_id)
        logger.info(f"开发者状态: {is_dev}")
        
        if not is_dev:
            logger.error("开发者状态检查失败")
            return 1
        
        # 获取用户信息
        logger.info(f"获取用户信息: {user_id}")
        user_info = user_manager.get_user_by_id(user_id)
        logger.info(f"用户信息: {user_info}")
        
        if not user_info:
            logger.error("获取用户信息失败")
            return 1
        
        # 尝试通过API调用来检查开发者状态
        logger.info("测试获取所有用户")
        all_users = user_manager.get_all_users()
        logger.info(f"获取到 {len(all_users)} 个用户")
        
        # 查找我们刚创建的用户
        for user in all_users:
            if user['id'] == user_id:
                logger.info(f"在用户列表中找到测试用户: {user}")
                if not user['is_developer']:
                    logger.error("用户列表中的开发者状态不正确")
                    return 1
                break
        else:
            logger.error("在用户列表中未找到测试用户")
            return 1
        
        logger.info("所有测试通过")
        return 0
    except Exception as e:
        logger.error(f"测试过程中出错: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 