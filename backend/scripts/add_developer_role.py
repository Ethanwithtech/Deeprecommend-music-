#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据库迁移脚本：添加开发者角色到用户表

用法：
python add_developer_role.py
"""

import sqlite3
import logging
import os
import sys
import traceback

# 添加父级目录到路径，以便导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 配置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_migration():
    """执行数据库迁移"""
    try:
        # 数据库路径
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                              'music_recommender.db')
        
        print(f"连接数据库: {db_path}")
        logger.info(f"连接数据库: {db_path}")
        
        # 检查数据库文件是否存在
        if not os.path.exists(db_path):
            error_msg = f"数据库文件不存在: {db_path}"
            print(error_msg)
            logger.error(error_msg)
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查users表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if not cursor.fetchone():
            print("创建users表")
            logger.info("users表不存在，创建新表")
            cursor.execute('''
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_developer INTEGER DEFAULT 0 NOT NULL
            )
            ''')
            conn.commit()
            print("users表创建成功")
            logger.info("users表创建成功")
        else:
            # 检查列是否已存在
            cursor.execute("PRAGMA table_info(users)")
            columns = cursor.fetchall()
            print(f"现有列: {columns}")
            column_names = [column[1] for column in columns]
            
            # 检查并添加是否缺少is_developer列
            if 'is_developer' not in column_names:
                print("添加 is_developer 列到 users 表")
                logger.info("添加 is_developer 列到 users 表")
                cursor.execute("ALTER TABLE users ADD COLUMN is_developer INTEGER DEFAULT 0 NOT NULL")
                conn.commit()
                print("添加 is_developer 列成功")
                logger.info("添加 is_developer 列成功")
            else:
                print("is_developer 列已存在")
                logger.info("is_developer 列已存在")
                
            # 检查其他必要字段
            required_columns = ['id', 'username', 'password', 'created_at']
            for col in required_columns:
                if col not in column_names:
                    error_msg = f"users表缺少必要字段: {col}"
                    print(error_msg)
                    logger.error(error_msg)
                    return False
            
        # 确认修改
        cursor.execute("PRAGMA table_info(users)")
        result = cursor.fetchall()
        print(f"最终用户表结构: {result}")
        logger.info(f"用户表结构: {result}")
        
        # 添加测试开发者账号
        from models.user_manager import UserManager
        user_manager = UserManager(db_path=db_path)
        
        # 检查开发者账号是否存在
        cursor.execute("SELECT id FROM users WHERE username = ?", ("admin",))
        if not cursor.fetchone():
            print("添加管理员账号")
            logger.info("添加管理员账号")
            admin_id = user_manager.register_user("admin", "admin123", is_developer=1)
            if admin_id:
                print(f"管理员账号创建成功，ID: {admin_id}")
                logger.info(f"管理员账号创建成功，ID: {admin_id}")
            else:
                print("创建管理员账号失败")
                logger.error("创建管理员账号失败")
        else:
            print("管理员账号已存在")
            logger.info("管理员账号已存在")
        
        # 关闭连接
        conn.close()
        print("数据库迁移成功完成")
        logger.info("数据库迁移成功完成")
        return True
    except Exception as e:
        error_msg = f"数据库迁移失败: {e}\n{traceback.format_exc()}"
        print(error_msg)
        logger.error(error_msg)
        return False

if __name__ == "__main__":
    success = run_migration()
    if success:
        print("迁移脚本执行成功")
        logger.info("迁移脚本执行成功")
        sys.exit(0)
    else:
        print("迁移脚本执行失败")
        logger.error("迁移脚本执行失败")
        sys.exit(1) 