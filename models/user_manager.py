#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用户管理模块

提供用户认证、注册和会话管理功能
"""

import sqlite3
import uuid
import hashlib
import datetime
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)

class UserManager:
    """用户管理类，提供用户相关操作"""
    
    def __init__(self, db_path='../../music_recommender.db'):
        """
        初始化用户管理器
        
        参数:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        logger.info(f"初始化用户管理器，数据库路径: {db_path}")
        
        # 确保数据库表存在
        self._ensure_tables()
    
    def _ensure_tables(self):
        """确保用户相关的数据库表存在"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建用户表
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TEXT NOT NULL,
                is_developer INTEGER DEFAULT 0 NOT NULL
            )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("用户数据库表初始化完成")
        except Exception as e:
            logger.error(f"初始化用户表时出错: {e}")
    
    def hash_password(self, password):
        """哈希密码"""
        salt = uuid.uuid4().hex
        hashed = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
        return f"{salt}:{hashed}"
    
    def verify_password(self, stored_password, provided_password):
        """验证密码"""
        salt, hashed = stored_password.split(':', 1)
        calculated_hash = hashlib.sha256(salt.encode() + provided_password.encode()).hexdigest()
        return calculated_hash == hashed
    
    def register_user(self, username, password, is_developer=0):
        """
        注册新用户
        
        参数:
            username: 用户名
            password: 密码
            is_developer: 是否为开发者（0表示否，1表示是）
            
        返回:
            成功返回用户ID，失败返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查用户名是否已存在
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            if cursor.fetchone():
                logger.warning(f"用户名已存在: {username}")
                conn.close()
                return None
            
            # 生成用户ID
            user_id = str(uuid.uuid4())
            
            # 哈希密码
            hashed_password = self.hash_password(password)
            
            # 创建时间
            created_at = datetime.datetime.now().isoformat()
            
            # 插入用户记录
            cursor.execute(
                "INSERT INTO users (id, username, password, created_at, is_developer) VALUES (?, ?, ?, ?, ?)",
                (user_id, username, hashed_password, created_at, is_developer)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"成功注册用户: {username} (ID: {user_id}, 开发者: {is_developer})")
            return user_id
        except Exception as e:
            logger.error(f"注册用户时出错: {e}")
            return None
    
    def authenticate_user(self, username, password):
        """
        验证用户身份
        
        参数:
            username: 用户名
            password: 密码
            
        返回:
            成功返回用户ID，失败返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询用户
            cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
            result = cursor.fetchone()
            
            conn.close()
            
            if not result:
                logger.warning(f"用户不存在: {username}")
                return None
            
            user_id, stored_password = result
            
            # 验证密码
            if self.verify_password(stored_password, password):
                logger.info(f"用户认证成功: {username} (ID: {user_id})")
                return user_id
            else:
                logger.warning(f"密码错误: {username}")
                return None
        except Exception as e:
            logger.error(f"用户认证时出错: {e}")
            return None
    
    def get_user_by_id(self, user_id):
        """
        根据ID获取用户信息
        
        参数:
            user_id: 用户ID
            
        返回:
            用户信息字典，失败返回None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询用户
            cursor.execute("SELECT id, username, created_at, is_developer FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if not result:
                logger.warning(f"用户ID不存在: {user_id}")
                return None
            
            # 构建用户信息
            user_info = {
                'id': result[0],
                'username': result[1],
                'created_at': result[2],
                'is_developer': bool(result[3])
            }
            
            return user_info
        except Exception as e:
            logger.error(f"获取用户信息时出错: {e}")
            return None
    
    def get_all_users(self):
        """
        获取所有用户列表
        
        返回:
            用户列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询所有用户
            cursor.execute("SELECT id, username, created_at, is_developer FROM users")
            results = cursor.fetchall()
            
            conn.close()
            
            # 构建用户信息列表
            users = []
            for row in results:
                users.append({
                    'id': row[0],
                    'username': row[1],
                    'created_at': row[2],
                    'is_developer': bool(row[3])
                })
            
            return users
        except Exception as e:
            logger.error(f"获取所有用户信息时出错: {e}")
            return []
    
    def set_developer_status(self, user_id, is_developer):
        """
        设置用户的开发者状态
        
        参数:
            user_id: 用户ID
            is_developer: 是否为开发者（True/False）
            
        返回:
            操作是否成功（True/False）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 转换布尔值为整数
            developer_value = 1 if is_developer else 0
            
            # 更新用户开发者状态
            cursor.execute(
                "UPDATE users SET is_developer = ? WHERE id = ?",
                (developer_value, user_id)
            )
            
            # 检查是否找到用户
            if cursor.rowcount == 0:
                logger.warning(f"未找到用户ID: {user_id}")
                conn.close()
                return False
            
            conn.commit()
            conn.close()
            
            logger.info(f"成功更新用户ID的开发者状态: {user_id}, 开发者状态: {is_developer}")
            return True
        except Exception as e:
            logger.error(f"更新用户开发者状态时出错: {e}")
            return False
    
    def is_developer(self, user_id):
        """
        检查用户是否为开发者
        
        参数:
            user_id: 用户ID
            
        返回:
            是否为开发者（True/False），出错时返回False
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询用户开发者状态
            cursor.execute("SELECT is_developer FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchone()
            
            conn.close()
            
            if not result:
                logger.warning(f"用户ID不存在: {user_id}")
                return False
            
            return bool(result[0])
        except Exception as e:
            logger.error(f"检查用户开发者状态时出错: {e}")
            return False

# 测试代码
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 测试用户管理器
    user_manager = UserManager(db_path='../../music_recommender.db')
    
    # 注册测试用户
    test_username = f"test_user_{uuid.uuid4().hex[:8]}"
    test_password = "password123"
    
    user_id = user_manager.register_user(test_username, test_password)
    print(f"注册用户结果: {user_id}")
    
    # 验证用户
    auth_id = user_manager.authenticate_user(test_username, test_password)
    print(f"验证用户结果: {auth_id}")
    
    # 错误密码
    auth_id = user_manager.authenticate_user(test_username, "wrong_password")
    print(f"错误密码验证结果: {auth_id}")
    
    # 获取用户信息
    user_info = user_manager.get_user_by_id(user_id)
    print(f"用户信息: {user_info}")
    
    # 设置开发者状态
    user_manager.set_developer_status(user_id, True)
    
    # 检查开发者状态
    is_dev = user_manager.is_developer(user_id)
    print(f"开发者状态: {is_dev}")
    
    # 获取所有用户
    all_users = user_manager.get_all_users()
    print(f"所有用户数量: {len(all_users)}") 