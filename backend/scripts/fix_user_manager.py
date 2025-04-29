#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
修复用户管理器模块

这个脚本会重新创建用户管理器文件，解决编码和缓存问题
"""

import os
import sys
import shutil

# 用户管理器内容
USER_MANAGER_CONTENT = """#!/usr/bin/env python
# -*- coding: utf-8 -*-

\"""
用户管理模块

提供用户认证、注册和会话管理功能
\"""

import sqlite3
import uuid
import hashlib
import datetime
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)

class UserManager:
    \"""用户管理类，提供用户相关操作\"""
    
    def __init__(self, db_path='../../music_recommender.db'):
        \"""
        初始化用户管理器
        
        参数:
            db_path: 数据库文件路径
        \"""
        self.db_path = db_path
        logger.info(f"初始化用户管理器，数据库路径: {db_path}")
        
        # 确保数据库表存在
        self._ensure_tables()
    
    def _ensure_tables(self):
        \"""确保用户相关的数据库表存在\"""
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
        \"""哈希密码\"""
        salt = uuid.uuid4().hex
        hashed = hashlib.sha256(salt.encode() + password.encode()).hexdigest()
        return f"{salt}:{hashed}"
    
    def verify_password(self, stored_password, provided_password):
        \"""验证密码\"""
        salt, hashed = stored_password.split(':', 1)
        calculated_hash = hashlib.sha256(salt.encode() + provided_password.encode()).hexdigest()
        return calculated_hash == hashed
    
    def register_user(self, username, password, is_developer=0):
        \"""
        注册新用户
        
        参数:
            username: 用户名
            password: 密码
            is_developer: 是否为开发者（0表示否，1表示是）
            
        返回:
            成功返回用户ID，失败返回None
        \"""
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
        \"""
        验证用户身份
        
        参数:
            username: 用户名
            password: 密码
            
        返回:
            成功返回用户ID，失败返回None
        \"""
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
        \"""
        根据ID获取用户信息
        
        参数:
            user_id: 用户ID
            
        返回:
            用户信息字典，失败返回None
        \"""
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
        \"""
        获取所有用户列表
        
        返回:
            用户列表
        \"""
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
        \"""
        设置用户的开发者状态
        
        参数:
            user_id: 用户ID
            is_developer: 是否为开发者（True/False）
            
        返回:
            操作是否成功（True/False）
        \"""
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
        \"""
        检查用户是否为开发者
        
        参数:
            user_id: 用户ID
            
        返回:
            是否为开发者（True/False），出错时返回False
        \"""
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
"""

def main():
    """主函数"""
    # 定义用户管理器路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    user_manager_path = os.path.join(base_dir, 'models', 'user_manager.py')
    
    print(f"将重新创建用户管理器文件: {user_manager_path}")
    
    # 备份现有文件
    try:
        if os.path.exists(user_manager_path):
            backup_path = user_manager_path + '.bak'
            print(f"备份现有文件: {backup_path}")
            shutil.copy2(user_manager_path, backup_path)
    except Exception as e:
        print(f"备份文件时出错: {e}")
    
    # 创建新文件
    try:
        with open(user_manager_path, 'w', encoding='utf-8') as f:
            f.write(USER_MANAGER_CONTENT)
        print("用户管理器文件创建成功")
        
        # 删除可能存在的Python缓存
        pycache_dir = os.path.join(base_dir, 'models', '__pycache__')
        if os.path.exists(pycache_dir):
            print(f"清理Python缓存: {pycache_dir}")
            for filename in os.listdir(pycache_dir):
                if filename.startswith('user_manager'):
                    cache_file = os.path.join(pycache_dir, filename)
                    print(f"删除缓存文件: {cache_file}")
                    os.remove(cache_file)
    except Exception as e:
        print(f"创建文件时出错: {e}")
        return 1
    
    print("处理完成")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 