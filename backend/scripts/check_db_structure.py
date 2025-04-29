#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查数据库结构的简单脚本
"""

import sqlite3
import os
import sys

def main():
    """主函数"""
    # 数据库路径
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                         '../music_recommender.db')
    
    print(f"数据库路径: {db_path}")
    
    # 检查数据库文件是否存在
    if not os.path.exists(db_path):
        print(f"错误: 数据库文件不存在: {db_path}")
        return 1
        
    try:
        # 连接数据库
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 列出所有表
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"数据库中的表: {[table[0] for table in tables]}")
        
        # 检查users表结构
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        
        print("\n用户表结构:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]}), {'Primary Key' if col[5] == 1 else 'Not Null' if col[3] == 1 else 'Nullable'}")
        
        # 关闭连接
        conn.close()
        return 0
    except Exception as e:
        print(f"错误: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 