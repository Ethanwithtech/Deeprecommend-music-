#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设置HKBU GenAI Platform API密钥的辅助脚本
"""

import os
import sys

def main():
    # 提示用户输入HKBU API密钥
    print("=" * 50)
    print("设置HKBU GenAI Platform API密钥")
    print("=" * 50)
    print("\n此脚本将帮助您设置HKBU API密钥，以启用AI聊天功能。")
    print("您需要从HKBU GenAI Platform获取API密钥。")
    print("\n如果您没有API密钥，可以访问 https://genai.hkbu.edu.hk/ 注册并获取。")
    
    api_key = input("\n请输入您的HKBU API密钥: ").strip()
    
    if not api_key:
        print("\n错误：API密钥不能为空。")
        return
    
    # 检查.env文件是否存在
    env_path = '.env'
    env_content = {}
    
    if os.path.exists(env_path):
        # 读取现有的.env文件
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    env_content[key] = value
    
    # 更新或添加HKBU API密钥
    env_content['HKBU_API_KEY'] = api_key
    
    # 写入.env文件
    with open(env_path, 'w', encoding='utf-8') as f:
        for key, value in env_content.items():
            f.write(f"{key}={value}\n")
    
    print("\n✅ HKBU API密钥已成功设置！")
    print("\n现在您可以启动应用程序并使用AI聊天功能。")
    print("执行以下命令启动应用程序:")
    print("    python app.py")
    
    return True

if __name__ == '__main__':
    main() 