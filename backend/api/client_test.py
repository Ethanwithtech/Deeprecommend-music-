#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
推荐系统API客户端测试脚本
用于测试推荐API的各个功能
"""

import requests
import json
import sys
import argparse
from pprint import pprint

def test_health(base_url):
    """测试健康检查端点"""
    print("测试健康检查...")
    response = requests.get(f"{base_url}/api/health")
    print(f"状态码: {response.status_code}")
    pprint(response.json())
    print()

def test_recommendation(base_url, user_id, message):
    """测试推荐功能"""
    print(f"测试推荐功能 (用户: {user_id}, 消息: {message})...")
    response = requests.post(
        f"{base_url}/api/recommend",
        json={"user_id": user_id, "message": message}
    )
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"情感: {result.get('emotion', '未知')}")
        print(f"回复: {result.get('message', '无回复')}")
        print("推荐:")
        for idx, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {idx}. {rec}")
    else:
        print(f"错误: {response.text}")
    print()

def test_feedback(base_url, user_id, song_id, rating):
    """测试反馈功能"""
    print(f"测试反馈功能 (用户: {user_id}, 歌曲: {song_id}, 评分: {rating})...")
    response = requests.post(
        f"{base_url}/api/feedback",
        json={"user_id": user_id, "song_id": song_id, "rating": rating}
    )
    print(f"状态码: {response.status_code}")
    pprint(response.json())
    print()

def test_mood_recommendation(base_url, user_id, mood):
    """测试基于情绪的推荐"""
    print(f"测试基于情绪的推荐 (用户: {user_id}, 情绪: {mood})...")
    response = requests.post(
        f"{base_url}/api/mood_recommend",
        json={"user_id": user_id, "mood": mood}
    )
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("推荐:")
        for idx, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {idx}. {rec}")
    else:
        print(f"错误: {response.text}")
    print()

def test_activity_recommendation(base_url, user_id, activity):
    """测试基于活动的推荐"""
    print(f"测试基于活动的推荐 (用户: {user_id}, 活动: {activity})...")
    response = requests.post(
        f"{base_url}/api/activity_recommend",
        json={"user_id": user_id, "activity": activity}
    )
    print(f"状态码: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print("推荐:")
        for idx, rec in enumerate(result.get('recommendations', []), 1):
            print(f"  {idx}. {rec}")
    else:
        print(f"错误: {response.text}")
    print()

def test_train_model(base_url):
    """测试模型训练"""
    print("测试模型训练...")
    response = requests.post(
        f"{base_url}/api/train",
        json={}
    )
    print(f"状态码: {response.status_code}")
    pprint(response.json())
    print()

def run_all_tests(base_url, user_id):
    """运行所有测试"""
    print("=" * 50)
    print(f"开始测试推荐系统API (服务器: {base_url})")
    print("=" * 50)
    
    # 测试健康检查
    test_health(base_url)
    
    # 测试模型训练
    test_train_model(base_url)
    
    # 测试各种消息推荐
    test_recommendation(base_url, user_id, "我今天心情很好")
    test_recommendation(base_url, user_id, "我有点难过")
    test_recommendation(base_url, user_id, "推荐一些流行音乐")
    
    # 测试反馈
    test_feedback(base_url, user_id, "song1", 5)
    test_feedback(base_url, user_id, "song2", 2)
    
    # 测试情绪推荐
    test_mood_recommendation(base_url, user_id, "happy")
    test_mood_recommendation(base_url, user_id, "sad")
    
    # 测试活动推荐
    test_activity_recommendation(base_url, user_id, "studying")
    test_activity_recommendation(base_url, user_id, "exercising")
    
    print("=" * 50)
    print("测试完成!")
    print("=" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='测试推荐系统API')
    parser.add_argument('--server', type=str, default='http://localhost:5000', help='API服务器地址')
    parser.add_argument('--user', type=str, default='test_user_1', help='测试用户ID')
    args = parser.parse_args()
    
    run_all_tests(args.server, args.user) 