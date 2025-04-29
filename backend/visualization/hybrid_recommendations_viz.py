#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音乐推荐可视化工具 - Blender版本
此脚本用于将混合推荐系统的结果可视化为3D交互式场景
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

# 在Blender内部运行时，需要手动导入bpy
try:
    import bpy
    in_blender = True
except ImportError:
    in_blender = False
    print("警告: 未在Blender环境中运行，仅能进行数据准备")

# 颜色映射
GENRE_COLORS = {
    'rock': (0.8, 0.2, 0.2),       # 红色
    'pop': (0.95, 0.6, 0.8),       # 粉色
    'electronic': (0.2, 0.8, 0.8),  # 青色
    'jazz': (0.2, 0.2, 0.8),       # 蓝色
    'classical': (0.8, 0.8, 0.2),   # 黄色
    'hip hop': (0.5, 0.2, 0.5),    # 紫色
    'folk': (0.5, 0.3, 0.1),       # 棕色
    'country': (0.1, 0.5, 0.1),    # 绿色
    'blues': (0.2, 0.4, 0.8),      # 深蓝色
    'r&b': (0.8, 0.4, 0.2),        # 橙色
    'default': (0.6, 0.6, 0.6)     # 灰色 - 默认
}

def load_recommendations(recommendations_file):
    """加载推荐结果"""
    print(f"正在加载推荐数据: {recommendations_file}")
    with open(recommendations_file, 'r', encoding='utf-8') as f:
        recommendations = json.load(f)
    return recommendations

def load_hybrid_model(model_file):
    """加载混合推荐模型"""
    print(f"正在加载混合模型: {model_file}")
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    return model

def prepare_visualization_data(recommendations, model, use_tsne=True):
    """
    准备可视化数据
    返回包含位置和颜色信息的字典列表
    """
    print("准备可视化数据...")
    
    # 提取所有推荐歌曲
    all_songs = []
    for user_id, recs in recommendations.items():
        for rec in recs:
            rec['user_id'] = user_id
            all_songs.append(rec)
    
    # 如果有内容模型，使用其特征进行可视化
    if model and model.get('content_model') and use_tsne:
        song_features = []
        valid_songs = []
        
        # 收集有效特征
        for song in all_songs:
            song_id = song['song_id']
            song_idx = model.get('song_id_map', {}).get(song_id)
            
            if song_idx is not None and model['content_model'].get('features') and song_idx in model['content_model']['features']:
                features = model['content_model']['features'][song_idx]
                if len(features) > 0:
                    song_features.append(features)
                    valid_songs.append(song)
        
        # 使用t-SNE降维到3D
        if len(song_features) > 3:
            print(f"使用t-SNE将{len(song_features)}首歌曲降维到3D空间...")
            tsne = TSNE(n_components=3, random_state=42)
            positions_3d = tsne.fit_transform(song_features)
            
            # 标准化到[-5, 5]范围
            scaler = MinMaxScaler(feature_range=(-5, 5))
            positions_3d = scaler.fit_transform(positions_3d)
            
            # 添加位置信息
            for i, song in enumerate(valid_songs):
                song['position'] = positions_3d[i].tolist()
        else:
            print("特征数量不足，使用随机位置...")
            for song in all_songs:
                song['position'] = [
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5),
                    np.random.uniform(-5, 5)
                ]
    else:
        # 如果没有足够的特征信息，使用随机位置
        print("使用随机位置...")
        for song in all_songs:
            song['position'] = [
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5),
                np.random.uniform(-5, 5)
            ]
    
    # 添加颜色信息（基于流派或得分）
    for song in all_songs:
        # 基于流派的颜色
        if 'genre' in song:
            genre = song['genre'].lower() if song.get('genre') else 'default'
            # 查找最匹配的流派
            matched_genre = 'default'
            for g in GENRE_COLORS.keys():
                if g in genre:
                    matched_genre = g
                    break
            song['color'] = GENRE_COLORS.get(matched_genre, GENRE_COLORS['default'])
        # 基于得分的颜色
        else:
            # 使用得分生成颜色 (红色到蓝色的渐变)
            score = min(max(song.get('score', 0.5), 0), 1)
            song['color'] = (score, 0.3, 1.0-score)
    
    return all_songs

def clear_scene():
    """清空Blender场景"""
    if not in_blender:
        return
    
    # 删除所有对象
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # 创建新集合
    if 'Recommendations' not in bpy.data.collections:
        recommendations_collection = bpy.data.collections.new('Recommendations')
        bpy.context.scene.collection.children.link(recommendations_collection)
    
    # 清除所有材质
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)

def create_song_sphere(song, size=0.2):
    """创建表示歌曲的球体"""
    if not in_blender:
        return None
    
    # 获取推荐集合
    if 'Recommendations' in bpy.data.collections:
        collection = bpy.data.collections['Recommendations']
    else:
        collection = bpy.context.scene.collection
    
    # 创建球体
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=size, 
        location=song['position']
    )
    
    sphere = bpy.context.active_object
    sphere.name = f"Song_{song['song_id']}"
    
    # 创建并分配材质
    mat = bpy.data.materials.new(name=f"Mat_{song['song_id']}")
    mat.use_nodes = True
    
    # 设置基础颜色
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        principled_bsdf.inputs[0].default_value = (*song['color'], 1.0)  # RGB + Alpha
    
    # 添加一些光泽
    if principled_bsdf:
        principled_bsdf.inputs['Metallic'].default_value = 0.7
        principled_bsdf.inputs['Specular'].default_value = 0.5
        principled_bsdf.inputs['Roughness'].default_value = 0.2
    
    # 分配材质
    if sphere.data.materials:
        sphere.data.materials[0] = mat
    else:
        sphere.data.materials.append(mat)
    
    # 将对象移动到正确的集合
    if sphere.name in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.unlink(sphere)
    if sphere.name not in collection.objects:
        collection.objects.link(sphere)
    
    # 添加自定义属性
    sphere['song_id'] = song['song_id']
    sphere['title'] = song.get('title', 'Unknown Title')
    sphere['artist'] = song.get('artist', 'Unknown Artist')
    sphere['score'] = song.get('score', 0.0)
    sphere['cf_score'] = song.get('cf_score', 0.0)
    sphere['content_score'] = song.get('content_score', 0.0)
    sphere['context_score'] = song.get('context_score', 0.0)
    sphere['deep_score'] = song.get('deep_score', 0.0)
    sphere['user_id'] = song.get('user_id', 'unknown')
    
    return sphere

def create_user_marker(user_id, user_songs):
    """创建表示用户的标记"""
    if not in_blender or not user_songs:
        return None
    
    # 计算用户位置（用户收到的推荐的平均位置）
    user_pos = np.mean([song['position'] for song in user_songs], axis=0)
    
    # 创建空物体作为用户标记
    bpy.ops.object.empty_add(
        type='PLAIN_AXES',
        radius=0.3,
        location=user_pos
    )
    
    user_obj = bpy.context.active_object
    user_obj.name = f"User_{user_id}"
    
    # 添加自定义属性
    user_obj['user_id'] = user_id
    user_obj['recommendation_count'] = len(user_songs)
    
    # 获取推荐集合
    if 'Recommendations' in bpy.data.collections:
        collection = bpy.data.collections['Recommendations']
        if user_obj.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(user_obj)
        if user_obj.name not in collection.objects:
            collection.objects.link(user_obj)
    
    return user_obj

def create_connections(user_obj, song_objects):
    """创建用户与推荐歌曲之间的连接"""
    if not in_blender:
        return
    
    user_pos = user_obj.location
    
    for song_obj in song_objects:
        # 创建曲线
        curve_data = bpy.data.curves.new(f'Curve_{user_obj.name}_{song_obj.name}', type='CURVE')
        curve_data.dimensions = '3D'
        
        # 创建样条
        polyline = curve_data.splines.new('BEZIER')
        polyline.bezier_points.add(1)  # 两个点
        
        # 设置起点（用户位置）
        polyline.bezier_points[0].co = user_pos
        polyline.bezier_points[0].handle_left_type = 'AUTO'
        polyline.bezier_points[0].handle_right_type = 'AUTO'
        
        # 设置终点（歌曲位置）
        polyline.bezier_points[1].co = song_obj.location
        polyline.bezier_points[1].handle_left_type = 'AUTO'
        polyline.bezier_points[1].handle_right_type = 'AUTO'
        
        # 创建曲线对象
        curve_obj = bpy.data.objects.new(f'Connection_{user_obj.name}_{song_obj.name}', curve_data)
        
        # 设置曲线厚度
        curve_data.bevel_depth = 0.02
        
        # 创建材质
        mat = bpy.data.materials.new(name=f"Connection_{user_obj.name}_{song_obj.name}")
        mat.use_nodes = True
        
        # 根据推荐分数设置颜色
        score = song_obj.get('score', 0.5)
        color = (0.8, 0.8, 0.2, 1.0)  # 默认黄色
        
        principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
        if principled_bsdf:
            principled_bsdf.inputs[0].default_value = color
            principled_bsdf.inputs['Alpha'].default_value = 0.7  # 半透明
        
        # 分配材质
        curve_obj.data.materials.append(mat)
        
        # 添加到场景
        if 'Recommendations' in bpy.data.collections:
            bpy.data.collections['Recommendations'].objects.link(curve_obj)
        else:
            bpy.context.scene.collection.objects.link(curve_obj)

def setup_lighting():
    """设置场景光照"""
    if not in_blender:
        return
    
    # 创建环境光
    bpy.ops.object.light_add(type='SUN', radius=1, location=(0, 0, 10))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    
    # 添加HDRI环境贴图
    world = bpy.context.scene.world
    world.use_nodes = True
    
    # 移除现有节点
    for node in world.node_tree.nodes:
        world.node_tree.nodes.remove(node)
    
    # 创建新节点
    node_tree = world.node_tree
    output_node = node_tree.nodes.new(type='ShaderNodeOutputWorld')
    background_node = node_tree.nodes.new(type='ShaderNodeBackground')
    
    # 设置背景颜色为暗灰色
    background_node.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)
    background_node.inputs[1].default_value = 1.0  # 强度
    
    # 连接节点
    node_tree.links.new(background_node.outputs[0], output_node.inputs[0])

def setup_camera():
    """设置场景相机"""
    if not in_blender:
        return
    
    # 创建相机
    bpy.ops.object.camera_add(location=(15, 15, 8), rotation=(1.0, 0.0, 0.8))
    camera = bpy.context.active_object
    
    # 设置相机为活动相机
    bpy.context.scene.camera = camera
    
    # 设置相机参数
    camera.data.lens = 35  # 焦距
    camera.data.clip_start = 0.1
    camera.data.clip_end = 100

def create_legend():
    """创建场景图例"""
    if not in_blender:
        return
    
    # 添加文本说明图例
    bpy.ops.object.text_add(location=(-8, -8, 8))
    text = bpy.context.active_object
    text.data.body = "音乐推荐可视化\n\n"
    text.data.body += "球体 = 推荐歌曲\n"
    text.data.body += "轴心 = 用户\n"
    text.data.body += "连线 = 推荐关系\n\n"
    text.data.body += "颜色表示音乐风格\n"
    text.data.body += "大小表示推荐分数"
    
    # 设置文本属性
    text.data.size = 0.5
    text.data.extrude = 0.02
    
    # 创建并分配文本材质
    mat = bpy.data.materials.new(name="LegendText")
    mat.use_nodes = True
    principled_bsdf = mat.node_tree.nodes.get('Principled BSDF')
    if principled_bsdf:
        principled_bsdf.inputs[0].default_value = (1, 1, 1, 1)  # 白色
    
    text.data.materials.append(mat)
    
    # 添加图例对象到场景
    if 'Recommendations' in bpy.data.collections:
        if text.name in bpy.context.scene.collection.objects:
            bpy.context.scene.collection.objects.unlink(text)
        if text.name not in bpy.data.collections['Recommendations'].objects:
            bpy.data.collections['Recommendations'].objects.link(text)

def create_visualization(visualization_data):
    """创建完整的可视化场景"""
    if not in_blender:
        print("警告: 未在Blender环境中运行，跳过可视化创建")
        return False
    
    # 清空当前场景
    clear_scene()
    
    # 设置光照
    setup_lighting()
    
    # 设置相机
    setup_camera()
    
    # 按用户分组
    user_songs = {}
    for song in visualization_data:
        user_id = song.get('user_id', 'unknown')
        if user_id not in user_songs:
            user_songs[user_id] = []
        user_songs[user_id].append(song)
    
    # 创建歌曲对象
    song_objects = {}
    for song in visualization_data:
        # 基于推荐分数调整大小
        score = song.get('score', 0.5)
        size = 0.1 + score * 0.3  # 大小范围: 0.1-0.4
        
        # 创建球体表示歌曲
        sphere = create_song_sphere(song, size=size)
        if sphere:
            song_objects[song['song_id']] = sphere
    
    # 创建用户及连接
    for user_id, songs in user_songs.items():
        # 创建用户标记
        user_obj = create_user_marker(user_id, songs)
        if not user_obj:
            continue
            
        # 获取用户的歌曲对象
        user_song_objects = [song_objects.get(song['song_id']) for song in songs 
                            if song['song_id'] in song_objects]
        user_song_objects = [obj for obj in user_song_objects if obj is not None]
        
        # 创建连接
        create_connections(user_obj, user_song_objects)
    
    # 添加图例
    create_legend()
    
    # 重置视图
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for region in area.regions:
                if region.type == 'WINDOW':
                    override = {'area': area, 'region': region}
                    bpy.ops.view3d.view_all(override)
                    break
    
    print("可视化场景创建完成")
    return True

def export_visualization(output_file):
    """导出可视化场景"""
    if not in_blender:
        return False
    
    # 设置渲染参数
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.resolution_percentage = 100
    
    # 设置输出路径
    bpy.context.scene.render.filepath = output_file
    
    # 渲染图像
    bpy.ops.render.render(write_still=True)
    
    print(f"可视化图像已导出到: {output_file}")
    return True

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='音乐推荐可视化工具')
    parser.add_argument('--recs', type=str, required=True,
                        help='推荐结果JSON文件路径')
    parser.add_argument('--model', type=str, required=False,
                        help='混合模型文件路径 (.pkl)')
    parser.add_argument('--output', type=str, default='recommendations_viz.blend',
                        help='输出的Blender文件路径')
    parser.add_argument('--render', type=str, default=None,
                        help='渲染输出图像路径')
    parser.add_argument('--no-tsne', action='store_true',
                        help='不使用t-SNE降维，使用随机位置')
    
    # 从bpy.data传递的参数或命令行参数
    if in_blender:
        if '--' in sys.argv:
            argv = sys.argv[sys.argv.index('--') + 1:]
        else:
            argv = []
        return parser.parse_args(argv)
    else:
        return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    recommendations = load_recommendations(args.recs)
    
    # 加载模型（如果有）
    model = None
    if args.model and os.path.exists(args.model):
        model = load_hybrid_model(args.model)
    
    # 准备可视化数据
    visualization_data = prepare_visualization_data(
        recommendations, model, use_tsne=not args.no_tsne
    )
    
    # 在Blender环境中创建可视化
    if in_blender:
        create_visualization(visualization_data)
        
        # 保存Blender文件
        bpy.ops.wm.save_as_mainfile(filepath=args.output)
        print(f"Blender场景已保存到: {args.output}")
        
        # 渲染图像（如果需要）
        if args.render:
            export_visualization(args.render)
    else:
        print("数据准备完成，但未在Blender环境中运行")
        print(f"准备了 {len(visualization_data)} 个数据点用于可视化")
        print("请在Blender中执行此脚本以创建可视化")

if __name__ == "__main__":
    main() 