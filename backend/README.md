# DeepRecommend Music - 混合推荐系统

## 混合推荐系统概述

DeepRecommend Music采用多算法混合的音乐推荐系统，支持MSD数据集预训练和实时调整。系统结合了协同过滤、内容特征、上下文感知以及深度学习推荐模型，能够提供高质量的个性化推荐。

## 主要特性

- **多数据源融合**：结合用户评分、对话内容、游戏互动和问卷调查等多种数据源
- **多算法混合**：集成SVD++协同过滤、内容特征匹配、上下文感知推荐和深度学习模型
- **深度学习模型**：整合MLP(多层感知机)和NCF(神经协同过滤)深度推荐算法
- **动态权重调整**：根据用户数据和情境自动调整各算法权重
- **情绪感知推荐**：根据用户情绪状态提供匹配的音乐推荐
- **活动场景推荐**：针对不同活动场景（学习、运动、放松等）推荐合适的音乐
- **MSD数据预训练**：支持使用百万歌曲数据集进行模型预训练
- **实时反馈调整**：根据用户实时反馈动态调整推荐策略

## 核心算法

系统整合了五种核心推荐算法：

1. **SVD++协同过滤**：基于矩阵分解的协同过滤算法，分析用户-物品交互模式
2. **内容特征匹配**：基于音乐声学特性(节奏、能量、情绪等)的内容推荐
3. **上下文感知推荐**：根据用户情绪状态和活动场景优化推荐
4. **MLP深度推荐**：使用多层感知机深度学习用户-物品交互模式
5. **NCF神经协同过滤**：结合矩阵分解和神经网络的混合深度学习推荐

## 使用方法

### 1. 准备MSD数据和预训练

```bash
# 创建示例数据
python models/pretrainer.py --create_sample --output_dir processed_data

# 使用真实MSD数据预处理和预训练(包括深度学习模型)
python models/pretrainer.py --msd_path /path/to/msd/data --output_dir processed_data --model_path models/pretrained_model.pkl --train_deep_models --gpu
```

### 2. 在应用中使用预训练模型

```python
from backend.models.ai_music_agent import MusicRecommenderAgent

# 初始化推荐代理，加载预训练模型
agent = MusicRecommenderAgent(
    data_dir="processed_data",
    use_msd=True,
    load_pretrained=True,
    pretrained_model_path="models/pretrained_model.pkl"
)

# 处理用户消息，获取推荐
response = agent.process_message("user123", "我想听一些轻松的音乐")
```

### 3. 实时处理用户反馈

```python
# 处理用户的新评分
agent.handle_new_user_feedback("user123", "song_id_123", 5)  # 5分好评

# 获取最新推荐
recommendations = agent.hybrid_recommender.recommend("user123", top_n=10)
```

## 模型预训练参数

MSD数据预训练支持以下参数：

- `--msd_path`: MSD原始数据路径
- `--output_dir`: 处理后数据输出目录
- `--model_path`: 预训练模型保存路径
- `--create_sample`: 是否创建示例数据
- `--sample_size`: 处理MSD数据时的样本大小限制
- `--batch_size`: 预训练时的批处理大小
- `--max_users`: 预训练时处理的最大用户数
- `--train_deep_models`: 是否训练深度学习模型(MLP和NCF)
- `--gpu`: 是否使用GPU加速深度学习模型训练

## 系统集成

混合推荐系统已与AI音乐代理完全集成，支持以下功能：

1. 智能对话中提取用户偏好信息
2. 情绪分析结果直接用于调整推荐
3. 实时处理用户评分和反馈
4. 多种推荐方式：基于情绪、基于艺术家、基于内容和深度学习推荐

## 注意事项

- 首次使用时，推荐先执行预训练以获得更好的初始推荐质量
- 深度学习模型(MLP和NCF)需要大量评分数据进行训练，建议使用更大的MSD数据集
- 实时调整功能要求有足够的用户反馈数据，建议收集足够的用户评分
- 情绪感知推荐需要先进行情绪分析
- GPU训练可大幅提升深度学习模型训练速度 