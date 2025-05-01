# DeepRecommend Music - 个性化音乐推荐系统

一个基于协同过滤、内容分析、上下文感知和深度学习的混合音乐推荐系统，支持使用Million Song Dataset(MSD)数据集训练，并集成了Spotify API和HKBU GenAI平台。

## 功能特点

### 推荐系统核心功能
- **混合推荐策略**：结合协同过滤、内容分析、上下文感知和深度学习推荐
- **多数据源融合**：结合用户评分、对话内容、游戏互动和问卷调查等多种数据源
- **用户向量支持**：可接收前端发送的实时用户行为向量
- **动态权重调整**：根据用户数据和情境自动调整各算法权重

### 数据处理和训练
- **可定制训练数据量**：支持使用不同规模的MSD数据子集
- **MSD数据集支持**：高效处理百万歌曲数据集，支持分块加载大型数据
- **多种评分转换**：支持多种播放次数到评分的转换方法（对数、线性、百分位）

### 外部集成
- **Spotify API集成**：获取Spotify音乐元数据和音频特征，丰富推荐数据
- **HKBU GenAI集成**：支持通过香港浸会大学AI平台提供情感分析和心理咨询功能

### 智能交互
- **情绪感知推荐**：根据用户情绪状态提供匹配的音乐推荐
- **活动场景推荐**：针对不同活动场景（学习、运动、放松等）推荐合适的音乐
- **实时反馈调整**：根据用户实时反馈动态调整推荐策略

### 技术实现
- **深度学习模型**：整合MLP(多层感知机)和NCF(神经协同过滤)深度推荐算法
- **实时向量更新**：根据用户情绪、对话和听歌行为实时更新用户向量
- **模型序列化**：支持保存和加载训练好的模型

## 安装

1. 克隆代码库：

```bash
git clone https://github.com/Ethanwithtech/Deeprecommend-music-.git
cd Deeprecommend-music-
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备MSD数据集（可选）：
   - 下载MSD数据集（[官方网站](http://millionsongdataset.com/pages/getting-dataset/)）
   - 将`msd_summary_file.h5`和`train_triplets.txt`放在项目根目录

4. (可选) 配置Spotify API：
   - 创建`.env`文件并添加以下内容：
   ```
   SPOTIFY_CLIENT_ID=你的Spotify客户端ID
   SPOTIFY_CLIENT_SECRET=你的Spotify客户端密钥
   ```

5. (可选) 配置HKBU GenAI API：
   - 在`.env`文件中添加以下内容：
   ```
   HKBU_API_KEY=你的HKBU API密钥
   ```

## 快速入门

### 使用样本数据启动

```bash
# 创建样本数据并启动应用
python backend/pretrainer.py --sample
python start_recommender.py
```

### 使用MSD数据集训练模型

```bash
# 使用脚本快速启动训练
python run_msd_training.py

# 或者使用自定义参数
python run_msd_training.py --path /your/msd/path --chunk_limit 5 --rating linear --epochs 10
```

### 启动API服务器

```bash
python api_server.py
```

默认地址为：http://localhost:5000

## API端点

- **健康检查**: `GET /api/health`
- **获取推荐**: `GET /api/recommend?user_id=USER123&top_n=10&context=morning`
- **更新用户向量**: `POST /api/user_vector` (JSON格式: `{"user_id": "USER123", "user_vector": [0.1, 0.2, ...]}`)
- **获取可用上下文**: `GET /api/contexts`
- **AI聊天**: `POST /api/chat_with_ai` (JSON格式: `{"message": "我今天很难过", "user_id": "USER123", "history": [...]}`)

## 核心算法

系统整合了五种核心推荐算法：

1. **SVD++协同过滤**：基于矩阵分解的协同过滤算法，分析用户-物品交互模式
2. **内容特征匹配**：基于音乐声学特性(节奏、能量、情绪等)的内容推荐
3. **上下文感知推荐**：根据用户情绪状态和活动场景优化推荐
4. **MLP深度推荐**：使用多层感知机深度学习用户-物品交互模式
5. **NCF神经协同过滤**：结合矩阵分解和神经网络的混合深度学习推荐

## 用户向量更新功能

系统支持实时更新和存储用户向量信息，包括：

### 1. 用户情绪向量更新
- 根据情绪类型赋予不同权重
- 将情绪信息存储到数据库中的`normalized_preferences`表
- 同时在`user_emotion_history`表中记录用户情绪历史

### 2. 对话向量更新
- 尝试使用jieba分词提取关键词，如不可用则使用简单分词方法
- 将对话内容、情绪和提取的关键词存储到`user_dialogue_history`表
- 识别对话中的音乐相关关键词，更新用户的音乐流派偏好

### 3. 歌曲偏好更新
- 支持不同偏好来源的权重调整（评分、听歌、AI推荐）
- 将偏好记录存储到`user_song_preferences`表

### 4. 听歌向量更新
- 根据播放时长和完成情况计算偏好权重
- 记录用户的听歌偏好

## 数据处理流程

处理Million Song Dataset的完整流程：

1. 解析HDF5文件获取歌曲元数据和音频特征
2. 处理triplet文件获取用户交互数据
3. 数据对齐和清洗
4. 特征规范化和编码
5. 集成Spotify API数据（可选）
6. 生成用户特征
7. 保存处理好的数据

## MSD数据集训练详情

### 数据集介绍

Million Song Dataset (MSD) 是一个免费可用的音乐数据集，包含百万首歌曲的音频特征和元数据。该数据集由哥伦比亚大学LabROSA实验室和The Echo Nest合作创建。

主要组成部分：
- **歌曲元数据**：歌曲名称、艺术家、专辑、发行年份等
- **音频特征**：音调、节奏、响度等声学特征
- **用户交互数据**：用户-歌曲-播放次数三元组

### 获取MSD数据集

Million Song Dataset非常大（约280GB），为方便使用，我们提供了两种方式：

#### 1. 使用完整MSD数据集

从官方网站下载数据集：http://millionsongdataset.com/

主要文件：
- **HDF5数据**：包含所有歌曲的音频特征和元数据
- **Triplet数据集**：包含用户-歌曲-播放次数
- **其他辅助数据**：标签、相似性等

#### 2. 使用样本数据（推荐用于测试）

如果你只想测试系统功能，可以使用我们提供的样本数据：

```bash
# 自动生成样本数据
python backend/pretrainer.py --sample
```

### 自定义训练参数

可以通过命令行参数自定义训练过程：

```bash
python run_msd_training.py --path /your/msd/path --chunk_limit 5 --rating linear --epochs 10
```

参数说明：

#### 数据路径
- `--path`：MSD数据根目录
- `--h5`：H5文件名
- `--triplet`：Triplet文件名

#### 数据处理
- `--chunk_limit`：处理的数据块数限制（测试用，每块约100万条记录）
- `--force_process`：强制重新处理数据，忽略缓存
- `--no_spotify`：不使用Spotify API
- `--spotify_max`：使用Spotify API处理的最大歌曲数
- `--rating`：播放次数转评分方法（log、linear、percentile）

#### 模型训练
- `--epochs`：训练轮数
- `--batch_size`：批次大小
- `--skip_deep`：跳过深度学习模型训练
- `--skip_hybrid`：跳过混合模型训练

### 播放次数转评分方法

本系统提供了三种播放次数到评分的转换方法：

1. **对数方法 (log)**：
   ```
   rating = min(5, max(1, int(log2(plays + 1) + 1)))
   ```
   这种方法适合播放次数分布不均的数据，将增长较快的播放次数映射到1-5分。

2. **线性方法 (linear)**：
   基于用户平均播放次数的相对比例，更好地反映用户个人偏好强度。

3. **百分位方法 (percentile)**：
   对每个用户的播放次数进行排序，按百分位分配评分，确保每个用户的评分分布均匀。

## Spotify API集成

### 设置步骤

1. 登录[Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. 创建一个应用获取Client ID和Client Secret
3. 将凭证写入`.env`文件

系统默认使用以下Spotify API凭证（demo用）：

- Client ID: 4f1a2f4e1e034050ac432f8ebba72484
- Client Secret: 4abd4c31749748c8b89f7807c61a3f11

如需使用您自己的凭证，可以通过以下参数指定：

```bash
python backend/train_msd_with_deep.py --use_spotify --spotify_client_id YOUR_ID --spotify_client_secret YOUR_SECRET
```

## HKBU GenAI API设置

### 功能概述

HKBU AI聊天功能允许用户与AI助手进行对话，AI助手能够：
- 分析用户的情绪状态
- 提供心理支持和建议
- 根据用户情绪推荐合适的音乐
- 解释音乐如何影响情绪和心理健康

### 设置步骤

1. 向HKBU申请API密钥
2. 将API密钥添加到环境变量或`.env`文件
3. 系统会在聊天时自动调用HKBU的AI服务，提供心理咨询和音乐推荐功能

#### 使用辅助脚本设置（推荐）

我们提供了一个简单的脚本来帮助您设置API密钥：

```bash
python setup_hkbu_api.py
```

按照提示输入您的API密钥即可。

## 模型训练

模型训练流程包括：

1. 协同过滤模型训练
2. 内容特征模型训练
3. 上下文感知模型训练
4. 深度学习模型训练（MLP和NCF）
5. 权重动态调整
6. 模型集成与保存

### 模型预训练参数

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

## 项目结构

```
deeprecommend-music/
├── api_server.py          # API服务器
├── start_recommender.py   # 启动应用
├── requirements.txt       # 依赖列表
├── .env                   # 环境变量配置
├── models/                # 模型存储目录
│   └── trained/           # 训练好的模型
├── data/                  # 数据目录
├── processed_data/        # 处理后的数据
└── backend/               # 后端代码
    ├── api/               # API接口
    ├── models/            # 模型实现
    │   ├── hybrid_music_recommender.py  # 混合推荐系统
    │   ├── ai_music_agent.py            # AI音乐代理
    │   ├── emotion_analyzer.py          # 情感分析器
    │   └── recommendation_engine.py     # 推荐引擎
    └── utils/             # 工具函数
```

## 常见问题

### 数据和训练相关
- **处理完整MSD数据需要多长时间？** 取决于硬件配置，使用标准配置（16GB RAM, 4核CPU）处理完整数据集大约需要6-8小时。
- **内存不足怎么办？** 可以设置较小的`chunk_limit`参数，或通过调整`sample_size`参数减少数据量。
- **如何只使用部分数据进行快速测试？** 使用`--chunk_limit 5`参数限制处理的数据块数，或使用`pretrainer.py --sample`生成样本数据。
- **模型训练失败如何调试？** 检查日志输出，确保数据路径正确，内存足够。也可以开启`--debug`模式获取更详细的日志。

### 其他常见问题
- **权限问题**：如果遇到文件权限问题，请使用`icacls train_triplets.txt /grant Everyone:R`授予读权限
- **模型加载失败**：确保API服务器和训练脚本使用相同版本的依赖库
- **HKBU API错误**：检查API密钥是否正确，以及网络连接是否正常
- **Spotify API限制**：减小`spotify_max`参数，或获取自己的API凭证
- **训练过慢**：使用`--skip_deep`跳过深度学习模型训练，或减小`epochs`参数
- **缺少jieba模块**：如果缺少分词模块，系统会使用简单分词方法

## 后续优化方向

1. 添加更多情绪分析和分类算法，提高情绪识别准确性
2. 优化关键词提取算法，更精确地捕捉用户音乐偏好
3. 扩展偏好权重调整算法，更好地平衡不同来源的偏好信息
4. 添加用户向量周期性聚合功能，减少数据库存储压力
5. 集成更多外部数据源，丰富推荐系统的数据基础
6. 优化深度学习模型的训练效率和性能

## 开发者

- 邮箱：22256342@life.hkbu.edu.hk
- GitHub：[Ethanwithtech](https://github.com/Ethanwithtech)

## 许可证

本项目使用MIT许可证。

## 鸣谢

- Million Song Dataset (MSD)
- Spotify Web API
- TensorFlow
- HKBU GenAI Platform 