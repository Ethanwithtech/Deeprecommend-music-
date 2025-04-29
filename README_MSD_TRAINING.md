# Million Song Dataset (MSD) 推荐系统训练指南

本文档提供了如何获取、处理Million Song Dataset并训练混合推荐模型的详细指导。

## 目录

- [数据集介绍](#数据集介绍)
- [环境准备](#环境准备)
- [获取MSD数据集](#获取msd数据集)
- [数据处理流程](#数据处理流程)
- [模型训练](#模型训练)
- [启动推荐系统](#启动推荐系统)
- [常见问题解答](#常见问题解答)

## 数据集介绍

Million Song Dataset (MSD) 是一个免费可用的音乐数据集，包含百万首歌曲的音频特征和元数据。该数据集由哥伦比亚大学LabROSA实验室和The Echo Nest合作创建。

主要组成部分：
- **歌曲元数据**：歌曲名称、艺术家、专辑、发行年份等
- **音频特征**：音调、节奏、响度等声学特征
- **用户交互数据**：用户-歌曲-播放次数三元组

## 环境准备

确保系统已安装以下软件：

1. Python 3.8+
2. pip或conda包管理器

安装所需依赖包：

```bash
pip install -r requirements.txt
```

## 获取MSD数据集

Million Song Dataset非常大（约280GB），为方便使用，我们提供了两种方式：

### 1. 使用完整MSD数据集

从官方网站下载数据集：http://millionsongdataset.com/

主要文件：
- **HDF5数据**：包含所有歌曲的音频特征和元数据
- **Triplet数据集**：包含用户-歌曲-播放次数
- **其他辅助数据**：标签、相似性等

### 2. 使用样本数据（推荐用于测试）

如果你只想测试系统功能，可以使用我们提供的样本数据：

```bash
# 自动生成样本数据
python backend/pretrainer.py --sample
```

## 数据处理流程

完整的数据处理流程包括：

1. 解析HDF5文件获取歌曲元数据和音频特征
2. 处理triplet文件获取用户交互数据
3. 数据对齐和清洗
4. 特征规范化和编码
5. 生成用户特征
6. 保存处理好的数据

使用下面的命令进行数据处理：

```bash
python backend/process_msd_data.py \
    --h5_path <HDF5数据路径> \
    --triplets_path <Triplet数据路径> \
    --output_dir models/trained/processed_data \
    --model_path models/trained/hybrid_recommender.pkl
```

参数说明：
- `--h5_path`：HDF5数据文件或目录路径
- `--triplets_path`：用户-歌曲-播放次数数据文件路径
- `--output_dir`：处理后数据保存目录
- `--model_path`：训练好的模型保存路径
- `--chunk_limit`：(可选) 处理的数据块数量，用于测试

## 模型训练

模型训练在数据处理之后自动进行，包括：

1. 协同过滤模型训练
2. 内容特征模型训练
3. 上下文感知模型训练
4. 权重动态调整
5. 模型集成与保存

如果你想单独进行模型训练，可以使用：

```bash
python backend/pretrainer.py \
    --data_dir models/trained/processed_data \
    --model_path models/trained/hybrid_recommender.pkl
```

## 启动推荐系统

训练完成后，可以启动Web演示界面：

```bash
python start_recommender.py
```

更多启动选项：

```bash
python start_recommender.py --help
```

参数包括：
- `--model_path`：指定模型路径
- `--data_path`：指定数据路径
- `--port`：指定服务器端口
- `--debug`：启用调试模式

## 常见问题解答

### Q: 处理完整MSD数据需要多长时间？
A: 取决于硬件配置，使用标准配置（16GB RAM, 4核CPU）处理完整数据集大约需要6-8小时。

### Q: 内存不足怎么办？
A: 可以设置较小的`chunk_limit`参数，这样程序会分批次处理数据，但总处理时间会变长。

### Q: 如何只使用部分数据进行快速测试？
A: 使用`--chunk_limit 5`参数限制处理的数据块数，或使用`pretrainer.py --sample`生成样本数据。

### Q: 模型训练失败如何调试？
A: 检查日志输出，确保数据路径正确，内存足够。也可以开启`--debug`模式获取更详细的日志。

---

如有更多问题，请提交Issue或联系项目维护者。 