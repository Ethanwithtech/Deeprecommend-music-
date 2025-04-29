# MSD + Spotify 混合推荐系统

本项目基于Million Song Dataset (MSD)和Spotify API数据，实现了一个混合推荐系统，结合了协同过滤、内容特征和深度学习。

## 功能特点

- **数据处理**：高效处理MSD数据集，支持分块加载以处理大型数据集
- **Spotify集成**：自动获取Spotify音乐元数据和音频特征，丰富推荐数据
- **多种评分转换**：支持多种播放次数到评分的转换方法（对数、线性、百分位）
- **深度学习推荐**：基于TensorFlow实现的深度学习推荐模型
- **混合推荐策略**：集成协同过滤、内容特征和深度学习的混合推荐系统
- **模型评估**：内置推荐系统评估指标（精确率、召回率、NDCG）

## 安装依赖

安装所需的依赖包：

```bash
pip install -r requirements.txt
```

## 快速开始

使用脚本快速启动训练：

```bash
python run_msd_training.py
```

这将使用默认参数开始处理MSD数据集并训练模型。

## 自定义参数

可以通过命令行参数自定义训练过程：

```bash
python run_msd_training.py --path /your/msd/path --chunk_limit 5 --rating linear --epochs 10
```

参数说明：

### 数据路径
- `--path`：MSD数据根目录（默认："C:/Users/dyc06/Desktop/Deeprecommend-music-"）
- `--h5`：H5文件名（默认："msd_summary_file.h5"）
- `--triplet`：Triplet文件名（默认："train_triplets.txt/train_triplets.txt"）

### 数据处理
- `--chunk_limit`：处理的数据块数限制（测试用，每块约100万条记录）
- `--force_process`：强制重新处理数据，忽略缓存
- `--no_spotify`：不使用Spotify API
- `--spotify_max`：使用Spotify API处理的最大歌曲数（默认：1000）
- `--rating`：播放次数转评分方法（log、linear、percentile）

### 模型训练
- `--epochs`：训练轮数（默认：20）
- `--batch_size`：批次大小（默认：256）
- `--skip_deep`：跳过深度学习模型训练
- `--skip_hybrid`：跳过混合模型训练

## 高级用法

### 直接使用训练脚本

如果需要更多自定义选项，可以直接使用训练脚本：

```bash
python backend/train_msd_with_deep.py --h5_file [h5_path] --triplet_file [triplet_path] --use_spotify --rating_method log
```

### 调整数据处理和评分方法

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

### Spotify API集成

系统默认使用以下Spotify API凭证：

- Client ID: 4f1a2f4e1e034050ac432f8ebba72484
- Client Secret: 4abd4c31749748c8b89f7807c61a3f11

如需使用您自己的凭证，可以通过以下参数指定：

```bash
python backend/train_msd_with_deep.py --use_spotify --spotify_client_id YOUR_ID --spotify_client_secret YOUR_SECRET
```

## Web演示

训练完成后，可以启动Web演示界面：

```bash
python start_recommender.py
```

访问 http://localhost:5000 查看推荐系统界面。

## 故障排除

1. **内存不足**：减小`chunk_limit`参数，限制处理的数据量
2. **Spotify API限制**：减小`spotify_max`参数，或获取自己的API凭证
3. **训练过慢**：使用`--skip_deep`跳过深度学习模型训练，或减小`epochs`参数
4. **文件路径错误**：确保正确设置了`--path`、`--h5`和`--triplet`参数

## 许可证

本项目使用MIT许可证。

## 鸣谢

- Million Song Dataset (MSD)
- Spotify Web API
- TensorFlow 