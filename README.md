# 混合音乐推荐系统

一个基于协同过滤、内容分析和上下文感知的混合音乐推荐系统，支持使用Million Song Dataset(MSD)数据集训练，并可集成Spotify API。

## 功能特点

- **混合推荐策略**：结合协同过滤、内容分析和上下文感知推荐
- **用户向量支持**：可接收前端发送的实时用户行为向量
- **可定制训练数据量**：支持使用不同规模的MSD数据子集
- **Spotify API集成**：可选择性集成Spotify API获取更丰富的音乐信息
- **模型序列化**：支持保存和加载训练好的模型
- **REST API接口**：提供HTTP接口供前端调用

## 安装

1. 克隆代码库：

```bash
git clone https://github.com/yourusername/deeprecommend-music.git
cd deeprecommend-music
```

2. 安装依赖：

```bash
pip install -r requirements.txt
```

3. 准备MSD数据集：

- 下载MSD数据集（[官方网站](http://millionsongdataset.com/pages/getting-dataset/)）
- 将`msd_summary_file.h5`和`train_triplets.txt`放在项目根目录

4. (可选) 配置Spotify API：
   - 创建`.env`文件并添加以下内容：
   ```
   SPOTIFY_CLIENT_ID=你的Spotify客户端ID
   SPOTIFY_CLIENT_SECRET=你的Spotify客户端密钥
   ```

## 使用方法

### 训练模型

使用以下命令训练模型：

```bash
python train_model_10k.py
```

修改样本大小（默认10000条）：

```python
# 在train_model_10k.py文件末尾修改参数
if __name__ == "__main__":
    # ...
    try:
        # 修改sample_size参数，例如使用50000条数据
        model = train_model(h5_path, triplet_path, sample_size=50000)
        print("✅ 模型训练成功")
    # ...
```

### 启动API服务器

```bash
python api_server.py
```

默认地址为：http://localhost:5000

### API端点

- **健康检查**: `GET /api/health`
- **获取推荐**: `GET /api/recommend?user_id=USER123&top_n=10&context=morning`
- **更新用户向量**: `POST /api/user_vector` (JSON格式: `{"user_id": "USER123", "user_vector": [0.1, 0.2, ...]}`)
- **获取可用上下文**: `GET /api/contexts`

## 项目结构

```
deeprecommend-music/
├── api_server.py          # API服务器
├── train_model_10k.py     # 训练脚本
├── requirements.txt       # 依赖列表
├── .env                   # 环境变量配置
├── models/                # 模型存储目录
│   └── trained/           # 训练好的模型
└── backend/               # 后端代码
    ├── data_processor/    # 数据处理
    ├── models/            # 各类模型实现
    └── utils/             # 工具函数
```

## Spotify API设置

1. 登录[Spotify Developer Dashboard](https://developer.spotify.com/dashboard/)
2. 创建一个应用获取Client ID和Client Secret
3. 将凭证写入`.env`文件：
```
SPOTIFY_CLIENT_ID=bdfa10b0a8bf49a3a413ba67d2ff1706
SPOTIFY_CLIENT_SECRET=b8e97ad8e96043b4b0d768d3e3c568b4
```

## 常见问题

- **权限问题**：如果遇到文件权限问题，请使用`icacls train_triplets.txt /grant Everyone:R`授予读权限
- **内存不足**：处理大规模数据时，可以通过调整`sample_size`参数减少数据量
- **模型加载失败**：确保API服务器和训练脚本使用相同版本的依赖库 