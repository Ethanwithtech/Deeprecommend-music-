# AI音乐推荐系统设置指南

本文档提供AI音乐推荐系统的设置和运行指南。

## 1. 环境准备

### 安装依赖
首先安装所有必要的Python依赖：

```bash
pip install -r requirements.txt
```

### 配置API密钥
复制`.env.example`到`.env`，并编辑该文件设置必要的API密钥：

```bash
cp .env.example .env
# 然后用文本编辑器打开.env文件
```

在`.env`文件中，需要设置以下关键参数：

- `ANTHROPIC_API_KEY`: Claude AI API密钥，从[Anthropic Console](https://console.anthropic.com/)获取
- `SPOTIFY_CLIENT_ID`和`SPOTIFY_CLIENT_SECRET`: 从[Spotify开发者控制台](https://developer.spotify.com/dashboard/)获取

## 2. 数据准备

### 使用百万歌曲数据集
如果要使用百万歌曲数据集(MSD)，请确保您有以下文件：
- `msd_summary_file.h5`: 包含歌曲元数据
- `train_triplets.txt`: 包含用户-歌曲-播放次数数据

将这些文件放置在项目根目录下。

### 使用自定义数据集
如果不使用MSD，请编辑`.env`文件中的`USE_MSD`设置为`false`。系统将使用内置的示例数据。

## 3. 测试API连接

在开始使用之前，建议测试API连接是否正常：

```bash
python test_ai_connection.py
```

如果连接测试成功，您将看到确认消息。

## 4. 启动系统

运行以下命令启动Flask应用服务器：

```bash
python app.py
```

系统将在`http://localhost:5000`上启动。在浏览器中访问该地址即可使用音乐推荐系统。

## 5. 与AI助手交互

系统启动后，您可以通过以下方式与AI助手交互：

1. 通过Web界面: 在主页上的聊天框中输入消息
2. 通过API: 向`/api/chat`端点发送POST请求，格式为：
   ```json
   {
     "user_id": "your_user_id",
     "message": "推荐一些流行歌曲"
   }
   ```

## 6. 常见问题

### API连接问题

如果遇到Claude API连接问题，请检查：
- API密钥是否正确设置
- 是否有有效的Claude API订阅
- 网络连接是否正常

### 数据加载问题

如果遇到数据加载问题，请检查：
- 数据文件路径是否正确
- 文件格式是否正确
- `.env`中的`DATA_DIR`配置是否正确

### 更多帮助

如需更多帮助，请参考[项目文档](README.md)或提交问题。 