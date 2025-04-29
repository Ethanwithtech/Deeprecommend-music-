# 音乐推荐系统故障排除指南

## 常见问题

### 1. 模块导入错误 - `MusicRecommender.__init__() got an unexpected keyword argument 'model_type'`

这个错误通常是由Python缓存导致的。当你修改了`recommendation_engine.py`文件后，Python可能仍然使用旧版本的缓存文件。

**解决方案:**

1. 清理Python缓存文件：
   
   ```bash
   # 在Windows上
   Remove-Item -Path . -Recurse -Include "__pycache__" -Force
   
   # 在Linux/Mac上
   find . -name "__pycache__" -type d -exec rm -rf {} +
   ```

2. 使用`-B`选项运行Python，以避免创建缓存文件：
   
   ```bash
   python -B app.py
   ```

### 2. 数据库连接错误

如果遇到数据库连接问题，可能是相对路径不正确。

**解决方案:**

确保从项目根目录运行应用程序：

```bash
cd /path/to/Deeprecommend-music-
python app.py
```

### 3. 内存不足错误 - 处理百万歌曲数据集

当处理大型数据集时，可能会遇到内存不足的问题。

**解决方案:**

1. 调整`SAMPLE_SIZE`环境变量以限制使用的数据量：
   
   ```bash
   # 在.env文件中设置
   SAMPLE_SIZE=10000
   ```

2. 增加系统虚拟内存（或者使用更大内存的机器）。

### 4. API密钥缺失警告

如果看到关于Spotify API或Claude API密钥缺失的警告，某些功能可能不可用。

**解决方案:**

在`.env`文件中设置必要的API密钥：

```
SPOTIFY_CLIENT_ID=your_spotify_client_id
SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
ANTHROPIC_API_KEY=your_claude_api_key
```

## 诊断工具

项目包含一个诊断工具，可以帮助识别问题：

```bash
python test_imports.py
```

该工具会清理缓存、检查导入路径，并尝试创建必要的组件实例。

## 联系支持

如果仍然遇到问题，请提交GitHub Issue，并附上以下信息：

1. Python版本
2. 操作系统
3. 错误信息完整截图
4. `test_imports.py`的运行输出 