# 7digital API 配置指南

本文档介绍如何获取和配置7digital API密钥，以便在音乐推荐系统中启用歌曲预览功能。

## 什么是7digital API？

7digital是一个数字音乐提供商，其API允许应用程序访问大量音乐元数据和音频预览。在我们的音乐推荐系统中，7digital API用于获取歌曲的30秒预览片段，让用户可以试听推荐的歌曲。

## 获取7digital API密钥

1. 访问[7digital Developer](https://developer.7digital.com/)网站并注册一个账户
2. 登录后，在开发者控制台中申请一个新的API密钥
3. 完成注册流程后，您将获得以下信息：
   - Consumer Key (API密钥)
   - Consumer Secret (API密钥密钥)

## 配置密钥

获取密钥后，您需要将它们配置到系统中：

### 方法1：环境变量

设置以下环境变量：

```bash
export SEVENDIGITAL_API_KEY=your_api_key_here
export SEVENDIGITAL_API_SECRET=your_api_secret_here
```

### 方法2：.env文件

编辑项目根目录下的`.env`文件，添加或修改以下行：

```
SEVENDIGITAL_API_KEY=your_api_key_here
SEVENDIGITAL_API_SECRET=your_api_secret_here
```

将`your_api_key_here`和`your_api_secret_here`替换为您从7digital获取的实际密钥。

## 测试配置

配置完成后，您可以运行测试脚本验证API连接：

```bash
cd backend
python test_7digital.py
```

如果一切正常，您应该看到类似以下的输出：

```
2025-04-19 20:10:12,123 - INFO - 7digital API配置已检测
2025-04-19 20:10:12,345 - INFO - 搜索歌曲: Shape of You - Ed Sheeran
2025-04-19 20:10:13,678 - INFO - 找到匹配歌曲: Shape of You (ID: 12345678)
2025-04-19 20:10:13,789 - INFO - 获取曲目ID 12345678 的预览URL
2025-04-19 20:10:14,890 - INFO - 找到预览URL: http://previews.7digital.com/...
```

## 使用提示

- 7digital API对请求数量有限制，请合理使用
- 预览URL通常有效期为24小时
- 如果您遇到错误，请检查API密钥是否正确配置

## 无法获取7digital API密钥？

如果您无法获取7digital API密钥，系统仍然可以正常工作，但将无法提供歌曲预览功能。用户仍然可以获取音乐推荐，只是无法试听推荐的歌曲。

## 疑难解答

如果您设置了正确的密钥但仍然遇到问题：

1. 检查API密钥是否正确复制（没有多余的空格）
2. 确认您的7digital开发者账户处于激活状态
3. 查看应用日志中的具体错误信息
4. 检查防火墙是否阻止了对7digital API的访问 