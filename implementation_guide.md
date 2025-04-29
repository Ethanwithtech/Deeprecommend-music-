# 音乐评分问卷整合实施指南

按照图片显示的需求，我们需要将问卷功能与评分功能整合在一起。以下是实施步骤：

## 前端修改

### 1. 修改导航栏链接
在`frontend/templates/index.html`文件的第54行，修改问卷链接：

```html
<!-- 修改前 -->
<a class="navbar-item" href="/questionnaire" v-if="isLoggedIn">
    <i class="fas fa-clipboard-list mr-2"></i>
    <span>音乐问卷</span>
</a>

<!-- 修改后 -->
<a class="navbar-item" data-tab="rate" v-if="isLoggedIn">
    <i class="fas fa-clipboard-list mr-2"></i>
    <span>音乐评分问卷</span>
</a>
```

### 2. 修改页脚链接
在`frontend/templates/index.html`文件的第700行，修改页脚问卷链接：

```html
<!-- 修改前 -->
<li><a href="/questionnaire">音乐偏好问卷</a></li>

<!-- 修改后 -->
<li><a href="#" data-tab="rate">音乐评分问卷</a></li>
```

### 3. 修改评分界面标题
在`frontend/templates/index.html`文件的第381-386行，修改评分界面标题：

```html
<!-- 修改前 -->
<h2 class="title is-4">
    <i class="fas fa-star"></i> {{ t('rate') }}
</h2>
<p class="subtitle is-6">{{ t('rateSubtitle') }}</p>

<!-- 修改后 -->
<h2 class="title is-4">
    <i class="fas fa-clipboard-list mr-2"></i> 音乐评分问卷
</h2>
<p class="subtitle is-6">为歌曲评分，帮助我们了解您的音乐偏好</p>
```

### 4. 更新语言翻译（确保中英文切换正常工作）
在`frontend/static/js/main.js`文件中的语言翻译部分（约215-255行）修改翻译文本：

```javascript
// 中文翻译
'zh': {
  // ...现有翻译...
  'rate': '音乐评分问卷',
  'rateSubtitle': '为歌曲评分，帮助我们了解您的音乐偏好',
  // ...其他翻译...
},

// 英文翻译
'en': {
  // ...现有翻译...
  'rate': 'Music Rating Questionnaire',
  'rateSubtitle': 'Rate songs to help us understand your music preferences',
  // ...其他翻译...
}
```

## 后端修改

### 1. 修改`/questionnaire`路由重定向
在`backend/api/app.py`文件中，修改问卷路由函数（约1193行）：

```python
# 需要导入redirect
from flask import Flask, request, jsonify, render_template, send_from_directory, render_template_string, make_response, redirect

@app.route('/questionnaire')
def questionnaire():
    """重定向到评分问卷界面"""
    logger.info("用户访问问卷调查页面，重定向到评分界面")
    return redirect('/?tab=rate')
```

同样在`backend/app.py`文件中修改相应函数：

```python
@app.route('/questionnaire')
def questionnaire():
    """重定向到评分问卷界面"""
    return redirect('/?tab=rate')
```

## 实施注意事项

1. **保留黑色和紫色主题**：
   - 无需修改CSS文件，因为已有的黑紫色主题已符合要求
   - 确保新修改内容使用现有的CSS类和颜色变量

2. **调整会话管理**：
   - 当用户从外部链接访问`/questionnaire`时，现在会重定向到主页的评分选项卡
   - 在`main.js`的`mounted`方法中，添加对URL参数`tab`的检查：

```javascript
// 在main.js的mounted方法中添加
mounted() {
  // 现有代码...
  
  // 检查URL参数
  const urlParams = new URLSearchParams(window.location.search);
  const tabParam = urlParams.get('tab');
  if (tabParam && ['welcome', 'rate', 'recommend', 'chat', 'game'].includes(tabParam)) {
    this.currentTab = tabParam;
  }
  
  // 现有代码...
}
```

## 验证测试

完成以上修改后，请通过以下步骤测试功能：

1. 导航栏中的"音乐评分问卷"链接应直接打开评分界面
2. 页脚中的"音乐评分问卷"链接应直接打开评分界面
3. 访问`/questionnaire`URL应重定向到带有评分选项卡的主页
4. 评分界面标题应显示为"音乐评分问卷"
5. 切换语言后，评分界面标题应相应更新为英文/中文
6. 评分功能应正常工作，包括星级评分和推荐功能 