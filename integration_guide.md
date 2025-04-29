# 音乐评分问卷集成方案

本方案提供将问卷调查页面(questionnaire.html)整合到主页面(index.html)评分功能中的详细实施步骤。整合后，用户在单一界面中即可完成所有音乐偏好相关的交互，保持黑紫色主题设计和中英文双语支持。

## 总体方案

1. 修改导航链接，使问卷调查链接指向评分标签页
2. 增强评分界面，添加问卷部分交互元素
3. 整合问卷UI与其交互逻辑到主脚本
4. 调整后端路由，重定向问卷请求到评分界面

## 具体修改步骤

### 1. 修改导航栏链接

将原本指向`/questionnaire`的链接修改为通过数据属性切换到评分标签页：

```html
<!-- 在导航栏中 (frontend/templates/index.html) -->
<a class="navbar-item" data-tab="rate">
    <i class="fas fa-clipboard-list"></i>
    <span>{{ t('questionnaire') }}</span>
</a>
```

### 2. 修改欢迎页面卡片链接

将问卷调查卡片的链接修改为指向评分标签页：

```html
<!-- 在欢迎页面问卷卡片中 (frontend/templates/index.html) -->
<div class="has-text-centered mt-3">
    <a data-tab="rate" class="button is-primary is-medium">
        <i class="fas fa-clipboard-list mr-2"></i> <span v-text="t('startQuestionnaire')"></span>
    </a>
</div>
```

### 3. 增强评分界面标题和说明

修改评分界面的标题和描述，使其更清晰地表明这是一个评分问卷：

```html
<!-- 评分界面头部 (frontend/templates/index.html) -->
<div v-else-if="currentTab === 'rate'" class="section" transition="fade">
    <div class="container">
        <h2 class="title is-4">
            <i class="fas fa-clipboard-list mr-2"></i> {{ t('rate') }}
        </h2>
        <p class="subtitle is-6">{{ t('rateSubtitle') }}</p>
        
        <div class="notification is-primary is-light mb-4">
            <p><i class="fas fa-info-circle mr-2"></i> {{ t('questionnaireContent') }}</p>
        </div>
```

### 4. 添加问卷界面组件

在评分界面中添加问卷UI组件，包括问题卡片和导航按钮：

```html
<!-- 问卷UI (frontend/templates/index.html) -->
<div v-if="showQuestionnaireUI">
    <!-- 当前问题 -->
    <div class="box music-questionnaire mb-4">
        <div class="questionnaire-header">
            <div class="questionnaire-icon">
                <i class="fas fa-music"></i>
            </div>
            <div class="questionnaire-title">
                <h3>{{ getCurrentQuestionStep().title }}</h3>
                <div class="questionnaire-subtitle">
                    {{ getCurrentQuestionStep().subtitle }}
                </div>
            </div>
        </div>
        
        <div class="answer-options">
            <div class="answer-option" 
                 v-for="option in getCurrentQuestionStep().options" 
                 :key="option.value"
                 :class="{'selected': isOptionSelected(getCurrentQuestionStep().dataCategory, option.value)}"
                 @click="toggleSelection(getCurrentQuestionStep().dataCategory, option.value)">
                <div class="answer-option-icon">
                    <i class="fas fa-music"></i>
                </div>
                <div class="answer-option-text">
                    {{ option.label }}
                </div>
            </div>
        </div>
    </div>
    
    <!-- 问卷导航按钮 -->
    <div class="questionnaire-actions">
        <button class="action-button" 
                :disabled="currentQuestionStep === 1"
                @click="prevQuestionStep">
            <i class="fas fa-arrow-left"></i> {{ currentLanguage === 'zh' ? '上一步' : 'Previous' }}
        </button>
        <button class="action-button primary" @click="nextQuestionStep">
            {{ currentQuestionStep < totalQuestionSteps ? (currentLanguage === 'zh' ? '下一步' : 'Next') : (currentLanguage === 'zh' ? '完成' : 'Finish') }}
            <i class="fas fa-arrow-right"></i>
        </button>
    </div>
</div>
```

### 5. 添加Vue.js数据和方法

在main.js中添加以下数据属性和方法来支持问卷功能：

```javascript
// 数据属性 (frontend/static/js/main.js)
data: {
    // ... 现有数据 ...
    
    // 问卷相关
    currentQuestionStep: 1,
    totalQuestionSteps: 8,
    showQuestionnaireUI: false,
    questionnaireProgress: 0,
    
    // 问卷相关数据模型
    questionnaireAnswers: {
        genres: [],
        moods: [],
        languages: [],
        scenarios: [],
        discovery: [],
        eras: [],
        artist_types: [],
        frequency: []
    },
    
    // 问卷步骤定义
    questionSteps: [
        {
            id: 1,
            title: '音乐风格偏好',
            subtitle: '请选择您喜欢的音乐风格 (可多选)',
            dataCategory: 'genres',
            options: [
                { value: 'pop', label: '流行音乐 (Pop)' },
                { value: 'rock', label: '摇滚音乐 (Rock)' },
                { value: 'classical', label: '古典音乐 (Classical)' },
                { value: 'jazz', label: '爵士乐 (Jazz)' },
                { value: 'electronic', label: '电子音乐 (Electronic)' },
                { value: 'hiphop', label: '嘻哈音乐 (Hip-hop)' },
                { value: 'folk', label: '民谣 (Folk)' },
                { value: 'rnb', label: 'R&B / 灵魂乐' }
            ]
        },
        {
            id: 2,
            title: '音乐心情',
            subtitle: '您通常在什么心情下听音乐？(可多选)',
            dataCategory: 'moods',
            options: [
                { value: 'happy', label: '愉快/兴奋' },
                { value: 'relax', label: '放松/冥想' },
                { value: 'sad', label: '忧伤/沉思' },
                { value: 'energetic', label: '精力充沛/运动' },
                { value: 'focus', label: '专注/工作' },
                { value: 'nostalgic', label: '怀旧/回忆' },
                { value: 'romantic', label: '浪漫/感性' }
            ]
        },
        // 定义其他问题步骤...
    ]
},

// 计算属性 (frontend/static/js/main.js)
computed: {
    // ... 现有计算属性 ...
    
    // 问卷进度
    questionnaireProgress() {
        return (this.currentQuestionStep / this.totalQuestionSteps) * 100;
    }
},

// 方法 (frontend/static/js/main.js)
methods: {
    // ... 现有方法 ...
    
    // 获取当前问题步骤
    getCurrentQuestionStep() {
        return this.questionSteps.find(step => step.id === this.currentQuestionStep) || this.questionSteps[0];
    },
    
    // 切换问卷选项选择状态
    toggleSelection(category, value) {
        if (!this.questionnaireAnswers[category]) {
            this.questionnaireAnswers[category] = [];
        }
        
        // 对于频率问题（单选题）
        if (category === 'frequency') {
            this.questionnaireAnswers[category] = [value];
            return;
        }
        
        // 对于多选题
        const index = this.questionnaireAnswers[category].indexOf(value);
        if (index === -1) {
            this.questionnaireAnswers[category].push(value);
        } else {
            this.questionnaireAnswers[category].splice(index, 1);
        }
    },
    
    // 检查选项是否被选中
    isOptionSelected(category, value) {
        return this.questionnaireAnswers[category] && 
               this.questionnaireAnswers[category].indexOf(value) !== -1;
    },
    
    // 下一个问题
    nextQuestionStep() {
        if (this.currentQuestionStep < this.totalQuestionSteps) {
            this.currentQuestionStep++;
        } else {
            this.submitQuestionnaire();
        }
    },
    
    // 上一个问题
    prevQuestionStep() {
        if (this.currentQuestionStep > 1) {
            this.currentQuestionStep--;
        }
    },
    
    // 提交问卷
    submitQuestionnaire() {
        // 合并问卷答案与用户偏好
        this.selectedMusicStyles = [...new Set([...this.selectedMusicStyles, ...this.questionnaireAnswers.genres])];
        this.selectedMusicScenes = [...new Set([...this.selectedMusicScenes, ...this.questionnaireAnswers.scenarios])];
        this.selectedMusicLanguages = [...new Set([...this.selectedMusicLanguages, ...this.questionnaireAnswers.languages])];
        this.selectedMusicEras = [...new Set([...this.selectedMusicEras, ...this.questionnaireAnswers.eras])];
        
        // 保存偏好到服务器
        this.saveUserPreferences();
        
        // 显示成功消息
        this.addNotification(
            this.currentLanguage === 'zh' ? 
            '问卷提交成功！感谢您的参与。' : 
            'Questionnaire submitted successfully! Thank you for your participation.',
            'is-success'
        );
        
        // 重置问卷状态
        this.showQuestionnaireUI = false;
        this.currentQuestionStep = 1;
    }
},

// 监听器 (frontend/static/js/main.js)
watch: {
    // ... 现有监听器 ...
    
    currentTab(newTab) {
        // 切换到评分标签页时，检查是否显示问卷UI
        if (newTab === 'rate') {
            // 检查URL参数中是否有questionnaire=true
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.get('questionnaire') === 'true') {
                this.showQuestionnaireUI = true;
                // 清除URL参数
                history.replaceState(null, '', window.location.pathname);
            }
        }
    }
}
```

### 6. 添加CSS样式

在main.css中添加问卷相关样式：

```css
/* 问卷部分样式 (frontend/static/css/main.css) */
.music-questionnaire {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    overflow: hidden;
    transition: all 0.3s ease;
}

.questionnaire-header {
    padding: 1.5rem;
    background: linear-gradient(to right, #8A2BE2, #9B4BFF);
    color: white;
    display: flex;
    align-items: center;
}

.questionnaire-icon {
    margin-right: 1rem;
    font-size: 2rem;
}

.questionnaire-progress {
    height: 8px;
    background-color: #f0f0f0;
    border-radius: 4px;
    overflow: hidden;
    margin: 1rem 0;
}

.questionnaire-progress-bar {
    height: 100%;
    background: linear-gradient(to right, #8A2BE2, #9B4BFF);
    transition: width 0.3s ease;
}

.answer-options {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 1rem;
    padding: 1.5rem;
}

.answer-option {
    border: 2px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    transition: all 0.2s ease;
}

.answer-option:hover {
    border-color: #9B4BFF;
    background-color: #f9f5ff;
}

.answer-option.selected {
    border-color: #8A2BE2;
    background-color: #f0e6ff;
}

.answer-option-icon {
    width: 40px;
    height: 40px;
    background-color: #9B4BFF;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
}

.answer-option-text {
    font-weight: 500;
}

.questionnaire-actions {
    display: flex;
    justify-content: space-between;
    margin-top: 1.5rem;
}

.action-button {
    padding: 0.5rem 1.5rem;
    border-radius: 4px;
    border: 1px solid #e0e0e0;
    background-color: white;
    cursor: pointer;
    transition: all 0.2s ease;
}

.action-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

.action-button.primary {
    background-color: #8A2BE2;
    color: white;
    border-color: #8A2BE2;
}

.action-button.primary:hover {
    background-color: #7B1FA2;
}
```

### 7. 扩展翻译字典

在`main.js`文件的`t()`函数的翻译字典中添加以下翻译键：

```javascript
'zh': {
    // ... 现有翻译 ...
    'questionnaire': '音乐评分问卷',
    'questionnaireDesc': '通过评分和问卷，获取个性化音乐推荐',
    'questionnaireContent': '完成此简短问卷和歌曲评分，帮助我们了解您的音乐偏好，提供更精准的推荐。',
    'startQuestionnaire': '开始问卷',
    'rate': '音乐评分问卷',
    'rateSubtitle': '评分喜欢的歌曲并回答问题，获取个性化推荐',
    'musicStyle': '音乐风格偏好',
    'mood': '心情偏好',
    'language': '语言偏好',
    'scenario': '场景偏好',
    'discovery': '发现方式',
    'era': '年代偏好',
    'artistType': '艺术家类型',
    'frequency': '收听频率',
    'prev': '上一步',
    'next': '下一步',
    'finish': '完成',
    'questionnaireComplete': '问卷完成',
    'thankYou': '感谢您的参与',
    'multipleChoice': '(可多选)',
    'singleChoice': '(单选)'
},
'en': {
    // ... 现有翻译 ...
    'questionnaire': 'Music Rating Questionnaire',
    'questionnaireDesc': 'Get personalized music recommendations through ratings and questionnaire',
    'questionnaireContent': 'Complete this short questionnaire and rate songs to help us understand your music preferences for more accurate recommendations.',
    'startQuestionnaire': 'Start Questionnaire',
    'rate': 'Music Rating Questionnaire',
    'rateSubtitle': 'Rate songs you like and answer questions for personalized recommendations',
    'musicStyle': 'Music Style Preferences',
    'mood': 'Mood Preferences',
    'language': 'Language Preferences',
    'scenario': 'Scenario Preferences',
    'discovery': 'Discovery Method',
    'era': 'Era Preferences',
    'artistType': 'Artist Type',
    'frequency': 'Listening Frequency',
    'prev': 'Previous',
    'next': 'Next',
    'finish': 'Finish',
    'questionnaireComplete': 'Questionnaire Complete',
    'thankYou': 'Thank you for your participation',
    'multipleChoice': '(multiple choice)',
    'singleChoice': '(single choice)'
}
```

### 8. 后端路由调整

后端已经有合适的路由，重定向问卷页面请求到主页面的评分标签页：

```python
@app.route('/questionnaire')
def questionnaire():
    """提供问卷调查页面"""
    logger.info("用户访问问卷调查页面，重定向到评分界面")
    return redirect('/?tab=rate&questionnaire=true')
```

## 相关API端点

保留现有的用户偏好API端点，用于保存和获取问卷结果：

1. 保存用户偏好: `/api/user/preferences` (POST)
2. 获取用户偏好: `/api/user/preferences/<user_id>` (GET)

## 集成后效果

1. 用户点击"音乐评分问卷"链接时，会进入评分界面，自动显示问卷UI
2. 问卷UI与评分界面采用一致的黑紫色主题
3. 用户可以在问卷中选择音乐风格、心情、场景等偏好
4. 提交问卷后，这些偏好与用户的歌曲评分结合，用于生成更精准的推荐
5. 整个流程支持中英文切换
6. 访问原问卷URL将自动重定向到整合后的界面

## 测试计划

1. 测试导航链接是否正确跳转到评分标签页
2. 测试URL参数`questionnaire=true`是否触发问卷UI显示
3. 测试问卷多步骤交互是否正常工作
4. 测试单选和多选题型是否按预期工作
5. 测试问卷提交功能是否正确保存用户偏好
6. 测试中英文切换功能是否正常
7. 测试原`/questionnaire`路由重定向是否正常工作

## 注意事项

1. 集成过程中保留了原有的评分功能，增强了问卷部分
2. 保持了黑紫色主题设计的一致性
3. 确保了所有新增内容都支持中英文双语显示
4. 优化了UI交互流程，使用户体验更加流畅