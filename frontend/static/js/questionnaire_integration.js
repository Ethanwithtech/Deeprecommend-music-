/**
 * 问卷与评分集成JavaScript文件
 * 用于整合问卷调查和歌曲评分功能，确保数据一致性和避免用户体验重复
 */

// 当DOM加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('问卷评分集成: DOM已加载');
    // 检查是否在Vue应用环境中
    if (window.app) {
        console.log('问卷评分集成: Vue应用已找到，立即初始化');
        // 初始化集成器
        initQuestionnaireRatingIntegration();
    } else {
        console.log('问卷评分集成: Vue应用未找到，将等待加载');
        // 延迟初始化，等待Vue应用加载完成
        let attempts = 0;
        const maxAttempts = 50; // 5秒超时
        const checkInterval = setInterval(() => {
            attempts++;
            if (window.app) {
                console.log('问卷评分集成: Vue应用已加载，开始初始化');
                initQuestionnaireRatingIntegration();
                clearInterval(checkInterval);
            } else if (attempts >= maxAttempts) {
                console.error('问卷评分集成: Vue应用加载超时');
                clearInterval(checkInterval);
            }
        }, 100);
    }
});

/**
 * 初始化问卷评分集成
 */
function initQuestionnaireRatingIntegration() {
    // 获取Vue应用实例
    const app = window.app;
    console.log('初始化问卷评分集成...');
    
    // 确保必要的数据结构存在
    if (!app.questionnaireAnswers) {
        app.questionnaireAnswers = {};
        console.log('创建问卷答案存储对象');
    }
    
    // 为Vue应用添加集成方法 - 用于问卷页面直接调用
    app.integrateQuestionnaireData = function(data) {
        console.log('接收到问卷数据，进行集成:', data);
        if (!data) return;
        
        Object.keys(data).forEach(key => {
            if (data[key]) {
                app.questionnaireAnswers[key] = data[key];
                updateVueDataFromQuestionnaire(app, key, data[key]);
            }
        });
        
        // 保存用户偏好到本地存储
        localStorage.setItem('musicPreferences', JSON.stringify(app.questionnaireAnswers));
        
        // 请求更新推荐
        if (app.getRecommendations && typeof app.getRecommendations === 'function') {
            console.log('基于集成数据请求推荐');
            setTimeout(() => app.getRecommendations(), 500);
        }
    };
    
    // 获取本地存储的问卷数据
    const storedPreferences = localStorage.getItem('musicPreferences');
    let userPreferences = null;
    
    // 处理本地存储中的问卷答案
    if (storedPreferences) {
        try {
            userPreferences = JSON.parse(storedPreferences);
            console.log('找到已存储的音乐偏好:', userPreferences);
            
            // 将问卷数据同步到Vue应用
            Object.keys(userPreferences).forEach(key => {
                if (userPreferences[key]) {
                    app.questionnaireAnswers[key] = userPreferences[key];
                    
                    // 同时更新相应的Vue数据属性
                    updateVueDataFromQuestionnaire(app, key, userPreferences[key]);
                }
            });
            
            // 通知用户
            if (app.addNotification && typeof app.addNotification === 'function') {
                app.addNotification('已加载您的音乐偏好设置', 'is-success');
            }
        } catch (e) {
            console.error('解析存储的偏好数据出错:', e);
        }
    } else {
        console.log('未找到存储的音乐偏好');
    }
    
    // 监听问卷完成事件 - 从一页式问卷页面发送过来的事件
    window.addEventListener('questionnaire_submitted', function(e) {
        console.log('收到问卷提交事件');
        if (e.detail && e.detail.answers) {
            const answers = e.detail.answers;
            console.log('问卷答案:', answers);
            
            // 将问卷答案同步到Vue应用
            Object.keys(answers).forEach(key => {
                app.questionnaireAnswers[key] = answers[key];
                updateVueDataFromQuestionnaire(app, key, answers[key]);
            });
            
            // 基于问卷答案请求推荐
            setTimeout(() => {
                if (app.getRecommendations && typeof app.getRecommendations === 'function') {
                    console.log('基于问卷答案请求推荐');
                    app.getRecommendations();
                    if (app.addNotification && typeof app.addNotification === 'function') {
                        app.addNotification('根据您的问卷答案生成推荐', 'is-info');
                    }
                }
            }, 1000);
        }
    });
    
    // 为评分页面添加查看/编辑问卷按钮
    addPreferenceButtonToRating(app);
    
    // 创建双向数据绑定，确保评分系统的变更反映到问卷系统
    setupDataBinding(app);
    
    // 覆盖Vue方法，使问卷和评分系统共享数据
    extendVueMethods(app);
    
    console.log('问卷评分集成初始化完成');
}

/**
 * 为评分页面添加查看/编辑问卷按钮
 */
function addPreferenceButtonToRating(app) {
    // 检查是否在评分页面
    if (document.querySelector('.rate-container')) {
        console.log('检测到评分页面，添加问卷偏好按钮');
        
        // 创建按钮
        const prefButton = document.createElement('button');
        prefButton.className = 'button is-primary is-outlined mb-4';
        prefButton.innerHTML = '<i class="fas fa-sliders-h mr-2"></i> 查看/编辑音乐偏好';
        
        // 添加点击事件
        prefButton.addEventListener('click', function() {
            // 打开问卷页面
            window.location.href = '/questionnaire';
        });
        
        // 查找合适位置插入按钮
        const rateContainer = document.querySelector('.rate-container');
        if (rateContainer) {
            // 尝试找到评分标题或说明元素
            const titleElement = rateContainer.querySelector('h3, h4, .title');
            if (titleElement) {
                titleElement.parentNode.insertBefore(prefButton, titleElement.nextSibling);
            } else {
                // 如果找不到标题，就在容器开头插入
                rateContainer.insertBefore(prefButton, rateContainer.firstChild);
            }
        }
    }
}

/**
 * 设置数据双向绑定
 */
function setupDataBinding(app) {
    console.log('设置数据双向绑定');
    
    // 初始同步一次数据以确保一致性
    if (app.questionnaireAnswers) {
        syncAppToQuestionnaire(app);
    }
}

/**
 * 同步应用数据到问卷数据
 */
function syncAppToQuestionnaire(app) {
    try {
        // 同步Vue应用数据到问卷答案
        app.questionnaireAnswers.genres = app.selectedMusicStyles || [];
        app.questionnaireAnswers.scenarios = app.selectedMusicScenes || [];
        app.questionnaireAnswers.languages = app.selectedMusicLanguages || [];
        app.questionnaireAnswers.eras = app.selectedMusicEras || [];
        app.questionnaireAnswers.favorite_artists = app.favoriteArtists || '';
        app.questionnaireAnswers.listening_time = app.dailyListeningTime ? [app.dailyListeningTime] : [];
        
        // 如果应用有这些属性则同步
        if (app.selectedMusicMoods !== undefined) {
            app.questionnaireAnswers.moods = app.selectedMusicMoods || [];
        }
        if (app.selectedMusicElements !== undefined) {
            app.questionnaireAnswers.elements = app.selectedMusicElements || [];
        }
        
        // 保存到本地存储
        localStorage.setItem('musicPreferences', JSON.stringify(app.questionnaireAnswers));
        console.log('应用数据已同步到问卷数据');
    } catch (e) {
        console.error('同步应用数据到问卷数据时出错:', e);
    }
}

/**
 * 从问卷答案更新Vue应用数据
 */
function updateVueDataFromQuestionnaire(app, category, values) {
    if (!app) {
        console.error('更新Vue数据失败: app对象不存在');
        return;
    }
    
    console.log(`从问卷答案更新Vue数据: ${category} = `, values);
    
    try {
        switch (category) {
            case 'genres':
                app.selectedMusicStyles = values;
                break;
            case 'scenarios':
                app.selectedMusicScenes = values;
                break;
            case 'languages':
                app.selectedMusicLanguages = values;
                break;
            case 'eras':
                app.selectedMusicEras = values;
                break;
            case 'favorite_artists':
                app.favoriteArtists = values;
                break;
            case 'listening_time':
                if (values && values.length > 0) {
                    app.dailyListeningTime = values[0]; // 单选题只取第一个值
                }
                break;
            case 'moods': 
                if (app.selectedMusicMoods !== undefined) {
                    app.selectedMusicMoods = values;
                }
                break;
            case 'elements': 
                if (app.selectedMusicElements !== undefined) {
                    app.selectedMusicElements = values;
                }
                break;
            default:
                console.log(`未知类别 ${category}，无法更新Vue数据`);
        }
    } catch (e) {
        console.error(`更新Vue数据时出错 (${category}):`, e);
    }
}

/**
 * 扩展Vue应用方法
 */
function extendVueMethods(app) {
    console.log('扩展Vue应用方法');
    
    // 保存原始的保存用户偏好方法
    const originalSavePreferences = app.saveUserPreferences;
    app.saveUserPreferences = function() {
        console.log('执行扩展的保存用户偏好方法');
        
        // 执行原始逻辑
        if (originalSavePreferences && typeof originalSavePreferences === 'function') {
            try {
                originalSavePreferences.call(app);
                console.log('原始保存方法已执行');
            } catch (e) {
                console.error('执行原始保存方法时出错:', e);
            }
        }
        
        // 额外逻辑：同步评分系统数据到问卷数据
        syncAppToQuestionnaire(app);
    };
    
    // 增强评分方法，考虑用户偏好
    const originalGetRecommendations = app.getRecommendations;
    app.getRecommendations = function() {
        console.log('执行扩展的获取推荐方法');
        
        // 执行原始逻辑
        if (originalGetRecommendations && typeof originalGetRecommendations === 'function') {
            try {
                originalGetRecommendations.call(app);
                console.log('原始推荐方法已执行');
            } catch (e) {
                console.error('执行原始推荐方法时出错:', e);
            }
        }
        
        // 额外逻辑：使用问卷数据增强推荐
        setTimeout(() => {
            try {
                const preferences = app.questionnaireAnswers;
                if (preferences && app.recommendations && app.recommendations.length > 0) {
                    console.log('使用用户偏好增强推荐:', preferences);
                    
                    // 为推荐添加详细理由
                    app.recommendations.forEach(rec => {
                        if (!rec.recommendationReason) {
                            // 根据用户偏好生成推荐理由
                            const reasons = [];
                            
                            // 基于流派偏好
                            if (preferences.genres && preferences.genres.length > 0) {
                                const preferredGenre = preferences.genres[Math.floor(Math.random() * preferences.genres.length)];
                                reasons.push(`匹配您喜欢的${getGenreChinese(preferredGenre)}风格`);
                            }
                            
                            // 基于语言偏好
                            if (preferences.languages && preferences.languages.length > 0) {
                                const preferredLang = preferences.languages[Math.floor(Math.random() * preferences.languages.length)];
                                reasons.push(`符合您对${getLanguageChinese(preferredLang)}的偏好`);
                            }
                            
                            // 组合理由
                            if (reasons.length > 0) {
                                rec.recommendationReason = reasons.join('，') + '。';
                            }
                        }
                    });
                    
                    console.log('推荐增强完成');
                }
            } catch (e) {
                console.error('使用用户偏好增强推荐时出错:', e);
            }
        }, 500);
    };
}

/**
 * 获取音乐风格的中文名称
 */
function getGenreChinese(genreId) {
    const genreMap = {
        'pop': '流行',
        'rock': '摇滚',
        'classical': '古典',
        'jazz': '爵士',
        'electronic': '电子',
        'hiphop': '嘻哈',
        'folk': '民谣',
        'rnb': 'R&B'
    };
    return genreMap[genreId] || genreId;
}

/**
 * 获取语言的中文名称
 */
function getLanguageChinese(langId) {
    const langMap = {
        'chinese': '中文歌曲',
        'english': '英文歌曲',
        'japanese': '日文歌曲',
        'korean': '韩文歌曲',
        'instrumental': '纯音乐',
        'other': '多语言'
    };
    return langMap[langId] || langId;
}

/**
 * 动态加载问卷CSS
 */
function loadQuestionnaireCss() {
    // 检查是否已加载
    if (document.getElementById('questionnaire-css')) return;
    
    // 创建链接元素
    const link = document.createElement('link');
    link.id = 'questionnaire-css';
    link.rel = 'stylesheet';
    link.href = '/static/css/questionnaire.css';
    
    // 添加到页面
    document.head.appendChild(link);
}

// 初始加载CSS
loadQuestionnaireCss();
