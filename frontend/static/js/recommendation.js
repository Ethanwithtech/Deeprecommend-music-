// 音乐推荐系统 - 前端JavaScript
// 负责处理API调用和用户交互逻辑

// API基础URL - 使用相对路径
const API_BASE_URL = '';

// 当DOM加载完成后执行
document.addEventListener('DOMContentLoaded', () => {
    // 初始化UI组件
    initComponents();
    
    // 加载系统状态
    loadSystemStatus();
    
    // 加载用户列表
    loadUsers();
    
    // 添加事件监听器
    setupEventListeners();
});

// 初始化UI组件
function initComponents() {
    // 检查并定义所有需要用到的DOM元素
    window.elements = {
        // 用户选择相关
        userSelect: document.getElementById('user-select'),
        topNInput: document.getElementById('top-n'),
        getRecommendationsBtn: document.getElementById('get-recommendations-btn'),
        createUserBtn: document.getElementById('create-user-btn'),
        newUserCard: document.getElementById('new-user-card'),
        
        // 推荐结果显示相关
        recommendationsContainer: document.getElementById('recommendations-container'),
        noRecommendation: document.getElementById('no-recommendation'),
        recommendationLoading: document.getElementById('recommendation-loading'),
        recommendationError: document.getElementById('recommendation-error'),
        currentUserBadge: document.getElementById('current-user-badge'),
        
        // 状态相关
        statusContainer: document.getElementById('status-container'),
        topNContainer: document.getElementById('top-n-container'),
        deepModelStatus: document.getElementById('deep-model-status'),
        deepModelStatusText: document.getElementById('deep-model-status-text'),
        
        // 模型设置相关
        useDeepModel: document.getElementById('use-deep-model'),
        cfWeight: document.getElementById('cf-weight'),
        contentWeight: document.getElementById('content-weight'),
        contextWeight: document.getElementById('context-weight'),
        deepWeight: document.getElementById('deep-weight'),
        cfWeightValue: document.getElementById('cf-weight-value'),
        contentWeightValue: document.getElementById('content-weight-value'),
        contextWeightValue: document.getElementById('context-weight-value'),
        deepWeightValue: document.getElementById('deep-weight-value'),
        deepWeightContainer: document.getElementById('deep-weight-container'),
        updateWeightsBtn: document.getElementById('update-weights-btn'),
        trainDeepModelBtn: document.getElementById('train-deep-model-btn')
    };
    
    // 初始设置
    elements.newUserCard.style.display = 'none';
    elements.recommendationsContainer.style.display = 'none';
    elements.recommendationError.style.display = 'none';
    elements.recommendationLoading.style.display = 'none';
    elements.currentUserBadge.style.display = 'none';
    elements.deepModelStatus.style.display = 'none';
    elements.deepWeightContainer.style.display = 'none';
    elements.trainDeepModelBtn.style.display = 'none';
}

// 加载系统状态
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        const status = await response.json();
        
        if (status.model_loaded && status.songs_loaded && status.users_loaded) {
            elements.statusContainer.className = 'alert alert-success';
            elements.statusContainer.textContent = `系统状态: 正常 | 已加载模型，${status.total_songs} 首歌曲，${status.total_users} 个用户`;
            
            // 更新深度学习模型状态
            updateDeepModelStatus(status.has_deep_model);
        } else {
            elements.statusContainer.className = 'alert alert-warning';
            elements.statusContainer.textContent = `系统状态: 部分功能不可用 | 模型: ${status.model_loaded ? '已加载' : '未加载'}, 歌曲: ${status.songs_loaded ? '已加载' : '未加载'}, 用户: ${status.users_loaded ? '已加载' : '未加载'}`;
        }
    } catch (error) {
        elements.statusContainer.className = 'alert alert-danger';
        elements.statusContainer.textContent = `系统状态: 错误 | 无法连接到服务器: ${error.message}`;
    }
}

// 更新深度学习模型状态
function updateDeepModelStatus(hasDeepModel) {
    elements.deepModelStatus.style.display = 'inline-block';
    
    if (hasDeepModel) {
        elements.deepModelStatus.className = 'badge bg-success mb-2';
        elements.deepModelStatusText.textContent = '已启用';
        elements.useDeepModel.checked = true;
        elements.deepWeightContainer.style.display = 'block';
        elements.trainDeepModelBtn.style.display = 'none';
    } else {
        elements.deepModelStatus.className = 'badge bg-secondary mb-2';
        elements.deepModelStatusText.textContent = '未启用';
        elements.useDeepModel.checked = false;
        elements.deepWeightContainer.style.display = 'none';
        elements.trainDeepModelBtn.style.display = 'block';
    }
}

// 加载用户列表
async function loadUsers() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/users`);
        const data = await response.json();
        
        // 清空当前选项（保留前两个选项）
        while (elements.userSelect.options.length > 2) {
            elements.userSelect.remove(2);
        }
        
        // 添加用户
        data.users.forEach(userId => {
            const option = document.createElement('option');
            option.value = userId;
            option.textContent = `用户 ${userId}`;
            elements.userSelect.appendChild(option);
        });
    } catch (error) {
        console.error('加载用户列表失败:', error);
        showError('无法加载用户列表，请检查网络连接');
    }
}

// 设置事件监听器
function setupEventListeners() {
    // 用户选择改变时的处理
    elements.userSelect.addEventListener('change', handleUserSelectChange);
    
    // 获取推荐按钮点击事件
    elements.getRecommendationsBtn.addEventListener('click', handleGetRecommendations);
    
    // 创建用户按钮点击事件
    elements.createUserBtn.addEventListener('click', handleCreateUser);
    
    // 更新权重按钮点击事件
    elements.updateWeightsBtn.addEventListener('click', handleUpdateWeights);
    
    // 训练深度学习模型按钮点击事件
    elements.trainDeepModelBtn.addEventListener('click', handleTrainDeepModel);
    
    // 使用深度学习模型切换事件
    elements.useDeepModel.addEventListener('change', handleUseDeepModelChange);
    
    // 权重滑块变化事件
    elements.cfWeight.addEventListener('input', () => {
        elements.cfWeightValue.textContent = elements.cfWeight.value;
    });
    elements.contentWeight.addEventListener('input', () => {
        elements.contentWeightValue.textContent = elements.contentWeight.value;
    });
    elements.contextWeight.addEventListener('input', () => {
        elements.contextWeightValue.textContent = elements.contextWeight.value;
    });
    elements.deepWeight.addEventListener('input', () => {
        elements.deepWeightValue.textContent = elements.deepWeight.value;
    });
}

// 处理用户选择变化
function handleUserSelectChange() {
    if (elements.userSelect.value === 'new') {
        elements.newUserCard.style.display = 'block';
        elements.topNContainer.style.display = 'none';
        elements.getRecommendationsBtn.style.display = 'none';
    } else {
        elements.newUserCard.style.display = 'none';
        elements.topNContainer.style.display = 'block';
        elements.getRecommendationsBtn.style.display = 'block';
    }
}

// 处理获取推荐请求
async function handleGetRecommendations() {
    const userId = elements.userSelect.value;
    const topN = elements.topNInput.value;
    
    if (!userId) {
        alert('请选择一个用户');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/recommend?user_id=${userId}&top_n=${topN}`);
        const data = await response.json();
        
        if (response.ok) {
            showRecommendations(data);
            elements.currentUserBadge.textContent = `用户: ${userId}`;
            elements.currentUserBadge.style.display = 'inline-block';
        } else {
            showError(data.error || '获取推荐失败');
        }
    } catch (error) {
        showError(error.message);
    }
}

// 处理创建用户请求
async function handleCreateUser() {
    // 收集用户偏好
    const genres = [];
    document.querySelectorAll('input[type=checkbox]:checked').forEach(checkbox => {
        genres.push(checkbox.value);
    });
    
    const data = {
        genres: genres,
        tempo: parseInt(document.getElementById('tempo-range').value),
        energy: parseInt(document.getElementById('energy-range').value),
        top_n: parseInt(elements.topNInput.value || 10)
    };
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/create_user`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });
        
        const responseData = await response.json();
        
        if (response.ok) {
            showRecommendations(responseData);
            elements.currentUserBadge.textContent = `新用户: ${responseData.user_id}`;
            elements.currentUserBadge.style.display = 'inline-block';
            
            // 添加此用户到下拉列表
            const option = document.createElement('option');
            option.value = responseData.user_id;
            option.textContent = `用户 ${responseData.user_id}`;
            elements.userSelect.appendChild(option);
            
            // 选中该用户
            elements.userSelect.value = responseData.user_id;
            
            // 隐藏新用户卡片
            elements.newUserCard.style.display = 'none';
            elements.topNContainer.style.display = 'block';
            elements.getRecommendationsBtn.style.display = 'block';
        } else {
            showError(responseData.error || '创建用户失败');
        }
    } catch (error) {
        showError(error.message);
    }
}

// 处理更新权重请求
async function handleUpdateWeights() {
    const userId = elements.userSelect.value;
    if (!userId || userId === 'new') {
        alert('请先选择一个用户');
        return;
    }
    
    const weights = {
        user_id: userId,
        cf_weight: parseFloat(elements.cfWeight.value),
        content_weight: parseFloat(elements.contentWeight.value),
        context_weight: parseFloat(elements.contextWeight.value),
        top_n: parseInt(elements.topNInput.value || 10)
    };
    
    // 如果启用深度学习模型，添加深度权重
    if (elements.useDeepModel.checked) {
        weights.deep_weight = parseFloat(elements.deepWeight.value);
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/update_weights`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(weights)
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showRecommendations(data);
            alert('权重更新成功');
        } else {
            showError(data.error || '更新权重失败');
        }
    } catch (error) {
        showError(error.message);
    }
}

// 处理训练深度学习模型请求
async function handleTrainDeepModel() {
    if (!confirm('训练深度学习模型可能需要较长时间，是否继续？')) {
        return;
    }
    
    showLoading();
    elements.statusContainer.className = 'alert alert-warning';
    elements.statusContainer.textContent = '正在训练深度学习模型，请耐心等待...';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/train_deep_model`, {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (response.ok) {
            elements.statusContainer.className = 'alert alert-success';
            elements.statusContainer.textContent = '深度学习模型训练成功！';
            updateDeepModelStatus(true);
            
            // 如果有当前选择的用户，刷新推荐
            const userId = elements.userSelect.value;
            if (userId && userId !== 'new') {
                handleGetRecommendations();
            } else {
                hideLoading();
            }
        } else {
            elements.statusContainer.className = 'alert alert-danger';
            elements.statusContainer.textContent = `训练失败: ${data.error}`;
            hideLoading();
        }
    } catch (error) {
        elements.statusContainer.className = 'alert alert-danger';
        elements.statusContainer.textContent = `训练失败: ${error.message}`;
        hideLoading();
    }
}

// 处理使用深度学习模型开关变化
function handleUseDeepModelChange() {
    const isChecked = elements.useDeepModel.checked;
    elements.deepWeightContainer.style.display = isChecked ? 'block' : 'none';
    
    // 重新计算其他权重
    if (isChecked) {
        // 添加了深度学习，需要调整其他权重
        const deepWeight = parseFloat(elements.deepWeight.value) || 0.2;
        const scaleRatio = (1 - deepWeight) / 1.0;
        
        elements.cfWeight.value = (0.5 * scaleRatio).toFixed(1);
        elements.contentWeight.value = (0.3 * scaleRatio).toFixed(1);
        elements.contextWeight.value = (0.2 * scaleRatio).toFixed(1);
        
        elements.cfWeightValue.textContent = elements.cfWeight.value;
        elements.contentWeightValue.textContent = elements.contentWeight.value;
        elements.contextWeightValue.textContent = elements.contextWeight.value;
    } else {
        // 恢复默认权重
        elements.cfWeight.value = 0.5;
        elements.contentWeight.value = 0.3;
        elements.contextWeight.value = 0.2;
        
        elements.cfWeightValue.textContent = "0.5";
        elements.contentWeightValue.textContent = "0.3";
        elements.contextWeightValue.textContent = "0.2";
    }
}

// 显示推荐结果
function showRecommendations(data) {
    if (!data.recommendations || data.recommendations.length === 0) {
        elements.noRecommendation.style.display = 'block';
        elements.recommendationsContainer.style.display = 'none';
        hideLoading();
        return;
    }
    
    elements.recommendationsContainer.innerHTML = '';
    
    data.recommendations.forEach(rec => {
        // 格式化评分为百分比
        const scorePercent = Math.round(rec.score * 100);
        
        // 创建推荐卡片
        const card = document.createElement('div');
        card.className = 'col-lg-6 col-xl-4';
        card.innerHTML = `
            <div class="card recommendation-card">
                <div class="card-body">
                    <div class="score-badge badge bg-primary">${scorePercent}%</div>
                    <div class="song-title">${rec.title || '未知歌曲'}</div>
                    <div class="song-artist">${rec.artist || '未知艺术家'}</div>
                    <div class="mt-2">
                        <small class="text-muted">时长: ${formatDuration(rec.duration || 0)}</small>
                    </div>
                    <div class="mt-3">
                        <span class="badge badge-cf me-1" title="协同过滤评分">CF: ${formatScore(rec.cf_score)}</span>
                        <span class="badge badge-content me-1" title="内容评分">内容: ${formatScore(rec.content_score)}</span>
                        <span class="badge badge-context" title="上下文评分">上下文: ${formatScore(rec.context_score)}</span>
                        ${rec.deep_score > 0 ? `<span class="badge bg-purple" title="深度学习评分">深度: ${formatScore(rec.deep_score)}</span>` : ''}
                    </div>
                </div>
            </div>
        `;
        elements.recommendationsContainer.appendChild(card);
    });
    
    hideLoading();
    elements.noRecommendation.style.display = 'none';
    elements.recommendationsContainer.style.display = 'flex';
}

// 显示加载中
function showLoading() {
    elements.noRecommendation.style.display = 'none';
    elements.recommendationsContainer.style.display = 'none';
    elements.recommendationError.style.display = 'none';
    elements.recommendationLoading.style.display = 'flex';
}

// 隐藏加载中
function hideLoading() {
    elements.recommendationLoading.style.display = 'none';
}

// 显示错误
function showError(message) {
    hideLoading();
    elements.recommendationsContainer.style.display = 'none';
    elements.noRecommendation.style.display = 'none';
    elements.recommendationError.style.display = 'block';
    elements.recommendationError.textContent = `错误: ${message}`;
}

// 格式化分数
function formatScore(score) {
    if (score === undefined || score === null) return 'N/A';
    return (score * 100).toFixed(0) + '%';
}

// 格式化时长为分:秒
function formatDuration(seconds) {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
} 