/**
 * 音乐偏好收集器
 * 用于通过互动游戏收集用户音乐偏好
 */

// 当文档加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
  // 确保Vue应用加载完成
  setTimeout(() => {
    initPreferenceCollector();
  }, 1500);
});

// 音乐流派列表
const MUSIC_GENRES = [
  {id: 'pop', name: '流行', color: '#FF5252'},
  {id: 'rock', name: '摇滚', color: '#AA00FF'},
  {id: 'hiphop', name: '嘻哈', color: '#2196F3'},
  {id: 'electronic', name: '电子', color: '#00BCD4'},
  {id: 'jazz', name: '爵士', color: '#FFC107'},
  {id: 'classical', name: '古典', color: '#4CAF50'},
  {id: 'rbsoul', name: 'R&B', color: '#9C27B0'},
  {id: 'country', name: '乡村', color: '#795548'},
  {id: 'folk', name: '民谣', color: '#8BC34A'},
  {id: 'metal', name: '金属', color: '#607D8B'},
  {id: 'blues', name: '蓝调', color: '#3F51B5'},
  {id: 'world', name: '世界音乐', color: '#009688'}
];

// 音乐情绪列表
const MUSIC_MOODS = [
  {id: 'happy', name: '欢快', color: '#FFC107'},
  {id: 'sad', name: '忧伤', color: '#3F51B5'},
  {id: 'energetic', name: '充满活力', color: '#F44336'},
  {id: 'calm', name: '平静', color: '#4CAF50'},
  {id: 'romantic', name: '浪漫', color: '#E91E63'},
  {id: 'dreamy', name: '梦幻', color: '#9C27B0'}
];

// 音乐年代列表
const MUSIC_ERAS = [
  {id: '60s', name: '60年代', color: '#795548'},
  {id: '70s', name: '70年代', color: '#FF9800'},
  {id: '80s', name: '80年代', color: '#9C27B0'},
  {id: '90s', name: '90年代', color: '#2196F3'},
  {id: '2000s', name: '2000年代', color: '#4CAF50'},
  {id: '2010s', name: '2010年代', color: '#F44336'},
  {id: '2020s', name: '2020年代', color: '#607D8B'}
];

// 初始化偏好收集器
function initPreferenceCollector() {
  const gameContainer = document.getElementById('music-game-container');
  if (!gameContainer) return;
  
  // 清空游戏容器
  gameContainer.innerHTML = '';
  
  // 添加游戏标题
  const gameHeader = document.createElement('div');
  gameHeader.className = 'game-header has-text-centered';
  gameHeader.innerHTML = `
    <h2 class="title is-4">音乐偏好收集游戏</h2>
    <p class="subtitle is-6">捕捉掉落的音乐元素，帮助我们了解您的偏好</p>
  `;
  gameContainer.appendChild(gameHeader);
  
  // 添加游戏画布
  const canvas = document.createElement('canvas');
  canvas.id = 'musicCanvas';
  canvas.width = gameContainer.clientWidth;
  canvas.height = 300;
  canvas.style.display = 'block';
  canvas.style.backgroundColor = '#111';
  canvas.style.borderRadius = '8px';
  canvas.style.boxShadow = '0 4px 15px rgba(138, 43, 226, 0.3)';
  gameContainer.appendChild(canvas);
  
  // 添加收集的偏好显示区域
  const preferencesContainer = document.createElement('div');
  preferencesContainer.className = 'collected-preferences mt-3';
  preferencesContainer.innerHTML = `
    <h3 class="is-size-5">已收集的偏好</h3>
    <div id="genre-preferences" class="tags are-medium mt-2"></div>
    <div id="mood-preferences" class="tags are-medium mt-2"></div>
    <div id="era-preferences" class="tags are-medium mt-2"></div>
  `;
  gameContainer.appendChild(preferencesContainer);
  
  // 添加游戏说明
  const gameInstructions = document.createElement('div');
  gameInstructions.className = 'game-instructions mt-3';
  gameInstructions.innerHTML = `
    <p class="is-size-6 has-text-grey">
      <i class="fas fa-info-circle mr-1"></i> 
      使用键盘左右方向键或点击屏幕左右区域移动收集器，捕捉您喜欢的音乐元素。
      收集的偏好将帮助我们提供更个性化的推荐。
    </p>
  `;
  gameContainer.appendChild(gameInstructions);
  
  // 初始化游戏
  initGame(canvas);
}

// 游戏状态
let game = {
  canvas: null,
  ctx: null,
  player: {
    x: 0,
    y: 0,
    width: 60,
    height: 20,
    speed: 5
  },
  elements: [],
  collectedPreferences: {
    genres: {},
    moods: {},
    eras: {}
  },
  keys: {
    left: false,
    right: false
  },
  running: false,
  animationFrame: null
};

// 初始化游戏
function initGame(canvas) {
  game.canvas = canvas;
  game.ctx = canvas.getContext('2d');
  
  // 设置玩家初始位置
  game.player.x = canvas.width / 2 - game.player.width / 2;
  game.player.y = canvas.height - game.player.height - 10;
  
  // 设置键盘事件监听
  window.addEventListener('keydown', handleKeyDown);
  window.addEventListener('keyup', handleKeyUp);
  
  // 设置触摸/点击事件
  canvas.addEventListener('mousedown', handleMouseDown);
  canvas.addEventListener('touchstart', handleTouchStart, {passive: true});
  
  // 更新canvas尺寸
  window.addEventListener('resize', resizeCanvas);
  
  // 开始游戏循环
  game.running = true;
  gameLoop();
  
  // 定时生成新元素
  setInterval(generateElement, 1200);
}

// 调整画布大小
function resizeCanvas() {
  if (!game.canvas) return;
  
  const container = game.canvas.parentElement;
  game.canvas.width = container.clientWidth;
  
  // 重新设置玩家位置
  game.player.x = game.canvas.width / 2 - game.player.width / 2;
}

// 生成新的音乐元素
function generateElement() {
  if (!game.running) return;
  
  // 随机选择元素类型
  const types = ['genre', 'mood', 'era'];
  const type = types[Math.floor(Math.random() * types.length)];
  
  let element;
  
  // 根据类型生成不同的元素
  switch (type) {
    case 'genre':
      const genre = MUSIC_GENRES[Math.floor(Math.random() * MUSIC_GENRES.length)];
      element = {
        type: 'genre',
        id: genre.id,
        name: genre.name,
        color: genre.color,
        x: Math.random() * (game.canvas.width - 50),
        y: -30,
        width: 50,
        height: 30,
        speed: 1 + Math.random() * 2
      };
      break;
    case 'mood':
      const mood = MUSIC_MOODS[Math.floor(Math.random() * MUSIC_MOODS.length)];
      element = {
        type: 'mood',
        id: mood.id,
        name: mood.name,
        color: mood.color,
        x: Math.random() * (game.canvas.width - 50),
        y: -30,
        width: 50,
        height: 30,
        speed: 1 + Math.random() * 2
      };
      break;
    case 'era':
      const era = MUSIC_ERAS[Math.floor(Math.random() * MUSIC_ERAS.length)];
      element = {
        type: 'era',
        id: era.id,
        name: era.name,
        color: era.color,
        x: Math.random() * (game.canvas.width - 50),
        y: -30,
        width: 50,
        height: 30,
        speed: 1 + Math.random() * 2
      };
      break;
  }
  
  game.elements.push(element);
}

// 游戏主循环
function gameLoop() {
  if (!game.running) return;
  
  // 清空画布
  game.ctx.clearRect(0, 0, game.canvas.width, game.canvas.height);
  
  // 更新玩家位置
  updatePlayer();
  
  // 更新和绘制元素
  updateElements();
  
  // 绘制玩家
  drawPlayer();
  
  // 继续游戏循环
  game.animationFrame = requestAnimationFrame(gameLoop);
}

// 更新玩家位置
function updatePlayer() {
  // 根据按键状态移动玩家
  if (game.keys.left) {
    game.player.x -= game.player.speed;
  }
  if (game.keys.right) {
    game.player.x += game.player.speed;
  }
  
  // 限制玩家在画布内
  if (game.player.x < 0) {
    game.player.x = 0;
  }
  if (game.player.x > game.canvas.width - game.player.width) {
    game.player.x = game.canvas.width - game.player.width;
  }
}

// 更新元素位置并检查碰撞
function updateElements() {
  for (let i = game.elements.length - 1; i >= 0; i--) {
    const element = game.elements[i];
    
    // 更新位置
    element.y += element.speed;
    
    // 检查是否超出画布底部
    if (element.y > game.canvas.height) {
      game.elements.splice(i, 1);
      continue;
    }
    
    // 检查碰撞
    if (checkCollision(game.player, element)) {
      // 收集偏好
      collectPreference(element);
      game.elements.splice(i, 1);
      continue;
    }
    
    // 绘制元素
    drawElement(element);
  }
}

// 检查两个矩形是否碰撞
function checkCollision(rect1, rect2) {
  return rect1.x < rect2.x + rect2.width &&
         rect1.x + rect1.width > rect2.x &&
         rect1.y < rect2.y + rect2.height &&
         rect1.y + rect1.height > rect2.y;
}

// 收集偏好
function collectPreference(element) {
  // 根据元素类型收集不同的偏好
  switch (element.type) {
    case 'genre':
      game.collectedPreferences.genres[element.id] = (game.collectedPreferences.genres[element.id] || 0) + 1;
      break;
    case 'mood':
      game.collectedPreferences.moods[element.id] = (game.collectedPreferences.moods[element.id] || 0) + 1;
      break;
    case 'era':
      game.collectedPreferences.eras[element.id] = (game.collectedPreferences.eras[element.id] || 0) + 1;
      break;
  }
  
  // 更新显示
  updatePreferencesDisplay();
  
  // 如果使用了Vue，同步偏好到Vue实例
  syncPreferencesToVue();
  
  // 播放收集音效
  playCollectSound();
}

// 更新偏好显示
function updatePreferencesDisplay() {
  // 更新流派偏好
  const genreContainer = document.getElementById('genre-preferences');
  if (genreContainer) {
    genreContainer.innerHTML = '';
    
    Object.entries(game.collectedPreferences.genres).forEach(([id, count]) => {
      const genre = MUSIC_GENRES.find(g => g.id === id);
      if (genre) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.style.backgroundColor = genre.color;
        tag.style.color = '#fff';
        tag.style.margin = '2px';
        tag.style.fontSize = `${Math.min(1 + count * 0.1, 1.5)}rem`;
        tag.textContent = genre.name + (count > 1 ? ` x${count}` : '');
        genreContainer.appendChild(tag);
      }
    });
  }
  
  // 更新情绪偏好
  const moodContainer = document.getElementById('mood-preferences');
  if (moodContainer) {
    moodContainer.innerHTML = '';
    
    Object.entries(game.collectedPreferences.moods).forEach(([id, count]) => {
      const mood = MUSIC_MOODS.find(m => m.id === id);
      if (mood) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.style.backgroundColor = mood.color;
        tag.style.color = '#fff';
        tag.style.margin = '2px';
        tag.style.fontSize = `${Math.min(1 + count * 0.1, 1.5)}rem`;
        tag.textContent = mood.name + (count > 1 ? ` x${count}` : '');
        moodContainer.appendChild(tag);
      }
    });
  }
  
  // 更新年代偏好
  const eraContainer = document.getElementById('era-preferences');
  if (eraContainer) {
    eraContainer.innerHTML = '';
    
    Object.entries(game.collectedPreferences.eras).forEach(([id, count]) => {
      const era = MUSIC_ERAS.find(e => e.id === id);
      if (era) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.style.backgroundColor = era.color;
        tag.style.color = '#fff';
        tag.style.margin = '2px';
        tag.style.fontSize = `${Math.min(1 + count * 0.1, 1.5)}rem`;
        tag.textContent = era.name + (count > 1 ? ` x${count}` : '');
        eraContainer.appendChild(tag);
      }
    });
  }
}

// 同步偏好到Vue实例
function syncPreferencesToVue() {
  if (window.app && typeof window.app.updateMusicPreferences === 'function') {
    window.app.updateMusicPreferences({
      genres: game.collectedPreferences.genres,
      moods: game.collectedPreferences.moods,
      eras: game.collectedPreferences.eras
    });
  }
}

// 绘制玩家
function drawPlayer() {
  game.ctx.fillStyle = '#8A2BE2';
  game.ctx.fillRect(game.player.x, game.player.y, game.player.width, game.player.height);
  
  // 添加渐变效果
  const gradient = game.ctx.createLinearGradient(
    game.player.x, game.player.y, 
    game.player.x, game.player.y + game.player.height
  );
  gradient.addColorStop(0, 'rgba(155, 75, 255, 0.8)');
  gradient.addColorStop(1, 'rgba(106, 27, 154, 0.8)');
  
  game.ctx.fillStyle = gradient;
  game.ctx.fillRect(game.player.x, game.player.y, game.player.width, game.player.height);
  
  // 添加发光效果
  game.ctx.shadowBlur = 10;
  game.ctx.shadowColor = '#8A2BE2';
  game.ctx.fillRect(game.player.x, game.player.y, game.player.width, game.player.height);
  game.ctx.shadowBlur = 0;
}

// 绘制元素
function drawElement(element) {
  // 绘制元素背景
  game.ctx.fillStyle = element.color;
  game.ctx.fillRect(element.x, element.y, element.width, element.height);
  
  // 添加发光效果
  game.ctx.shadowBlur = 5;
  game.ctx.shadowColor = element.color;
  game.ctx.fillRect(element.x, element.y, element.width, element.height);
  game.ctx.shadowBlur = 0;
  
  // 绘制元素文本
  game.ctx.fillStyle = '#fff';
  game.ctx.font = '10px Arial';
  game.ctx.textAlign = 'center';
  game.ctx.fillText(element.name, element.x + element.width / 2, element.y + element.height / 2 + 3);
}

// 播放收集音效
function playCollectSound() {
  // 创建音频上下文
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // 创建音频节点
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();
    
    // 连接节点
    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);
    
    // 设置音频参数
    oscillator.type = 'sine';
    oscillator.frequency.value = 600;
    gainNode.gain.value = 0.1;
    
    // 设置音量淡出
    gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioContext.currentTime + 0.3);
    
    // 播放和停止
    oscillator.start();
    oscillator.stop(audioContext.currentTime + 0.3);
  } catch (e) {
    console.log('无法播放音效:', e);
  }
}

// 键盘按下处理
function handleKeyDown(e) {
  if (e.key === 'ArrowLeft') {
    game.keys.left = true;
  } else if (e.key === 'ArrowRight') {
    game.keys.right = true;
  }
}

// 键盘释放处理
function handleKeyUp(e) {
  if (e.key === 'ArrowLeft') {
    game.keys.left = false;
  } else if (e.key === 'ArrowRight') {
    game.keys.right = false;
  }
}

// 鼠标点击处理
function handleMouseDown(e) {
  const rect = game.canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  
  // 点击左半部分向左移动，右半部分向右移动
  if (x < game.canvas.width / 2) {
    movePlayerLeft();
  } else {
    movePlayerRight();
  }
}

// 触摸开始处理
function handleTouchStart(e) {
  if (e.touches.length > 0) {
    const rect = game.canvas.getBoundingClientRect();
    const x = e.touches[0].clientX - rect.left;
    
    // 触摸左半部分向左移动，右半部分向右移动
    if (x < game.canvas.width / 2) {
      movePlayerLeft();
    } else {
      movePlayerRight();
    }
    
    // 防止默认行为和滚动
    e.preventDefault();
  }
}

// 向左移动玩家
function movePlayerLeft() {
  game.player.x -= 30;
  if (game.player.x < 0) {
    game.player.x = 0;
  }
}

// 向右移动玩家
function movePlayerRight() {
  game.player.x += 30;
  if (game.player.x > game.canvas.width - game.player.width) {
    game.player.x = game.canvas.width - game.player.width;
  }
} 