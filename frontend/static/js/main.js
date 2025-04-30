/**
 * 深度推荐音乐系统 - 主JavaScript文件
 * 包含Vue.js应用初始化和核心功能实现
 */

console.log('深度推荐音乐系统初始化开始');

// 音乐游戏变量
let musicGame = null;
let previewGame = null;

// 初始化游戏预览
setTimeout(() => {
  const previewCanvas = document.getElementById('musicPreviewCanvas');
  if (previewCanvas) {
    initGamePreview();
  }
}, 500);

function initGamePreview() {
  const canvas = document.getElementById('musicPreviewCanvas');
  if (!canvas) return;
  
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  
  // 调整canvas大小
  canvas.width = canvas.parentElement.clientWidth;
  
  // 绘制预览
  let particles = [];
  const genres = [
      { name: "流行", color: "#9B4BFF" },
      { name: "摇滚", color: "#FF5252" },
      { name: "电子", color: "#2196F3" },
      { name: "嘻哈", color: "#4CAF50" },
      { name: "古典", color: "#FFEB3B" },
      { name: "爵士", color: "#FF9800" },
  ];
  
  // 创建粒子
  for (let i = 0; i < 12; i++) {
    const genre = genres[Math.floor(Math.random() * genres.length)];
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height - canvas.height,
      size: 30,
      speed: Math.random() * 1 + 1,
      color: genre.color,
      genre: genre.name
    });
  }
  
  // 创建玩家
  const player = {
    x: canvas.width / 2 - 25,
    y: canvas.height - 50,
    width: 50,
    height: 50,
    color: '#8A2BE2'
  };
  
  function drawPreview() {
    // 清除画布
    ctx.fillStyle = '#191919';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 添加背景元素
    ctx.strokeStyle = 'rgba(138, 43, 226, 0.2)';
    for (let i = 0; i < 8; i++) {
      const x = Math.sin(Date.now() / 2000 + i) * canvas.width / 3 + canvas.width / 2;
      const y = i * canvas.height / 8;
      const width = Math.cos(Date.now() / 3000 + i) * 15 + 30;
      
      ctx.beginPath();
      ctx.arc(x, y, width, 0, Math.PI * 2);
      ctx.stroke();
    }
    
    // 绘制玩家
    ctx.fillStyle = player.color;
    ctx.beginPath();
    ctx.arc(
        player.x + player.width / 2,
        player.y + player.height / 2,
        player.width / 2,
        0,
        Math.PI * 2
    );
    ctx.fill();
    
    // 渲染粒子
    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      
      // 更新位置
      p.y += p.speed;
      
      // 如果超出屏幕底部，重置到顶部
      if (p.y > canvas.height) {
        p.y = -p.size;
        p.x = Math.random() * canvas.width;
      }
      
      // 绘制粒子
      ctx.fillStyle = p.color;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // 添加文字
      ctx.font = 'bold 14px Arial';
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.lineWidth = 3;
      ctx.textAlign = 'center';
      ctx.strokeText(p.genre, p.x, p.y + 4);
      ctx.fillStyle = '#FFFFFF';
      ctx.fillText(p.genre, p.x, p.y + 4);
    }
    
    // 添加"点击开始游戏"文字
    ctx.fillStyle = '#FFFFFF';
    ctx.font = '18px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('点击开始音乐收集游戏', canvas.width / 2, 30);
    
    requestAnimationFrame(drawPreview);
  }
  
  drawPreview();
}

// Vue.js应用实例 - 直接初始化，不包装在DOMContentLoaded事件中
window.app = new Vue({
  el: '#app',
  
  // 数据层
  data: {
    // 应用状态
    currentTab: 'welcome',
    isLoading: false,
    isLoadingRecommendations: false,
    currentLanguage: 'zh',
    isDeveloperMode: false,
    notifications: [],
    
    // 用户相关
    currentUser: null,
    isLoggedIn: false,
    username: '',
    email: '',
    password: '',
    newUsername: '',
    newEmail: '',
    newPassword: '',
    loginError: '',
    registerError: '',
    
    // 歌曲数据
    sampleSongs: [],
    recommendations: [],
    
    // 聊天功能
    chatMessages: [],
    currentMessage: '',
    isChatLoading: false,
    
    // 游戏数据
    gameResults: null,
    
    // 音频播放器
    currentAudio: null,
    isPlaying: false,
    currentPlayingId: null,
    
    // 情感相关数据
    userEmotion: null,
    emotionInput: '',
    showEmotionDetector: false,
    emotionDetector: null,
    
    // 用户偏好数据
    selectedMusicStyles: [],
    selectedMusicScenes: [],
    selectedMusicLanguages: [],
    selectedMusicEras: [],
    favoriteArtists: '',
    dailyListeningTime: '',
    
    // 音乐风格
    musicGenres: [
        { id: 'pop', zh: '流行音乐', en: 'Pop Music' },
        { id: 'rock', zh: '摇滚音乐', en: 'Rock Music' },
        { id: 'classical', zh: '古典音乐', en: 'Classical Music' },
        { id: 'jazz', zh: '爵士音乐', en: 'Jazz Music' },
        { id: 'electronic', zh: '电子音乐', en: 'Electronic Music' },
        { id: 'folk', zh: '民谣音乐', en: 'Folk Music' },
        { id: 'hip-hop', zh: '嘻哈音乐', en: 'Hip-Hop Music' },
        { id: 'r&b', zh: 'R&B音乐', en: 'R&B Music' },
        { id: 'country', zh: '乡村音乐', en: 'Country Music' },
        { id: 'metal', zh: '金属音乐', en: 'Metal Music' },
        { id: 'blues', zh: '蓝调音乐', en: 'Blues Music' },
        { id: 'reggae', zh: '雷鬼音乐', en: 'Reggae Music' }
    ],
    
    // 音乐风格选项
    musicStyleOptions: [
      { value: 'pop', label: '流行 (Pop)' },
      { value: 'rock', label: '摇滚 (Rock)' },
      { value: 'classical', label: '古典 (Classical)' },
      { value: 'jazz', label: '爵士 (Jazz)' },
      { value: 'electronic', label: '电子 (Electronic)' },
      { value: 'hiphop', label: '嘻哈 (Hip-hop)' },
      { value: 'folk', label: '民谣 (Folk)' },
      { value: 'rb', label: 'R&B (R&B)' },
      { value: 'country', label: '乡村 (Country)' },
      { value: 'metal', label: '金属 (Metal)' }
    ],
    
    // 场景选项
    musicSceneOptions: [
      { value: 'relax', label: '放松/休息时' },
      { value: 'work', label: '工作/学习时' },
      { value: 'exercise', label: '运动时' },
      { value: 'travel', label: '旅行/通勤时' },
      { value: 'party', label: '聚会/社交场合' },
      { value: 'sleep', label: '睡前/冥想时' }
    ],
    
    // 语言选项
    musicLanguageOptions: [
      { value: 'chinese', label: '中文' },
      { value: 'english', label: '英文' },
      { value: 'japanese', label: '日文' },
      { value: 'korean', label: '韩文' },
      { value: 'other', label: '其他' }
    ],
    
    // 年代选项
    musicEraOptions: [
      { value: '60s', label: '60年代' },
      { value: '70s', label: '70年代' },
      { value: '80s', label: '80年代' },
      { value: '90s', label: '90年代' },
      { value: '00s', label: '2000年代' },
      { value: '10s', label: '2010年代' },
      { value: 'current', label: '现代/最新' }
    ],
    
    // 听歌时长选项
    listeningTimeOptions: [
      { value: 'less1', label: '少于1小时' },
      { value: '1to2', label: '1-2小时' },
      { value: '2to4', label: '2-4小时' },
      { value: 'more4', label: '4小时以上' }
    ],
    
    // 添加预设音乐预览URL（使用相对路径，指向项目本地音频）
    previewUrls: [
        '/static/audio/preview1.mp3', // 示例本地预览1
        '/static/audio/preview2.mp3', // 示例本地预览2 
        '/static/audio/preview3.mp3', // 示例本地预览3
        '/static/audio/preview4.mp3', // 示例本地预览4
        '/static/audio/preview5.mp3', // 示例本地预览5
        // 备用：Spotify预览URLs（如果本地音频不可用）
        'https://p.scdn.co/mp3-preview/3eb16018c2a700240e9dfb8817b6f2d041f15eb1', // Shape of You
        'https://p.scdn.co/mp3-preview/e2f5edb569c73916235f2cadc8290b3dde522179', // Blinding Lights
        'https://p.scdn.co/mp3-preview/74456889dc17ca44897559c14ec7de20f431dd82', // Dance Monkey
        'https://p.scdn.co/mp3-preview/84a68eef8a7d26be04b81c21621f32adcf44b825', // Circles
        'https://p.scdn.co/mp3-preview/8250dc653c7abe6e89552a22c30b52b4d7414b41'  // Watermelon Sugar
    ],
    
    // 其他现有数据
    userRatings: {}, // 用户评分记录
    
    // 问卷相关
    showQuestionnaireUI: false,
    currentQuestionStep: 1,
    totalQuestionSteps: 8,
    questionnaireProgress: 0,
    
    // 问卷相关数据
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
            title: '音乐情绪偏好',
            subtitle: '您通常希望音乐带给您什么样的情绪？ (可多选)',
            dataCategory: 'moods',
            options: [
                { value: 'happy', label: '愉快/兴奋' },
                { value: 'relax', label: '放松/平静' },
                { value: 'sad', label: '忧伤/沉思' },
                { value: 'energetic', label: '精力充沛' },
                { value: 'focus', label: '专注/集中' },
                { value: 'nostalgic', label: '怀旧/回忆' }
            ]
        },
        {
            id: 3,
            title: '音乐场景偏好',
            subtitle: '您在什么场景下最常听音乐？ (可多选)',
            dataCategory: 'scenarios',
            options: [
                { value: 'work', label: '工作/学习时' },
                { value: 'exercise', label: '运动时' },
                { value: 'commute', label: '通勤/旅行时' },
                { value: 'relax', label: '休息放松时' },
                { value: 'party', label: '社交/聚会时' },
                { value: 'sleep', label: '睡前/冥想时' }
            ]
        },
        {
            id: 4,
            title: '音乐语言偏好',
            subtitle: '您偏好哪种语言的歌曲？ (可多选)',
            dataCategory: 'languages',
            options: [
                { value: 'chinese', label: '中文歌曲' },
                { value: 'english', label: '英文歌曲' },
                { value: 'japanese', label: '日文歌曲' },
                { value: 'korean', label: '韩文歌曲' },
                { value: 'other', label: '其他语言' },
                { value: 'instrumental', label: '纯音乐(无歌词)' }
            ]
        }
    ],
    
    // 新增音乐心理咨询师相关状态
    therapistMode: true, // 启用心理咨询师模式
    userSentiments: [], // 记录用户情感历史
    musicPreferences: {}, // 用户音乐偏好
    conversationHistory: [], // 对话历史
    lastProactiveQuestion: null, // 上一次主动提问
    therapistSuggestions: [] // 心理咨询师的建议
  },
  
  // 计算属性
  computed: {
    // 检查是否评分了足够的歌曲
    hasRatedEnoughSongs() {
      let ratedCount = 0;
      this.sampleSongs.forEach(song => {
        if (song.rating > 0) ratedCount++;
      });
      console.log('已评分歌曲数量:', ratedCount);
      return ratedCount >= 5; // 至少需要评分5首歌曲
    },
    
    // 翻译函数
    t() {
      return (key) => {
        const translations = {
          'zh': {
            'home': '首页',
            'login': '登录',
            'register': '注册',
            'username': '用户名',
            'email': '邮箱',
            'password': '密码',
            'loginPrompt': '已有账号？点击登录',
            'registerPrompt': '没有账号？点击注册',
            'logout': '退出',
            'user': '用户',
            'welcome': '欢迎',
            'rate': '音乐评分问卷',
            'rateSubtitle': '为歌曲评分，帮助我们了解您的音乐偏好',
            'notRated': '尚未评分',
            'recommend': '推荐',
            'recommendSubtitle': '基于您的评分和偏好推荐的音乐',
            'loading': '加载中...',
            'noRecommendations': '暂无推荐，请先评分一些歌曲',
            'rateMore': '去评分更多歌曲',
            'getRecommendations': '获取推荐',
            'needMoreRatings': '请至少对5首歌曲进行评分',
            'chat': '聊天',
            'chatSubtitle': '与AI助手聊天，获取个性化音乐推荐',
            'chatWelcome': '你好！我是AI音乐助手，可以帮你找到你喜欢的音乐。试着告诉我你喜欢什么类型的音乐或者你喜欢的歌手吧！',
            'typeSomething': '输入消息...',
            'game': '游戏',
            'gameSubtitle': '通过游戏收集音乐道具，表达您的音乐偏好',
            'questionnaire': '音乐评分问卷',
            'questionnaireDesc': '评分歌曲帮助我们了解您的音乐偏好，为您提供更准确的推荐',
            'questionnaireContent': '完成这份音乐评分问卷，帮助我们了解您的音乐偏好。通过评分歌曲、分享您的心情和听歌场景，我们能够为您提供更加个性化的音乐推荐。您的每一次评分都能让推荐系统更了解您！',
            'startQuestionnaire': '开始评分问卷',
            'mood': '您当前的心情',
            'tellMood': '告诉我们您的心情',
            'saveMood': '保存心情',
            'moodPlaceholder': '例如：开心、放松、忧郁、精力充沛...',
            'moodSaved': '已记录您的心情！',
            'preferences': '音乐偏好设置',
            'musicStyle': '您喜欢的音乐风格',
            'musicScene': '您通常在什么场景下听音乐',
            'musicLanguage': '您喜欢的歌曲语言',
            'musicEra': '您喜欢的音乐年代',
            'favoriteArtists': '您喜欢的歌手/艺术家',
            'artistPlaceholder': '输入您喜欢的歌手名称，用逗号分隔',
            'dailyListening': '您平均每天听音乐的时长',
            'savePreferences': '保存偏好设置'
          },
          'en': {
            'home': 'Home',
            'login': 'Login',
            'register': 'Register',
            'username': 'Username',
            'email': 'Email',
            'password': 'Password',
            'loginPrompt': 'Already have an account? Login',
            'registerPrompt': 'No account? Register',
            'logout': 'Logout',
            'user': 'User',
            'welcome': 'Welcome',
            'rate': 'Music Rating Questionnaire',
            'rateSubtitle': 'Rate songs to help us understand your music preferences',
            'notRated': 'Not rated yet',
            'recommend': 'Recommend',
            'recommendSubtitle': 'Music recommendations based on your ratings and preferences',
            'loading': 'Loading...',
            'noRecommendations': 'No recommendations yet. Please rate some songs first.',
            'rateMore': 'Rate more songs',
            'getRecommendations': 'Get Recommendations',
            'needMoreRatings': 'Please rate at least 5 songs',
            'chat': 'Chat',
            'chatSubtitle': 'Chat with AI assistant to get personalized music recommendations',
            'chatWelcome': 'Hello! I\'m the AI Music Assistant. I can help you find music you\'ll love. Try telling me what genres or artists you like!',
            'typeSomething': 'Type a message...',
            'game': 'Game',
            'gameSubtitle': 'Collect music items through a game to express your music preferences',
            'questionnaire': 'Music Rating Questionnaire',
            'questionnaireDesc': 'Rate songs to help us understand your music preferences for more accurate recommendations',
            'questionnaireContent': 'Complete this music rating questionnaire to help us understand your music preferences. By rating songs, sharing your mood and listening scenarios, we can provide more personalized music recommendations for you. Each rating helps our system understand you better!',
            'startQuestionnaire': 'Start Questionnaire',
            'mood': 'Your Current Mood',
            'tellMood': 'Tell us your mood',
            'saveMood': 'Save Mood',
            'moodPlaceholder': 'For example: happy, relaxed, melancholic, energetic...',
            'moodSaved': 'Your mood has been recorded!',
            'preferences': 'Music Preferences',
            'musicStyle': 'Music styles you like',
            'musicScene': 'When do you usually listen to music',
            'musicLanguage': 'Languages of songs you prefer',
            'musicEra': 'Music eras you enjoy',
            'favoriteArtists': 'Your favorite artists/singers',
            'artistPlaceholder': 'Enter artist names separated by commas',
            'dailyListening': 'Average daily music listening time',
            'savePreferences': 'Save Preferences'
          }
        };
        
        return translations[this.currentLanguage][key] || key;
      };
    }
  },
  
  // 方法
  methods: {
    // 语言切换
    switchLanguage(lang) {
      if (lang === 'zh' || lang === 'en') {
      this.currentLanguage = lang;
      // 本地存储用户语言偏好
      localStorage.setItem('preferredLanguage', lang);
        localStorage.setItem('language', lang);
      this.addNotification(lang === 'zh' ? '已切换到中文' : 'Switched to English', 'is-success');
        // 强制更新所有绑定
        this.$forceUpdate();
      }
    },
    
    // 添加格式化消息方法
    formatMessage(text) {
      if (!text) return '';
      // 将换行符转换为HTML换行
      return text.replace(/\n/g, '<br>');
    },
    
    // 添加格式化时间方法
    formatTime(timestamp) {
      if (!timestamp) return '';
      const date = new Date(timestamp);
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    },
    
    // 登录
    login() {
      this.isLoading = true;
      this.loginError = '';
      
      // 发送真实的登录请求
      fetch('/api/user/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            username: this.username,
          password: this.password
        })
      })
      .then(response => response.json())
      .then(data => {
        this.isLoading = false;
        
        if (data.error) {
          this.loginError = data.error;
          this.addNotification(data.error, 'is-danger');
        } else {
          // 登录成功
          console.log('登录成功:', data);
          
          // 设置用户信息
          this.currentUser = {
            id: data.user_id,
            username: data.username,
            email: data.email || '',
            isDeveloper: data.is_developer || false
          };
          
          this.isLoggedIn = true;
          
          // 保存到本地存储
          localStorage.setItem('user', JSON.stringify(this.currentUser));
          localStorage.setItem('userId', this.currentUser.id);
          localStorage.setItem('username', this.currentUser.username);
          
          // 加载初始数据
          this.currentTab = 'welcome';
          this.loadSampleSongs();
          this.addNotification('登录成功！欢迎，' + this.currentUser.username, 'is-success');
        }
      })
      .catch(error => {
        console.error('登录请求失败:', error);
        this.isLoading = false;
        this.loginError = '登录失败，请稍后再试';
        this.addNotification('登录失败，请稍后再试', 'is-danger');
      });
    },
    
    // 注册
    register() {
      this.isLoading = true;
      this.registerError = '';
      
      // 发送真实的注册请求
      fetch('/api/user/register', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          username: this.newUsername,
          password: this.newPassword,
          email: this.newEmail
        })
      })
      .then(response => response.json())
      .then(data => {
          this.isLoading = false;
        
        if (data.error) {
          this.registerError = data.error;
          this.addNotification(data.error, 'is-danger');
        } else {
          // 注册成功
          console.log('注册成功:', data);
          this.addNotification('注册成功！请登录', 'is-success');
          
          // 自动填充登录表单，方便用户登录
          this.username = this.newUsername;
          this.password = this.newPassword;
          
          // 切换到登录页面
          this.currentTab = 'login';
        }
      })
      .catch(error => {
        console.error('注册请求失败:', error);
        this.isLoading = false;
        this.registerError = '注册失败，请稍后再试';
        this.addNotification('注册失败，请稍后再试', 'is-danger');
      });
    },
    
    // 登出
    logout() {
      // 清除用户状态
      this.isLoggedIn = false;
      this.currentUser = null;
      
      // 清除本地存储中的所有用户相关数据
      localStorage.removeItem('user');
      localStorage.removeItem('userId');
      localStorage.removeItem('username');
      
      // 切换到登录页面
      this.currentTab = 'login';
      this.addNotification('您已成功登出', 'is-info');
      
      // 清除游戏状态
      if (musicGame) {
        musicGame.stopGame();
        musicGame = null;
      }
      
      console.log('用户已登出，所有状态已重置');
    },
    
    // 检查会话
    checkSession() {
      const savedUser = localStorage.getItem('user');
      if (savedUser) {
        try {
          this.currentUser = JSON.parse(savedUser);
          this.isLoggedIn = true;
          this.currentTab = 'welcome';
          this.loadSampleSongs();
        } catch (e) {
          console.error('无法解析保存的用户数据', e);
          localStorage.removeItem('user');
        }
      }
      
      // 恢复语言设置
      const savedLanguage = localStorage.getItem('language');
      if (savedLanguage) {
        this.currentLanguage = savedLanguage;
      }
    },
    
    // 加载示例歌曲
    loadSampleSongs() {
      this.isLoading = true;
      
      // 模拟API请求
      setTimeout(() => {
        this.sampleSongs = [
          { id: 101, title: "Shape of You", artist: "Ed Sheeran", album_image: "https://i.scdn.co/image/ab67616d0000b273ba5db46f4b838ef6027e6f96", preview_url: this.previewUrls[0] },
          { id: 102, title: "Blinding Lights", artist: "The Weeknd", album_image: "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb36", preview_url: this.previewUrls[1] },
          { id: 103, title: "Dance Monkey", artist: "Tones and I", album_image: "https://i.scdn.co/image/ab67616d0000b2739f39192ec5a1f04f7c08d9ab", preview_url: this.previewUrls[2] },
          { id: 104, title: "Circles", artist: "Post Malone", album_image: "https://i.scdn.co/image/ab67616d0000b27399e211c11052dcb57a592f6c", preview_url: this.previewUrls[3] },
          { id: 105, title: "Watermelon Sugar", artist: "Harry Styles", album_image: "https://i.scdn.co/image/ab67616d0000b273da5d5aeeabacacc1263c0f4b", preview_url: this.previewUrls[4] },
          { id: 106, title: "Bad Guy", artist: "Billie Eilish", album_image: "https://i.scdn.co/image/ab67616d0000b273a91c10fe9472d9bd89802e5a", preview_url: this.previewUrls[5] },
          { id: 107, title: "Don't Start Now", artist: "Dua Lipa", album_image: "https://i.scdn.co/image/ab67616d0000b273bd26ede1ae69327010d49946", preview_url: this.previewUrls[6] },
          { id: 108, title: "Everything I Wanted", artist: "Billie Eilish", album_image: "https://i.scdn.co/image/ab67616d0000b273a91c10fe9472d9bd89802e5a", preview_url: this.previewUrls[7] },
          { id: 109, title: "Memories", artist: "Maroon 5", album_image: "https://i.scdn.co/image/ab67616d0000b273b25ef9c9015bdd771fbda74d", preview_url: this.previewUrls[8] },
          { id: 110, title: "Someone You Loved", artist: "Lewis Capaldi", album_image: "https://i.scdn.co/image/ab67616d0000b2733c65bbfd4c0f45af8c4b6e59", preview_url: this.previewUrls[9] }
        ];
        this.isLoading = false;
      }, 1000);
    },
    
    // 评分歌曲
    rateSong(song, rating) {
      // 确保song对象存在
      if (!song) return;
      
      // 直接设置评分到歌曲对象上
      this.$set(song, 'rating', rating);
      
      // 更新全局用户评分对象（如果需要）
      if (!this.userRatings) this.userRatings = {};
      this.userRatings[song.id] = rating;
      
      console.log(`为 "${song.title}" 评分 ${rating} 星`, song);
      this.addNotification(`已为 "${song.title}" 评分 ${rating} 星`, 'is-success');
      
      // 检查是否可以获取推荐
      if (this.hasRatedEnoughSongs) {
        this.addNotification('您已评分足够的歌曲，可以获取个性化推荐!', 'is-success');
      }
    },
    
    // 获取推荐
    getRecommendations() {
      if (!this.hasRatedEnoughSongs) {
        this.addNotification('请至少对5首歌曲进行评分', 'is-warning');
        return;
      }
      
      this.isLoadingRecommendations = true;
      this.currentTab = 'recommend';
      
      // 模拟API请求
      setTimeout(() => {
        // 根据评分生成模拟推荐
        const ratedSongs = this.sampleSongs.filter(song => song.rating > 0);
        const highRatedArtists = {};
        
        ratedSongs.forEach(song => {
          if (song.rating >= 4) {
            if (!highRatedArtists[song.artist]) {
              highRatedArtists[song.artist] = 0;
            }
            highRatedArtists[song.artist] += song.rating;
          }
        });
        
        // 模拟推荐结果
        this.recommendations = [
          { id: 101, title: "November Rain", artist: "Guns N' Roses", album_image: "https://via.placeholder.com/150", explanation: "因为你喜欢摇滚音乐和Guns N' Roses的其他歌曲" },
          { id: 102, title: "Hello", artist: "Adele", album_image: "https://via.placeholder.com/150", explanation: "基于你对Adele的高评分" },
          { id: 103, title: "Thriller", artist: "Michael Jackson", album_image: "https://via.placeholder.com/150", explanation: "与你喜欢的流行音乐风格相似" },
          { id: 104, title: "Wonderwall", artist: "Oasis", album_image: "https://via.placeholder.com/150", explanation: "摇滚音乐爱好者的经典选择" },
          { id: 105, title: "Nothing Else Matters", artist: "Metallica", album_image: "https://via.placeholder.com/150", explanation: "为摇滚音乐爱好者推荐" },
          { id: 106, title: "Someone Like You", artist: "Adele", album_image: "https://via.placeholder.com/150", explanation: "基于你对Adele的高评分" },
          { id: 107, title: "布拉格广场", artist: "周杰伦", album_image: "https://via.placeholder.com/150", explanation: "因为你喜欢周杰伦的其他歌曲" },
          { id: 108, title: "稻香", artist: "周杰伦", album_image: "https://via.placeholder.com/150", explanation: "因为你喜欢周杰伦的其他歌曲" }
        ];
        
        this.isLoadingRecommendations = false;
        this.addNotification('根据您的评分生成了推荐', 'is-success');
      }, 1500);
    },
    
    // 处理图片加载错误
    handleImageError(event) {
      event.target.src = '/static/img/default-album.png';
    },
    
    // 喜欢歌曲
    likeSong(song) {
      this.addNotification(`已添加 "${song.title}" 到我喜欢的音乐`, 'is-success');
    },
    
    // 不喜欢歌曲
    dislikeSong(song) {
      this.addNotification(`已将 "${song.title}" 标记为不喜欢`, 'is-warning');
      // 从推荐列表中移除
      this.recommendations = this.recommendations.filter(s => s.id !== song.id);
    },
    
    // 发送聊天消息
    async sendMessage() {
      if (!this.currentMessage.trim()) return;
      
      const userMessage = this.currentMessage.trim();
      this.chatMessages.push({
          content: userMessage,
          isUser: true,
          timestamp: new Date()
      });
      this.currentMessage = '';
      
      // 记录对话历史
      this.conversationHistory.push({
          role: 'user',
          message: userMessage,
          timestamp: new Date()
      });
      
      // 滚动到底部
      this.$nextTick(() => {
          const chatMessages = document.querySelector('.chat-messages');
          if (chatMessages) {
              chatMessages.scrollTop = chatMessages.scrollHeight;
          }
      });
      
      // 处理用户发送的消息
      this.isChatLoading = true;
      
      try {
          // 检查消息中是否包含特定指令
          const hasSpecificRequest = this.checkForSpecificRequests(userMessage);
          
          if (!hasSpecificRequest) {
              // 发送到后端AI服务处理
              const response = await this.sendMessageToAI(userMessage);
              
              // 如果启用了心理咨询师模式，处理回复
              if (this.therapistMode) {
                  this.handleTherapistResponse(response);
              } else {
                  // 常规模式处理
              this.chatMessages.push({
                      content: response.message,
                  isUser: false,
                  timestamp: new Date()
              });
              
                  // 如果有推荐，处理推荐
                  if (response.recommendations && response.recommendations.length > 0) {
                      this.handleRecommendations(response.recommendations, response.emotion);
                  }
              }
          }
          
          this.isChatLoading = false;
      } catch (error) {
          console.error('处理聊天消息时出错:', error);
          this.addNotification('抱歉，处理消息时出现了问题', 'is-danger');
          this.isChatLoading = false;
      }
    },
    
    /**
     * 检查消息是否包含特定指令
     */
    checkForSpecificRequests(message) {
        // 特定指令处理
        const disableTherapistCmd = /关闭心理咨询/i;
        const enableTherapistCmd = /开启心理咨询/i;
        
        if (disableTherapistCmd.test(message)) {
            this.therapistMode = false;
                  this.chatMessages.push({
                content: "已关闭音乐心理咨询师模式。我将以常规音乐助手的方式为您服务。",
                      isUser: false,
                timestamp: new Date()
            });
            return true;
        }
        
        if (enableTherapistCmd.test(message)) {
            this.therapistMode = true;
            this.chatMessages.push({
                content: "已开启音乐心理咨询师模式。我将帮助您探索音乐与情感的联系，并提供个性化的音乐建议。",
                isUser: false,
                timestamp: new Date()
            });
            return true;
        }
        
        return false;
    },
    
    /**
     * 发送消息到后端AI服务
     */
    async sendMessageToAI(message) {
        try {
            const response = await axios.post('/api/chat', {
                user_id: this.currentUser.id || 'guest_user',
                message: message
            });
            
            if (response.data && response.data.response) {
                return response.data.response;
            } else {
                throw new Error('无效的响应数据');
            }
        } catch (error) {
            console.error('AI服务请求失败:', error);
                  return {
                message: "抱歉，我无法连接到服务器。请检查您的网络连接后再试。",
                emotion: "neutral",
                recommendations: []
            };
        }
    },
    
    /**
     * 处理心理咨询师模式下的回复
     */
    handleTherapistResponse(response) {
        // 记录情感状态
        if (response.emotion && response.emotion !== 'neutral') {
            this.userSentiments.push({
                emotion: response.emotion,
                timestamp: new Date()
            });
        }
        
        // 处理回复消息
        const message = response.message;
        
        // 分离主动问题和回复部分（如果可能）
        const parts = message.split(/(?<=。|！|\?|？)(?=\S)/);
        let mainResponse = message;
        let proactiveQuestion = '';
        
        if (parts.length > 1) {
            // 假设最后一部分是主动问题
            mainResponse = parts.slice(0, -1).join('');
            proactiveQuestion = parts[parts.length - 1];
            this.lastProactiveQuestion = proactiveQuestion;
        }
        
        // 添加主要回复
        if (mainResponse.trim()) {
              this.chatMessages.push({
                content: mainResponse,
                  isUser: false,
                  timestamp: new Date()
              });
        }
        
        // 如果有推荐，处理推荐
        if (response.recommendations && response.recommendations.length > 0) {
            this.handleRecommendations(response.recommendations, response.emotion);
        }
        
        // 添加主动问题（如果有）
        if (proactiveQuestion && proactiveQuestion.trim()) {
            setTimeout(() => {
                  this.chatMessages.push({
                    content: proactiveQuestion,
                      isUser: false,
                    timestamp: new Date(),
                    isProactiveQuestion: true
                  });
                  
                  // 滚动到底部
                  this.$nextTick(() => {
                      const chatMessages = document.querySelector('.chat-messages');
                      if (chatMessages) {
                          chatMessages.scrollTop = chatMessages.scrollHeight;
                      }
                  });
            }, 1500); // 延迟显示主动问题，更自然
        }
        
        // 记录对话历史
        this.conversationHistory.push({
            role: 'assistant',
            message: message,
            timestamp: new Date(),
            emotion: response.emotion
        });
    },
    
    /**
     * 处理音乐推荐
     */
    handleRecommendations(recommendations, emotion) {
        // 处理推荐
        if (!recommendations || recommendations.length === 0) return;
        
        // 构建推荐歌曲列表
        const formattedSongs = recommendations.map(song => ({
            ...song,
            // 确保有预览URL
            preview_url: song.preview_url || this.previewUrls[Math.floor(Math.random() * this.previewUrls.length)]
        }));
        
        // 生成推荐消息
        let songRecommendations = '根据我的分析，为您推荐以下歌曲：\n\n';
        formattedSongs.forEach((song, index) => {
            songRecommendations += `${index + 1}. ${song.title} - ${song.artist}\n`;
            
            // 添加推荐理由
            const reason = song.reason || this.emotionDetector.generateRecommendationReason(emotion);
            songRecommendations += `   ${reason.zh}\n\n`;
        });
        
        songRecommendations += '您可以点击"试听"按钮来收听这些歌曲。';
        
        // 添加推荐消息
        setTimeout(() => {
              this.chatMessages.push({
                content: songRecommendations,
                  isUser: false,
                timestamp: new Date(),
                songs: formattedSongs  // 附加歌曲数据供显示
              });
          
          // 滚动到底部
          this.$nextTick(() => {
              const chatMessages = document.querySelector('.chat-messages');
              if (chatMessages) {
                  chatMessages.scrollTop = chatMessages.scrollHeight;
              }
          });
        }, 1000);
        
        // 同时更新推荐页面
        this.recommendations = formattedSongs;
    },
    
    // 播放歌曲预览
    async playSongPreview(songOrUrl, songTitle, songArtist) {
      try {
        let url = '';
        let title = '';
        let artist = '';
        
        // 判断参数类型
        if (typeof songOrUrl === 'object') {
          // 如果是歌曲对象
          url = songOrUrl.preview_url;
          title = songOrUrl.title || songOrUrl.track_name || '未知歌曲';
          artist = songOrUrl.artist || songOrUrl.artist_name || '未知艺术家';
        } else {
          // 如果是直接传递URL和标题
          url = songOrUrl;
          title = songTitle || '未知歌曲';
          artist = songArtist || '未知艺术家';
        }
        
        console.log(`正在尝试播放: ${title} - ${artist}, URL: ${url}`);
        
        if (!url || url === 'null' || url === 'undefined') {
          this.addNotification('没有可用的试听链接', 'is-warning');
          console.warn('歌曲没有预览URL:', title);
          
          // 使用默认备用URL
          url = 'https://p.scdn.co/mp3-preview/3eb16018c2a700240e9dfb8817b6f2d041f15eb1';
          console.log('使用备用URL:', url);
        }

        console.log("开始播放音乐预览:", url, title, artist);

        // 使用audio元素播放
        const audioPlayer = document.getElementById('audioPlayer');
        const playerContainer = document.getElementById('audioPlayerContainer');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const audioTitle = document.getElementById('audioTitle');
        const audioArtist = document.getElementById('audioArtist');
        const closeBtn = document.getElementById('closeAudioPlayer');
        
        if (!audioPlayer || !playerContainer) {
            console.error('找不到音频播放器元素');
            this.addNotification('音频播放器初始化失败', 'is-danger');
            return;
        }
        
        // 设置音频属性
        audioPlayer.crossOrigin = "anonymous";  // 添加跨域支持
        audioPlayer.preload = "auto";           // 预加载音频
        audioPlayer.src = url;
        
        // 添加音频加载错误处理
        audioPlayer.onerror = (e) => {
            console.error("音频加载错误:", e);
            this.addNotification('音频文件无法加载，可能是URL无效或跨域限制', 'is-danger');
            
            // 尝试使用备用URL
            const backupUrl = this.previewUrls[Math.floor(Math.random() * this.previewUrls.length)];
            console.log("尝试使用备用URL:", backupUrl);
            
            // 使用备用链接重试
            try {
                audioPlayer.src = backupUrl;
                audioPlayer.load();
                audioPlayer.play().catch(innerError => {
                    console.error("备用音频播放失败:", innerError);
                    this.addNotification('播放失败，请尝试使用现代浏览器并确保网络连接正常', 'is-warning');
                });
            } catch (retryError) {
                console.error("备用播放尝试失败:", retryError);
            }
        };
        
        // 更新播放器信息
        audioTitle.textContent = title;
        audioArtist.textContent = artist;
        
        // 显示播放器
        playerContainer.classList.remove('hidden');
        
        // 加载音频
        audioPlayer.load();
        
        // 播放音频，添加自动重试逻辑
        const tryPlay = (retryCount = 0) => {
            if (retryCount >= 3) {
                this.addNotification('无法自动播放音频，请点击播放按钮手动播放', 'is-warning');
                return;
            }
            
            const playPromise = audioPlayer.play();
            
            if (playPromise !== undefined) {
                playPromise.then(_ => {
                    // 播放成功
                    console.log("音频播放成功");
                    this.addNotification(`正在播放: ${title} - ${artist}`, 'is-info');
                    playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                })
                .catch(error => {
                    // 播放失败
                    console.error("音频播放失败:", error);
                    
                    if (error.name === "NotAllowedError") {
                        this.addNotification('浏览器阻止了自动播放，请点击播放按钮手动播放', 'is-info');
                    } else if (retryCount < 2) {
                        console.log(`播放失败，${retryCount + 1}秒后重试...`);
                        setTimeout(() => tryPlay(retryCount + 1), 1000);
                    } else {
                        this.addNotification('无法播放音频，请检查网络连接并确保使用支持HTML5的浏览器', 'is-warning');
                    }
                    
                    playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                });
            }
        };
        
        // 开始尝试播放
        tryPlay();
        
        // 添加播放/暂停切换
        playPauseBtn.onclick = function() {
            if (audioPlayer.paused) {
                audioPlayer.play().catch(e => {
                    console.error("手动播放失败:", e);
                    playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                });
                playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
            } else {
                audioPlayer.pause();
                playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
            }
        };
        
        // 音频结束事件
        audioPlayer.onended = function() {
            playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        };
        
        // 关闭按钮
        closeBtn.onclick = function() {
            audioPlayer.pause();
            playerContainer.classList.add('hidden');
        };
      } catch (error) {
          console.error("播放音乐出错:", error);
          this.addNotification('播放音乐时发生错误', 'is-danger');
      }
    },
    
    // 初始化音乐游戏
    initMusicGame() {
      if (this.currentTab === 'game') {
        // 清除先前的游戏
        if (musicGame) {
          musicGame.stopGame();
        }
        
        // 初始化新游戏
        musicGame = initMusicGame('game-canvas-container', this.handleGameComplete);
      }
    },
    
    // 处理游戏完成
    handleGameComplete(results) {
      console.log('游戏结果:', results);
      this.gameResults = results;
      this.showGameResults = true;
      
      // 添加情感分析
      const dominantGenre = Object.entries(results)
          .sort((a, b) => b[1] - a[1])[0][0];
          
      // 使用情感检测器将音乐风格映射到情感
      const emotionResult = this.emotionDetector.mapGenreToEmotion(dominantGenre);
      this.userEmotion = emotionResult;
      
      this.addNotification(`根据您喜欢的${dominantGenre}音乐，分析出您可能的情绪是: ${emotionResult.emotion}`, 'is-info');
    },
    
    // 使用游戏结果获取推荐
    useGameResultsForRecommendations() {
      if (!this.gameResults) {
          this.addNotification('请先完成音乐游戏', 'is-warning');
          return;
      }
      
      this.loading = true;
      this.currentTab = 'recommend'; // 确保切换到推荐页面
      
      // 如果已经有情绪分析，使用它来获取推荐
      if (this.userEmotion) {
          // 获取最受欢迎的流派
          const genres = Object.entries(this.gameResults)
              .sort((a, b) => b[1] - a[1])
              .map(entry => entry[0]);
              
          // 前端模拟实现：基于随机挑选歌曲并添加推荐理由
          setTimeout(() => {
              this.recommendations = this.sampleSongs
                  .sort(() => 0.5 - Math.random())
                  .slice(0, 6)
                  .map(song => {
                      return {
                          ...song,
                          // 加入游戏收集到的流派信息
                          recommendationReason: `根据您在游戏中喜欢的${genres[0]}音乐，以及检测到的"${this.userEmotion.emotion}"情绪，${this.emotionDetector.generateRecommendationReason(this.userEmotion.emotion)}`
                      };
                  });
              
              this.loading = false;
              this.addNotification('根据游戏结果生成了新推荐', 'is-success');
          }, 1000);
      } else {
          // 如果没有情绪分析，使用原本的游戏结果获取推荐
          setTimeout(() => {
              this.recommendations = this.sampleSongs
                  .sort(() => 0.5 - Math.random())
                  .slice(0, 6);
              this.loading = false;
              this.addNotification('根据游戏结果生成了新推荐', 'is-success');
          }, 1000);
      }
      
      this.showGameResults = false;
    },
    
    // 添加通知
    addNotification(message, type = 'is-info') {
      const id = Date.now();
      const icon = this.getNotificationIcon(type);
      
      this.notifications.push({
        id,
        message,
        type,
        icon
      });
      
      // 5秒后自动移除
      setTimeout(() => {
        this.removeNotification(id);
      }, 5000);
    },
    
    // 获取通知图标
    getNotificationIcon(type) {
      switch (type) {
        case 'is-success': return 'check-circle';
        case 'is-danger': return 'exclamation-circle';
        case 'is-warning': return 'exclamation-triangle';
        default: return 'info-circle';
      }
    },
    
    // 移除通知
    removeNotification(id) {
      this.notifications = this.notifications.filter(n => n.id !== id);
    },
    
    // 初始化情感检测器
    initEmotionDetector() {
      this.emotionDetector = new EmotionDetector();
    },
    
    // 切换情感输入界面
    toggleEmotionInput() {
      this.showEmotionDetector = !this.showEmotionDetector;
      if (this.showEmotionDetector) {
          // 如果打开了情感输入，滚动到该区域
          this.$nextTick(() => {
              const container = document.querySelector('.emotion-input-container');
              if (container) {
                  container.scrollIntoView({ behavior: 'smooth' });
              }
          });
      }
    },
    
    // 处理情感输入
    async detectEmotion() {
      if (!this.emotionInput.trim()) {
          this.addNotification('请输入您当前的心情', 'is-warning');
          return;
      }
      
      this.addNotification('正在分析您的情绪...', 'is-info');
      const result = await this.emotionDetector.detectFromText(this.emotionInput);
      
      this.userEmotion = result;
      this.addNotification(`检测到您当前的情绪: ${result.emotion}`, 'is-success');
      
      // 自动获取情感推荐
      this.getEmotionBasedRecommendations();
    },
    
    // 获取基于情感的音乐推荐
    async getEmotionBasedRecommendations() {
      if (!this.userEmotion) {
          this.addNotification('请先输入您的心情', 'is-warning');
          return;
      }
      
      this.loading = true;
      try {
          // 在实际项目中，应该调用后端API获取推荐
          // const response = await axios.post('/api/recommend_by_emotion', {
          //     user_id: this.userId,
          //     emotion: this.userEmotion.emotion,
          //     valence: this.userEmotion.valence,
          //     energy: this.userEmotion.energy
          // });
          // this.recommendations = response.data.recommendations;
          
          // 前端模拟实现：基于随机挑选歌曲并添加推荐理由
          this.recommendations = this.sampleSongs
              .sort(() => 0.5 - Math.random())
              .slice(0, 6)
              .map(song => {
                  return {
                      ...song,
                      // 使用情感检测器生成推荐理由
                      recommendationReason: this.emotionDetector.generateRecommendationReason(this.userEmotion.emotion)
                  };
              });
          
          this.addNotification('根据您的情绪推荐了新音乐', 'is-success');
      } catch (error) {
          console.error('获取推荐失败:', error);
          this.addNotification('推荐失败，请稍后再试', 'is-danger');
      } finally {
          this.loading = false;
      }
    },
    
    // 导航到情感推荐页面
    navigateToEmotionRecommend() {
      this.currentTab = 'recommend'; // 切换到推荐标签页
      this.$nextTick(() => {
        this.showEmotionDetector = true; // 显示情感输入界面
        // 滚动到情感输入区域
        setTimeout(() => {
          const container = document.querySelector('.emotion-input-container');
          if (container) {
            container.scrollIntoView({ behavior: 'smooth' });
          }
        }, 300);
      });
    },
    
    // 使用建议的聊天提示
    useSuggestion(suggestion) {
      this.currentMessage = suggestion;
      this.sendMessage();
    },
    
    // 初始化页面按钮事件绑定
    initButtonEvents() {
      console.log('初始化按钮事件');
      
      // 确保Vue已完全渲染DOM
      this.$nextTick(() => {
        // 为所有带有data-tab属性的元素添加点击事件
        document.querySelectorAll('[data-tab]').forEach(el => {
          el.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = el.getAttribute('data-tab');
            if (tab) {
              console.log('切换到标签页:', tab);
              this.currentTab = tab;
            }
          });
        });
        
        // 为所有试听按钮添加事件监听器
        const previewButtons = document.querySelectorAll('.button[data-preview-id]');
        if (previewButtons.length > 0) {
          previewButtons.forEach(button => {
            console.log('找到试听按钮:', button);
            button.addEventListener('click', (event) => {
              event.preventDefault();
              const songId = button.getAttribute('data-preview-id');
              const song = this.findSongById(songId);
              if (song) {
                this.playSongPreview(song);
              }
            });
          });
        }
      });
    },
    
    // 通过ID查找歌曲
    findSongById(id) {
      // 在sampleSongs和recommendations中查找
      const allSongs = [...this.sampleSongs, ...this.recommendations];
      return allSongs.find(song => song.id === parseInt(id));
    },
    
    // 添加新的方法用于保存心情
    submitMood() {
      if (this.emotionInput) {
        // 保存用户心情数据
        this.userEmotion = {
          emotion: this.emotionInput,
          timestamp: new Date().toISOString()
        };
        this.addNotification(
          this.currentLanguage === 'zh' ? 
          '已记录您的心情！' : 
          'Your mood has been recorded!', 
          'is-success'
        );
      }
    },
    
    // 保存用户偏好设置
    saveUserPreferences() {
      // 收集表单数据
      const preferences = {
        musicStyles: this.selectedMusicStyles,
        musicScenes: this.selectedMusicScenes,
        musicLanguages: this.selectedMusicLanguages,
        musicEras: this.selectedMusicEras,
        favoriteArtists: this.favoriteArtists,
        dailyListeningTime: this.dailyListeningTime
      };
      
      // 保存到本地存储或发送到服务器
      localStorage.setItem('userMusicPreferences', JSON.stringify(preferences));
      
      // 添加通知
      this.addNotification(
        this.currentLanguage === 'zh' ? 
        '偏好设置已保存！' : 
        'Preferences saved!', 
        'is-success'
      );
    },
    
    // 获取当前问题步骤
    getCurrentQuestionStep() {
        // 确保questionSteps存在
        if (!this.questionSteps || !this.questionSteps.length) {
            return {
                title: '音乐风格偏好',
                subtitle: '请选择您喜欢的音乐风格 (可多选)',
                dataCategory: 'genres',
                options: this.musicStyleOptions || []
            };
        }
        return this.questionSteps.find(step => step.id === this.currentQuestionStep) || this.questionSteps[0];
    },
    
    // 检查选项是否被选中
    isOptionSelected(category, value) {
        if (!this.questionnaireAnswers[category]) {
            return false;
        }
        return this.questionnaireAnswers[category].indexOf(value) !== -1;
    },
    
    // 切换问卷选项选择状态
    toggleSelection(category, value) {
        if (!this.questionnaireAnswers[category]) {
            this.questionnaireAnswers[category] = [];
        }
        
        const index = this.questionnaireAnswers[category].indexOf(value);
        if (index === -1) {
            this.questionnaireAnswers[category].push(value);
        } else {
            this.questionnaireAnswers[category].splice(index, 1);
        }
    },
    
    // 下一个问题
    nextQuestionStep() {
        if (this.currentQuestionStep < this.totalQuestionSteps) {
            this.currentQuestionStep++;
            this.questionnaireProgress = (this.currentQuestionStep / this.totalQuestionSteps) * 100;
        } else {
            // 提交问卷
            this.submitQuestionnaire();
        }
    },
    
    // 上一个问题
    prevQuestionStep() {
        if (this.currentQuestionStep > 1) {
            this.currentQuestionStep--;
            this.questionnaireProgress = (this.currentQuestionStep / this.totalQuestionSteps) * 100;
        }
    },
    
    // 提交问卷
    submitQuestionnaire() {
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
    },
  },
  
  // 侦听器
  watch: {
    currentTab(newTab, oldTab) {
      if (newTab === 'game') {
        // 初始化游戏
        this.$nextTick(() => {
          this.initMusicGame();
        });
      } else if (oldTab === 'game' && musicGame) {
        // 停止游戏
        musicGame.stopGame();
      }
      
      // 新增：检查切换到rate标签页时是否需要显示问卷UI
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
  },
  
  // 组件挂载后
  mounted() {
    console.log('Vue应用已挂载');
    this.checkSession();
    
    // 初始化汉堡菜单
    const $navbarBurgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);
    if ($navbarBurgers.length > 0) {
      $navbarBurgers.forEach(el => {
        el.addEventListener('click', () => {
          const target = el.dataset.target;
          const $target = document.getElementById(target);
          el.classList.toggle('is-active');
          $target.classList.toggle('is-active');
        });
      });
    }
    
    // 初始化用户数据
    if (!this.userRatings) {
      this.userRatings = {};
    }

    // 加载示例歌曲数据
    this.loadSampleSongs();
    
    // 初始化页面按钮事件绑定
    this.initButtonEvents();
    
    // 初始化情感检测器
    this.initEmotionDetector();
    
    // 初始化心理咨询师模式
    this.therapistMode = true;
    console.log("已启用音乐心理咨询师模式");
    
    // 获取URL参数
    const urlParams = new URLSearchParams(window.location.search);
    
    // 如果有tab参数，切换到相应标签
    if (urlParams.has('tab')) {
      const tab = urlParams.get('tab');
      if (['welcome', 'rate', 'recommend', 'chat', 'game'].includes(tab)) {
        this.currentTab = tab;
      }
    }
    
    // 如果有问卷参数，显示问卷
    if (urlParams.has('questionnaire') && urlParams.get('questionnaire') === 'true') {
      this.showQuestionnaireUI = true;
    }
    
    // 绑定关闭音频播放器事件
    const closeAudioBtn = document.getElementById('closeAudioPlayer');
    if (closeAudioBtn) {
      closeAudioBtn.onclick = function() {
    const audioPlayer = document.getElementById('audioPlayer');
        const playerContainer = document.getElementById('audioPlayerContainer');
        if (audioPlayer && playerContainer) {
          audioPlayer.pause();
          playerContainer.classList.add('hidden');
        }
      };
    }
  }
});
