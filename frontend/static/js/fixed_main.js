/**
 * 音乐推荐系统主JavaScript文件
 * 包含Vue.js应用初始化和核心功能实现
 */

// 等待页面加载完成
document.addEventListener('DOMContentLoaded', function() {
  console.log('音乐推荐系统应用已初始化');
  
  // 全局事件总线，用于组件间通信
  const EventBus = new Vue();
  
  // 音乐粒子游戏对象
  const musicGame = {
    canvas: null,
    ctx: null,
    particles: [],
    animationFrame: null,
    
    init(canvasId) {
      this.canvas = document.getElementById(canvasId);
      if (!this.canvas) return false;
      
      this.ctx = this.canvas.getContext('2d');
      if (!this.ctx) return false;
      
      // 调整canvas大小
      this.resizeCanvas();
      window.addEventListener('resize', () => this.resizeCanvas());
      
      // 初始化粒子
      this.initParticles(50);
      
      // 开始动画
      this.startAnimation();
      
      return true;
    },
    
    resizeCanvas() {
      if (!this.canvas) return;
      const container = this.canvas.parentElement;
      this.canvas.width = container.clientWidth;
      this.canvas.height = container.clientHeight;
    },
    
    initParticles(count) {
      this.particles = [];
      
      for (let i = 0; i < count; i++) {
        this.particles.push({
          x: Math.random() * this.canvas.width,
          y: Math.random() * this.canvas.height,
          size: Math.random() * 4 + 1,
          speedX: Math.random() * 2 - 1,
          speedY: Math.random() * 2 - 1,
          color: this.getRandomColor()
        });
      }
    },
    
    getRandomColor() {
      const colors = ['#FF5252', '#4CAF50', '#2196F3', '#FFEB3B', '#9C27B0'];
      return colors[Math.floor(Math.random() * colors.length)];
    },
    
    startAnimation() {
      if (this.animationFrame) {
        cancelAnimationFrame(this.animationFrame);
      }
      
      const animate = () => {
        this.draw();
        this.updateParticles();
        this.animationFrame = requestAnimationFrame(animate);
      };
      
      animate();
    },
    
    draw() {
      if (!this.canvas || !this.ctx) return;
      
      // 清除画布
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      
      // 绘制粒子
      for (let i = 0; i < this.particles.length; i++) {
        const p = this.particles[i];
        
        this.ctx.beginPath();
        this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        this.ctx.fillStyle = p.color;
        this.ctx.fill();
      }
    },
    
    // 更新粒子
    updateParticles() {
      if (!this.particles || !Array.isArray(this.particles)) return;
      
      for (let i = 0; i < this.particles.length; i++) {
        let p = this.particles[i];
        
        // 更新粒子位置
        p.x += p.speedX;
        p.y += p.speedY;
        
        // 边界检查
        if (p.x < 0 || p.x > this.canvas.width) {
          p.speedX *= -1;
        }
        
        if (p.y < 0 || p.y > this.canvas.height) {
          p.speedY *= -1;
        }
      }
    }
  }; // 结束对象定义

  // 当Vue应用加载完成后(或一段时间后)初始化游戏
  setTimeout(() => {
    initGameWhenCanvasReady();
  }, 1000);

  function initGameWhenCanvasReady() {
    const canvasElement = document.getElementById('musicCanvas');
    if (canvasElement) {
      musicGame.init('musicCanvas');
    } else {
      console.log('音乐Canvas尚未就绪，等待下一次尝试');
      setTimeout(initGameWhenCanvasReady, 500);
    }
  }
  
  // Vue.js应用实例
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
      isRegistering: false,
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
      ratedSongs: {},
      recommendations: [],
      
      // 聊天功能
      chatMessages: [],
      currentMessage: '',
      isChatLoading: false,
      
      // 评价功能
      evaluation: {
        accuracy: 3,
        usefulness: 3,
        satisfaction: 3,
        feedback: ''
      }
    },
    
    // 计算属性
    computed: {
      // 用户界面文本翻译
      t() {
        return (key) => {
          // 根据当前语言返回对应文本
          const translations = {
            'zh': {
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
              'ratings': '评分',
              'recommendations': '推荐',
              'chat': '聊天',
              'evaluate': '评价',
              // 其他翻译...
            },
            'en': {
              'login': 'Login',
              'register': 'Register',
              'username': 'Username',
              'email': 'Email',
              'password': 'Password',
              'loginPrompt': 'Have an account? Login',
              'registerPrompt': 'No account? Register',
              'logout': 'Logout',
              'user': 'User',
              'welcome': 'Welcome',
              'ratings': 'Ratings',
              'recommendations': 'Recommendations',
              'chat': 'Chat',
              'evaluate': 'Evaluate',
              // 其他翻译...
            }
          };
          
          return translations[this.currentLanguage][key] || key;
        };
      }
    },
    
    // 方法
    methods: {
      // 切换语言
      switchLanguage(lang) {
        if (lang === 'zh' || lang === 'en') {
          this.currentLanguage = lang;
          localStorage.setItem('preferred_language', lang);
        }
      },
      
      // 登录方法
      login() {
        this.isLoading = true;
        this.loginError = '';
        
        // API调用登录
        fetch('/api/user/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            username: this.username,
            password: this.password,
            email: this.email
          })
        })
        .then(response => response.json())
        .then(data => {
          this.isLoading = false;
          
          if (data.error) {
            this.loginError = data.error;
          } else {
            // 登录成功
            this.currentUser = {
              id: data.user_id,
              username: data.username,
              isDeveloper: data.is_developer
            };
            
            this.isLoggedIn = true;
            this.currentTab = 'welcome';
            
            // 保存用户会话
            localStorage.setItem('user_session', JSON.stringify({
              user_id: data.user_id,
              username: data.username,
              is_developer: data.is_developer
            }));
            
            // 显示成功通知
            this.addNotification('登录成功', 'is-success');
          }
        })
        .catch(error => {
          this.isLoading = false;
          this.loginError = '登录失败，请稍后再试';
          console.error('登录错误:', error);
        });
      },
      
      // 注册方法
      register() {
        this.isLoading = true;
        this.registerError = '';
        
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
          } else {
            // 注册成功
            this.addNotification('注册成功，请登录', 'is-success');
            this.currentTab = 'login';
            
            // 自动填充登录表单
            this.username = this.newUsername;
            this.password = this.newPassword;
            this.email = this.newEmail;
          }
        })
        .catch(error => {
          this.isLoading = false;
          this.registerError = '注册失败，请稍后再试';
          console.error('注册错误:', error);
        });
      },
      
      // 退出登录
      logout() {
        this.currentUser = null;
        this.isLoggedIn = false;
        this.currentTab = 'welcome';
        
        // 清除本地会话
        localStorage.removeItem('user_session');
        
        this.addNotification('已退出登录', 'is-info');
      },
      
      // 检查用户会话
      checkSession() {
        const session = localStorage.getItem('user_session');
        if (session) {
          try {
            const sessionData = JSON.parse(session);
            this.currentUser = {
              id: sessionData.user_id,
              username: sessionData.username,
              isDeveloper: sessionData.is_developer
            };
            this.isLoggedIn = true;
          } catch (e) {
            console.error('解析会话数据出错:', e);
            localStorage.removeItem('user_session');
          }
        }
        
        // 检查语言偏好
        const lang = localStorage.getItem('preferred_language');
        if (lang && (lang === 'zh' || lang === 'en')) {
          this.currentLanguage = lang;
        }
      },
      
      // 添加通知
      addNotification(message, type = 'is-info') {
        const id = Date.now();
        this.notifications.push({
          id,
          message,
          type,
          icon: this.getNotificationIcon(type)
        });
        
        // 自动移除通知
        setTimeout(() => {
          this.removeNotification(id);
        }, 5000);
      },
      
      // 根据类型获取通知图标
      getNotificationIcon(type) {
        switch (type) {
          case 'is-success': return 'check-circle';
          case 'is-danger': return 'exclamation-circle';
          case 'is-warning': return 'exclamation-triangle';
          case 'is-info': 
          default: return 'info-circle';
        }
      },
      
      // 移除通知
      removeNotification(id) {
        const index = this.notifications.findIndex(n => n.id === id);
        if (index !== -1) {
          this.notifications.splice(index, 1);
        }
      }
    },
    
    // 生命周期钩子
    mounted() {
      this.checkSession();
      
      // 检查是否当前是开发者面板页面
      if (window.location.pathname === '/developer') {
        this.isDeveloperMode = true;
      }
    }
  });
  
  // 修复模板渲染
  fixTemplateRendering();
  
  function fixTemplateRendering() {
    // 查找所有包含Vue双大括号语法的元素
    const allElements = document.querySelectorAll('*');
    allElements.forEach(el => {
      if (el.hasAttribute('src') && el.getAttribute('src').includes('{{')) {
        // 修复src属性
        const srcAttr = el.getAttribute('src');
        el.setAttribute('src', srcAttr.replace(/\{\{\s*url_for\('static',\s*filename='(.+?)'\)\s*\}\}/g, '/static/$1'));
      }
      
      if (el.hasAttribute('href') && el.getAttribute('href').includes('{{')) {
        // 修复href属性
        const hrefAttr = el.getAttribute('href');
        el.setAttribute('href', hrefAttr.replace(/\{\{\s*url_for\('static',\s*filename='(.+?)'\)\s*\}\}/g, '/static/$1'));
      }
    });
  }
}); 