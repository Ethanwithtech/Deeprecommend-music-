/**
 * 音乐收集游戏
 * 让用户通过收集飘落的音乐道具来表达对音乐类型的偏好
 */

class MusicCollectionGame {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        this.player = {
            x: this.width / 2,
            y: this.height - 80,
            width: 50,
            height: 50,
            speed: 7,
            color: '#8A2BE2',
            isJumping: false,
            jumpHeight: 120,
            jumpSpeed: 8,
            jumpProgress: 0,
            originalY: this.height - 80
        };
        this.genres = [
            { name: "流行", color: "#9B4BFF" },
            { name: "摇滚", color: "#FF5252" },
            { name: "电子", color: "#2196F3" },
            { name: "嘻哈", color: "#4CAF50" },
            { name: "古典", color: "#FFEB3B" },
            { name: "爵士", color: "#FF9800" },
            { name: "蓝调", color: "#3F51B5" },
            { name: "民谣", color: "#009688" }
        ];
        this.musicItems = [];
        this.collectedGenres = {};
        this.score = 0;
        this.isGameStarted = false;
        this.isGameOver = false;
        this.animationFrame = null;
        this.lastItemTime = 0;
        this.keys = {
            ArrowLeft: false,
            ArrowRight: false,
            ArrowUp: false,
            Space: false
        };
        
        // 粒子效果系统
        this.particles = [];
        
        // 游戏结束后的回调
        this.onGameComplete = null;
        
        // 事件监听器
        this.initEventListeners();
    }

    init() {
        // 重置游戏状态
        this.musicItems = [];
        this.collectedGenres = {};
        this.score = 0;
        this.isGameOver = false;
        this.isGameStarted = false;
        
        // 初始化游戏界面
        this.renderStartScreen();
    }

    initEventListeners() {
        // 键盘控制
        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || 
                e.key === 'ArrowUp' || e.key === ' ') {
                
                if (e.key === ' ') {
                    this.keys.Space = true;
                } else {
                    this.keys[e.key] = true;
                }
                
                // 处理跳跃
                if ((e.key === 'ArrowUp' || e.key === ' ') && !this.player.isJumping) {
                    this.player.isJumping = true;
                    this.player.jumpProgress = 0;
                }
                
                e.preventDefault(); // 防止页面滚动
            }
            
            // 按空格开始游戏
            if (e.key === ' ' && !this.isGameStarted) {
                this.startGame();
                e.preventDefault();
            }
        });

        document.addEventListener('keyup', (e) => {
            if (e.key === 'ArrowLeft' || e.key === 'ArrowRight' || 
                e.key === 'ArrowUp') {
                this.keys[e.key] = false;
            }
            if (e.key === ' ') {
                this.keys.Space = false;
            }
        });

        // 触摸控制（适用于移动设备）
        this.canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            if (!this.isGameStarted) {
                this.startGame();
                return;
            }
            
            const touch = e.touches[0];
            const rect = this.canvas.getBoundingClientRect();
            const touchX = touch.clientX - rect.left;
            const touchY = touch.clientY - rect.top;
            
            // 上半部分触摸 - 跳跃
            if (touchY < this.height / 2) {
                if (!this.player.isJumping) {
                    this.player.isJumping = true;
                    this.player.jumpProgress = 0;
                }
            } 
            // 左侧触摸 - 向左移动
            else if (touchX < this.width / 2) {
                this.keys.ArrowLeft = true;
                this.keys.ArrowRight = false;
            } 
            // 右侧触摸 - 向右移动
            else {
                this.keys.ArrowLeft = false;
                this.keys.ArrowRight = true;
            }
        });

        this.canvas.addEventListener('touchend', () => {
            this.keys.ArrowLeft = false;
            this.keys.ArrowRight = false;
        });

        // 鼠标点击开始游戏
        this.canvas.addEventListener('click', (e) => {
            if (!this.isGameStarted) {
                this.startGame();
            } else {
                // 游戏中点击上半部分跳跃
                const rect = this.canvas.getBoundingClientRect();
                const clickY = e.clientY - rect.top;
                
                if (clickY < this.height / 2 && !this.player.isJumping) {
                    this.player.isJumping = true;
                    this.player.jumpProgress = 0;
                }
            }
        });
    }

    startGame() {
        if (this.isGameStarted) return;
        
        this.isGameStarted = true;
        this.isGameOver = false;
        this.score = 0;
        this.collectedGenres = {};
        this.genres.forEach(genre => {
            this.collectedGenres[genre.name] = 0;
        });
        this.lastItemTime = Date.now();
        
        // 启动游戏循环
        this.gameLoop();
    }

    gameLoop() {
        if (this.isGameOver) {
            this.renderGameOver();
            return;
        }

        // 清除画布
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // 更新玩家位置
        this.updatePlayer();
        
        // 生成音乐道具
        this.generateMusicItems();
        
        // 更新音乐道具位置
        this.updateMusicItems();
        
        // 检测碰撞
        this.checkCollisions();
        
        // 更新粒子效果
        this.updateParticles();
        
        // 渲染游戏元素
        this.renderGame();
        
        // 渲染粒子效果
        this.renderParticles();
        
        // 检查游戏是否结束
        if (this.score >= 30) {
            this.isGameOver = true;
            if (typeof this.onGameComplete === 'function') {
                this.onGameComplete(this.collectedGenres);
            }
        } else {
            // 继续游戏循环
            this.animationFrame = requestAnimationFrame(() => this.gameLoop());
        }
    }

    updatePlayer() {
        // 左右移动
        if (this.keys.ArrowLeft && this.player.x > 0) {
            this.player.x -= this.player.speed;
        }
        if (this.keys.ArrowRight && this.player.x < this.width - this.player.width) {
            this.player.x += this.player.speed;
        }
        
        // 处理跳跃
        if (this.player.isJumping) {
            // 跳跃动画使用正弦曲线实现更自然的跳跃效果
            this.player.jumpProgress += this.player.jumpSpeed;
            
            if (this.player.jumpProgress >= 180) {
                // 一个完整的跳跃周期结束
                this.player.isJumping = false;
                this.player.y = this.player.originalY;
            } else {
                // 计算当前跳跃高度
                const jumpY = Math.sin(this.player.jumpProgress * Math.PI / 180) * this.player.jumpHeight;
                this.player.y = this.player.originalY - jumpY;
            }
        }
    }

    generateMusicItems() {
        const currentTime = Date.now();
        const timeDiff = currentTime - this.lastItemTime;
        
        // 每500ms生成一个新音乐道具
        if (timeDiff > 500) {
            const randomGenre = this.genres[Math.floor(Math.random() * this.genres.length)];
            
            this.musicItems.push({
                x: Math.random() * (this.width - 40),
                y: -30,
                width: 40,
                height: 40,
                speed: Math.random() * 2 + 2,
                genre: randomGenre.name,
                color: randomGenre.color
            });
            
            this.lastItemTime = currentTime;
        }
    }

    updateMusicItems() {
        for (let i = 0; i < this.musicItems.length; i++) {
            // 更新位置
            this.musicItems[i].y += this.musicItems[i].speed;
            
            // 如果道具超出屏幕底部，移除它
            if (this.musicItems[i].y > this.height) {
                this.musicItems.splice(i, 1);
                i--;
            }
        }
    }

    checkCollisions() {
        for (let i = 0; i < this.musicItems.length; i++) {
            const item = this.musicItems[i];
            
            // 检查玩家是否与道具碰撞
            if (this.player.x < item.x + item.width &&
                this.player.x + this.player.width > item.x &&
                this.player.y < item.y + item.height &&
                this.player.y + this.player.height > item.y) {
                
                // 增加分数
                this.score++;
                
                // 更新收集的音乐类型
                this.collectedGenres[item.genre]++;
                
                // 添加粒子效果
                this.createCollectParticles(item.x + item.width/2, item.y + item.height/2, item.color);
                
                // 移除已收集的道具
                this.musicItems.splice(i, 1);
                i--;
            }
        }
    }

    createCollectParticles(x, y, color) {
        for (let i = 0; i < 12; i++) {
            const angle = Math.random() * Math.PI * 2;
            const speed = Math.random() * 3 + 1;
            
            this.particles.push({
                x: x,
                y: y,
                vx: Math.cos(angle) * speed,
                vy: Math.sin(angle) * speed,
                color: color,
                radius: Math.random() * 5 + 2,
                alpha: 1,
                shrink: Math.random() * 0.05 + 0.01
            });
        }
    }

    updateParticles() {
        for (let i = 0; i < this.particles.length; i++) {
            const p = this.particles[i];
            p.x += p.vx;
            p.y += p.vy;
            p.alpha -= p.shrink;
            
            if (p.alpha <= 0) {
                this.particles.splice(i, 1);
                i--;
            }
        }
    }

    renderParticles() {
        for (const p of this.particles) {
            this.ctx.save();
            this.ctx.globalAlpha = p.alpha;
            this.ctx.fillStyle = p.color;
            this.ctx.beginPath();
            this.ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.strokeStyle = '#FFFFFF';
            this.ctx.lineWidth = 2;
            this.ctx.stroke();
            this.ctx.restore();
        }
    }

    renderGame() {
        // 渲染背景
        this.ctx.fillStyle = '#191919';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // 添加背景元素
        this.renderBackgroundElements();
        
        // 渲染玩家
        this.renderPlayer();
        
        // 渲染音乐道具
        this.renderMusicItems();
        
        // 显示分数
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '20px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`得分: ${this.score}/30`, 10, 30);
        
        // 显示跳跃提示
        if (!this.player.isJumping) {
            this.ctx.font = '14px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillStyle = 'rgba(255,255,255,0.5)';
            this.ctx.fillText('按空格或上箭头跳跃', this.width / 2, 30);
        }
    }

    renderPlayer() {
        // 创建渐变色彩
        const gradient = this.ctx.createRadialGradient(
            this.player.x + this.player.width / 2,
            this.player.y + this.player.height / 2,
            0,
            this.player.x + this.player.width / 2,
            this.player.y + this.player.height / 2,
            this.player.width / 2
        );
        gradient.addColorStop(0, '#9B4BFF');
        gradient.addColorStop(1, '#8A2BE2');
        
        // 绘制玩家主体
        this.ctx.fillStyle = gradient;
        this.ctx.beginPath();
        this.ctx.arc(
            this.player.x + this.player.width / 2,
            this.player.y + this.player.height / 2,
            this.player.width / 2,
            0,
            Math.PI * 2
        );
        this.ctx.fill();
        
        // 绘制玩家面部
        // 头部装饰 - 耳机
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.beginPath();
        this.ctx.arc(
            this.player.x + this.player.width / 2,
            this.player.y + this.player.height / 2 - 5,
            this.player.width / 6,
            0,
            Math.PI * 2
        );
        this.ctx.fill();
        
        // 添加眼睛
        this.ctx.fillStyle = '#000000';
        // 左眼
        this.ctx.beginPath();
        this.ctx.arc(
            this.player.x + this.player.width / 2 - 8,
            this.player.y + this.player.height / 2,
            3,
            0,
            Math.PI * 2
        );
        this.ctx.fill();
        // 右眼
        this.ctx.beginPath();
        this.ctx.arc(
            this.player.x + this.player.width / 2 + 8,
            this.player.y + this.player.height / 2,
            3,
            0,
            Math.PI * 2
        );
        this.ctx.fill();
        
        // 添加笑脸
        this.ctx.strokeStyle = '#000000';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.arc(
            this.player.x + this.player.width / 2,
            this.player.y + this.player.height / 2 + 5,
            8,
            0.1 * Math.PI,
            0.9 * Math.PI,
            false
        );
        this.ctx.stroke();
        
        // 如果正在跳跃，添加一些效果
        if (this.player.isJumping) {
            // 添加跳跃波纹
            this.ctx.strokeStyle = 'rgba(138, 43, 226, 0.3)';
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.arc(
                this.player.x + this.player.width / 2,
                this.player.y + this.player.height / 2,
                this.player.width / 2 + 5 + Math.sin(Date.now() / 100) * 3,
                0,
                Math.PI * 2
            );
            this.ctx.stroke();
        }
    }

    renderMusicItems() {
        for (let i = 0; i < this.musicItems.length; i++) {
            const item = this.musicItems[i];
            
            // 创建渐变
            const gradient = this.ctx.createRadialGradient(
                item.x + item.width / 2,
                item.y + item.height / 2,
                0,
                item.x + item.width / 2,
                item.y + item.height / 2,
                item.width / 2
            );
            gradient.addColorStop(0, this.lightenColor(item.color, 30));
            gradient.addColorStop(1, item.color);
            
            // 音符外圈
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(
                item.x + item.width / 2,
                item.y + item.height / 2,
                item.width / 2,
                0,
                Math.PI * 2
            );
            this.ctx.fill();
            
            // 音乐符号装饰
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
            
            // 根据不同类型的音乐绘制不同形状
            switch (item.genre) {
                case "流行":
                    // 音符形状 ♪
                    this.drawMusicNote(item.x + item.width / 2, item.y + item.height / 2, 10);
                    break;
                case "摇滚":
                    // 闪电形状
                    this.drawLightning(item.x + item.width / 2, item.y + item.height / 2, 10);
                    break;
                case "电子":
                    // 声波形状
                    this.drawSoundwave(item.x + item.width / 2, item.y + item.height / 2, 10);
                    break;
                default:
                    // 默认音符形状
                    this.drawMusicNote(item.x + item.width / 2, item.y + item.height / 2, 8);
                    break;
            }
            
            // 音符标签
            this.ctx.font = 'bold 14px Arial';
            this.ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
            this.ctx.lineWidth = 3;
            this.ctx.strokeText(item.genre, item.x + item.width / 2, item.y + item.height / 2 + 4);
            this.ctx.fillStyle = '#FFFFFF';
            this.ctx.fillText(item.genre, item.x + item.width / 2, item.y + item.height / 2 + 4);
        }
    }

    lightenColor(color, percent) {
        const num = parseInt(color.replace("#", ""), 16);
        const amt = Math.round(2.55 * percent);
        const R = (num >> 16) + amt;
        const G = (num >> 8 & 0x00FF) + amt;
        const B = (num & 0x0000FF) + amt;
        
        return "#" + (
            0x1000000 + 
            (R < 255 ? (R < 1 ? 0 : R) : 255) * 0x10000 + 
            (G < 255 ? (G < 1 ? 0 : G) : 255) * 0x100 + 
            (B < 255 ? (B < 1 ? 0 : B) : 255)
        ).toString(16).slice(1);
    }

    drawMusicNote(x, y, size) {
        this.ctx.beginPath();
        this.ctx.arc(x - size/3, y + size/3, size/2, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.fillRect(x - size/3 + size/2 - 1, y - size/2, 2, size);
    }

    drawLightning(x, y, size) {
        this.ctx.beginPath();
        this.ctx.moveTo(x, y - size);
        this.ctx.lineTo(x - size/2, y);
        this.ctx.lineTo(x, y);
        this.ctx.lineTo(x - size/2, y + size);
        this.ctx.closePath();
        this.ctx.fill();
    }

    drawSoundwave(x, y, size) {
        for (let i = 0; i < 3; i++) {
            this.ctx.beginPath();
            this.ctx.arc(x, y, size * (i+1)/3, 0, Math.PI * 2);
            this.ctx.stroke();
        }
    }

    renderBackgroundElements() {
        // 星星背景
        for (let i = 0; i < 50; i++) {
            const x = Math.sin(Date.now() / 5000 + i) * this.width / 2 + this.width / 2;
            const y = (i / 50) * this.height;
            const size = Math.cos(Date.now() / 3000 + i) * 1.5 + 2;
            
            this.ctx.fillStyle = `rgba(255, 255, 255, ${0.3 + Math.sin(Date.now() / 1000 + i) * 0.2})`;
            this.ctx.beginPath();
            this.ctx.arc(x, y, size, 0, Math.PI * 2);
            this.ctx.fill();
        }
        
        // 网格线背景
        this.ctx.strokeStyle = 'rgba(138, 43, 226, 0.1)';
        this.ctx.lineWidth = 1;
        
        // 横线
        for (let y = 0; y < this.height; y += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.width, y);
            this.ctx.stroke();
        }
        
        // 纵线
        for (let x = 0; x < this.width; x += 30) {
            this.ctx.beginPath();
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.height);
            this.ctx.stroke();
        }
    }

    renderStartScreen() {
        // 渲染背景
        this.ctx.fillStyle = '#191919';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // 添加背景元素
        this.renderBackgroundElements();
        
        // 游戏标题
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = 'bold 36px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('音乐收集游戏', this.width / 2, this.height / 3 - 20);
        
        // 彩色标题下划线
        const gradient = this.ctx.createLinearGradient(
            this.width / 2 - 150, this.height / 3 - 10, 
            this.width / 2 + 150, this.height / 3 - 10
        );
        gradient.addColorStop(0, "#9B4BFF");
        gradient.addColorStop(0.25, "#FF5252");
        gradient.addColorStop(0.5, "#2196F3");
        gradient.addColorStop(0.75, "#4CAF50");
        gradient.addColorStop(1, "#FF9800");
        
        this.ctx.strokeStyle = gradient;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        this.ctx.moveTo(this.width / 2 - 150, this.height / 3);
        this.ctx.lineTo(this.width / 2 + 150, this.height / 3);
        this.ctx.stroke();
        
        // 游戏说明
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
        this.ctx.font = '18px Arial';
        this.ctx.fillText('使用← →箭头键控制角色', this.width / 2, this.height / 2 - 10);
        this.ctx.fillText('使用↑箭头键或空格键跳跃', this.width / 2, this.height / 2 + 20);
        this.ctx.fillText('收集落下的音乐道具来表达您的偏好', this.width / 2, this.height / 2 + 50);
        
        // 动态的开始按钮
        const buttonY = this.height / 2 + 100;
        const buttonWidth = 180;
        const buttonHeight = 60;
        const buttonX = this.width / 2 - buttonWidth / 2;
        
        // 动态阴影效果
        const shadowSize = 5 + Math.sin(Date.now() / 300) * 2;
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        this.ctx.fillRect(
            buttonX + shadowSize, 
            buttonY + shadowSize, 
            buttonWidth, 
            buttonHeight
        );
        
        // 按钮主体
        const buttonGradient = this.ctx.createLinearGradient(
            buttonX, buttonY, 
            buttonX + buttonWidth, buttonY + buttonHeight
        );
        buttonGradient.addColorStop(0, "#9B4BFF");
        buttonGradient.addColorStop(1, "#8A2BE2");
        
        this.ctx.fillStyle = buttonGradient;
        this.ctx.fillRect(buttonX, buttonY, buttonWidth, buttonHeight);
        
        // 按钮边框
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        this.ctx.lineWidth = 2;
        this.ctx.strokeRect(buttonX, buttonY, buttonWidth, buttonHeight);
        
        // 按钮文字
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = 'bold 24px Arial';
        this.ctx.fillText('开始游戏', this.width / 2, buttonY + buttonHeight / 2 + 8);
        
        // 添加闪烁的提示
        const alpha = Math.sin(Date.now() / 500) * 0.3 + 0.7;
        this.ctx.fillStyle = `rgba(255, 255, 255, ${alpha})`;
        this.ctx.font = '16px Arial';
        this.ctx.fillText('点击或按空格开始', this.width / 2, buttonY + buttonHeight + 30);
        
        // 绘制示例角色
        this.player.x = this.width / 2 - this.player.width / 2;
        this.player.y = this.height / 3 + 40;
        this.renderPlayer();
    }

    renderGameOver() {
        // 渲染背景
        this.ctx.fillStyle = '#191919';
        this.ctx.fillRect(0, 0, this.width, this.height);
        
        // 添加背景元素
        this.renderBackgroundElements();
        
        // 游戏结束标题
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '30px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText('游戏结束！', this.width / 2, this.height / 4);
        
        // 显示收集结果
        this.ctx.font = '20px Arial';
        this.ctx.fillText('您收集的音乐偏好:', this.width / 2, this.height / 3);
        
        let y = this.height / 3 + 40;
        Object.entries(this.collectedGenres)
            .sort((a, b) => b[1] - a[1])
            .forEach(([genre, count], index) => {
                if (count > 0) {
                    const genreObj = this.genres.find(g => g.name === genre);
                    this.ctx.fillStyle = genreObj ? genreObj.color : '#FFFFFF';
                    this.ctx.font = index === 0 ? 'bold 18px Arial' : '18px Arial';
                    this.ctx.fillText(`${genre}: ${count}`, this.width / 2, y);
                    y += 30;
                }
            });
        
        // 返回按钮
        this.ctx.fillStyle = '#8A2BE2';
        this.ctx.fillRect(this.width / 2 - 80, this.height - 100, 160, 50);
        this.ctx.fillStyle = '#FFFFFF';
        this.ctx.font = '20px Arial';
        this.ctx.fillText('返回', this.width / 2, this.height - 70);
    }

    resizeCanvas() {
        const container = this.canvas.parentElement;
        this.canvas.width = container.clientWidth;
        this.canvas.height = container.clientHeight;
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        // 重新绘制当前场景
        if (!this.isGameStarted) {
            this.renderStartScreen();
        } else if (this.isGameOver) {
            this.renderGameOver();
        }
    }

    stopGame() {
        if (this.animationFrame) {
            cancelAnimationFrame(this.animationFrame);
        }
    }
}

// 游戏初始化函数
function initMusicGame(canvasId, onComplete) {
    // 创建canvas元素
    const container = document.getElementById(canvasId);
    if (!container) return null;
    
    // 清空容器
    container.innerHTML = '';
    
    // 创建canvas
    const canvas = document.createElement('canvas');
    canvas.id = 'musicGameCanvas';
    canvas.width = container.clientWidth;
    canvas.height = 400; // 固定高度
    container.appendChild(canvas);
    
    // 初始化游戏
    const game = new MusicCollectionGame('musicGameCanvas');
    game.onGameComplete = onComplete;
    game.init();
    
    // 监听窗口大小变化
    window.addEventListener('resize', () => {
        game.resizeCanvas();
    });
    
    return game;
} 