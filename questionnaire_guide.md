# 添加问卷调查卡片指南

## 已完成的工作

1. CSS样式已添加到 `frontend/static/css/main.css`，包含了问卷调查卡片的样式定义
2. 问卷卡片的HTML代码已创建在 `frontend/templates/add_questionnaire_card.html`

## 添加问卷卡片步骤

由于编辑器可能存在问题，请按照以下步骤手动添加问卷卡片：

1. 打开 `frontend/templates/index.html` 文件
2. 查找游戏预览容器的闭合标签：
   ```html
   <div class="column is-12 mb-5">
       <!-- 音乐游戏预览 -->
       <div id="music-game-container" class="game-container">
           <div class="game-header">
               <h3><i class="fas fa-music mr-2"></i> 音乐收集游戏</h3>
               <button class="button is-small is-primary" data-tab="game">
                   开始游戏
               </button>
           </div>
           <canvas id="musicPreviewCanvas" height="200"></canvas>
       </div>
   </div>
   ```

3. 在此div闭合标签`</div>`后，粘贴来自`frontend/templates/add_questionnaire_card.html`的HTML代码
4. 保存文件

## 卡片位置

这将把问卷卡片放在游戏预览和其他功能卡片之间，使其成为首页上的一个醒目元素。

## 效果说明

1. 问卷卡片有特殊的边框和右上角装饰
2. 卡片在鼠标悬停时会微微上浮
3. 按钮采用渐变色设计，点击时有放大效果
4. 卡片使用了animate.css库实现脉动动画，吸引用户注意

这个醒目的问卷入口将增加用户填写问卷的可能性，从而提高个性化推荐的准确度。 