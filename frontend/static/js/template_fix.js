/**
 * 修复模板渲染问题的脚本
 * 添加到页面底部，用于解决Flask+Vue模板变量冲突
 */

document.addEventListener('DOMContentLoaded', function() {
  // 修复静态资源链接
  fixStaticResourceLinks();
  
  // 初始化游戏环境
  setTimeout(function() {
    initMusicGame();
  }, 1000);
});

/**
 * 修复静态资源链接
 * 将模板中未正确渲染的{{ url_for('static', filename='...') }}替换为正确的链接
 */
function fixStaticResourceLinks() {
  // 修复CSS链接
  document.querySelectorAll('link[rel="stylesheet"]').forEach(function(link) {
    if (link.getAttribute('href') && link.getAttribute('href').includes('{{')) {
      const href = link.getAttribute('href');
      const newHref = href.replace(/\{\{\s*url_for\('static',\s*filename='(.+?)'\)\s*\}\}/g, '/static/$1');
      link.setAttribute('href', newHref);
      console.log('修复CSS链接:', newHref);
    }
  });
  
  // 修复JS链接
  document.querySelectorAll('script').forEach(function(script) {
    if (script.getAttribute('src') && script.getAttribute('src').includes('{{')) {
      const src = script.getAttribute('src');
      const newSrc = src.replace(/\{\{\s*url_for\('static',\s*filename='(.+?)'\)\s*\}\}/g, '/static/$1');
      script.setAttribute('src', newSrc);
      console.log('修复JS链接:', newSrc);
    }
  });
  
  // 修复图像链接
  document.querySelectorAll('img').forEach(function(img) {
    if (img.getAttribute('src') && img.getAttribute('src').includes('{{')) {
      const src = img.getAttribute('src');
      const newSrc = src.replace(/\{\{\s*url_for\('static',\s*filename='(.+?)'\)\s*\}\}/g, '/static/$1');
      img.setAttribute('src', newSrc);
      console.log('修复图像链接:', newSrc);
    }
  });
  
  // 自动加载补充的CSS和JS文件
  loadRequiredResources();
}

/**
 * 加载必要的资源
 */
function loadRequiredResources() {
  // 检查CSS是否已加载
  if (!isResourceLoaded('link', 'href', '/static/css/main.css')) {
    const cssLink = document.createElement('link');
    cssLink.rel = 'stylesheet';
    cssLink.href = '/static/css/main.css';
    document.head.appendChild(cssLink);
    console.log('手动加载CSS:', cssLink.href);
  }
  
  // 检查JS是否已加载
  if (!isResourceLoaded('script', 'src', '/static/js/fixed_main.js')) {
    const scriptTag = document.createElement('script');
    scriptTag.src = '/static/js/fixed_main.js';
    document.body.appendChild(scriptTag);
    console.log('手动加载JS:', scriptTag.src);
  }
}

/**
 * 检查资源是否已加载
 */
function isResourceLoaded(tagName, attrName, value) {
  const elements = document.querySelectorAll(tagName);
  for (let i = 0; i < elements.length; i++) {
    const element = elements[i];
    if (element.getAttribute(attrName) === value) {
      return true;
    }
  }
  return false;
}

/**
 * 初始化音乐游戏
 */
function initMusicGame() {
  // 检查音乐游戏容器是否存在
  const gameContainer = document.getElementById('music-game-container');
  if (!gameContainer) return;
  
  // 创建音乐游戏画布
  const canvas = document.createElement('canvas');
  canvas.id = 'musicCanvas';
  canvas.width = gameContainer.clientWidth;
  canvas.height = 300;
  canvas.style.display = 'block';
  canvas.style.backgroundColor = '#111';
  canvas.style.borderRadius = '8px';
  canvas.style.boxShadow = '0 4px 15px rgba(138, 43, 226, 0.3)';
  
  // 添加到容器
  gameContainer.appendChild(canvas);
  
  console.log('音乐游戏画布已创建');
  
  // 如果window.app存在，强制更新
  if (window.app && typeof window.app.$forceUpdate === 'function') {
    window.app.$forceUpdate();
    console.log('已强制更新Vue应用');
  }
} 