/**
 * 闊充箰鎺ㄨ崘绯荤粺涓籎avaScript鏂囦欢
 * 鍖呭惈Vue.js搴旂敤鍒濆鍖栧拰鏍稿績鍔熻兘瀹炵幇
 */

// 绛夊緟椤甸潰鍔犺浇瀹屾垚
document.addEventListener('DOMContentLoaded', function() {
  console.log('闊充箰鎺ㄨ崘绯荤粺搴旂敤宸插垵濮嬪寲');
  
  // 鍏ㄥ眬浜嬩欢鎬荤嚎锛岀敤浜庣粍浠堕棿閫氫俊
  const EventBus = new Vue();
  
  // Vue.js搴旂敤瀹炰緥
  const app = new Vue({
    el: '#app',
    
    // 鏁版嵁灞烇拷?
    data: {
      // 搴旂敤鐘讹拷?
      currentTab: 'welcome',
      isLoading: false,
      isLoadingRecommendations: false,
      currentLanguage: 'zh',
      isDeveloperMode: false,
      isRegistering: false,
      loginUsername: "",
      loginPassword: "",
      loginEmail: "",
      showLoginForm: true,
      showWelcomeOptions: false,
      showPreferencesSurvey: false,
      surveyCompleted: false,
      
      // 鐢ㄦ埛鐩稿叧
      username: '',
      email: '',
      password: '',
      loginErrorMessage: '',
      loginForm: {
        username: '',
        email: '',
        password: ''
      },
      
      // 闊充箰鍋忓ソ閫夐」
      musicGenres: [
        '娴佽', '鎽囨粴', '鍢诲搱', '鐢靛瓙', '鐖靛＋', '鍙ゅ吀', 
        'R&B', '涔℃潙', '姘戣埃', '閲戝睘', '钃濊皟', '涓栫晫闊充箰',
        '鐙珛', '瀹為獙', '姘涘洿', '鏈嬪厠', '闆烽', '鐏甸瓊',
        '鏀惧厠', '杩柉锟?, '甯冮瞾锟?, '鎷変竵', '鑻变鸡', '鍙︾被'
      ],
      selectedGenres: [],
      
      // 鎺ㄨ崘绠楁硶璁剧疆
      recommendationAlgorithms: {
        hybrid: {
          name: '娣峰悎鎺ㄨ崘',
          description: '缁撳悎SVD++銆丯CF銆丮LP绠楁硶鍜屽崗鍚岃繃婊ゆ彁渚涙洿鍑嗙‘鐨勬帹锟?,
          selected: true
        },
        collaborative: {
          name: '鍗忓悓杩囨护',
          description: '鍩轰簬鐢ㄦ埛琛屼负鍜屽亸濂界殑鐩镐技鎬ф帹鑽愰煶锟?,
          selected: false
        },
        svdpp: {
          name: 'SVD++绠楁硶',
          description: '鍩轰簬鐭╅樀鍒嗚В鐨勬帹鑽愮畻锟?,
          selected: false
        },
        content: {
          name: '鍐呭鎺ㄨ崘',
          description: '鍩轰簬闊充箰鐗瑰緛鎺ㄨ崘鐩镐技椋庢牸鐨勬瓕锟?,
          selected: false
        }
      },
      
      // 闊充箰璋冩煡闂嵎
      surveyQuestions: [
        {
          id: 'music_genres',
          question: '鎮ㄥ枩娆㈠摢浜涢煶涔愮被鍨嬶紵',
          type: 'multiple',
          options: [], // 灏嗗湪鍒濆鍖栨椂濉厖
          answer: []
        },
        {
          id: 'listening_frequency',
          question: '鎮ㄥ涔呭惉涓€娆￠煶涔愶紵',
          type: 'single',
          options: ['姣忓ぉ', '姣忓懆鍑犳', '鍋跺皵', '寰堝皯'],
          answer: ''
        },
        {
          id: 'preferred_era',
          question: '鎮ㄥ亸濂藉摢涓勾浠ｇ殑闊充箰锟?,
          type: 'multiple',
          options: ['60骞翠唬', '70骞翠唬', '80骞翠唬', '90骞翠唬', '2000骞翠唬', '2010骞翠唬', '2020骞翠唬'],
          answer: []
        },
        {
          id: 'mood_preference',
          question: '鎮ㄩ€氬父鍦ㄤ粈涔堝績鎯呬笅鍚煶涔愶紵',
          type: 'multiple',
          options: ['鏀炬澗锟?, '宸ヤ綔/瀛︿範锟?, '杩愬姩锟?, '绀句氦锟?, '浼ゅ績锟?, '寮€蹇冩椂'],
          answer: []
        },
        {
          id: 'discovery_preference',
          question: '鎮ㄦ洿鍠滄鍙戠幇鏂伴煶涔愯繕鏄惉鐔熸倝鐨勬瓕鏇诧紵',
          type: 'single',
          options: ['鎬绘槸瀵绘壘鏂伴煶锟?, '鍋跺皵灏濊瘯鏂伴煶锟?, '涓昏鍚啛鎮夌殑姝屾洸', '鍙惉鎴戝凡鐭ョ殑姝屾洸'],
          answer: ''
        }
      ],
      
      // 绠＄悊鍛樺姛鑳界浉锟?
      allUsers: [],
      newUser: {
        username: '',
        email: '',
        password: '',
        isDeveloper: false
      },
      editingUser: null,
      
      // 鍐呭瀵硅薄 - 淇妯℃澘涓娇鐢ㄧ殑content瀵硅薄
      content: {
        welcome: {
          title: '娆㈣繋浣跨敤闊充箰鎺ㄨ崘绯荤粺',
          subtitle: '閫夋嫨涓嬮潰鐨勯€夐」寮€濮嬫偍鐨勯煶涔愪箣锟?,
          talkToAI: '涓嶢I闊充箰鍔╂墜鑱婂ぉ',
          talkToAIDesc: '鍚慉I鍔╂墜璇㈤棶闊充箰鎺ㄨ崘銆佽壓鏈淇℃伅鎴栬〃杈炬偍鐨勬儏缁紝鑾峰彇涓€у寲闊充箰寤鸿',
          fillQuestionnaire: '濉啓闊充箰闂嵎',
          fillQuestionnaireDesc: '閫氳繃瀵规瓕鏇茶瘎鍒嗭紝甯姪鎴戜滑浜嗚В鎮ㄧ殑闊充箰鍋忓ソ锛岃幏鍙栨洿绮惧噯鐨勬帹锟?
        },
        rate: {
          title: '瀵规瓕鏇茶瘎锟?,
          subtitle: '璇峰浠ヤ笅姝屾洸杩涜璇勫垎锛屼互甯姪鎴戜滑浜嗚В鎮ㄧ殑闊充箰鍝佸懗',
          notRated: '鏈瘎锟?,
          continueButton: '鑾峰彇鎺ㄨ崘',
          needMoreRatings: '璇疯嚦灏戝5棣栨瓕鏇茶瘎锟?
        },
        recommend: {
          title: '涓€у寲鎺ㄨ崘',
          subtitle: '鏍规嵁鎮ㄧ殑璇勫垎锛屾垜浠帹鑽愪互涓嬫瓕锟?,
          loading: '姝ｅ湪涓烘偍鐢熸垚鎺ㄨ崘...',
          noRecommendations: '鏆傛棤鎺ㄨ崘锛岃鍏堣瘎鍒嗘洿澶氭瓕锟?,
          rateMore: '杩斿洖璇勫垎鏇村姝屾洸'
        },
        chat: {
          title: 'AI闊充箰鍔╂墜',
          subtitle: '涓嶢I鍔╂墜浜ゆ祦锛岃幏鍙栭煶涔愭帹鑽愬拰淇℃伅',
          welcome: '浣犲ソ锛佹垜鏄綘鐨凙I闊充箰鍔╂墜銆傛垜鍙互甯綘鎵炬瓕鏇层€佷簡瑙ｈ壓鏈銆佽幏鍙栨帹鑽愶紝鎴栬€呭洖绛旈煶涔愮浉鍏抽棶棰樸€傝闅忔椂鍚戞垜鎻愰棶锟?,
          inputPlaceholder: '杈撳叆鎮ㄧ殑闂鎴栬锟?..'
        },
        evaluate: {
          title: '绯荤粺璇勪及',
          subtitle: '璇峰鎺ㄨ崘绯荤粺杩涜璇勪环锛屽府鍔╂垜浠敼锟?,
          submit: '鎻愪氦璇勪及',
          thanks: '鎰熻阿鎮ㄧ殑鍙嶉锟?,
          select: '璇烽€夋嫨',
          rating: {
            veryDissatisfied: '闈炲父涓嶆弧锟?,
            dissatisfied: '涓嶆弧锟?,
            neutral: '涓€锟?,
            satisfied: '婊℃剰',
            verySatisfied: '闈炲父婊℃剰'
          },
          feedback: '鍏朵粬寤鸿鎴栨剰锟?,
          feedbackPlaceholder: '璇疯緭鍏ユ偍鐨勫缓璁垨鎰忚...',
          submitButton: '鎻愪氦璇勪环',
          thankYou: '鎰熻阿鎮ㄧ殑鍙嶉锟?
        },
        header: {
          title: '娆㈣繋锟?,
          subtitle: '鎺㈢储鎮ㄧ殑涓撳睘闊充箰涓栫晫',
          logout: '閫€锟?
        },
        tabs: {
          welcome: '娆㈣繋',
          rate: '璇勫垎',
          recommend: '鎺ㄨ崘',
          chat: '鑱婂ぉ',
          evaluate: '璇勪及'
        },
        footer: {
          title: '闊充箰鎺ㄨ崘绯荤粺',
          description: '涓€涓熀浜嶢I鐨勪釜鎬у寲闊充箰鎺ㄨ崘绯荤粺'
        },
        errors: {
          emptyUsername: '璇疯緭鍏ョ敤鎴峰悕',
          loginFailed: '鐧诲綍澶辫触锛岃妫€鏌ユ偍鐨勭敤鎴峰悕鍜屽瘑锟?
        },
        success: {
          login: '鐧诲綍鎴愬姛锟?
        }
      },
      
      // 缈昏瘧瀵硅薄
      translationsZh: {
        welcome: '娆㈣繋浣跨敤闊充箰鎺ㄨ崘绯荤粺',
        description: '鎺㈢储涓€у寲闊充箰鎺ㄨ崘',
        login: '鐧诲綍',
        register: '娉ㄥ唽',
        userId: '鐢ㄦ埛ID',
        username: '鐢ㄦ埛锟?,
        password: '瀵嗙爜',
        submit: '鎻愪氦',
        cancel: '鍙栨秷',
        rate: '璇勫垎',
        recommend: '鎺ㄨ崘',
        chat: '鑱婂ぉ',
        evaluate: '璇勪及',
        moreInfo: '鏇村淇℃伅',
        rateThisSong: '涓鸿繖棣栨瓕璇勫垎',
        similar: '鐩镐技姝屾洸',
        artist: '鑹烘湳锟?,
        album: '涓撹緫',
        releaseDate: '鍙戣鏃ユ湡',
        popularity: '娴佽锟?,
        listen: '鏀跺惉',
        chatWithAI: '涓嶢I闊充箰鍔╂墜鑱婂ぉ',
        sendMessage: '鍙戦€佹秷锟?,
        typeMessage: '杈撳叆娑堟伅...',
        loading: '鍔犺浇锟?..',
        error: '鍑洪敊锟?,
        retry: '閲嶈瘯',
        noResults: '娌℃湁缁撴灉',
        welcome_message: '浣犲ソ锛佹垜鏄綘鐨凙I闊充箰鍔╂墜銆傛垜鍙互甯綘鎵炬瓕鏇层€佷簡瑙ｈ壓鏈銆佽幏鍙栨帹鑽愶紝鎴栬€呭洖绛旈煶涔愮浉鍏抽棶棰樸€傝闅忔椂鍚戞垜鎻愰棶锟?,
        developer: '寮€鍙戯拷?,
        logout: '閫€鍑虹櫥锟?,
        enterUsername: '璇疯緭鍏ョ敤鎴峰悕',
        enterEmail: '璇疯緭鍏ラ偖锟?,
        enterPassword: '璇疯緭鍏ュ瘑锟?
      },
      translationsEn: {
        welcome: 'Welcome to Music Recommendation System',
        description: 'Explore personalized music recommendations',
        login: 'Login',
        register: 'Register',
        userId: 'User ID',
        username: 'Username',
        password: 'Password',
        submit: 'Submit',
        cancel: 'Cancel',
        rate: 'Rate',
        recommend: 'Recommend',
        chat: 'Chat',
        evaluate: 'Evaluate',
        moreInfo: 'More Info',
        rateThisSong: 'Rate this song',
        similar: 'Similar Songs',
        artist: 'Artist',
        album: 'Album',
        releaseDate: 'Release Date',
        popularity: 'Popularity',
        listen: 'Listen',
        chatWithAI: 'Chat with AI Music Assistant',
        sendMessage: 'Send Message',
        typeMessage: 'Type a message...',
        loading: 'Loading...',
        error: 'Error',
        retry: 'Retry',
        noResults: 'No Results',
        welcome_message: 'Hello! I am your AI music assistant. I can help you find songs, learn about artists, get recommendations, or answer music-related questions. Feel free to ask me anything!',
        developer: 'Developer',
        logout: 'Logout',
        welcome_title: 'Welcome to Music Recommendation System',
        welcome_subtitle: 'Choose an option below to start your music journey',
        welcome_talkToAI: 'Chat with AI Music Assistant',
        welcome_talkToAIDesc: 'Ask the AI assistant for music recommendations, artist information, or express your emotions for personalized music suggestions',
        welcome_fillQuestionnaire: 'Fill Music Questionnaire',
        welcome_fillQuestionnaireDesc: 'Rate songs to help us understand your music preferences and get more accurate recommendations'
      },
      
      // 璇勪及闂
      evaluationQuestions: [
        { id: 'recommendation_quality', text: '鎮ㄥ绯荤粺鐨勬帹鑽愯川閲忔弧鎰忓悧锟? },
        { id: 'ui_experience', text: '鎮ㄥ绯荤粺鐨勭敤鎴风晫闈綋楠屾弧鎰忓悧锟? },
        { id: 'overall_satisfaction', text: '鎮ㄥ绯荤粺鐨勬暣浣撲綋楠屾弧鎰忓悧锟? }
      ],
      evaluationResponses: [],
      evaluationSubmitted: false,
      
      // 鐢ㄦ埛淇℃伅
      user: {
        id: null,
        username: '',
        email: '',
        isLoggedIn: false,
        isDeveloper: false
      },
      
      // 闊充箰鏁版嵁
      sampleSongs: [
        {
          track_id: '1',
          track_name: '鏅村ぉ',
          artist_name: '鍛ㄦ澃锟?,
          title: '鏅村ぉ',
          artist: '鍛ㄦ澃锟?,
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: '2',
          track_name: 'Shape of You',
          artist_name: 'Ed Sheeran',
          title: 'Shape of You',
          artist: 'Ed Sheeran',
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: '3',
          track_name: '婕斿憳',
          artist_name: '钖涗箣锟?,
          title: '婕斿憳',
          artist: '钖涗箣锟?,
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: '4',
          track_name: 'Uptown Funk',
          artist_name: 'Mark Ronson ft. Bruno Mars',
          title: 'Uptown Funk',
          artist: 'Mark Ronson ft. Bruno Mars',
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: '5',
          track_name: '婕傛磱杩囨捣鏉ョ湅锟?,
          artist_name: '鍒樻槑锟?,
          title: '婕傛磱杩囨捣鏉ョ湅锟?,
          artist: '鍒樻槑锟?,
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
      ],
      userRatings: {},
      recommendations: [
        {
          track_id: 'rec1',
          track_name: '鐖辨儏杞Щ',
          artist_name: '闄堝锟?,
          title: '鐖辨儏杞Щ',
          artist: '闄堝锟?,
          explanation: '鍩轰簬鎮ㄥ鍛ㄦ澃浼︾殑鍠滃ソ鎺ㄨ崘',
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: 'rec2',
          track_name: 'Thinking Out Loud',
          artist_name: 'Ed Sheeran',
          title: 'Thinking Out Loud',
          artist: 'Ed Sheeran',
          explanation: '涓庢偍鍠滄锟?Shape of You 椋庢牸鐩镐技',
          album_image: '/static/img/default-album.png',
          preview_url: null
        },
        {
          track_id: 'rec3',
          track_name: '涓戝叓锟?,
          artist_name: '钖涗箣锟?,
          title: '涓戝叓锟?,
          artist: '钖涗箣锟?,
          explanation: '鏉ヨ嚜鎮ㄥ枩娆㈢殑鑹烘湳瀹惰枦涔嬭唉',
          album_image: '/static/img/default-album.png',
          preview_url: null
        }
      ],
      
      // 鑱婂ぉ鐩稿叧
      chatMessage: '',
      chatHistory: [],
      chatMessages: [],
      chatInput: '',
      isTyping: false,
      
      // 璇勪及鏁版嵁
      satisfactionLevel: 0,
      feedbackText: '',
      evaluationComment: '',
      
      // 绯荤粺娑堟伅
      notification: {
        message: '',
        type: 'info',
        isVisible: false
      },
      
      // 閫氱煡鍒楄〃
      notifications: [],
      
      // 闊抽鎾斁
      currentPreviewUrl: null,
      audioPlayer: null,
      
      // 鎯呯华鍒嗘瀽鐩稿叧
      emotionKeywords: [
        '闅捐繃', '浼ゅ績', '鎮蹭激', '鍘嬪姏', '鐒﹁檻', '寮€锟?, '楂樺叴', '鍏村', 
        '鐢熸皵', '鎰わ拷?, '鏃犺亰', '鐤叉儷', '瀛ょ嫭', '鎬濆康', '澶辫惤', 
        '鎯冲摥', '涓嶅紑锟?, '鎶戦儊', '鐑﹁簛', '蹇冩儏'
      ],
      lastEmotionAnalysis: null,
      isEmotionAnalysing: false,
      
      translations: {
        zh: {
          appTitle: '娣卞害鎺ㄨ崘闊充箰',
          recommendations: '鎺ㄨ崘',
          chat: '鑱婂ぉ',
          evaluation: '璇勪环',
          developer: '寮€鍙戣€呮ā锟?,
          logout: '閫€鍑虹櫥锟?
        },
        en: {
          appTitle: 'Deep Recommend Music',
          recommendations: 'Recommendations',
          chat: 'Chat',
          evaluation: 'Evaluation',
          developer: 'Developer Mode',
          logout: 'Log Out'
        }
      },
      
      // 搴曢儴鎻忚堪鏇存柊浠ユ彁鍙婃贩鍚堟帹鑽愮畻锟?
      footerDescription: {
        zh: '鏈郴缁熼噰鐢ㄦ贩鍚堟帹鑽愮畻锟?(SVD++, NCF, MLP 鍜屽崗鍚岃繃锟?锛岀粨鍚堝唴瀹瑰垎鏋愬拰鐢ㄦ埛琛屼负锛屾彁渚涗釜鎬у寲闊充箰鎺ㄨ崘锟?,
        en: 'This system uses a hybrid recommendation algorithm (SVD++, NCF, MLP, and Collaborative Filtering), combining content analysis and user behavior to provide personalized music recommendations.'
      },
      
      // 鐢ㄦ埛鍋忓ソ
      preferences: [],
      
      // 榛樿鎺ㄨ崘姝屾洸
      defaultRecommendations: [
        {
          track_id: 'default_1',
          track_name: '鍗冮噷涔嬪',
          artist_name: '鍛ㄦ澃锟?,
          title: '鍗冮噷涔嬪',
          artist: '鍛ㄦ澃锟?,
          explanation: '鐑棬鍗庤姝屾洸鎺ㄨ崘'
        },
        {
          track_id: 'default_2',
          track_name: '璧烽锟?,
          artist_name: '涔拌荆妞掍篃鐢ㄥ埜',
          title: '璧烽锟?,
          artist: '涔拌荆妞掍篃鐢ㄥ埜',
          explanation: '杩戞湡娴佽姝屾洸鎺ㄨ崘'
        }
      ]
    },
    
    // 璁＄畻灞烇拷?
    computed: {
      // 宸茶瘎鍒嗙殑姝屾洸鏁伴噺
      ratedSongsCount() {
        return Object.keys(this.userRatings).length;
      },
      
      // 鐢ㄦ埛鏄惁鍙互鑾峰彇鎺ㄨ崘
      canGetRecommendations() {
        return this.ratedSongsCount >= 5 && this.user.isLoggedIn;
      },
      
      // 鐢ㄦ埛鏄惁宸茬粡璇勫垎瓒冲鐨勬瓕锟?
      hasRatedEnoughSongs() {
        return this.ratedSongsCount >= 5;
      },
      
      // 鏄剧ず璇█
      t() {
        return (key) => {
          const translations = this.currentLanguage === 'zh' ? this.translationsZh : this.translationsEn;
          return translations[key] || key;
        };
      },
      
      // 璇勪及鏄惁瀹屾垚
      isEvaluationComplete() {
        return this.evaluationResponses.filter(r => r !== '').length === this.evaluationQuestions.length;
      },
      
      // 鐧诲綍鐘讹拷?(鍏煎妯℃澘涓殑鍙橀噺锟?
      isLoggedIn() {
        return this.user.isLoggedIn;
      }
    },
    
    // 鐩戝惉灞炴€у彉锟?
    watch: {
      currentLanguage(newVal) {
        localStorage.setItem('preferredLanguage', newVal);
      },
      'user.isLoggedIn'(newValue) {
        if (newValue) {
          // 淇濆瓨鐢ㄦ埛ID鍒版湰鍦板瓨锟?
          localStorage.setItem('musicRecommendUserId', this.user.id);
        }
      }
    },
    
    // 鍒涘缓鏃舵墽锟?
    created() {
      // 鍒濆鍖栨儏缁叧閿瘝鍒楄〃
      this.emotionKeywords = [
          '闅捐繃', '浼ゅ績', '鎮蹭激', '鍘嬪姏', '鐒﹁檻', '寮€锟?, '楂樺叴', '鍏村', 
          '鐢熸皵', '鎰わ拷?, '鏃犺亰', '鐤叉儷', '瀛ょ嫭', '鎬濆康', '澶辫惤', 
          '鎯冲摥', '涓嶅紑锟?, '鎶戦儊', '鐑﹁簛', '蹇冩儏'
      ];
      
      // 妫€鏌ョ敤鎴蜂細锟?
      this.checkUserSession();
      
      // 璁剧疆榛樿璇█
      this.currentLanguage = localStorage.getItem('language') || 'zh';
      
      // 浠庢湰鍦板瓨鍌ㄥ姞杞借瑷€鍋忓ソ
      const savedLanguage = localStorage.getItem('preferredLanguage');
      if (savedLanguage) {
        this.currentLanguage = savedLanguage;
      }
      
      // 鍒濆鍖栧姞杞界姸锟?
      this.isLoading = false;
      this.isLoadingRecommendations = false;
      
      // 璁剧疆translations瀵硅薄
      if (!this.translations || !this.translations.zh || !this.translations.zh.welcome) {
        // 纭繚translations瀵硅薄瀹屾暣
        this.translations = {
          zh: {
            appTitle: '娣卞害鎺ㄨ崘闊充箰',
            recommendations: '鎺ㄨ崘',
            chat: '鑱婂ぉ',
            evaluation: '璇勪环',
            developer: '寮€鍙戣€呮ā锟?,
            logout: '閫€鍑虹櫥锟?,
            tabs: {
              welcome: '娆㈣繋',
              rate: '璇勫垎',
              recommend: '鎺ㄨ崘',
              chat: '鑱婂ぉ',
              evaluate: '璇勪及',
              admin: '鐢ㄦ埛绠＄悊'
            },
            welcome: {
              title: '娆㈣繋浣跨敤闊充箰鎺ㄨ崘绯荤粺',
              subtitle: '閫夋嫨涓嬮潰鐨勯€夐」寮€濮嬫偍鐨勯煶涔愪箣锟?,
              talkToAI: '涓嶢I闊充箰鍔╂墜鑱婂ぉ',
              talkToAIDesc: '鍚慉I鍔╂墜璇㈤棶闊充箰鎺ㄨ崘銆佽壓鏈淇℃伅鎴栬〃杈炬偍鐨勬儏缁紝鑾峰彇涓€у寲闊充箰寤鸿',
              fillQuestionnaire: '濉啓闊充箰闂嵎',
              fillQuestionnaireDesc: '閫氳繃瀵规瓕鏇茶瘎鍒嗭紝甯姪鎴戜滑浜嗚В鎮ㄧ殑闊充箰鍋忓ソ锛岃幏鍙栨洿绮惧噯鐨勬帹锟?
            }
          },
          en: {
            appTitle: 'Deep Recommend Music',
            recommendations: 'Recommendations',
            chat: 'Chat',
            evaluation: 'Evaluation',
            developer: 'Developer Mode',
            logout: 'Log Out',
            tabs: {
              welcome: 'Welcome',
              rate: 'Rate',
              recommend: 'Recommend',
              chat: 'Chat',
              evaluate: 'Evaluate',
              admin: 'User Admin'
            },
            welcome: {
              title: 'Welcome to Music Recommendation System',
              subtitle: 'Choose an option below to start your music journey',
              talkToAI: 'Chat with AI Music Assistant',
              talkToAIDesc: 'Ask the AI assistant for music recommendations, artist information, or express your emotions for personalized music suggestions',
              fillQuestionnaire: 'Fill Music Questionnaire',
              fillQuestionnaireDesc: 'Rate songs to help us understand your music preferences and get more accurate recommendations'
            }
          }
        };
      }
      
      // 璁剧疆榛樿绀轰緥姝屾洸锛岄槻姝㈡覆鏌撻敊锟?
      this.sampleSongs = [
        { 
          track_id: '1', 
          track_name: '鏅村ぉ', 
          artist_name: '鍛ㄦ澃锟?, 
          album_name: '鍙舵儬锟?,
          title: '鏅村ぉ',
          artist: '鍛ㄦ澃锟?,
          rating: 5,  // 榛樿璇勫垎锟?
          album_image: '/static/img/default-album.png'
        },
        { 
          track_id: '2', 
          track_name: 'Shape of You', 
          artist_name: 'Ed Sheeran', 
          album_name: 'Divide',
          title: 'Shape of You',
          artist: 'Ed Sheeran',
          rating: 4,  // 榛樿璇勫垎锟?
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: '3',
          track_name: '婕斿憳',
          artist_name: '钖涗箣锟?,
          album_name: '缁呭＋',
          title: '婕斿憳',
          artist: '钖涗箣锟?,
          rating: 5,
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: '4',
          track_name: 'Uptown Funk',
          artist_name: 'Mark Ronson ft. Bruno Mars',
          album_name: 'Uptown Special',
          title: 'Uptown Funk',
          artist: 'Mark Ronson ft. Bruno Mars',
          rating: 4,
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: '5',
          track_name: '婕傛磱杩囨捣鏉ョ湅锟?,
          artist_name: '鍒樻槑锟?,
          album_name: '婕傛磱杩囨捣鏉ョ湅锟?,
          title: '婕傛磱杩囨捣鏉ョ湅锟?,
          artist: '鍒樻槑锟?,
          rating: 5,
          album_image: '/static/img/default-album.png'
        }
      ];
      
      // 璁剧疆榛樿鎺ㄨ崘鏁版嵁锛岄槻姝㈡覆鏌撻敊锟?
      this.recommendations = [
        {
          track_id: 'rec1',
          track_name: '鍛婄櫧姘旂悆',
          artist_name: '鍛ㄦ澃锟?,
          explanation: '鏍规嵁鎮ㄥ枩娆㈢殑鍛ㄦ澃浼︾殑浣滃搧鎺ㄨ崘',
          title: '鍛婄櫧姘旂悆',
          artist: '鍛ㄦ澃锟?,
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: 'rec2',
          track_name: 'Perfect',
          artist_name: 'Ed Sheeran',
          explanation: '涓庢偍鍠滄鐨凷hape of You椋庢牸鐩镐技',
          title: 'Perfect',
          artist: 'Ed Sheeran',
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: 'rec3',
          track_name: '鍏夊勾涔嬪',
          artist_name: '閭撶传锟?,
          explanation: '鍩轰簬鎮ㄧ殑娴佽闊充箰鍋忓ソ鎺ㄨ崘',
          title: '鍏夊勾涔嬪',
          artist: '閭撶传锟?,
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: 'rec4',
          track_name: 'Thinking Out Loud',
          artist_name: 'Ed Sheeran',
          explanation: '涓庢偍鍠滄鐨凷hape of You鐨勮壓鏈鐩稿悓',
          title: 'Thinking Out Loud',
          artist: 'Ed Sheeran',
          album_image: '/static/img/default-album.png'
        },
        {
          track_id: 'rec5',
          track_name: '瀹夐潤',
          artist_name: '鍛ㄦ澃锟?,
          explanation: '鏍规嵁鎮ㄥ枩娆㈢殑鍛ㄦ澃浼︾殑浣滃搧鎺ㄨ崘',
          title: '瀹夐潤',
          artist: '鍛ㄦ澃锟?,
          album_image: '/static/img/default-album.png'
        }
      ];
      
      // 鍒濆鍖栧繀瑕佺殑鏁版嵁锛岄伩鍏島ndefined閿欒
      this.chatMessages = [];
      if (!Array.isArray(this.chatMessages)) {
        this.chatMessages = [];
      }
      
      // 纭繚鍐呭瀵硅薄瀛樺湪
      if (!this.content) {
        this.content = {
          welcome: {
            title: '娆㈣繋浣跨敤闊充箰鎺ㄨ崘绯荤粺',
            subtitle: '閫夋嫨涓嬮潰鐨勯€夐」寮€濮嬫偍鐨勯煶涔愪箣锟?,
            talkToAI: '涓嶢I闊充箰鍔╂墜鑱婂ぉ',
            talkToAIDesc: '鍚慉I鍔╂墜璇㈤棶闊充箰鎺ㄨ崘銆佽壓鏈淇℃伅鎴栬〃杈炬偍鐨勬儏缁紝鑾峰彇涓€у寲闊充箰寤鸿',
            fillQuestionnaire: '濉啓闊充箰闂嵎',
            fillQuestionnaireDesc: '閫氳繃瀵规瓕鏇茶瘎鍒嗭紝甯姪鎴戜滑浜嗚В鎮ㄧ殑闊充箰鍋忓ソ锛岃幏鍙栨洿绮惧噯鐨勬帹锟?
          },
          rate: {
            title: '瀵规瓕鏇茶瘎锟?,
            subtitle: '璇峰浠ヤ笅姝屾洸杩涜璇勫垎锛屼互甯姪鎴戜滑浜嗚В鎮ㄧ殑闊充箰鍝佸懗',
            notRated: '鏈瘎锟?,
            continueButton: '鑾峰彇鎺ㄨ崘',
            needMoreRatings: '璇疯嚦灏戝5棣栨瓕鏇茶瘎锟?
          },
          recommend: {
            title: '涓€у寲鎺ㄨ崘',
            subtitle: '鏍规嵁鎮ㄧ殑璇勫垎锛屾垜浠帹鑽愪互涓嬫瓕锟?,
            loading: '姝ｅ湪涓烘偍鐢熸垚鎺ㄨ崘...',
            noRecommendations: '鏆傛棤鎺ㄨ崘锛岃鍏堣瘎鍒嗘洿澶氭瓕锟?,
            rateMore: '杩斿洖璇勫垎鏇村姝屾洸'
          },
          chat: {
            title: 'AI闊充箰鍔╂墜',
            subtitle: '涓嶢I鍔╂墜浜ゆ祦锛岃幏鍙栭煶涔愭帹鑽愬拰淇℃伅',
            welcome: '浣犲ソ锛佹垜鏄綘鐨凙I闊充箰鍔╂墜銆傛垜鍙互甯綘鎵炬瓕鏇层€佷簡瑙ｈ壓鏈銆佽幏鍙栨帹鑽愶紝鎴栬€呭洖绛旈煶涔愮浉鍏抽棶棰樸€傝闅忔椂鍚戞垜鎻愰棶锟?,
            inputPlaceholder: '杈撳叆鎮ㄧ殑闂鎴栬锟?..'
          },
          evaluate: {
            title: '绯荤粺璇勪及',
            subtitle: '璇峰鎺ㄨ崘绯荤粺杩涜璇勪环锛屽府鍔╂垜浠敼锟?,
            submit: '鎻愪氦璇勪及',
            thanks: '鎰熻阿鎮ㄧ殑鍙嶉锟?
          },
          header: {
            title: '娆㈣繋锟?,
            subtitle: '鎺㈢储鎮ㄧ殑涓撳睘闊充箰涓栫晫',
            logout: '閫€锟?
          },
          tabs: {
            welcome: '娆㈣繋',
            rate: '璇勫垎',
            recommend: '鎺ㄨ崘',
            chat: '鑱婂ぉ',
            evaluate: '璇勪及'
          },
          footer: {
            title: '闊充箰鎺ㄨ崘绯荤粺',
            description: '涓€涓熀浜嶢I鐨勪釜鎬у寲闊充箰鎺ㄨ崘绯荤粺'
          },
          errors: {
            emptyUsername: '璇疯緭鍏ョ敤鎴峰悕',
            loginFailed: '鐧诲綍澶辫触锛岃妫€鏌ユ偍鐨勭敤鎴峰悕鍜屽瘑锟?
          },
          success: {
            login: '鐧诲綍鎴愬姛锟?
          }
        };
      }
      
      // 鍒濆鍖栨儏缁叧閿瘝鏁扮粍
      if (!this.emotionKeywords || !Array.isArray(this.emotionKeywords)) {
        this.emotionKeywords = [
          '闅捐繃', '浼ゅ績', '鎮蹭激', '鍘嬪姏', '鐒﹁檻', '寮€锟?, '楂樺叴', '鍏村', 
          '鐢熸皵', '鎰わ拷?, '鏃犺亰', '鐤叉儷', '瀛ょ嫭', '鎬濆康', '澶辫惤', 
          '鎯冲摥', '涓嶅紑锟?, '鎶戦儊', '鐑﹁簛', '蹇冩儏'
        ];
      }
      
      // 纭繚鍏朵粬鍙橀噺鏈夐粯璁わ拷?
      this.chatInput = '';
      this.notifications = [];
      this.evaluationResponses = [];
      this.currentPreviewUrl = null;
      
      // 棰勮鐢ㄦ埛璇勫垎鏁版嵁
      this.userRatings = {
        '1': 5, // 鏅村ぉ璇勫垎锟?
        '2': 4, // Shape of You璇勫垎锟?
        '3': 5, // 婕斿憳璇勫垎锟?
        '4': 4, // Uptown Funk璇勫垎锟?
        '5': 5  // 婕傛磱杩囨捣鏉ョ湅浣犺瘎鍒嗕负5
      };
      
      // 娣诲姞AI娆㈣繋娑堟伅
      this.chatMessages.push({
        content: `娆㈣繋锟?{this.user.username}锛佹垜鏄偍鐨凙I闊充箰鍔╂墜銆傝闂偍鎯充簡瑙ｄ粈涔堥煶涔愪俊鎭垨鑾峰彇浠€涔堟帹鑽愶紵`,
        isUser: false
      });
      
      // 鏄剧ず娆㈣繋閫氱煡
      this.showNotification('娆㈣繋浣跨敤闊充箰鎺ㄨ崘绯荤粺锛佹垜浠凡涓烘偍鍑嗗浜嗕竴浜涙帹鑽愶拷?, 'success');
      
      // 鑷姩鍔犺浇绀轰緥姝屾洸锛堜絾涓嶈皟鐢ˋPI锟?
      console.log('宸插姞杞界ず渚嬫瓕锟?', this.sampleSongs.length);
      console.log('宸茶缃敤鎴疯瘎锟?', Object.keys(this.userRatings).length);
      
      // 瀹氭椂鍔ㄧ敾鏁堟灉
      setTimeout(() => {
        const cards = document.querySelectorAll('.recommendation-item');
        if (cards && cards.length > 0) {
          cards.forEach((card, index) => {
            setTimeout(() => {
              card.classList.add('animate__animated', 'animate__fadeInUp');
            }, index * 100);
          });
        }
      }, 500);
    },
    
    // 鏂规硶瀹氫箟
    methods: {
      // 鑾峰彇瀵瑰簲璇█鐨勭炕锟?
      getTranslation(key) {
        const translations = this.currentLanguage === 'zh' ? 
          (this.translations.zh || this.translationsZh) : 
          (this.translations.en || this.translationsEn);
        return translations[key] || key;
      },
      
      // 鍒囨崲璇█
      switchLanguage(lang) {
        if (lang) {
          this.currentLanguage = lang;
        } else {
          // 濡傛灉娌℃湁鎻愪緵鍙傛暟锛屽垯鍒囨崲璇█
          this.currentLanguage = this.currentLanguage === 'zh' ? 'en' : 'zh';
        }
        document.documentElement.lang = this.currentLanguage;
      },
      
      // 鍒囨崲鏍囩锟?
      switchTab(tab) {
        if (tab === 'welcome') {
          console.log("鍒囨崲鍒版杩庨〉锟?);
        }
        this.currentTab = tab;
        
        // 濡傛灉鍒囨崲鍒版帹鑽愭爣绛鹃〉锛岃幏鍙栨帹锟?
        if (tab === 'recommend' && this.canGetRecommendations) {
          this.getRecommendations();
        }
        
        // 濡傛灉鍒囨崲鍒拌亰澶╂爣绛鹃〉锛屽姞杞借亰澶╁巻锟?
        if (tab === 'chat' && this.user.isLoggedIn) {
          this.getChatHistory();
        }
      },
      
      // 妫€鏌ョ敤鎴蜂細锟?
      checkUserSession() {
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
          this.user = JSON.parse(storedUser);
        }
      },
      
      // 鍔犺浇鍒濆鏁版嵁
      loadInitialData() {
        // 鍔犺浇绀轰緥姝屾洸
        this.loadSampleSongs();
        
        // 濡傛灉鐢ㄦ埛宸茬櫥褰曪紝鑾峰彇鐢ㄦ埛璇勫垎璁板綍
        if (this.user.isLoggedIn) {
          this.getUserRatings();
        }
      },
      
      // 鍔犺浇绀轰緥姝屾洸
      loadSampleSongs() {
        // 浣跨敤棰勮鐨勭ず渚嬫瓕鏇叉暟鎹紝涓嶅彂閫丄PI璇锋眰
        console.log('浣跨敤棰勮鐨勭ず渚嬫瓕鏇叉暟锟?);
        
        // 璁剧疆鍔犺浇鐘讹拷?
        this.isLoading = true;
        
        // 妯℃嫙缃戠粶寤惰繜
        setTimeout(() => {
          this.isLoading = false;
          console.log('宸插姞杞界ず渚嬫瓕锟?', this.sampleSongs.length);
          
          // 濡傛灉鏈夌敤鎴疯瘎鍒嗘暟鎹紝搴旂敤鍒版瓕鏇蹭笂
          if (Object.keys(this.userRatings).length > 0) {
            this.sampleSongs.forEach(song => {
              if (this.userRatings[song.track_id]) {
                song.rating = this.userRatings[song.track_id];
              }
            });
          }
        }, 500);
      },
      
      // 鑾峰彇鐢ㄦ埛璇勫垎璁板綍
      getUserRatings() {
        if (!this.user.isLoggedIn) return;
        
        this.isLoading = true;
        
        // 浣跨敤鐢ㄦ埛鍚嶄綔涓篒D
        const userId = this.user.id || this.user.username;
        
        axios.get(`/api/user_ratings/${userId}`)
          .then(response => {
            this.userRatings = response.data || {};
            console.log('宸插姞杞界敤鎴疯瘎锟?', Object.keys(this.userRatings).length);
            
            // 鏇存柊绀轰緥姝屾洸鐨勮瘎锟?
            if (this.sampleSongs.length > 0) {
              this.sampleSongs.forEach(song => {
                if (this.userRatings[song.track_id]) {
                  this.$set(song, 'rating', this.userRatings[song.track_id]);
                }
              });
            }
          })
          .catch(error => {
            console.error('鍔犺浇鐢ㄦ埛璇勫垎澶辫触:', error);
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 璇勫垎姝屾洸
      rateSong(trackId, rating) {
        if (!this.user.isLoggedIn) {
          this.showNotification('璇峰厛鐧诲綍鍚庡啀璇勫垎', 'warning');
          return;
        }
        
        this.isLoading = true;
        
        // 鍔ㄧ敾鏁堟灉鏍囪璇ユ瓕鏇插凡璇勫垎
        const songElement = document.querySelector(`[data-track-id="${trackId}"]`);
        if (songElement) {
          songElement.classList.add('animate__animated', 'animate__pulse');
          setTimeout(() => {
            songElement.classList.remove('animate__animated', 'animate__pulse');
          }, 1000);
        }
        
        axios.post('/api/rate_song', {
          user_id: this.user.id,
          track_id: trackId,
          rating: rating
        })
          .then(response => {
            // 鏇存柊鏈湴璇勫垎璁板綍
            this.$set(this.userRatings, trackId, rating);
            console.log('姝屾洸璇勫垎鎴愬姛:', trackId, rating);
            
            // 妫€鏌ユ槸鍚﹀彲浠ヨ幏鍙栨帹锟?
            if (this.canGetRecommendations) {
              this.showNotification('鎮ㄥ凡璇勫垎瓒冲鐨勬瓕鏇诧紝鍙互鑾峰彇涓€у寲鎺ㄨ崘锟?, 'success');
            }
          })
          .catch(error => {
            console.error('姝屾洸璇勫垎澶辫触:', error);
            this.showNotification('璇勫垎澶辫触锛岃閲嶈瘯', 'danger');
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 鑾峰彇鎺ㄨ崘姝屾洸
      getRecommendations() {
        // 鎴戜滑宸茬粡鏈夐璁剧殑鎺ㄨ崘鏁版嵁锛岀洿鎺ヤ娇鐢ㄥ畠锟?
        console.log('浣跨敤棰勮鐨勬帹鑽愭暟锟?);
        
        // 璁剧疆鍔犺浇鐘讹拷?
        this.isLoadingRecommendations = true;
        
        // 妯℃嫙缃戠粶璇锋眰寤惰繜
        setTimeout(() => {
          this.isLoadingRecommendations = false;
          
          // 娣诲姞鍔ㄧ敾鏁堟灉
          setTimeout(() => {
            const cards = document.querySelectorAll('.recommendation-item');
            if (cards && cards.length > 0) {
              cards.forEach((card, index) => {
                setTimeout(() => {
                  card.classList.add('animate__animated', 'animate__fadeInUp');
                }, index * 100);
              });
            }
          }, 100);
          
          this.showNotification('宸蹭负鎮ㄧ敓鎴愭帹鑽愭瓕鏇诧紒', 'success');
        }, 1000);
      },
      
      // 鍙戦€佽亰澶╂秷锟?
      sendChatMessage() {
        if (!this.chatInput.trim()) {
            return;
        }
        
        // 娣诲姞鐢ㄦ埛娑堟伅鍒拌亰锟?
        this.addChatMessage(this.chatInput, true);
        
        // 淇濆瓨鐢ㄦ埛杈撳叆
        const userMessage = this.chatInput;
        this.chatInput = '';
        
        // 鏄剧ずAI姝ｅ湪杈撳叆鐨勭姸锟?
        this.isTyping = true;
        
        // 鍒嗘瀽娑堟伅鏄惁鍖呭惈鎯呯华鍐呭
        if (this.containsEmotionKeywords(userMessage)) {
            this.analyzeEmotionAndRecommend(userMessage);
        } else {
            this.sendRegularChatMessage(userMessage);
        }
      },
      
      /**
       * 鍙戦€佸父瑙勮亰澶╂秷鎭埌API
       */
      sendRegularChatMessage(message) {
        // 鍙戦€佹秷鎭埌鍚庣
        axios.post('/api/chat', {
            user_id: this.user.id,
            message: message
        })
        .then(response => {
            this.isTyping = false;
            
            if (response.data && response.data.response) {
                // 娣诲姞AI鍥炲鍒拌亰锟?
                this.addChatMessage(response.data.response);
                
                // 淇濆瓨鑱婂ぉ璁板綍
                this.saveChatHistory();
            } else {
                this.addChatMessage("鎶辨瓑锛屾垜鏆傛椂鏃犳硶鍥炲簲鎮ㄧ殑闂銆傝绋嶅悗鍐嶈瘯锟?);
            }
        })
        .catch(error => {
            this.isTyping = false;
            console.error('鑱婂ぉ娑堟伅鍙戦€侀敊锟?', error);
            this.addChatMessage("缃戠粶閿欒锛屾棤娉曡幏鍙栧洖澶嶃€傝妫€鏌ユ偍鐨勭綉缁滆繛鎺ュ悗鍐嶈瘯锟?);
        });
      },
      
      /**
       * 鍒嗘瀽鐢ㄦ埛鎯呯华骞舵帹鑽愰煶锟?
       */
      analyzeEmotionAndRecommend(message) {
        if (!message || !this.user || !this.user.id) {
          console.error('analyzeEmotionAndRecommend: 鍙傛暟涓嶅畬锟?);
          this.showNotification('鏃犳硶鍒嗘瀽鎯呯华锛岃绋嶅悗鍐嶈瘯', 'error');
          this.isLoading = false;
          return;
        }
        
        this.isEmotionAnalysing = true;
        
        axios.post('/api/emotion/analyze', {
          user_id: this.user.id,
          message: message
        })
          .then(response => {
            // 淇濆瓨鎯呯华鍒嗘瀽缁撴灉
            this.lastEmotionAnalysis = {
              emotion: response.data.emotion || 'neutral',
              intensity: response.data.intensity || 0.5,
              description: response.data.description || '鎮ㄧ殑鎯呯华鐘讹拷?,
              music_suggestion: response.data.music_suggestion || '閫傚悎鎮ㄥ綋鍓嶆儏缁殑闊充箰'
            };
            
            // 娣诲姞 AI 鍥炲鍒拌亰澶╁巻锟?
            this.chatMessages.push({
              content: response.data.response || '鎴戜簡瑙ｄ簡鎮ㄧ殑鎯呯华锛岃鎴戜负鎮ㄦ帹鑽愪竴浜涢€傚悎鐨勯煶涔愶拷?,
              isUser: false
            });
            
            // 濡傛灉鐢ㄦ埛褰撳墠鍦ㄨ亰澶╂爣绛撅紝鎻愮ず鍙互鍦ㄦ帹鑽愭爣绛炬煡鐪嬬浉鍏抽煶锟?
            if (this.currentTab === 'chat') {
              this.showNotification('AI鍔╂墜宸插垎鏋愭偍鐨勬儏缁苟鎺ㄨ崘浜嗛€傚悎鐨勯煶锟?, 'info');
            }
            
            // 灏濊瘯鑾峰彇鍩轰簬鎯呯华鐨勯煶涔愭帹锟?
            if (response.data.emotion) {
              this.getEmotionBasedMusic(response.data.emotion);
            }
          })
          .catch(error => {
            console.error('鎯呯华鍒嗘瀽璇锋眰鍑洪敊:', error);
            
            // 娣诲姞閿欒娑堟伅
            this.chatMessages.push({
              content: '鎶辨瓑锛屾垜鏆傛椂鏃犳硶鍒嗘瀽鎮ㄧ殑鎯呯华銆傝绋嶅悗鍐嶈瘯鎴栧皾璇曚笉鍚岀殑琛ㄨ揪鏂瑰紡锟?,
              isUser: false
            });
            
            this.showNotification('鎯呯华鍒嗘瀽澶辫触锛岃绋嶅悗鍐嶈瘯', 'danger');
          })
          .finally(() => {
            this.isLoading = false;
            this.isEmotionAnalysing = false;
            
            // 婊氬姩鍒板簳锟?
            this.$nextTick(() => {
              const chatContainer = document.querySelector('.chat-messages');
              if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
              }
            });
          });
      },
      
      /**
       * 鑾峰彇鍩轰簬鎯呯华鐨勯煶涔愭帹锟?
       */
      getEmotionBasedMusic(emotion) {
        if (!this.user || !this.user.isLoggedIn || !emotion) {
          console.error('getEmotionBasedMusic: 缂哄皯蹇呰鍙傛暟');
          return;
        }
        
        this.isLoading = true;
        
        // 纭繚鍏堣缃粯璁ゆ帹鑽愶紝闃叉undefined閿欒
        this.recommendations = [
          {
            track_id: 'default1',
            track_name: '鎯呯华鎺ㄨ崘姝屾洸1',
            artist_name: '鎯呯华鑹烘湳锟?',
            explanation: `鍩轰簬鎮ㄧ殑${emotion}鎯呯华鎺ㄨ崘`,
            title: '鎯呯华鎺ㄨ崘姝屾洸1',
            artist: '鎯呯华鑹烘湳锟?',
            album_image: '/static/img/default-album.png'
          },
          {
            track_id: 'default2',
            track_name: '鎯呯华鎺ㄨ崘姝屾洸2',
            artist_name: '鎯呯华鑹烘湳锟?',
            explanation: `鍩轰簬鎮ㄧ殑${emotion}鎯呯华鎺ㄨ崘`,
            title: '鎯呯华鎺ㄨ崘姝屾洸2',
            artist: '鎯呯华鑹烘湳锟?',
            album_image: '/static/img/default-album.png'
          }
        ];
        
        axios.get(`/api/emotion/music?user_id=${this.user.id}&emotion=${emotion}`)
          .then(response => {
            // 纭繚杩斿洖鏁版嵁鏄暟锟?
            if (response.data && Array.isArray(response.data) && response.data.length > 0) {
              // 纭繚姣忎釜鎺ㄨ崘瀵硅薄閮芥湁title鍜宎rtist瀛楁
              const processedRecommendations = response.data.map(rec => {
                // 纭繚rec鏄竴涓锟?
                if (!rec || typeof rec !== 'object') {
                  return {
                    track_id: '鏈煡ID',
                    track_name: '鏈煡鏍囬',
                    artist_name: '鏈煡鑹烘湳锟?,
                    title: '鏈煡鏍囬',
                    artist: '鏈煡鑹烘湳锟?,
                    explanation: `鍩轰簬鎮ㄧ殑${emotion}鎯呯华鎺ㄨ崘`,
                    album_image: '/static/img/default-album.png'
                  };
                }
                
                return {
                  ...rec,
                  track_id: rec.track_id || rec.id || `鏈煡ID_${Math.random()}`,
                  track_name: rec.track_name || rec.title || '鏈煡鏍囬',
                  artist_name: rec.artist_name || rec.artist || '鏈煡鑹烘湳锟?,
                  title: rec.track_name || rec.title || '鏈煡鏍囬',
                  artist: rec.artist_name || rec.artist || '鏈煡鑹烘湳锟?,
                  explanation: rec.explanation || `鍩轰簬鎮ㄧ殑${emotion}鎯呯华鎺ㄨ崘`,
                  album_image: rec.album_image || rec.image || '/static/img/default-album.png'
                };
              });
              
              // 鏇存柊鎺ㄨ崘鍒楄〃
              this.recommendations = processedRecommendations;
            }
            
            // 濡傛灉鐢ㄦ埛褰撳墠鍦ㄨ亰澶╂爣绛撅紝鍙互鎻愪緵涓€涓寜閽垏鎹㈠埌鎺ㄨ崘鏍囩
            if (this.currentTab === 'chat') {
              // 娣诲姞涓€涓彁绀猴紝鍙互鐢ㄦ寜閽垏锟?
              this.showNotification('鎯呯华闊充箰宸插噯澶囧ソ锛屽彲鍦ㄦ帹鑽愭爣绛炬煡锟?, 'success');
            }
          })
          .catch(error => {
            console.error('鑾峰彇鎯呯华闊充箰澶辫触:', error);
            this.showNotification('鑾峰彇鎯呯华闊充箰鎺ㄨ崘澶辫触锛屼娇鐢ㄩ粯璁ゆ帹锟?, 'danger');
            // 榛樿鎺ㄨ崘宸插湪鏂规硶寮€濮嬫椂璁剧疆
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 鑾峰彇鑱婂ぉ鍘嗗彶
      getChatHistory() {
        if (!this.user.isLoggedIn) return;
        
        this.isLoading = true;
        
        axios.get(`/api/chat/history?user_id=${this.user.id}`)
          .then(response => {
            this.chatHistory = response.data.history || [];
            console.log('宸插姞杞借亰澶╁巻锟?', this.chatHistory.length);
            
            // 杞崲涓哄吋瀹规ā鏉跨殑鏍煎紡
            this.chatMessages = this.chatHistory.flatMap(record => [
              { content: record.user_message, isUser: true },
              { content: record.ai_response, isUser: false }
            ]);
          })
          .catch(error => {
            console.error('鍔犺浇鑱婂ぉ鍘嗗彶澶辫触:', error);
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      /**
       * 浣跨敤棰勮鎻愮ず娑堟伅
       */
      usePrompt(prompt) {
        if (!prompt) return;
        
        this.chatInput = prompt;
        
        this.$nextTick(() => {
          // 璁╄緭鍏ユ鑾峰彇鐒︾偣
          const inputElement = document.querySelector('.chat-input-container input');
          if (inputElement) {
            inputElement.focus();
          }
          
          // 涔熷彲浠ョ洿鎺ュ彂閫佹秷锟?
          // this.sendChatMessage();
        });
      },
      
      // 鎻愪氦鍙嶉
      submitFeedback(songId, feedbackType) {
        if (!this.user.isLoggedIn) return;
        
        axios.post('/api/feedback', {
          user_id: this.user.id,
          track_id: songId,
          feedback_type: feedbackType
        })
          .then(response => {
            console.log('鍙嶉鎻愪氦鎴愬姛:', songId, feedbackType);
            this.showNotification('鎰熻阿鎮ㄧ殑鍙嶉锟?, 'success');
            
            // 浠庢帹鑽愬垪琛ㄤ腑绉婚櫎璇ユ瓕鏇插苟娣诲姞娣″嚭鍔ㄧ敾
            if (feedbackType === 'dislike') {
              const index = this.recommendations.findIndex(song => song.track_id === songId);
              if (index !== -1) {
                const songElement = document.querySelectorAll('.card')[index];
                if (songElement) {
                  songElement.classList.add('animate__animated', 'animate__fadeOut');
                  
                  setTimeout(() => {
                    this.recommendations.splice(index, 1);
                  }, 500);
                }
              }
            }
          })
          .catch(error => {
            console.error('鍙嶉鎻愪氦澶辫触:', error);
            this.showNotification('鍙嶉鎻愪氦澶辫触锛岃閲嶈瘯', 'danger');
          });
      },
      
      // 鎻愪氦婊℃剰搴﹁瘎锟?
      submitEvaluation() {
        if (!this.user.isLoggedIn || !this.isEvaluationComplete) return;
        
        this.isLoading = true;
        
        axios.post('/api/evaluation', {
          user_id: this.user.id,
          responses: this.evaluationResponses,
          comment: this.evaluationComment
        })
          .then(response => {
            console.log('婊℃剰搴﹁瘎浼版彁浜ゆ垚锟?', this.evaluationResponses);
            this.showNotification('鎰熻阿鎮ㄧ殑璇勪环锟?, 'success');
            this.evaluationSubmitted = true;
            this.satisfactionLevel = 0;
            this.feedbackText = '';
          })
          .catch(error => {
            console.error('婊℃剰搴﹁瘎浼版彁浜ゅけ锟?', error);
            this.showNotification('璇勪环鎻愪氦澶辫触锛岃閲嶈瘯', 'danger');
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 鏄剧ず閫氱煡
      showNotification(message, type = 'info') {
        this.notification.message = message;
        this.notification.type = type;
        this.notification.isVisible = true;
        
        // 娣诲姞鍒伴€氱煡鍒楄〃
        const notificationType = type === 'danger' ? 'is-danger' : 
                                type === 'warning' ? 'is-warning' : 
                                type === 'success' ? 'is-success' : 'is-info';
        this.notifications.push({
          message: message,
          type: notificationType
        });
        
        // 3绉掑悗鑷姩闅愯棌閫氱煡
        setTimeout(() => {
          this.notification.isVisible = false;
        }, 3000);
        
        // 3绉掑悗浠庨€氱煡鍒楄〃涓Щ锟?
        setTimeout(() => {
          if (this.notifications.length > 0) {
            this.notifications.shift();
          }
        }, 3000);
      },
      
      // 鐢ㄦ埛娉ㄥ唽
      register() {
        if (!this.username.trim()) {
          this.loginErrorMessage = '璇疯緭鍏ョ敤鎴峰悕';
          return;
        }
        
        if (!this.email.trim()) {
          this.loginErrorMessage = '璇疯緭鍏ラ偖锟?;
          return;
        }
        
        if (!this.password.trim()) {
          this.loginErrorMessage = '璇疯緭鍏ュ瘑锟?;
          return;
        }
        
        this.isLoading = true;
        this.loginErrorMessage = '';
        
        // 鍑嗗娉ㄥ唽鏁版嵁
        const registerData = {
          username: this.username,
          password: this.password,
          email: this.email
        };
        
        // 鍙戦€佹敞鍐岃锟?
        axios.post('/api/user/register', registerData)
          .then(response => {
            console.log('娉ㄥ唽鎴愬姛:', response.data);
            
            if (response.data && response.data.user_id) {
              // 娉ㄥ唽鎴愬姛锛岃嚜鍔ㄧ櫥锟?
              this.user = {
                id: response.data.user_id,
                username: response.data.username || this.username,
                email: this.email,
                isLoggedIn: true,
                isDeveloper: response.data.is_developer || false
              };
              
              // 淇濆瓨鍒版湰鍦板瓨锟?
              localStorage.setItem('userId', this.user.id);
              localStorage.setItem('username', this.user.username);
              localStorage.setItem('email', this.user.email);
              localStorage.setItem('isLoggedIn', 'true');
              localStorage.setItem('isDeveloper', this.user.isDeveloper ? 'true' : 'false');
              
              // 鏄剧ず鎴愬姛閫氱煡
              this.showNotification('娉ㄥ唽骞剁櫥褰曟垚鍔燂紒', 'success');
              
              // 鍔犺浇鍒濆鏁版嵁
              this.loadSampleSongs();
              
              // 娣诲姞AI娆㈣繋娑堟伅
              this.chatMessages.push({
                content: `娆㈣繋锟?{this.username}锛佹垜鏄偍鐨凙I闊充箰鍔╂墜銆傝闂偍鎯充簡瑙ｄ粈涔堥煶涔愪俊鎭垨鑾峰彇浠€涔堟帹鑽愶紵`,
                isUser: false
              });
            } else {
              // 娉ㄥ唽鎴愬姛浣嗘病鏈夎繑鍥炵敤鎴稩D锛屽皾璇曠櫥锟?
              this.login();
            }
          })
          .catch(error => {
            console.error('娉ㄥ唽澶辫触:', error);
            
            // 鏄剧ず閿欒娑堟伅
            if (error.response && error.response.data && error.response.data.error) {
              this.loginErrorMessage = error.response.data.error;
            } else {
              this.loginErrorMessage = '娉ㄥ唽澶辫触锛岃閲嶈瘯';
            }
            
            this.isLoading = false;
          });
      },
      
      /**
       * 鐢ㄦ埛鐧诲綍
       */
      login: function() {
        console.log("寮€濮嬬櫥褰曟祦绋嬶紝褰撳墠鐢ㄦ埛鍚嶏細", this.username, "褰撳墠缁戝畾瀛楁:", document.querySelector("input[v-model='username']") ? true : false);
        
        if (this.user && this.user.isLoggedIn) {
            console.log("鐢ㄦ埛宸茬櫥褰曪紝璺宠繃鐧诲綍杩囩▼");
            return;
        }
        
        if (!this.username) {
            // 淇閿欒锛氱‘淇濆嵆浣垮湪鑻辨枃鐣岄潰涓嬩篃鑳借闂敊璇秷锟?
            let errorMessage = "璇疯緭鍏ョ敤鎴峰悕";
            
            // 瀹夊叏鍦拌闂甧rrors瀵硅薄
            if (this.content && this.content[this.currentLanguage] && 
                this.content[this.currentLanguage].errors && 
                this.content[this.currentLanguage].errors.emptyUsername) {
                errorMessage = this.content[this.currentLanguage].errors.emptyUsername;
            }
            
            this.addNotification(errorMessage, 'is-danger');
            return;
        }
        
        // 寮€濮嬬櫥褰曟祦锟?
        this.isLoading = true;
        
        // 鍑嗗鐧诲綍鏁版嵁
        var loginData = {
            username: this.username,
            email: this.email || "",
            password: this.password || ""
        };
        
        console.log("鐧诲綍鏁版嵁:", loginData);
        
        // 寮€鍙戣€呯櫥褰曢€昏緫绠€锟?
        if (this.isDeveloperMode && !this.password) {
            loginData.password = "test123";
        }
        
        axios.post('/api/user/login', loginData)
            .then(response => {
                console.log("鐧诲綍鎴愬姛:", response.data);
                
                // 纭繚鐢ㄦ埛鏁版嵁鍖呭惈鎵€鏈夊繀瑕佸瓧锟?
                const userData = {
                    username: response.data.username || this.username,
                    email: response.data.email || this.email || "",
                    isDeveloper: response.data.is_developer || false,
                    isLoggedIn: true,
                    id: response.data.id || response.data.user_id || Date.now()
                };
                
                // 淇濆瓨鐢ㄦ埛浼氳瘽鍒版湰鍦板瓨锟?
                localStorage.setItem('user', JSON.stringify(userData));
                localStorage.setItem('user_session', JSON.stringify(userData)); // 鍚屾椂淇濆瓨鍒颁袱涓猭ey
                localStorage.setItem('userId', userData.id);
                localStorage.setItem('username', userData.username);
                localStorage.setItem('isLoggedIn', 'true');
                
                // 鏇存柊搴旂敤鐘讹拷?
                this.user = userData;
                this.username = userData.username;
                this.isDeveloperMode = userData.isDeveloper;
                
                // 閲嶇疆鐧诲綍琛ㄥ崟
                this.loginUsername = "";
                this.password = "";
                this.loginEmail = "";
                this.isRegistering = false;
                
                // 鍔犺浇鍒濆鏁版嵁
                this.loadInitialData();
                
                // 淇敼杩欓噷锛氱櫥褰曟垚鍔熷悗鏄剧ず娆㈣繋椤甸潰
                this.currentTab = 'welcome';
                
                // 瀹夊叏鍦拌闂垚鍔熸秷锟?
                let successMessage = "鐧诲綍鎴愬姛锟?;
                if (this.content && this.content[this.currentLanguage] && 
                    this.content[this.currentLanguage].success && 
                    this.content[this.currentLanguage].success.login) {
                    successMessage = this.content[this.currentLanguage].success.login;
                }
                
                this.addNotification(successMessage, 'is-success');
                
                // 鍚戞帶鍒跺彴鎵撳嵃鐧诲綍鎴愬姛淇℃伅
                console.log("鐢ㄦ埛鐧诲綍鎴愬姛", this.user);
                console.log("褰撳墠Tab:", this.currentTab);
            })
            .catch(error => {
                console.error("鐧诲綍閿欒:", error.response ? error.response.data : error);
                
                // 瀹夊叏鍦拌闂敊璇秷锟?
                let errorMessage = "鐧诲綍澶辫触锛岃妫€鏌ユ偍鐨勭敤鎴峰悕鍜屽瘑锟?;
                if (this.content && this.content[this.currentLanguage] && 
                    this.content[this.currentLanguage].errors && 
                    this.content[this.currentLanguage].errors.loginFailed) {
                    errorMessage = this.content[this.currentLanguage].errors.loginFailed;
                }
                
                if (error.response && error.response.data && error.response.data.error) {
                    errorMessage = error.response.data.error;
                }
                
                this.addNotification(errorMessage, 'is-danger');
            })
            .finally(() => {
                this.isLoading = false;
            });
      },
      
      /**
       * 閫夋嫨濉啓璋冩煡闂嵎
       */
      chooseSurvey: function() {
        // 鍒濆鍖栬皟鏌ラ棶锟?
        this.surveyQuestions.find(q => q.id === 'music_genres').options = this.musicGenres;
        this.surveyCompleted = false;
        this.showWelcomeOptions = false;
        this.showPreferencesSurvey = true;
        this.currentTab = 'survey';
      },
      
      /**
       * 閫夋嫨AI鑱婂ぉ
       */
      chooseAIChat: function() {
        this.showWelcomeOptions = false;
        this.currentTab = 'chat';
        this.loadChatHistory();
      },
      
      /**
       * 鍥炵瓟璋冩煡闂
       */
      answerQuestion: function(questionId, answer) {
        const question = this.surveyQuestions.find(q => q.id === questionId);
        if (!question) return;
        
        if (question.type === 'single') {
          // 鍗曢€夐鐩存帴璁剧疆绛旀
          question.answer = answer;
        } else if (question.type === 'multiple') {
          // 澶氶€夐澶勭悊閫変腑/鍙栨秷閫変腑
          const index = question.answer.indexOf(answer);
          if (index === -1) {
            // 娣诲姞閫夐」
            question.answer.push(answer);
          } else {
            // 绉婚櫎閫夐」
            question.answer.splice(index, 1);
          }
        }
      },
      
      /**
       * 鎻愪氦璋冩煡闂嵎
       */
      submitSurvey: function() {
        // 鏀堕泦鐢ㄦ埛鍋忓ソ
        this.preferences = [];
        
        // 灏嗛棶鍗风瓟妗堣浆鎹负鍋忓ソ
        this.surveyQuestions.forEach(question => {
          if (question.id === 'music_genres' && question.answer.length > 0) {
            this.selectedGenres = [...question.answer];
          }
          
          if (question.answer && 
             (question.type === 'single' && question.answer !== '') || 
             (question.type === 'multiple' && question.answer.length > 0)) {
            this.preferences.push({
              preference_id: question.id,
              preference_type: question.type,
              preference_value: JSON.stringify(question.answer)
            });
          }
        });
        
        // 淇濆瓨鐢ㄦ埛鍋忓ソ鍒版湰鍦板瓨锟?
        localStorage.setItem('user_preferences', JSON.stringify(this.preferences));
        
        // 鏍囪璋冩煡瀹屾垚
        this.surveyCompleted = true;
        this.showPreferencesSurvey = false;
        
        // 鍔犺浇涓€у寲鎺ㄨ崘
        this.getPersonalizedRecommendations();
        
        // 鏄剧ず閫氱煡
        this.showNotification(
          this.currentLanguage === 'zh' ? '鎰熻阿鎮ㄥ畬鎴愯皟鏌ワ紒' : 'Thank you for completing the survey!',
          'success'
        );
      },
      
      /**
       * 璺宠繃璋冩煡
       */
      skipSurvey: function() {
        this.surveyCompleted = true;
        this.showPreferencesSurvey = false;
        
        // 鍔犺浇榛樿鎺ㄨ崘
        this.recommendations = [...this.defaultRecommendations];
        this.currentTab = 'recommendations';
        
        // 閫氱煡鐢ㄦ埛
        this.showNotification(
          this.currentLanguage === 'zh' ? '宸茶烦杩囪皟锟? : 'Survey skipped',
          'info'
        );
      },
      
      /**
       * 鑾峰彇涓€у寲鎺ㄨ崘
       */
      getPersonalizedRecommendations: function() {
        // 璁剧疆鍔犺浇鐘讹拷?
        this.isLoading = true;
        this.recommendations = [...this.defaultRecommendations]; // 璁剧疆榛樿锟?
        
        // 鍑嗗璇锋眰鍙傛暟
        const params = {
          user_id: this.user.id,
          genres: this.selectedGenres.join(',')
        };
        
        // 鍒囨崲鍒版帹鑽愰€夐」锟?
        this.currentTab = 'recommendations';
        
        // 鍚慉PI鍙戦€佽锟?
        axios.get('/api/recommendations/personalized', { params: params })
          .then(response => {
            this.isLoading = false;
            
            if (response.data && response.data.recommendations && response.data.recommendations.length > 0) {
              // 纭繚姣忎釜鎺ㄨ崘椤归兘鏈塼itle鍜宎rtist瀛楁
              this.recommendations = response.data.recommendations.map(rec => {
                return {
                  track_id: rec.track_id || '',
                  track_name: rec.track_name || '',
                  artist_name: rec.artist_name || '',
                  title: rec.title || rec.track_name || '',
                  artist: rec.artist || rec.artist_name || '',
                  explanation: rec.explanation || '鏍规嵁鎮ㄧ殑鍋忓ソ鎺ㄨ崘'
                };
              });
              
              // 璁板綍鍔犺浇鐨勬帹鑽愭暟锟?
              console.log('宸插姞锟? + this.recommendations.length + '鏉′釜鎬у寲鎺ㄨ崘');
              
              // 娣诲姞鍔ㄧ敾鏁堟灉
              setTimeout(() => {
                const cards = document.querySelectorAll('.recommendation-card');
                cards.forEach((card, index) => {
                  setTimeout(() => {
                    card.classList.add('show');
                  }, index * 100);
                });
              }, 100);
            } else {
              // 鎺ㄨ崘鍔犺浇澶辫触锛屼娇鐢ㄩ粯璁ゆ帹锟?
              console.log('鏈兘鑾峰彇涓€у寲鎺ㄨ崘锛屼娇鐢ㄩ粯璁ゆ帹锟?);
            }
          })
          .catch(error => {
            this.isLoading = false;
            console.error('鑾峰彇涓€у寲鎺ㄨ崘鍑洪敊:', error);
            
            // 鏄剧ず閿欒閫氱煡
            this.showNotification(
              this.currentLanguage === 'zh' ? '鑾峰彇鎺ㄨ崘澶辫触锛岃绋嶅悗鍐嶈瘯' : 'Failed to get recommendations, please try again later',
              'error'
            );
          });
      },
      
      // 鐧诲嚭鐢ㄦ埛
      logoutUser: function() {
        // 娓呴櫎鐢ㄦ埛鐘讹拷?
        this.user.id = null;
        this.user.username = '';
        this.user.email = '';
        this.user.isLoggedIn = false;
        this.user.isDeveloper = false;
        
        // 娓呴櫎鏈湴瀛樺偍
        localStorage.removeItem('user');
        localStorage.removeItem('user_preferences');
        
        // 閲嶇疆鐣岄潰鐘讹拷?
        this.showLoginForm = true;
        this.showWelcomeOptions = false;
        this.showPreferencesSurvey = false;
        this.currentTab = 'welcome';
        this.selectedGenres = [];
        
        // 娓呯┖闂嵎绛旀
        this.surveyQuestions.forEach(question => {
          if (question.type === 'single') {
            question.answer = '';
          } else {
            question.answer = [];
          }
        });
        
        // 鏄剧ず閫氱煡
        this.showNotification(
          this.currentLanguage === 'zh' ? '宸叉垚鍔熼€€鍑虹櫥锟? : 'Successfully logged out',
          'info'
        );
      },
      
      // 鎾斁闊抽棰勮
      playPreview: function(previewUrl, trackName) {
        if (!previewUrl) {
          this.showNotification(
            this.currentLanguage === 'zh' ? '鏃犳硶鎾斁锛岄瑙堥摼鎺ヤ笉鍙敤' : 'Cannot play, preview link unavailable',
            'error'
          );
          return;
        }
        
        // 鍋滄褰撳墠鎾斁鐨勯煶锟?
        if (this.currentAudio) {
          this.currentAudio.pause();
          this.currentAudio = null;
        }
        
        // 鍒涘缓鏂伴煶棰戝锟?
        const audio = new Audio(previewUrl);
        this.currentAudio = audio;
        
        // 寮€濮嬫挱锟?
        audio.play().then(() => {
          this.showNotification(
            this.currentLanguage === 'zh' ? '姝ｅ湪鎾斁: ' + (trackName || '闊充箰') : 'Now playing: ' + (trackName || 'music'),
            'info'
          );
        }).catch(error => {
          console.error('鎾斁闊抽鍑洪敊:', error);
          this.showNotification(
            this.currentLanguage === 'zh' ? '鎾斁澶辫触锛岃绋嶅悗鍐嶈瘯' : 'Playback failed, please try again later',
            'error'
          );
        });
        
        // 鎾斁缁撴潫鏃舵竻锟?
        audio.onended = () => {
          this.currentAudio = null;
        };
      },
      
      // 澶勭悊鍥剧墖鍔犺浇閿欒
      handleImageError: function(event) {
        event.target.src = 'static/img/music-pattern.svg';
      },
      
      // 鎾斁姝屾洸棰勮
      playSongPreview(previewUrl) {
        this.playPreview(previewUrl);
      },
      
      // 瀵瑰崟棣栨瓕鏇茶瘎锟?
      rateSongItem(song, rating) {
        if (!song || !song.track_id) return;
        
        // 璁剧疆姝屾洸璇勫垎
        song.rating = rating;
        
        // 璋冪敤璇勫垎API
        this.rateSong(song.track_id, rating);
      },
      
      // 鐐硅禐姝屾洸
      likeSong(song) {
        if (!song || !song.track_id) return;
        
        this.submitFeedback(song.track_id, 'like');
        this.showNotification('鎰熻阿鎮ㄧ殑鍙嶉锟?, 'success');
      },
      
      // 鐐硅俯姝屾洸
      dislikeSong(song) {
        if (!song || !song.track_id) return;
        
        this.submitFeedback(song.track_id, 'dislike');
        this.showNotification('鎰熻阿鎮ㄧ殑鍙嶉锛佹垜浠細鏀硅繘鎺ㄨ崘', 'info');
      },
      
      // 鍔犺浇鑱婂ぉ鍘嗗彶
      loadChatHistory() {
        // 濡傛灉鐢ㄦ埛娌℃湁鐧诲綍锛屼笉鍔犺浇鑱婂ぉ鍘嗗彶
        if (!this.user.isLoggedIn) return;
        
        // 璋冪敤鑾峰彇鑱婂ぉ鍘嗗彶鐨凙PI
        this.getChatHistory();
      },
      
      // 绉婚櫎閫氱煡
      removeNotification(index) {
        this.notifications.splice(index, 1);
      },
      
      /**
       * 璺宠浆鍒伴棶鍗疯皟鏌ラ〉锟?
       */
      goToQuestionnaire() {
        // 淇濆瓨褰撳墠鐢ㄦ埛ID鍒版湰鍦板瓨鍌紝浠ヤ究闂嵎椤甸潰浣跨敤
        if (this.user && this.user.id) {
          localStorage.setItem('userId', this.user.id);
        }
        // 璺宠浆鍒伴棶鍗烽〉锟?
        window.location.href = '/questionnaire';
      },
      
      /**
       * 鑾峰彇閫氱煡鍥炬爣
       */
      getNotificationIcon(type) {
        switch(type) {
          case 'success':
            return 'fas fa-check-circle';
          case 'error':
            return 'fas fa-exclamation-circle';
          case 'warning':
            return 'fas fa-exclamation-triangle';
          default:
            return 'fas fa-info-circle';
        }
      },
      
      // 娣诲姞閫氱煡
      addNotification(message, type = 'is-info') {
        console.log("娣诲姞閫氱煡:", message, type);
        if (!this.notifications) {
          this.notifications = [];
        }
        this.notifications.push({
          message: message,
          type: type
        });
        
        // 鍚屾椂鏇存柊鍗曚竴閫氱煡瀵硅薄锛屽吋瀹规棫浠ｇ爜
        this.notification = {
          message: message,
          type: type.replace('is-', ''),
          isVisible: true
        };
        
        // 3绉掑悗鑷姩绉婚櫎閫氱煡
        setTimeout(() => {
          if (this.notifications && this.notifications.length > 0) {
            this.notifications.shift();
          }
          this.notification.isVisible = false;
        }, 3000);
      },
      
      // 鍔犺浇鐢ㄦ埛璇勫垎璁板綍
      loadUserRatings() {
        if (!this.user.isLoggedIn) return;
        
        this.isLoading = true;
        
        axios.get(`/api/user_ratings/${this.user.id}`)
          .then(response => {
            this.userRatings = response.data || {};
            console.log('宸插姞杞界敤鎴疯瘎锟?', Object.keys(this.userRatings).length);
          })
          .catch(error => {
            console.error('鍔犺浇鐢ㄦ埛璇勫垎澶辫触:', error);
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 鍔犺浇鎵€鏈夌敤锟?(绠＄悊鍛樺姛锟?
      loadAllUsers() {
        if (!this.user.isDeveloper) {
          this.showNotification('鍙湁寮€鍙戣€呮墠鑳芥煡鐪嬬敤鎴峰垪锟?, 'warning');
          this.currentTab = 'home';
          return;
        }
        
        this.isLoading = true;
        
        axios.get(`/api/user/all?admin_id=${this.user.id}`)
          .then(response => {
            this.allUsers = response.data || [];
            console.log('宸插姞杞芥墍鏈夌敤锟?', this.allUsers.length);
          })
          .catch(error => {
            console.error('鍔犺浇鐢ㄦ埛鍒楄〃澶辫触:', error);
            this.showNotification('鍔犺浇鐢ㄦ埛鍒楄〃澶辫触', 'error');
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 娣诲姞鏂扮敤锟?(绠＄悊鍛樺姛锟?
      addUser() {
        if (!this.user.isDeveloper) {
          this.showNotification('鍙湁寮€鍙戣€呮墠鑳芥坊鍔犵敤锟?, 'warning');
          return;
        }
        
        if (!this.newUser.username.trim()) {
          this.showNotification('璇疯緭鍏ョ敤鎴峰悕', 'warning');
          return;
        }
        
        if (!this.newUser.password.trim()) {
          this.showNotification('璇疯緭鍏ュ瘑锟?, 'warning');
          return;
        }
        
        this.isLoading = true;
        
        // 鍑嗗娣诲姞鐢ㄦ埛鐨勬暟锟?
        const userData = {
          admin_id: this.user.id,
          username: this.newUser.username,
          password: this.newUser.password,
          email: this.newUser.email,
          is_developer: this.newUser.isDeveloper
        };
        
        axios.post('/api/user/register', userData)
          .then(response => {
            console.log('娣诲姞鐢ㄦ埛鎴愬姛:', response.data);
            this.showNotification('娣诲姞鐢ㄦ埛鎴愬姛', 'success');
            
            // 閲嶇疆琛ㄥ崟
            this.newUser = {
              username: '',
              email: '',
              password: '',
              isDeveloper: false
            };
            
            // 閲嶆柊鍔犺浇鐢ㄦ埛鍒楄〃
            this.loadAllUsers();
          })
          .catch(error => {
            console.error('娣诲姞鐢ㄦ埛澶辫触:', error);
            
            // 鏄剧ず閿欒娑堟伅
            if (error.response && error.response.data && error.response.data.error) {
              this.showNotification(error.response.data.error, 'error');
            } else {
              this.showNotification('娣诲姞鐢ㄦ埛澶辫触锛岃閲嶈瘯', 'error');
            }
          })
          .finally(() => {
            this.isLoading = false;
          });
      },
      
      // 缂栬緫鐢ㄦ埛 (绠＄悊鍛樺姛锟?
      editUser(user) {
        this.editingUser = { ...user };
        
        // 杩欓噷鍙互鎵撳紑涓€涓紪杈戞ā鎬佹
        // 涓虹畝鍖栬捣瑙侊紝鎴戜滑鐩存帴鍦ㄦ帶鍒跺彴涓樉绀轰竴鏉℃秷锟?
        console.log('缂栬緫鐢ㄦ埛:', this.editingUser);
        this.showNotification('缂栬緫鐢ㄦ埛鍔熻兘寰呭疄锟?, 'info');
      },
      
      // 鍒犻櫎鐢ㄦ埛 (绠＄悊鍛樺姛锟?
      deleteUser(user) {
        if (user.id === 'dev-001') {
          this.showNotification('涓嶈兘鍒犻櫎涓诲紑鍙戣€呰处锟?, 'warning');
          return;
        }
        
        if (confirm(`纭畾瑕佸垹闄ょ敤锟?${user.username} 鍚楋紵`)) {
          this.isLoading = true;
          
          axios.delete(`/api/user/delete?admin_id=${this.user.id}&user_id=${user.id}`)
            .then(response => {
              console.log('鍒犻櫎鐢ㄦ埛鎴愬姛:', response.data);
              this.showNotification('鍒犻櫎鐢ㄦ埛鎴愬姛', 'success');
              
              // 浠庡垪琛ㄤ腑绉婚櫎璇ョ敤锟?
              this.allUsers = this.allUsers.filter(u => u.id !== user.id);
            })
            .catch(error => {
              console.error('鍒犻櫎鐢ㄦ埛澶辫触:', error);
              this.showNotification('鍒犻櫎鐢ㄦ埛澶辫触锛岃閲嶈瘯', 'error');
            })
            .finally(() => {
              this.isLoading = false;
            });
        }
      },
      
      /**
       * 鐢ㄦ埛鐧诲嚭
       */
      logout: function() {
        console.log("鎵ц鐧诲嚭鎿嶄綔");
        // 璋冪敤logoutUser鏂规硶
        this.logoutUser();
        
        // 娓呴櫎鎵€鏈夌浉鍏冲瓨锟?
        localStorage.removeItem('user');
        localStorage.removeItem('user_session');
        localStorage.removeItem('userId');
        localStorage.removeItem('username');
        localStorage.removeItem('email');
        localStorage.removeItem('isLoggedIn');
        localStorage.removeItem('isDeveloper');
        
        // 閲嶇疆鐢ㄦ埛鐩稿叧瀛楁
        this.username = '';
        this.email = '';
        this.password = '';
        
        // 娣诲姞鐧诲嚭鎴愬姛閫氱煡
        this.addNotification(
          this.currentLanguage === 'zh' ? '宸叉垚鍔熼€€鍑虹櫥锟? : 'Successfully logged out',
          'is-success'
        );
        
        console.log("鐧诲嚭瀹屾垚锛岀敤鎴风姸鎬侊細", this.user);
      },
      
      // 鏌ユ壘骞舵挱鏀炬瓕鏇查锟?
      playSongPreview(previewUrl) {
        console.log("灏濊瘯鎾斁棰勮:", previewUrl);
        
        if (!previewUrl || previewUrl === "") {
          this.showNotification("鎶辨瓑锛岃姝屾洸娌℃湁鍙敤鐨勯瑙堬拷?, "warning");
          return;
        }
        
        // 浣跨敤鐜版湁鐨勯煶棰戝厓绱犺€屼笉鏄垱寤烘柊锟?
        const audioPlayer = document.getElementById('audio-player');
        if (!audioPlayer) {
          console.error("鎵句笉鍒伴煶棰戞挱鏀惧櫒鍏冪礌");
          this.showNotification("闊抽鎾斁鍣ㄥ姞杞藉け锟?, "error");
          return;
        }
        
        // 璁剧疆闊抽婧愬苟鎾斁
        audioPlayer.src = previewUrl;
        audioPlayer.style.display = 'block'; // 鏄剧ず闊抽鎾斁锟?
        
        // 鎾斁澶辫触鏃舵樉绀洪敊锟?
        audioPlayer.onerror = (e) => {
          console.error("闊抽鎾斁閿欒:", e);
          this.showNotification("鎾斁棰勮鏃跺嚭閿欙紝璇风◢鍚庡啀璇曪拷?, "error");
        };
        
        // 灏濊瘯鎾斁
        try {
          const playPromise = audioPlayer.play();
          
          if (playPromise !== undefined) {
            playPromise.then(() => {
              this.showNotification("寮€濮嬫挱鏀鹃锟?, "success");
            }).catch(error => {
              console.error("鎾斁澶辫触:", error);
              this.showNotification("鎾斁琚祻瑙堝櫒闃绘锛岃鐐瑰嚮椤甸潰鍚庡啀锟?, "warning");
            });
          }
        } catch (e) {
          console.error("鎾斁寮傚父:", e);
          this.showNotification("鎾斁鍑虹幇寮傚父", "error");
        }
        
        // 鎾斁瀹屾垚鍚庨殣钘忔挱鏀惧櫒
        audioPlayer.onended = () => {
          audioPlayer.style.display = 'none';
        };
      },
      
      // 鏍煎紡鍖栬亰澶╂秷鎭紝妫€娴嬪苟杞崲闊充箰閾炬帴涓哄彲鐐瑰嚮鐨勬挱鏀炬寜锟?
      formatChatMessage(message) {
        // 妫€鏌ユ秷鎭腑鏄惁鍖呭惈棰勮URL鐨勬ā锟?
        const urlPattern = /(https?:\/\/[^\s]+)/g;
        const spotifyPattern = /(https?:\/\/(?:open\.spotify\.com|api\.spotify\.com)[^\s]+)/g;
        
        // 鏇挎崲Spotify閾炬帴涓哄彲鐐瑰嚮鐨勬挱鏀炬寜锟?
        if (spotifyPattern.test(message)) {
            return message.replace(spotifyPattern, url => {
                return `<div class="music-preview-link">
                        <a href="${url}" target="_blank" class="music-link">${url}</a>
                        <button class="button is-small is-primary play-button" onclick="app.playSongPreview('${url}')">
                            <span class="icon"><i class="fas fa-play"></i></span> 鎾斁
                        </button>
                    </div>`;
            });
        }
        
        // 鏇挎崲鏅€歎RL涓哄彲鐐瑰嚮閾炬帴
        if (urlPattern.test(message)) {
            return message.replace(urlPattern, url => {
                // 妫€鏌RL鏄惁涓洪煶棰戞牸锟?
                if (url.match(/\.(mp3|wav|ogg)$/i)) {
                    return `<div class="music-preview-link">
                            <a href="${url}" target="_blank" class="music-link">${url}</a>
                            <button class="button is-small is-primary play-button" onclick="app.playSongPreview('${url}')">
                                <span class="icon"><i class="fas fa-play"></i></span> 鎾斁
                            </button>
                        </div>`;
                } else {
                    return `<a href="${url}" target="_blank">${url}</a>`;
                }
            });
        }
        
        return message;
      },
      
      // 娣诲姞鍒版秷鎭巻鍙插苟鏄剧ず
      addChatMessage(message, isUser = false) {
        // 鏍煎紡鍖栨秷鎭腑鐨勯摼锟?
        const formattedContent = isUser ? message : this.formatChatMessage(message);
        
        this.chatMessages.push({
            content: formattedContent,
            isUser: isUser,
            time: new Date().toLocaleTimeString()
        });
        
        // 婊氬姩鍒版渶鏂版秷锟?
        this.$nextTick(() => {
            const chatContainer = this.$refs.chatMessages;
            if (chatContainer) {
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });
      },
      
      // 淇濆瓨鑱婂ぉ璁板綍
      saveChatHistory() {
        // 灏嗚亰澶╄褰曚繚瀛樺埌鏈湴瀛樺偍
        localStorage.setItem('chatHistory', JSON.stringify(this.chatHistory));
      },
      
      // 妫€鏌ユ秷鎭槸鍚﹀寘鍚儏缁叧閿瘝
      containsEmotionKeywords(message) {
        return this.emotionKeywords.some(keyword => message.includes(keyword));
      },
      
      setupEventListeners() {
        console.log('璁剧疆娓告垙浜嬩欢鐩戝惉锟?..');
        
        // 绉婚櫎浠ュ墠鐨勪簨浠剁洃鍚櫒
        if (this._keydownHandler) {
          document.removeEventListener('keydown', this._keydownHandler);
          console.log('宸茬Щ闄ゆ棫鐨勯敭鐩樹簨浠剁洃鍚櫒');
        }
        
        // 濡傛灉宸茬粡璁剧疆杩囦簨浠剁洃鍚櫒锛屼笉瑕侀噸澶嶈缃寜閽簨锟?
        if (this._eventListenersSet) {
          // 鍙噸鏂扮粦瀹氶敭鐩樹簨锟?
          this._keydownHandler = this._createKeydownHandler();
          document.addEventListener('keydown', this._keydownHandler);
          console.log('宸叉坊鍔犳柊鐨勯敭鐩樹簨浠剁洃鍚櫒');
          return;
        }
        
        // 閿洏鎺у埗
        this._keydownHandler = this._createKeydownHandler();
        
        // 娣诲姞閿洏浜嬩欢锛岀‘淇濆湪鏁翠釜鏂囨。涓婄洃锟?
        document.addEventListener('keydown', this._keydownHandler);
        console.log('宸叉坊鍔犳柊鐨勯敭鐩樹簨浠剁洃鍚櫒');
        
        // 寮€濮嬫父鎴忔寜锟?
        const startButton = document.getElementById('game-start');
        console.log('鏌ユ壘寮€濮嬫父鎴忔寜锟?', startButton ? '鎴愬姛' : '澶辫触');
        
        if (startButton) {
          console.log('鎵惧埌寮€濮嬫父鎴忔寜閽紝娣诲姞鐐瑰嚮浜嬩欢...');
          // 绉婚櫎鎵€鏈夌幇鏈変簨浠剁洃鍚櫒锛岄槻姝㈤噸锟?
          const newStartButton = startButton.cloneNode(true);
          startButton.parentNode.replaceChild(newStartButton, startButton);
          
          // 娣诲姞鏂扮殑鐐瑰嚮浜嬩欢澶勭悊绋嬪簭
          this._startButtonHandler = (e) => {
            console.log('寮€濮嬫父鎴忔寜閽鐐瑰嚮!', e);
            e.preventDefault();
            e.stopPropagation();
            this.startGame();
          };
          
          // 鍚屾椂娣诲姞榧犳爣鐐瑰嚮鍜岃Е鎽镐簨浠讹紝澧炲姞鍏煎锟?
          newStartButton.addEventListener('click', this._startButtonHandler);
          newStartButton.addEventListener('touchend', this._startButtonHandler);
          
          // 娣诲姞棰濆鐨勮皟璇曚俊锟?
          newStartButton.style.pointerEvents = 'auto';
          newStartButton.style.cursor = 'pointer';
          newStartButton.setAttribute('data-listener-set', 'true');
          
          // 娣诲姞鍐呰仈鐐瑰嚮浜嬩欢锛屼綔涓哄锟?
          newStartButton.onclick = (e) => {
            console.log('娓告垙鎸夐挳鍐呰仈鐐瑰嚮浜嬩欢瑙﹀彂');
            this._startButtonHandler(e);
          };
          
          // 寮鸿皟鎸夐挳鍙偣鍑伙拷?
          newStartButton.style.zIndex = '1000';
          
          console.log('寮€濮嬫父鎴忔寜閽簨浠跺凡璁剧疆');
        } else {
          console.error('鎵句笉鍒板紑濮嬫父鎴忔寜锟?');
        }
        
        // 瀹屾垚娓告垙鎸夐挳
        const finishButton = document.getElementById('game-finish');
        if (finishButton) {
          const newFinishButton = finishButton.cloneNode(true);
          finishButton.parentNode.replaceChild(newFinishButton, finishButton);
          
          this._finishButtonHandler = () => {
            // 閫€鍑哄叏锟?
            if (document.fullscreenElement) {
              document.exitFullscreen().catch(err => {
                console.error('鏃犳硶閫€鍑哄叏灞忔ā锟?', err);
              });
            }
            
            this.finishGame();
          };
          
          newFinishButton.addEventListener('click', this._finishButtonHandler);
          newFinishButton.setAttribute('data-listener-set', 'true');
        }
        
        this._eventListenersSet = true;
      },
      
      // 鍒涘缓閿洏浜嬩欢澶勭悊鍑芥暟
      _createKeydownHandler() {
        return (e) => {
          console.log('閿洏浜嬩欢:', e.key, '娓告垙杩愯鐘讹拷?', this.gameRunning);
          
          if (!this.gameRunning || !this.player) return;
          
          // 涓轰簡澧炲姞杩愬姩鐨勫搷搴旀€э紝鎴戜滑绔嬪嵆鏇存柊鐜╁浣嶇疆骞堕噸锟?
          let needsRedraw = false;
          
          if (e.key === 'ArrowLeft') {
            this.player.x -= this.player.speed;
            console.log('鐜╁宸︾Щ:', this.player.x);
            needsRedraw = true;
          }
          
          if (e.key === 'ArrowRight') {
            this.player.x += this.player.speed;
            console.log('鐜╁鍙崇Щ:', this.player.x);
            needsRedraw = true;
          }
          
          if (e.key === 'ArrowUp' && this.player.grounded) {
            this.player.dy = -12; // 璺宠穬鍔涘害
            this.player.grounded = false;
            console.log('鐜╁璺宠穬:', this.player.dy);
            needsRedraw = true;
          }
          
          // 纭繚鐜╁涓嶄細瓒呭嚭鐢诲竷
          if (this.player.x < 0) this.player.x = 0;
          if (this.player.x + this.player.width > this.canvas.width) {
            this.player.x = this.canvas.width - this.player.width;
          }
          
          // 鎸塃SC閿€€鍑哄叏锟?
          if (e.key === 'Escape' && document.fullscreenElement) {
            document.exitFullscreen().catch(err => {
              console.error('鏃犳硶閫€鍑哄叏灞忔ā锟?', err);
            });
          }
          
          // 濡傛灉鏈変綅缃洿鏂帮紝绔嬪嵆閲嶇粯娓告垙鐢婚潰
          if (needsRedraw && this.canvas && this.ctx) {
            // 寮哄埗閲嶇粯涓€锟?
            cancelAnimationFrame(this._gameLoopId);
            this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
          }
        };
      },
      
      updatePreferenceDisplay() {
        try {
          const list = document.getElementById('preference-list');
          if (!list) {
            console.log('鍋忓ソ鍒楄〃鍏冪礌涓嶅瓨锟?);
            return;
          }
          
          // 娓呯┖鍒楄〃
          list.innerHTML = '';
          
          // 鑾峰彇鍞竴鍋忓ソ
          const uniquePreferences = [...new Set(this.preferences)];
          
          // 涓烘瘡涓亸濂藉垱寤轰竴涓垪琛ㄩ」
          uniquePreferences.forEach(prefName => {
            // 鎵惧埌瀵瑰簲鐨勯煶涔愰锟?
            const genre = this.findGenreByName(prefName);
            
            const item = document.createElement('li');
            
            if (genre) {
              // 濡傛灉鎵惧埌浜嗛鏍间俊鎭紝浣跨敤璇︾粏淇℃伅
              item.innerHTML = `${genre.icon} ${prefName}`;
              item.style.backgroundColor = genre.color;
            } else {
              // 鍚﹀垯鍙樉绀哄悕锟?
              item.innerHTML = prefName;
              item.style.backgroundColor = '#333333';
            }
            
            // 娣诲姞鏍峰紡
            item.style.padding = '8px 12px';
            item.style.margin = '5px 0';
            item.style.borderRadius = '4px';
            item.style.color = 'white';
            item.style.listStyleType = 'none';
            
            list.appendChild(item);
          });
          
          // 濡傛灉鏀堕泦浜嗚冻澶熺殑鍋忓ソ锛屾樉绀哄畬鎴愭寜锟?
          if (uniquePreferences.length >= 5) {
            const finishButton = document.getElementById('game-finish');
            if (finishButton) {
              finishButton.classList.remove('is-hidden');
            }
          }
        } catch (error) {
          console.error('鏇存柊鍋忓ソ鏄剧ず鏃跺嚭锟?', error);
        }
      },
      
      // 閫氳繃鍚嶇О鏌ユ壘闊充箰椋庢牸
      findGenreByName(name) {
        const genres = [
          { name: '娴佽闊充箰', icon: '馃幍', color: '#FF5733' },
          { name: '鎽囨粴', icon: '馃', color: '#C70039' },
          { name: '鍙ゅ吀', icon: '馃幓', color: '#900C3F' },
          { name: '鐖靛＋', icon: '馃幏', color: '#581845' },
          { name: '鐢靛瓙', icon: '馃帶', color: '#FFC300' },
          { name: '鍢诲搱', icon: '馃帳', color: '#DAF7A6' },
          { name: '姘戣埃', icon: '馃獣', color: '#FF5733' },
          { name: '钃濊皟', icon: '馃幐', color: '#C70039' }
        ];
        
        return genres.find(genre => genre.name === name);
      },
      
      // 娓告垙寰幆
      gameLoop() {
        try {
          // 纭繚蹇呰鐨勫璞″瓨锟?
          if (!this.canvas || !this.ctx || !this.player) {
            console.error('gameLoop: 缂哄皯蹇呰瀵硅薄', {
              canvas: !!this.canvas,
              ctx: !!this.ctx,
              player: !!this.player
            });
            return;
          }
          
          // 娓呴櫎鐢诲竷
          this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
          
          // 缁樺埗鑳屾櫙 - 娣诲姞鑳屾櫙鑹蹭互閬垮厤榛戝睆
          this.ctx.fillStyle = '#191919';
          this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
          
          // 缁樺埗鍦伴潰
          this.ctx.fillStyle = '#333';
          this.ctx.fillRect(0, this.ground + this.player.height, this.canvas.width, 2);
          
          // 濡傛灉娓告垙宸叉殏鍋滐紝鏄剧ず鏆傚仠灞忓箷
          if (this.gamePaused) {
            this.drawPauseScreen();
            return;
          }
          
          // 濡傛灉娓告垙宸插畬鎴愶紝鏄剧ず瀹屾垚灞忓箷
          if (this.gameCompleted) {
            this.showFinishScreen();
            return;
          }
          
          // 鏇存柊鐜╁
          this.updatePlayer();
          
          // 鏇存柊鏀堕泦鐗╀綅锟?
          this.updateCollectibles();
          
          // 妫€鏌ョ锟?
          this.checkCollisions();
          
          // 鏇存柊绮掑瓙鏁堟灉
          if (typeof this.updateParticles === 'function') {
            this.updateParticles();
          }
          
          // 缁樺埗鐜╁
          this.drawPlayer();
          
          // 缁樺埗鏀堕泦锟?
          this.drawCollectibles();
          
          // 缁樺埗绮掑瓙鏁堟灉
          if (typeof this.drawParticles === 'function') {
            this.drawParticles();
          }
          
          // 缁樺埗鍒嗘暟
          this.drawScore();
          
          // 缁樺埗鍋忓ソ鏄剧ず
          if (typeof this.drawPreferences === 'function') {
            this.drawPreferences();
          }
          
          // 缁х画娓告垙寰幆
          if (this.gameRunning) {
            this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
          }
        } catch (error) {
          console.error('娓告垙寰幆涓嚭锟?', error);
          
          // 灏濊瘯鎭㈠娓告垙杩愯
          if (this.gameRunning) {
            setTimeout(() => {
              this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
            }, 1000);
          }
        }
      },
      
      // 缁樺埗鍋忓ソ
      drawPreferences() {
        try {
          if (!this.ctx) return;
          
          this.ctx.fillStyle = 'white';
          this.ctx.font = '16px Arial';
          this.ctx.textAlign = 'left';
          this.ctx.fillText('宸叉敹闆嗙殑闊充箰鍋忓ソ:', 10, 30);
          
          // 鏄剧ず锟?涓凡鏀堕泦鐨勫亸锟?
          const uniquePreferences = [...new Set(this.preferences)];
          const displayPreferences = uniquePreferences.slice(0, 5);
          
          displayPreferences.forEach((pref, index) => {
            // 鎵惧埌瀵瑰簲鐨勯煶涔愰锟?
            const genre = this.findGenreByName(pref);
            
            // 缁樺埗鑳屾櫙
            if (genre) {
              this.ctx.fillStyle = genre.color;
            } else {
              this.ctx.fillStyle = '#333333';
            }
            
            this.ctx.fillRect(10, 40 + index * 25, 150, 20);
            
            // 缁樺埗鏂囧瓧
            this.ctx.fillStyle = 'white';
            this.ctx.font = '14px Arial';
            
            // 濡傛灉鏈夊浘鏍囷紝缁樺埗鍥炬爣鍜屽悕锟?
            if (genre) {
              this.ctx.fillText(`${genre.icon} ${pref}`, 15, 55 + index * 25);
            } else {
              this.ctx.fillText(pref, 15, 55 + index * 25);
            }
          });
          
          // 濡傛灉鏈夋洿澶氬亸濂斤紝鏄剧ず"鏌ョ湅鏇村"
          if (uniquePreferences.length > 5) {
            this.ctx.fillStyle = 'yellow';
            this.ctx.fillText(`+ 杩樻湁 ${uniquePreferences.length - 5} 椤筦, 15, 60 + 5 * 25);
          }
        } catch (error) {
          console.error('缁樺埗鍋忓ソ鏃跺嚭锟?', error);
        }
      },
      
      // 娣诲姞鏆傚仠鐢婚潰缁樺埗
      drawPauseScreen() {
        try {
          if (!this.ctx || !this.canvas) return;
          
          // 缁樺埗鍗婇€忔槑鑳屾櫙
          this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
          
          // 缁樺埗鏆傚仠鏂囧瓧
          this.ctx.fillStyle = '#ffffff';
          this.ctx.font = 'bold 36px Arial';
          this.ctx.textAlign = 'center';
          this.ctx.fillText('娓告垙鏆傚仠', this.canvas.width / 2, this.canvas.height / 2);
          
          // 缁樺埗鎻愮ず淇℃伅
          this.ctx.font = '20px Arial';
          this.ctx.fillText('鎸夌┖鏍奸敭缁х画', this.canvas.width / 2, this.canvas.height / 2 + 40);
          
          // 缁樺埗宸叉敹闆嗙殑鍋忓ソ
          this.drawPreferences();
        } catch (error) {
          console.error('缁樺埗鏆傚仠鐢婚潰鏃跺嚭锟?', error);
        }
      },
      
      // 娣诲姞绮掑瓙鏁堟灉鍒涘缓
      createParticles(x, y, color = '#ffff00') {
        if (!this.particles) this.particles = [];
        
        // 鍒涘缓鐖嗗彂鏁堟灉鐨勭矑锟?
        for (let i = 0; i < 15; i++) {
          const angle = Math.random() * Math.PI * 2;
          const speed = Math.random() * 3 + 1;
          
          this.particles.push({
            x,
            y,
            radius: Math.random() * 4 + 1,
            color,
            speedX: Math.cos(angle) * speed,
            speedY: Math.sin(angle) * speed,
            life: 30 + Math.random() * 20
          });
        }
      },
      
      // 鏇存柊绮掑瓙鏁堟灉
      updateParticles() {
        if (!this.particles || !Array.isArray(this.particles)) return;
        
        for (let i = 0; i < this.particles.length; i++) {
          let p = this.particles[i];
          
          // 鏇存柊绮掑瓙浣嶇疆
          p.x += p.speedX;
          p.y += p.speedY;
          
          // 杈圭晫妫€鏌?
          if (p.x < 0 || p.x > this.canvas.width) {
            p.speedX *= -1;
          }
          
          
          if (p.y < 0 || p.y > this.canvas.height) {
            p.speedY *= -1;
          }
        }
      },
      
      // 缁樺埗绮掑瓙鏁堟灉
      drawParticles() {
        if (!this.ctx || !this.particles) return;
        
        this.particles.forEach(particle => {
          // 璁剧疆閫忔槑搴︽牴鎹敓鍛斤拷?
          const alpha = particle.life / 50;
          
          this.ctx.globalAlpha = alpha;
          this.ctx.fillStyle = particle.color;
          
          // 缁樺埗鍦嗗舰绮掑瓙
          this.ctx.beginPath();
          this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
          this.ctx.fill();
          
          // 閲嶇疆閫忔槑锟?
          this.ctx.globalAlpha = 1;
        });
      },
      
      // 娓告垙鍚姩鏂规硶
      startGame() {
        try {
          console.log('娓告垙寮€锟?');
          
          // 濡傛灉娌℃湁canvas鎴栦笂涓嬫枃锛屽皾璇曞啀娆″垵濮嬪寲
          if (!this.canvas || !this.ctx) {
            console.log('娓告垙鏈垵濮嬪寲锛屽皾璇曞垵濮嬪寲');
            if (!this.init()) {
              console.error('鏃犳硶鍒濆鍖栨父锟?');
              return;
            }
          }
          
          // 纭繚player瀵硅薄宸插垵濮嬪寲
          if (!this.player) {
            console.log('鐜╁瀵硅薄鏈垵濮嬪寲锛岄噸鏂板垱寤虹帺锟?);
            this.player = {
              x: 100,
              y: 300,
              width: 40,
              height: 40,
              speed: 5,
              dy: 0,
              jumping: false,
              grounded: true
            };
            this.ground = this.canvas.height - this.player.height;
          } else {
            // 閲嶇疆鐜╁浣嶇疆
            this.player.x = 100;
            this.player.y = 300;
            this.player.dy = 0;
            this.player.grounded = true;
          }
          
          // 杩涘叆鍏ㄥ睆妯″紡
          this.toggleFullscreen();
          
          this.gameRunning = true;
          this.gamePaused = false;
          this.gameCompleted = false;
          
          // 娣诲姞閲嶅姏鍙傛暟
          this.gravity = 0.5;
          
          // 瀹氫箟闊充箰椋庢牸鍒楄〃锛岀‘淇濆瓨锟?
          this.genres = [
            { name: '娴佽闊充箰', icon: '馃幍', color: '#FF5733' },
            { name: '鎽囨粴', icon: '馃', color: '#C70039' },
            { name: '鍙ゅ吀', icon: '馃幓', color: '#900C3F' },
            { name: '鐖靛＋', icon: '馃幏', color: '#581845' },
            { name: '鐢靛瓙', icon: '馃帶', color: '#FFC300' },
            { name: '鍢诲搱', icon: '馃帳', color: '#DAF7A6' },
            { name: '姘戣埃', icon: '馃獣', color: '#FF5733' },
            { name: '钃濊皟', icon: '馃幐', color: '#C70039' }
          ];
          
          // 鍒濆鍖栨垨淇濈暀涔嬪墠鐨勭敤鎴峰亸锟?
          if (!this.preferences || this.preferences.length === 0) {
            this.preferences = [];
            // 鍔犺浇涔嬪墠淇濆瓨鐨勫亸锟?
            if (typeof this.loadUserPreferences === 'function') {
              this.loadUserPreferences();
            }
          }
          
          // 閲嶇疆鏀堕泦鐗╁拰鍒嗘暟
          this.collectibles = [];
          this.particles = [];
          this.score = 0;
          
          // 鐢熸垚鏀堕泦锟?
          this.generateCollectibles();
          
          // 鏇存柊鍋忓ソ鏄剧ず锛屽鏋滄柟娉曞瓨锟?
          if (typeof this.updatePreferenceDisplay === 'function') {
            this.updatePreferenceDisplay();
          }
          
          // 鍚姩娓告垙寰幆
          if (this._gameLoopId) {
            cancelAnimationFrame(this._gameLoopId);
          }
          
          // 绔嬪嵆鎵ц涓€娆＄粯鍒讹紝纭繚灞忓箷涓嶆槸榛戠殑
          this.gameLoop();
          
          // 闅愯棌寮€濮嬫寜閽紝鏄剧ず瀹屾垚鎸夐挳
          const startButton = document.getElementById('game-start');
          if (startButton) {
            startButton.disabled = true;
            console.log('宸茬鐢ㄥ紑濮嬫父鎴忔寜锟?);
          }
          
          // 纭繚瀹屾垚鎸夐挳闅愯棌
          const finishButton = document.getElementById('game-finish');
          if (finishButton) {
            finishButton.classList.add('is-hidden');
            console.log('宸查殣钘忓畬鎴愭父鎴忔寜锟?);
          }
          
          // 閲嶆柊缁戝畾閿洏浜嬩欢
          this.setupEventListeners();
          
          console.log('娓告垙鍚姩鎴愬姛!');
          console.log('璇蜂娇鐢ㄦ柟鍚戦敭绉诲姩: 锟?锟?宸﹀彸绉诲姩, 锟?璺宠穬');
        } catch (error) {
          console.error('鍚姩娓告垙鏃跺嚭锟?', error);
        }
      },
      
      // 淇敼閿洏浜嬩欢澶勭悊鍣紝娣诲姞鏆傚仠鍔熻兘
      _createKeydownHandler() {
        return (e) => {
          console.log('閿洏浜嬩欢:', e.key, '娓告垙杩愯鐘讹拷?', this.gameRunning);
          
          if (!this.gameRunning) return;
          
          // 绌烘牸閿殏锟?缁х画娓告垙
          if (e.key === ' ' || e.code === 'Space') {
            this.gamePaused = !this.gamePaused;
            console.log(this.gamePaused ? '娓告垙宸叉殏锟? : '娓告垙宸茬户锟?);
            
            // 濡傛灉缁х画娓告垙锛岀珛鍗冲惎鍔ㄦ父鎴忓惊锟?
            if (!this.gamePaused) {
              cancelAnimationFrame(this._gameLoopId);
              this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
            }
            return;
          }
          
          // 濡傛灉娓告垙鏆傚仠锛屼笉澶勭悊鍏朵粬鎸夐敭
          if (this.gamePaused || !this.player) return;
          
          // 涓轰簡澧炲姞杩愬姩鐨勫搷搴旀€э紝鎴戜滑绔嬪嵆鏇存柊鐜╁浣嶇疆骞堕噸锟?
          let needsRedraw = false;
          
          if (e.key === 'ArrowLeft') {
            this.player.x -= this.player.speed;
            console.log('鐜╁宸︾Щ:', this.player.x);
            needsRedraw = true;
          }
          
          if (e.key === 'ArrowRight') {
            this.player.x += this.player.speed;
            console.log('鐜╁鍙崇Щ:', this.player.x);
            needsRedraw = true;
          }
          
          if (e.key === 'ArrowUp' && this.player.grounded) {
            this.player.dy = -12; // 璺宠穬鍔涘害
            this.player.grounded = false;
            console.log('鐜╁璺宠穬:', this.player.dy);
            needsRedraw = true;
          }
          
          // 纭繚鐜╁涓嶄細瓒呭嚭鐢诲竷
          if (this.player.x < 0) this.player.x = 0;
          if (this.player.x + this.player.width > this.canvas.width) {
            this.player.x = this.canvas.width - this.player.width;
          }
          
          // 鎸塃SC閿€€鍑哄叏锟?
          if (e.key === 'Escape' && document.fullscreenElement) {
            document.exitFullscreen().catch(err => {
              console.error('鏃犳硶閫€鍑哄叏灞忔ā锟?', err);
            });
          }
          
          // 濡傛灉鏈変綅缃洿鏂帮紝绔嬪嵆閲嶇粯娓告垙鐢婚潰
          if (needsRedraw && this.canvas && this.ctx) {
            // 寮哄埗閲嶇粯涓€锟?
            cancelAnimationFrame(this._gameLoopId);
            this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
          }
        };
      },
      
      // 缁樺埗鐜╁
      drawPlayer() {
        try {
          if (!this.ctx || !this.player) {
            console.error('drawPlayer: 缂哄皯蹇呰瀵硅薄', {
              ctx: !!this.ctx,
              player: !!this.player
            });
            return;
          }
          
          // 鐜╁瑙掕壊
          this.ctx.fillStyle = 'red';  // 鏇撮矞鑹崇殑棰滆壊锛屼究浜庤瘑锟?
          this.ctx.fillRect(
            this.player.x,
            this.player.y,
            this.player.width,
            this.player.height
          );
          
          // 娣诲姞鐜╁杞粨锛屾洿瀹规槗鐪嬭
          this.ctx.strokeStyle = 'white';
          this.ctx.lineWidth = 2;
          this.ctx.strokeRect(
            this.player.x,
            this.player.y,
            this.player.width,
            this.player.height
          );
          
          // 娣诲姞鐪肩潧锛岃鐜╁鐪嬭捣鏉ユ洿鐢熷姩
          this.ctx.fillStyle = 'white';
          this.ctx.fillRect(
            this.player.x + 8, 
            this.player.y + 10,
            6,
            6
          );
          this.ctx.fillRect(
            this.player.x + this.player.width - 14, 
            this.player.y + 10,
            6,
            6
          );
        } catch (error) {
          console.error('缁樺埗鐜╁鏃跺嚭锟?', error);
        }
      },
      
      // 淇敼鏀堕泦鐗╃粯鍒舵柟娉曪紝纭繚鍙
      drawCollectibles() {
        try {
          // 纭繚collectibles鏁扮粍瀛樺湪
          if (!this.collectibles || !this.ctx) {
            console.error('drawCollectibles: 缂哄皯蹇呰瀵硅薄', {
              collectibles: !!this.collectibles,
              ctx: !!this.ctx
            });
            return;
          }
          
          // 缁樺埗鍙敹闆嗙殑闊充箰椋庢牸鍥炬爣
          this.collectibles.forEach(collectible => {
            // 鍙粯鍒舵湭鏀堕泦鍜屾湭閬垮紑鐨勬敹闆嗙墿
            if (!collectible.collected && !collectible.avoided) {
              // 缁樺埗鍥炬爣鑳屾櫙
              this.ctx.fillStyle = collectible.genre.color;
              this.ctx.beginPath();
              this.ctx.arc(
                collectible.x + collectible.width/2,
                collectible.y + collectible.height/2,
                collectible.width/2,
                0,
                Math.PI * 2
              );
              this.ctx.fill();
              
              // 娣诲姞杈规锛屼娇鍏舵洿瀹规槗鐪嬭
              this.ctx.strokeStyle = 'white';
              this.ctx.lineWidth = 2;
              this.ctx.stroke();
              
              // 缁樺埗鍥炬爣
              this.ctx.fillStyle = '#FFF';
              this.ctx.font = '20px Arial';
              this.ctx.textAlign = 'center';
              this.ctx.textBaseline = 'middle';
              this.ctx.fillText(
                collectible.genre.icon,
                collectible.x + collectible.width/2,
                collectible.y + collectible.height/2
              );
              
              // 缁樺埗闊充箰椋庢牸鍚嶇О
              this.ctx.fillStyle = '#FFF';
              this.ctx.font = 'bold 14px Arial';
              this.ctx.textAlign = 'center';
              
              // 娣诲姞鍚嶇О鑳屾櫙浠ユ彁楂樺彲璇伙拷?
              const textWidth = this.ctx.measureText(collectible.genre.name).width;
              this.ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
              this.ctx.fillRect(
                collectible.x + collectible.width/2 - textWidth/2 - 5,
                collectible.y + collectible.height + 10,
                textWidth + 10,
                20
              );
              
              // 閲嶆柊缁樺埗鏂囧瓧浣垮叾鍦ㄨ儗鏅笂锟?
              this.ctx.fillStyle = '#FFF';
              this.ctx.fillText(
                collectible.genre.name,
                collectible.x + collectible.width/2,
                collectible.y + collectible.height + 20
              );
            }
          });
          
          // 娣诲姞鏃ュ織锛屾樉绀烘敹闆嗙墿鏁伴噺
          if (this.collectibles.length > 0) {
            const activeCount = this.collectibles.filter(c => !c.collected && !c.avoided).length;
            if (activeCount === 0) {
              // 濡傛灉娌℃湁娲诲姩鐨勬敹闆嗙墿锛岀敓鎴愭洿锟?
              console.log('娌℃湁娲诲姩鐨勬敹闆嗙墿锛岀敓鎴愭洿锟?);
              this.generateMoreCollectibles(5);
            }
          }
        } catch (error) {
          console.error('缁樺埗鏀堕泦鐗╂椂鍑洪敊:', error);
        }
      },
      
      drawScore() {
        // 缁樺埗寰楀垎
        this.ctx.fillStyle = '#FFF';
        this.ctx.font = '16px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`寰楀垎: ${this.score}`, 10, 20);
        this.ctx.fillText(`宸叉敹锟? ${this.preferences.length}/${this.genres.length}`, 10, 40);
      },
      
      showFinishScreen() {
        this.gameRunning = false;
        
        // 娓呯┖鐢诲竷
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 缁樺埗鑳屾櫙
        this.ctx.fillStyle = '#191919';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        // 鍒涘缓鏇村搴嗙绮掑瓙鏁堟灉
        const celebrationParticles = [];
        for (let i = 0; i < 100; i++) {
          celebrationParticles.push({
            x: Math.random() * this.canvas.width,
            y: Math.random() * this.canvas.height,
            radius: Math.random() * 4 + 1,
            color: `rgba(155, 75, 255, ${Math.random() * 0.7 + 0.3})`,
            speedX: (Math.random() - 0.5) * 3,
            speedY: (Math.random() - 0.5) * 3
          });
        }
        
        // 缁樺埗绱壊鏍囬
        this.ctx.fillStyle = '#9B4BFF';
        this.ctx.font = '32px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
          '鎭枩瀹屾垚鏀堕泦!',
          this.canvas.width / 2,
          this.canvas.height / 2 - 50
        );
        
        // 缁樺埗瀹屾垚淇℃伅
        this.ctx.fillStyle = '#FFF';
        this.ctx.font = '24px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
          '鏀堕泦瀹屾垚! 鐐瑰嚮涓嬫柟"瀹屾垚"鎸夐挳淇濆瓨鎮ㄧ殑闊充箰鍋忓ソ!',
          this.canvas.width / 2,
          this.canvas.height / 2
        );
        
        // 鏄剧ず瀹屾垚鎸夐挳
        const finishButton = document.getElementById('game-finish');
        if (finishButton) {
          finishButton.classList.remove('is-hidden');
        }
        
        // 閲嶇疆寮€濮嬫寜锟?
        const startButton = document.getElementById('game-start');
        if (startButton) {
          startButton.disabled = false;
        }
        
        // 鎸佺画鏇存柊搴嗙绮掑瓙鏁堟灉
        const updateCelebration = () => {
          if (!this.gameRunning) {
            // 娓呯┖鐢诲竷
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            this.ctx.fillStyle = '#191919';
            this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
            
            // 鏇存柊骞剁粯鍒剁矑锟?
            celebrationParticles.forEach(particle => {
              particle.x += particle.speedX;
              particle.y += particle.speedY;
              
              // 濡傛灉绮掑瓙瓒呭嚭杈圭晫锛屽皢鍏剁Щ鍔ㄥ埌鍙︿竴锟?
              if (particle.x < 0) particle.x = this.canvas.width;
              if (particle.x > this.canvas.width) particle.x = 0;
              if (particle.y < 0) particle.y = this.canvas.height;
              if (particle.y > this.canvas.height) particle.y = 0;
              
              // 缁樺埗绮掑瓙
              this.ctx.beginPath();
              this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
              this.ctx.fillStyle = particle.color;
              this.ctx.fill();
            });
            
            // 閲嶆柊缁樺埗鏂囧瓧
            this.ctx.fillStyle = '#9B4BFF';
            this.ctx.font = '32px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
              '鎭枩瀹屾垚鏀堕泦!',
              this.canvas.width / 2,
              this.canvas.height / 2 - 50
            );
            
            this.ctx.fillStyle = '#FFF';
            this.ctx.font = '24px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText(
              '鏀堕泦瀹屾垚! 鐐瑰嚮涓嬫柟"瀹屾垚"鎸夐挳淇濆瓨鎮ㄧ殑闊充箰鍋忓ソ!',
              this.canvas.width / 2,
              this.canvas.height / 2
            );
            
            requestAnimationFrame(updateCelebration);
          }
        };
        
        // 鍚姩搴嗙鍔ㄧ敾
        updateCelebration();
        
        console.log('娓告垙瀹屾垚锛屾樉绀虹粨鏉熺晫锟?);
      },
      
      finishGame() {
        console.log('瀹屾垚娓告垙锛屼繚瀛樺亸锟?);
        // 鑾峰彇鐢ㄦ埛鍦ㄦ父鎴忎腑鏀堕泦鐨勯煶涔愬亸锟?
        const preferences = this.preferences.map(p => p.id);
        
        // 璋冪敤Vue搴旂敤涓殑鏂规硶淇濆瓨鍋忓ソ
        if (window.app) {
          console.log('姝ｅ湪淇濆瓨娓告垙鏀堕泦鐨勯煶涔愬亸锟?', preferences);
          window.app.saveGamePreferences(preferences);
        } else {
          console.error('鏃犳硶璁块棶Vue搴旂敤瀹炰緥锛屾棤娉曚繚瀛橀煶涔愬亸锟?);
        }
        
        // 闅愯棌瀹屾垚鎸夐挳
        const finishButton = document.getElementById('game-finish');
        if (finishButton) {
          finishButton.classList.add('is-hidden');
        }
      },
      
      // 娣诲姞娴嬭瘯鍔熻兘鍜岃瘖鏂俊锟?
      diagnoseGameState() {
        console.log('==================');
        console.log('娓告垙璇婃柇淇℃伅:');
        console.log('Canvas:', !!this.canvas);
        console.log('涓婁笅锟?', !!this.ctx);
        console.log('鐜╁:', !!this.player);
        if (this.player) {
          console.log('鐜╁浣嶇疆:', this.player.x, this.player.y);
        }
        console.log('娓告垙杩愯鐘讹拷?', this.gameRunning);
        console.log('娓告垙鏆傚仠鐘讹拷?', this.gamePaused);
        console.log('鏀堕泦鐗╂暟锟?', this.collectibles ? this.collectibles.length : 0);
        console.log('娲诲姩鏀堕泦锟?', this.collectibles ? this.collectibles.filter(c => !c.collected && !c.avoided).length : 0);
        console.log('閿洏浜嬩欢缁戝畾:', !!this._keydownHandler);
        console.log('娓告垙寰幆ID:', !!this._gameLoopId);
        console.log('==================');
        
        // 灏濊瘯淇甯歌闂
        if (!this.gameRunning && this.player) {
          console.log('娓告垙鏈繍琛岋紝灏濊瘯閲嶆柊鍚姩');
          this.gameRunning = true;
          this.gamePaused = false;
          this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
        }
        
        if (this.gameRunning && (!this.collectibles || this.collectibles.length === 0)) {
          console.log('娌℃湁鏀堕泦鐗╋紝灏濊瘯鐢熸垚');
          this.generateCollectibles();
        }
        
        if (!this._keydownHandler) {
          console.log('閿洏浜嬩欢鏈粦瀹氾紝灏濊瘯閲嶆柊缁戝畾');
          this.setupEventListeners();
        }
        
        // 寮哄埗閲嶇粯涓€锟?
        if (this.canvas && this.ctx) {
          this.gameLoop();
          console.log('寮哄埗閲嶇粯瀹屾垚');
        }
      }
    }
  });
  
  // 灏嗕簨浠舵€荤嚎鏆撮湶缁欏叏灞€锛屼互渚跨粍浠堕棿閫氫俊
  window.EventBus = EventBus;
  
  // 娣诲姞Vue鍏ㄥ眬閿欒澶勭悊
  Vue.config.errorHandler = function(err, vm, info) {
    console.error('Vue閿欒:', err);
    console.error('缁勪欢:', vm);
    console.error('淇℃伅:', info);
  };
}); 

// 闊充箰鍋忓ソ娓告垙瀵硅薄
const musicGame = {
  // 娓告垙鍏ㄥ眬灞炴€э紝纭繚杩欎簺鍙橀噺涓嶄細琚噸锟?
  canvas: null,
  ctx: null,
  player: null,
  preferences: [],
  genres: [
    { id: 'pop', name: '娴佽', icon: '馃幍', color: '#ff5252' },
    { id: 'rock', name: '鎽囨粴', icon: '馃幐', color: '#ff9800' },
    { id: 'classical', name: '鍙ゅ吀', icon: '馃幓', color: '#9c27b0' },
    { id: 'electronic', name: '鐢靛瓙', icon: '馃帶', color: '#2196f3' },
    { id: 'jazz', name: '鐖靛＋', icon: '馃幏', color: '#4caf50' },
    { id: 'hiphop', name: '鍢诲搱', icon: '馃帳', color: '#795548' },
    { id: 'folk', name: '姘戣埃', icon: '馃獣', color: '#607d8b' },
    { id: 'rb', name: 'R&B', icon: '馃幑', color: '#e91e63' }
  ],
  collectibles: [],
  gameRunning: false,
  score: 0,
  gravity: 0.5,
  ground: 0,
  particles: [],
  _initialized: false,
  _eventListenersSet: false,
  _gameLoopId: null,
  
  init() {
    try {
      console.log('姝ｅ湪鍒濆鍖栭煶涔愭父锟?..');
      
      // 鑾峰彇canvas鍏冪礌
      this.canvas = document.getElementById('game-canvas');
      if (!this.canvas) {
        console.error('鎵句笉鍒版父鎴忕敾甯冨厓锟?');
        
        // 灏濊瘯鍒涘缓canvas
        const container = document.querySelector('.game-container');
        if (container) {
          console.log('灏濊瘯鍒涘缓canvas鍏冪礌...');
          this.canvas = document.createElement('canvas');
          this.canvas.id = 'game-canvas';
          this.canvas.width = container.clientWidth;
          this.canvas.height = 400;
          container.appendChild(this.canvas);
          console.log('宸插垱寤篶anvas鍏冪礌:', this.canvas);
        } else {
          console.error('鎵句笉鍒癵ame-container锛屾棤娉曞垱寤篶anvas');
          return false;
        }
      }
      
      // 鑾峰彇2D涓婁笅锟?
      this.ctx = this.canvas.getContext('2d');
      if (!this.ctx) {
        console.error('鏃犳硶鑾峰彇canvas 2D涓婁笅锟?);
        return false;
      }
      
      console.log('Canvas鍑嗗灏辩华:', this.canvas.width, 'x', this.canvas.height);
      
      // 鍒濆鍖栨父鎴忓彉锟?
      this.gravity = 0.5;
      this.score = 0;
      this.preferences = [];
      this.collectibles = [];
      this.particles = [];
      this.gameRunning = false;
      this.gamePaused = false;
      this.gameCompleted = false;
      
      // 瀹氫箟闊充箰椋庢牸
      this.genres = [
        { name: '娴佽闊充箰', icon: '馃幍', color: '#FF5733' },
        { name: '鎽囨粴', icon: '馃', color: '#C70039' },
        { name: '鍙ゅ吀', icon: '馃幓', color: '#900C3F' },
        { name: '鐖靛＋', icon: '馃幏', color: '#581845' },
        { name: '鐢靛瓙', icon: '馃帶', color: '#FFC300' },
        { name: '鍢诲搱', icon: '馃帳', color: '#DAF7A6' },
        { name: '姘戣埃', icon: '馃獣', color: '#FF5733' },
        { name: '钃濊皟', icon: '馃幐', color: '#C70039' }
      ];
      
      // 鍒涘缓鐜╁瀵硅薄
      this.player = {
        x: 100,
        y: 300,
        width: 40,
        height: 40,
        speed: 5,
        dy: 0,
        jumping: false,
        grounded: true
      };
      
      // 璁＄畻鍦伴潰楂樺害
      this.ground = this.canvas.height - this.player.height;
      
      // 璁剧疆浜嬩欢鐩戝惉锟?
      this.setupEventListeners();
      
      // 缁樺埗鍒濆灞忓箷
      this.drawInitialScreen();
      
      console.log('娓告垙鍒濆鍖栧畬鎴愶紝鍦伴潰浣嶇疆:', this.ground);
      return true;
      
    } catch (error) {
      console.error('鍒濆鍖栨父鎴忔椂鍑洪敊:', error);
      return false;
    }
  },
  
  // 寮€濮嬫父锟?
  startGame() {
    try {
      console.log('===== 寮€濮嬫父锟?=====');
      
      // 1. 纭canvas鍜宑tx宸插垵濮嬪寲
      if (!this.canvas || !this.ctx) {
        console.log('Canvas鏈垵濮嬪寲锛屽皾璇曞垵濮嬪寲...');
        if (!this.init()) {
          alert('鏃犳硶鍒濆鍖栨父鎴忥紝璇峰埛鏂伴〉闈㈤噸锟?);
          return;
        }
      }
      
      console.log('Canvas鐘讹拷?', this.canvas.width, 'x', this.canvas.height);
      
      // 2. 閲嶇疆娓告垙鐘讹拷?
      this.gameRunning = true;
      this.gamePaused = false;
      this.gameCompleted = false;
      this.score = 0;
      
      // 3. 閲嶇疆鐜╁浣嶇疆
      this.player = {
        x: 100,
        y: 300,
        width: 40,
        height: 40,
        speed: 5,
        dy: 0,
        jumping: false,
        grounded: true
      };
      
      this.ground = this.canvas.height - this.player.height;
      console.log('鐜╁宸查噸锟?', this.player);
      
      // 4. 娓呯┖骞堕噸鏂扮敓鎴愭敹闆嗙墿
      this.collectibles = [];
      this.generateCollectibles();
      console.log('鏀堕泦鐗╁凡鐢熸垚:', this.collectibles.length);
      
      // 5. 娓呯┖绮掑瓙鏁堟灉
      this.particles = [];
      
      // 6. 灏濊瘯杩涘叆鍏ㄥ睆
      this.toggleFullscreen();
      
      // 7. 绔嬪嵆缁樺埗涓€甯ф父鎴忕敾闈紝閬垮厤榛戝睆
      this.gameLoop();
      console.log('绗竴甯ф父鎴忓凡缁樺埗');
      
      // 8. 闅愯棌/鏄剧ず鐩稿叧鎸夐挳
      const startButton = document.getElementById('game-start');
      if (startButton) {
        startButton.disabled = true;
      }
      
      const finishButton = document.getElementById('game-finish');
      if (finishButton) {
        finishButton.classList.add('is-hidden');
      }
      
      // 9. 璋冭瘯杈撳嚭
      console.log('娓告垙鎴愬姛鍚姩! 璇蜂娇鐢ㄦ柟鍚戦敭鎺у埗:');
      console.log('- 宸﹀彸绠ご: 绉诲姩瑙掕壊');
      console.log('- 涓婄锟? 璺宠穬');
      console.log('- 绌烘牸锟? 鏆傚仠/缁х画');
      console.log('===================');
      
    } catch (error) {
      console.error('鍚姩娓告垙鏃跺嚭锟?', error);
      alert('鍚姩娓告垙鏃跺嚭閿欙紝璇峰皾璇曞埛鏂伴〉锟?);
    }
  },
  
  // 娓告垙寰幆
  gameLoop() {
    try {
      // 纭繚蹇呰鐨勫璞″瓨锟?
      if (!this.canvas || !this.ctx || !this.player) {
        console.error('gameLoop: 缂哄皯蹇呰瀵硅薄', {
          canvas: !!this.canvas,
          ctx: !!this.ctx,
          player: !!this.player
        });
        return;
      }
      
      // 娓呴櫎鐢诲竷
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      
      // 缁樺埗鑳屾櫙
      this.ctx.fillStyle = '#191919';
      this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
      
      // 缁樺埗鍦伴潰
      this.ctx.fillStyle = '#333';
      this.ctx.fillRect(0, this.ground + this.player.height, this.canvas.width, 5);
      
      // 濡傛灉娓告垙宸叉殏鍋滐紝鏄剧ず鏆傚仠灞忓箷
      if (this.gamePaused) {
        this.drawPauseScreen();
        return;
      }
      
      // 濡傛灉娓告垙宸插畬鎴愶紝鏄剧ず瀹屾垚灞忓箷
      if (this.gameCompleted) {
        this.showFinishScreen();
        return;
      }
      
      // 鏇存柊鐜╁
      this.updatePlayer();
      
      // 鏇存柊鏀堕泦鐗╀綅锟?
      this.updateCollectibles();
      
      // 妫€鏌ョ锟?
      this.checkCollisions();
      
      // 鏇存柊绮掑瓙鏁堟灉
      if (typeof this.updateParticles === 'function') {
        this.updateParticles();
      }
      
      // 缁樺埗鐜╁
      this.drawPlayer();
      
      // 缁樺埗鏀堕泦锟?
      this.drawCollectibles();
      
      // 缁樺埗绮掑瓙鏁堟灉
      if (typeof this.drawParticles === 'function') {
        this.drawParticles();
      }
      
      // 缁樺埗鍒嗘暟
      this.drawScore();
      
      // 缁樺埗鍋忓ソ鏄剧ず
      if (typeof this.drawPreferences === 'function') {
        this.drawPreferences();
      }
      
      // 缁х画娓告垙寰幆
      if (this.gameRunning) {
        this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
      }
    } catch (error) {
      console.error('娓告垙寰幆涓嚭锟?', error);
      
      // 灏濊瘯鎭㈠娓告垙杩愯
      if (this.gameRunning) {
        this._gameLoopId = requestAnimationFrame(() => this.gameLoop());
      }
    }
  },
  
  // 鍒涘缓绮掑瓙鏁堟灉
  createParticles() {
    // 娓呯┖鐜版湁绮掑瓙
    this.particles = [];
    
    // 鍒涘缓鏂扮矑锟?
    for (let i = 0; i < 50; i++) {
      this.particles.push({
        x: Math.random() * this.canvas.width,
        y: Math.random() * this.canvas.height,
        radius: Math.random() * 3 + 1,
        color: `rgba(155, 75, 255, ${Math.random() * 0.5 + 0.1})`,
        speedX: Math.random() * 0.5 - 0.25,
        speedY: Math.random() * 0.5 - 0.25
      });
    }
  },
  
  // 鏇存柊绮掑瓙
  updateParticles() {
    if (!this.particles || !Array.isArray(this.particles)) return;
    
    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];
      
      // 鏇存柊绮掑瓙浣嶇疆
      p.x += p.speedX;
      p.y += p.speedY;
      
      // 杈圭晫妫€鏌?
      if (p.x < 0 || p.x > this.canvas.width) {
        p.speedX *= -1;
      }
      
      if (p.y < 0 || p.y > this.canvas.height) {
        p.speedY *= -1;
      }
    }
  },
  
  // 褰揤ue搴旂敤鍔犺浇瀹屾垚锟?鎴栦竴娈垫椂闂村悗)鍒濆鍖栨父锟?
  setTimeout(() => {
    initGameWhenCanvasReady();
  }, 1000);
}; 

// 鍦╠ocument.addEventListener('DOMContentLoaded')鍚庢坊鍔犳覆鏌撲慨澶嶅嚱锟?

// 娣诲姞淇Vue.js妯℃澘娓叉煋闂鐨勫嚱锟?
function fixTemplateRendering() {
  console.log('姝ｅ湪淇妯℃澘娓叉煋闂...');
  
  // 妫€鏌ue瀹炰緥鏄惁宸叉纭垵濮嬪寲
  if (!window.app) {
    console.error('Vue瀹炰緥鏈壘鍒帮紝灏濊瘯閲嶆柊鍒濆锟?);
    
    // 纭繚translations瀵硅薄瀛樺湪锛岀敤浜庢ā鏉挎覆锟?
    const translations = {
      zh: {
        'username': '鐢ㄦ埛锟?,
        'email': '閭',
        'password': '瀵嗙爜',
        'login': '鐧诲綍',
        'register': '娉ㄥ唽',
        'logout': '閫€锟?,
        'developer': '寮€鍙戯拷?,
        'registerPrompt': '杩樻病鏈夎处鍙凤紵鐐瑰嚮娉ㄥ唽',
        'loginPrompt': '宸叉湁璐﹀彿锛熺偣鍑荤櫥锟?,
        'welcome': '娆㈣繋',
        'survey': '闂嵎',
        'recommendations': '鎺ㄨ崘',
        'chat': '鑱婂ぉ',
        'evaluation': '璇勪环',
        'musicPreference': '闊充箰鍋忓ソ',
        'submit': '鎻愪氦',
        'next': '涓嬩竴锟?,
        'previous': '涓婁竴锟?,
        'searchPlaceholder': '鎼滅储闊充箰...',
        'songName': '姝屾洸锟?,
        'artist': '鑹烘湳锟?,
        'rating': '璇勫垎'
      },
      en: {
        'username': 'Username',
        'email': 'Email',
        'password': 'Password',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'developer': 'Developer',
        'registerPrompt': 'No account? Register here',
        'loginPrompt': 'Already have an account? Login here',
        'welcome': 'Welcome',
        'survey': 'Survey',
        'recommendations': 'Recommendations',
        'chat': 'Chat',
        'evaluation': 'Evaluation',
        'musicPreference': 'Music Preference',
        'submit': 'Submit',
        'next': 'Next',
        'previous': 'Previous',
        'searchPlaceholder': 'Search music...',
        'songName': 'Song Name',
        'artist': 'Artist',
        'rating': 'Rating'
      }
    };
    
    // 閲嶆柊鍒濆鍖朧ue搴旂敤
    try {
      window.app = new Vue({
        el: '#app',
        data: {
          currentTab: 'welcome',
          username: '',
          email: '',
          password: '',
          currentLanguage: 'zh',
          isLoggedIn: false,
          currentUser: null,
          loginError: '',
          registerError: '',
          translations: translations,
          notifications: []
        },
        methods: {
          t(key) {
            if (!key) return '';
            try {
              const translation = this.translations[this.currentLanguage][key];
              return translation || key;
            } catch (e) {
              console.error('缈昏瘧閿欒:', e);
              return key;
            }
          },
          switchLanguage(lang) {
            console.log('鍒囨崲璇█锟?', lang);
            this.currentLanguage = lang;
            localStorage.setItem('language', lang);
          },
          login() {
            console.log('灏濊瘯鐧诲綍...');
            // 鐧诲綍閫昏緫
          },
          logout() {
            console.log('鐧诲嚭...');
            this.isLoggedIn = false;
            this.currentUser = null;
          },
          register() {
            console.log('灏濊瘯娉ㄥ唽...');
            // 娉ㄥ唽閫昏緫
          }
        },
        mounted() {
          // 寮哄埗绔嬪嵆鏇存柊鎵€鏈夌粦瀹氾紝纭繚妯℃澘娓叉煋
          this.$nextTick(() => {
            this.$forceUpdate();
            console.log('寮哄埗鏇存柊Vue瀹炰緥瀹屾垚');
          });
        }
      });
      
      console.log('Vue瀹炰緥宸查噸鏂板垵濮嬪寲');
    } catch (e) {
      console.error('閲嶆柊鍒濆鍖朧ue瀹炰緥澶辫触:', e);
    }
  } else {
    // 寮哄埗Vue瀹炰緥鏇存柊
    try {
      window.app.$forceUpdate();
      console.log('寮哄埗鍒锋柊Vue瀹炰緥');
    } catch (e) {
      console.error('寮哄埗鍒锋柊Vue瀹炰緥澶辫触:', e);
    }
  }
  
  // 鎵嬪姩鏇挎崲鎵€鏈夋湭娓叉煋鐨勬ā鏉胯锟?
  setTimeout(() => {
    console.log('妫€鏌ユ湭娓叉煋鐨勬ā鏉挎爣锟?..');
    
    // 鑾峰彇鎵€鏈夊彲鑳藉寘鍚湭娓叉煋妯℃澘鐨勫厓锟?
    const elements = document.querySelectorAll('*:not(script):not(style)');
    
    // 璁℃暟淇鐨勬ā鏉胯〃杈惧紡
    let fixCount = 0;
    
    elements.forEach(el => {
      // 妫€鏌ュ厓绱犵殑textContent鍜宨nnerHTML
      if (el.innerHTML && el.innerHTML.includes('{{') && el.innerHTML.includes('}}')) {
        const originalHTML = el.innerHTML;
        
        // 鏇挎崲 {{ t('key') }} 鏍煎紡鐨勬ā锟?
        let newHTML = originalHTML.replace(/\{\{\s*t\('([^']+)'\)\s*\}\}/g, (match, key) => {
          fixCount++;
          
          // 鑾峰彇缈昏瘧鏂囨湰
          let replacement = key;
          if (window.app && window.app.translations && 
              window.app.translations[window.app.currentLanguage] && 
              window.app.translations[window.app.currentLanguage][key]) {
            replacement = window.app.translations[window.app.currentLanguage][key];
          }
          
          return replacement;
        });
        
        // 濡傛灉鍙戠敓浜嗘浛鎹紝鏇存柊HTML
        if (newHTML !== originalHTML) {
          el.innerHTML = newHTML;
        }
      }
      
      // 妫€鏌ュ厓绱犵殑瀛愭枃鏈妭锟?
      if (el.childNodes && el.childNodes.length > 0) {
        el.childNodes.forEach(node => {
          if (node.nodeType === Node.TEXT_NODE) {
            const text = node.textContent;
            
            // 妫€鏌ユ槸鍚﹀寘鍚ā鏉胯锟?{{ ... }}
            if (text.includes('{{') && text.includes('}}')) {
              
              // 鏇挎崲鎵€鏈夋ā鏉胯〃杈惧紡
              let newText = text.replace(/\{\{\s*t\('([^']+)'\)\s*\}\}/g, (match, key) => {
                fixCount++;
                
                // 鑾峰彇缈昏瘧鏂囨湰
                let replacement = key;
                if (window.app && window.app.translations && 
                    window.app.translations[window.app.currentLanguage] && 
                    window.app.translations[window.app.currentLanguage][key]) {
                  replacement = window.app.translations[window.app.currentLanguage][key];
                }
                
                return replacement;
              });
              
              // 濡傛灉鍙戠敓浜嗘浛鎹紝鏇存柊鏂囨湰
              if (newText !== text) {
                node.textContent = newText;
              }
            }
          }
        });
      }
    });
    
    console.log(`淇锟?${fixCount} 涓ā鏉胯〃杈惧紡`);
    
    // 妫€鏌ユ槸鍚︿粛鏈夋湭娓叉煋鐨勬ā锟?
    const stillHasUnrenderedTemplates = document.body.innerHTML.includes('{{') && 
                                      document.body.innerHTML.includes('}}');
    
    if (stillHasUnrenderedTemplates) {
      console.log('浠嶆湁鏈覆鏌撶殑妯℃澘锛屽皾璇曢噸鏂颁慨锟?);
      setTimeout(fixTemplateRendering, 500);
    }
  }, 100);
}

// 纭繚椤甸潰瀹屽叏鍔犺浇鍚庤皟鐢ㄤ慨澶嶅嚱锟?
window.addEventListener('load', function() {
  console.log('椤甸潰瀹屽叏鍔犺浇锛屾鏌ユā鏉挎覆鏌撶姸锟?);
  
  // 妫€鏌ユ槸鍚︽湁鏈覆鏌撶殑妯℃澘鏍囪
  setTimeout(() => {
    const hasUnrenderedTemplates = document.body.innerHTML.includes('{{') && 
                                  document.body.innerHTML.includes('}}');
    
    if (hasUnrenderedTemplates) {
      console.log('妫€娴嬪埌鏈覆鏌撶殑妯℃澘锛屽皾璇曚慨锟?);
      fixTemplateRendering();
    } else {
      console.log('妯℃澘娓叉煋姝ｅ父');
    }
  }, 500); // 寤惰繜妫€鏌ヤ互纭繚Vue鏈夋椂闂存覆锟?
});

// 鐩戝惉DOM鍙樺寲锛屾鏌ユ槸鍚︽湁鏂版坊鍔犵殑鏈覆鏌撴ā锟?
document.addEventListener('DOMContentLoaded', function() {
  // 鍒涘缓MutationObserver瑙傚療DOM鍙樺寲
  const observer = new MutationObserver((mutations) => {
    let needsFix = false;
    
    // 妫€鏌ュ彉鍖栫殑鑺傜偣
    mutations.forEach(mutation => {
      if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
        // 妫€鏌ユ槸鍚︽坊鍔犱簡娓告垙鐢诲竷鍏冪礌
        if (checkAndInitCanvas()) {
          console.log('閫氳繃MutationObserver妫€娴嬪埌canvas骞跺垵濮嬪寲娓告垙');
          tabObserver.disconnect();
          return;
        }
      }
    });
  });
  
  // 鐩戝惉鏁翠釜鏂囨。鐨勫彉锟?
  tabObserver.observe(document.body, { 
    childList: true, 
    subtree: true 
  });
  
  // 澶囩敤鏂规硶锛氫娇鐢ㄩ棿闅旇疆璇㈡鏌anvas鏄惁宸插姞锟?
  let attempts = 0;
  const maxAttempts = 10;
  
  const canvasCheckInterval = setInterval(() => {
    attempts++;
    console.log(`妫€鏌ユ父鎴忕敾锟?- 灏濊瘯 ${attempts}/${maxAttempts}`);
    
    if (checkAndInitCanvas()) {
      console.log('閫氳繃杞妫€娴嬪埌canvas骞跺垵濮嬪寲娓告垙');
      clearInterval(canvasCheckInterval);
      return;
    }
    
    if (attempts >= maxAttempts) {
      console.error('澶氭灏濊瘯鍚庝粛鎵句笉鍒版父鎴忕敾甯冨厓锟?');
      clearInterval(canvasCheckInterval);
      
      // 灏濊瘯寮哄埗娣诲姞canvas鍏冪礌
      const gameScreen = document.querySelector('.game-screen');
      if (gameScreen) {
        console.log('灏濊瘯寮哄埗鍒涘缓canvas鍏冪礌...');
        const canvas = document.createElement('canvas');
        canvas.id = 'game-canvas';
        canvas.width = 800;
        canvas.height = 400;
        gameScreen.innerHTML = '';
        gameScreen.appendChild(canvas);
        
        setTimeout(() => {
          checkAndInitCanvas();
        }, 100);
      }
    }
  }, 1000);
}); 

/**
 * 淇妯℃澘娓叉煋闂
 * 妫€鏌ラ〉闈笂鏈覆鏌撶殑妯℃澘琛ㄨ揪寮忓苟鏇挎崲涓烘纭殑鏂囨湰
 */
const fixTemplateRendering = function() {
  console.log('姝ｅ湪妫€鏌ュ拰淇鏈覆鏌撶殑妯℃澘...');
  
  // 妫€鏌ue瀹炰緥鏄惁瀛樺湪
  if (!window.app) {
    console.error('Vue瀹炰緥涓嶅瓨鍦紝鏃犳硶淇妯℃澘');
    return;
  }
  
  // 瀹氫箟缈昏瘧瀵硅薄
  if (!window.app.translations) {
    window.app.translations = {
      zh: {
        'username': '鐢ㄦ埛锟?,
        'email': '閭',
        'password': '瀵嗙爜',
        'login': '鐧诲綍',
        'register': '娉ㄥ唽',
        'logout': '閫€锟?,
        'developer': '寮€鍙戯拷?,
        'registerPrompt': '杩樻病鏈夎处鍙凤紵鐐瑰嚮娉ㄥ唽',
        'loginPrompt': '宸叉湁璐﹀彿锛熺偣鍑荤櫥锟?,
        'welcome': '娆㈣繋',
        'survey': '闂嵎',
        'recommendations': '鎺ㄨ崘',
        'chat': '鑱婂ぉ',
        'evaluation': '璇勪环',
        'musicPreference': '闊充箰鍋忓ソ',
        'submit': '鎻愪氦',
        'next': '涓嬩竴锟?,
        'previous': '涓婁竴锟?,
        'searchPlaceholder': '鎼滅储闊充箰...',
        'songName': '姝屾洸锟?,
        'artist': '鑹烘湳锟?,
        'rating': '璇勫垎'
      },
      en: {
        'username': 'Username',
        'email': 'Email',
        'password': 'Password',
        'login': 'Login',
        'register': 'Register',
        'logout': 'Logout',
        'developer': 'Developer',
        'registerPrompt': 'No account? Click to register',
        'loginPrompt': 'Already have an account? Click to login',
        'welcome': 'Welcome',
        'survey': 'Survey',
        'recommendations': 'Recommendations',
        'chat': 'Chat',
        'evaluation': 'Evaluation',
        'musicPreference': 'Music Preference',
        'submit': 'Submit',
        'next': 'Next',
        'previous': 'Previous',
        'searchPlaceholder': 'Search music...',
        'songName': 'Song Name',
        'artist': 'Artist',
        'rating': 'Rating'
      }
    };
  }
  
  // 璁剧疆褰撳墠璇█
  if (!window.app.currentLanguage) {
    window.app.currentLanguage = 'zh';
  }
  
  let fixCount = 0;
  
  // 鏌ユ壘鎵€鏈夋枃鏈妭锟?
  const textNodes = [];
  function findTextNodes(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      textNodes.push(node);
    } else {
      for (let i = 0; i < node.childNodes.length; i++) {
        findTextNodes(node.childNodes[i]);
      }
    }
  }
  
  // 浠庢枃妗ｄ綋寮€濮嬫煡锟?
  findTextNodes(document.body);
  
  // 妫€鏌ユ瘡涓枃鏈妭锟?
  textNodes.forEach(node => {
    const text = node.textContent;
    
    // 妫€鏌ユ枃鏈槸鍚﹀寘鍚ā鏉胯锟?
    if (text.includes('{{') && text.includes('}}')) {
      // 鏇挎崲鎵€鏈夋ā鏉胯〃杈惧紡
      const newText = text.replace(/\{\{\s*t\('([^']+)'\)\s*\}\}/g, function(match, key) {
        fixCount++;
        
        // 鑾峰彇缈昏瘧鏂囨湰
        let replacement = key;
        if (window.app.translations && 
            window.app.translations[window.app.currentLanguage] && 
            window.app.translations[window.app.currentLanguage][key]) {
          replacement = window.app.translations[window.app.currentLanguage][key];
        }
        
        return replacement;
      });
      
      // 鏇存柊鑺傜偣鍐呭
      if (text !== newText) {
        node.textContent = newText;
      }
    }
  });
  
  // 妫€鏌ユ槸鍚︿粛鏈夋湭娓叉煋鐨勬ā锟?
  const stillHasUnrenderedTemplates = document.body.innerHTML.includes('{{') && 
                                     document.body.innerHTML.includes('}}');
  
  if (stillHasUnrenderedTemplates) {
    console.log('浠嶆湁鏈覆鏌撶殑妯℃澘锛屽皾璇曢噸鏂板垵濮嬪寲Vue');
    
    // 鏌ユ壘鍖呭惈鏈覆鏌撴ā鏉跨殑鍏冪礌
    const elementsWithTemplates = [];
    document.querySelectorAll('*').forEach(el => {
      if (el.innerHTML.includes('{{') && el.innerHTML.includes('}}')) {
        elementsWithTemplates.push(el);
      }
    });
    
    console.log('鍖呭惈鏈覆鏌撴ā鏉跨殑鍏冪礌鏁伴噺:', elementsWithTemplates.length);
    
    // 灏濊瘯鐩存帴鏇挎崲妯℃澘鍐呭
    elementsWithTemplates.forEach(el => {
      const html = el.innerHTML;
      const newHtml = html.replace(/\{\{\s*t\('([^']+)'\)\s*\}\}/g, function(match, key) {
        fixCount++;
        
        // 鑾峰彇缈昏瘧鏂囨湰
        let replacement = key;
        if (window.app.translations && 
            window.app.translations[window.app.currentLanguage] && 
            window.app.translations[window.app.currentLanguage][key]) {
          replacement = window.app.translations[window.app.currentLanguage][key];
        }
        
        return replacement;
      });
      
      if (html !== newHtml) {
        el.innerHTML = newHtml;
      }
    });
  }
  
  console.log(`妯℃澘淇瀹屾垚锛屾浛鎹簡 ${fixCount} 涓ā鏉胯〃杈惧紡`);
  return fixCount;
};

// 鐩戝惉DOM鍙樺寲锛屾鏌ユ槸鍚︽湁鏂版坊鍔犵殑鏈覆鏌撴ā锟?
document.addEventListener('DOMContentLoaded', function() {
  // 棣栨杩愯淇
  setTimeout(fixTemplateRendering, 500);
  
  // 鍒涘缓MutationObserver瑙傚療DOM鍙樺寲
  const observer = new MutationObserver(function(mutations) {
    let needsFix = false;
    
    // 妫€鏌ュ彉鍖栫殑鑺傜偣
    mutations.forEach(function(mutation) {
      if (mutation.addedNodes && mutation.addedNodes.length) {
        mutation.addedNodes.forEach(function(node) {
          // 妫€鏌ユ柊娣诲姞鐨勮妭鐐规槸鍚﹀寘鍚湭娓叉煋鐨勬ā锟?
          if (node.nodeType === Node.ELEMENT_NODE && 
              node.innerHTML && 
              node.innerHTML.includes('{{') && 
              node.innerHTML.includes('}}')) {
            needsFix = true;
          }
        });
      }
    });
    
    // 濡傛灉妫€娴嬪埌鏈覆鏌撶殑妯℃澘锛屽皾璇曚慨锟?
    if (needsFix) {
      console.log('妫€娴嬪埌DOM鍙樺寲涓殑鏈覆鏌撴ā鏉匡紝灏濊瘯淇');
      fixTemplateRendering();
    }
  });
  
  // 寮€濮嬭瀵熸暣涓枃锟?
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}); 

// 瀹氫箟淇妯℃澘娓叉煋鐨勫嚱锟?
function fixTemplateRendering() {
  console.log('姝ｅ湪妫€鏌ュ拰淇鏈覆鏌撶殑妯℃澘...');
  
  // 妫€鏌ユ槸鍚︽湁鏈覆鏌撶殑妯℃澘
  const hasUnrenderedTemplates = document.body.innerHTML.includes('{{') && 
                                document.body.innerHTML.includes('}}');
  
  if (!hasUnrenderedTemplates) {
    console.log('娌℃湁鍙戠幇鏈覆鏌撶殑妯℃澘');
    return 0;
  }
  
  console.log('鍙戠幇鏈覆鏌撶殑妯℃澘锛屽紑濮嬩慨锟?);
  
  // 瀹氫箟缈昏瘧瀵硅薄
  const translations = {
    zh: {
      'username': '鐢ㄦ埛锟?,
      'email': '閭',
      'password': '瀵嗙爜',
      'login': '鐧诲綍',
      'register': '娉ㄥ唽',
      'logout': '閫€锟?,
      'developer': '寮€鍙戯拷?,
      'registerPrompt': '杩樻病鏈夎处鍙凤紵鐐瑰嚮娉ㄥ唽',
      'loginPrompt': '宸叉湁璐﹀彿锛熺偣鍑荤櫥锟?,
      'welcome': '娆㈣繋',
      'survey': '闂嵎',
      'recommendations': '鎺ㄨ崘',
      'chat': '鑱婂ぉ',
      'evaluation': '璇勪环',
      'musicPreference': '闊充箰鍋忓ソ',
      'submit': '鎻愪氦',
      'next': '涓嬩竴锟?,
      'previous': '涓婁竴锟?,
      'searchPlaceholder': '鎼滅储闊充箰...',
      'songName': '姝屾洸锟?,
      'artist': '鑹烘湳锟?,
      'rating': '璇勫垎'
    },
    en: {
      'username': 'Username',
      'email': 'Email',
      'password': 'Password',
      'login': 'Login',
      'register': 'Register',
      'logout': 'Logout',
      'developer': 'Developer',
      'registerPrompt': 'No account? Click to register',
      'loginPrompt': 'Already have an account? Click to login',
      'welcome': 'Welcome',
      'survey': 'Survey',
      'recommendations': 'Recommendations',
      'chat': 'Chat',
      'evaluation': 'Evaluation',
      'musicPreference': 'Music Preference',
      'submit': 'Submit',
      'next': 'Next',
      'previous': 'Previous',
      'searchPlaceholder': 'Search music...',
      'songName': 'Song Name',
      'artist': 'Artist',
      'rating': 'Rating'
    }
  };
  
  let fixCount = 0;
  
  // 鏌ユ壘鍖呭惈鏈覆鏌撴ā鏉跨殑鍏冪礌
  const elementsWithTemplates = [];
  document.querySelectorAll('*').forEach(el => {
    if (el.innerHTML && el.innerHTML.includes('{{') && el.innerHTML.includes('}}')) {
      elementsWithTemplates.push(el);
    }
  });
  
  console.log('鍖呭惈鏈覆鏌撴ā鏉跨殑鍏冪礌鏁伴噺:', elementsWithTemplates.length);
  
  // 璁剧疆褰撳墠璇█
  const currentLanguage = 'zh'; // 榛樿涓枃
  
  // 鏇挎崲妯℃澘
  elementsWithTemplates.forEach(el => {
    const original = el.innerHTML;
    const newHtml = original.replace(/\{\{\s*t\('([^']+)'\)\s*\}\}/g, function(match, key) {
      fixCount++;
      
      // 鑾峰彇缈昏瘧鏂囨湰
      let replacement = key;
      if (translations[currentLanguage] && translations[currentLanguage][key]) {
        replacement = translations[currentLanguage][key];
      }
      
      return replacement;
    });
    
    if (original !== newHtml) {
      el.innerHTML = newHtml;
    }
  });
  
  console.log(`妯℃澘淇瀹屾垚锛屾浛鎹簡 ${fixCount} 涓ā鏉胯〃杈惧紡`);
  return fixCount;
}

// 椤甸潰鍔犺浇瀹屾垚鍚庢墽琛屼慨锟?
document.addEventListener('DOMContentLoaded', function() {
  console.log('椤甸潰鍔犺浇瀹屾垚锛屽嵆灏嗗紑濮嬫鏌ユā鏉挎覆锟?);
  
  // 寤惰繜鎵ц淇锛岀‘淇漋ue鏈夋満浼氬厛娓叉煋
  setTimeout(function() {
    const fixedCount = fixTemplateRendering();
    if (fixedCount > 0) {
      console.log(`鎴愬姛淇锟?${fixedCount} 涓ā鏉胯〃杈惧紡`);
    }
  }, 1000);
}); 
