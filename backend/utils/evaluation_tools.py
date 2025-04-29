import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy import stats
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from backend.models.recommendation_engine import MusicRecommender

def load_ratings_data():
    """从数据库加载评分数据"""
    conn = sqlite3.connect('music_recommender.db')
    
    # 读取用户歌曲评分表
    ratings_df = pd.read_sql(
        "SELECT user_id, track_id, rating, timestamp FROM user_ratings",
        conn
    )
    
    # 读取用户反馈表
    feedback_df = pd.read_sql(
        "SELECT user_id, track_id, feedback, timestamp FROM user_feedback",
        conn
    )
    
    # 读取用户信息表
    users_df = pd.read_sql(
        "SELECT user_id, signup_date, last_login FROM users",
        conn
    )
    
    conn.close()
    
    return ratings_df, feedback_df, users_df

def load_evaluation_results():
    """加载用户提交的评估问卷结果"""
    results = []
    
    eval_dir = 'evaluation_results'
    if not os.path.exists(eval_dir):
        return pd.DataFrame()
    
    for file in os.listdir(eval_dir):
        if file.startswith('evaluation_') and file.endswith('.json'):
            try:
                with open(os.path.join(eval_dir, file), 'r') as f:
                    data = json.load(f)
                    
                    # 整理问卷结果
                    user_id = data['user_id']
                    timestamp = data['timestamp']
                    
                    for resp in data['responses']:
                        results.append({
                            'user_id': user_id,
                            'timestamp': timestamp,
                            'question': resp['question'],
                            'score': resp['score']
                        })
                    
            except Exception as e:
                print(f"读取评估文件 {file} 时出错: {e}")
    
    return pd.DataFrame(results)

def analyze_hybrid_performance():
    """分析混合推荐算法与单一算法的性能差异"""
    conn = sqlite3.connect('music_recommender.db')
    
    # 获取用户评分数据
    ratings_df = pd.read_sql(
        "SELECT user_id, track_id, rating FROM user_ratings",
        conn
    )
    
    conn.close()
    
    if ratings_df.empty:
        print("没有足够的评分数据用于分析")
        return
    
    # 加载推荐器
    recommender = MusicRecommender()
    
    # 随机选择50名用户进行评估（或全部用户，如果少于50名）
    user_ids = ratings_df['user_id'].unique()
    eval_users = np.random.choice(user_ids, min(50, len(user_ids)), replace=False)
    
    # 存储三种算法的MSE和MAE指标
    metrics = {
        'algorithm': [],
        'user_id': [],
        'mse': [],
        'mae': []
    }
    
    for user_id in eval_users:
        # 获取用户的评分记录
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        
        if len(user_ratings) < 5:  # 跳过评分太少的用户
            continue
            
        # 随机分割训练集和测试集
        msk = np.random.rand(len(user_ratings)) < 0.8
        train = user_ratings[msk]
        test = user_ratings[~msk]
        
        if len(test) == 0:  # 确保测试集非空
            continue
        
        # 创建一个临时数据库存储训练数据
        temp_db_path = f'temp_{user_id}.db'
        temp_conn = sqlite3.connect(temp_db_path)
        train.to_sql('user_ratings', temp_conn, if_exists='replace', index=False)
        temp_conn.close()
        
        # 在训练集上训练三种算法
        test_ids = test['track_id'].tolist()
        actual_ratings = test['rating'].tolist()
        
        # 获取协同过滤预测
        cf_preds = []
        for track_id in test_ids:
            try:
                cf_pred = recommender.get_cf_recommendations(user_id, top_n=1)[0][1] if track_id not in train['track_id'].tolist() else 0
                cf_preds.append(cf_pred)
            except:
                cf_preds.append(3)  # 默认中间值
        
        # 获取基于内容的预测
        cbf_preds = []
        for track_id in test_ids:
            try:
                cbf_rec = recommender.get_content_recommendations(user_id, top_n=1)[0][1] if track_id not in train['track_id'].tolist() else 0
                # 归一化到1-5的范围
                cbf_pred = 1 + 4 * cbf_rec
                cbf_preds.append(cbf_pred)
            except:
                cbf_preds.append(3)  # 默认中间值
        
        # 获取混合算法预测
        hybrid_preds = []
        for i, track_id in enumerate(test_ids):
            try:
                # 直接计算加权平均
                hybrid_pred = 0.6 * cf_preds[i] + 0.4 * cbf_preds[i]
                hybrid_preds.append(hybrid_pred)
            except:
                hybrid_preds.append(3)  # 默认中间值
        
        # 计算指标
        metrics['algorithm'].extend(['cf'] * len(test_ids))
        metrics['user_id'].extend([user_id] * len(test_ids))
        metrics['mse'].extend([(pred - actual) ** 2 for pred, actual in zip(cf_preds, actual_ratings)])
        metrics['mae'].extend([abs(pred - actual) for pred, actual in zip(cf_preds, actual_ratings)])
        
        metrics['algorithm'].extend(['cbf'] * len(test_ids))
        metrics['user_id'].extend([user_id] * len(test_ids))
        metrics['mse'].extend([(pred - actual) ** 2 for pred, actual in zip(cbf_preds, actual_ratings)])
        metrics['mae'].extend([abs(pred - actual) for pred, actual in zip(cbf_preds, actual_ratings)])
        
        metrics['algorithm'].extend(['hybrid'] * len(test_ids))
        metrics['user_id'].extend([user_id] * len(test_ids))
        metrics['mse'].extend([(pred - actual) ** 2 for pred, actual in zip(hybrid_preds, actual_ratings)])
        metrics['mae'].extend([abs(pred - actual) for pred, actual in zip(hybrid_preds, actual_ratings)])
        
        # 删除临时数据库
        os.remove(temp_db_path)
    
    # 转换为DataFrame
    metrics_df = pd.DataFrame(metrics)
    
    # 按算法分组计算平均指标
    results = metrics_df.groupby('algorithm').agg({
        'mse': 'mean',
        'mae': 'mean'
    }).reset_index()
    
    # 进行统计显著性测试
    cf_mse = metrics_df[metrics_df['algorithm'] == 'cf']['mse']
    cbf_mse = metrics_df[metrics_df['algorithm'] == 'cbf']['mse']
    hybrid_mse = metrics_df[metrics_df['algorithm'] == 'hybrid']['mse']
    
    # t检验: hybrid vs cf
    t_stat_cf, p_val_cf = stats.ttest_ind(hybrid_mse, cf_mse)
    
    # t检验: hybrid vs cbf
    t_stat_cbf, p_val_cbf = stats.ttest_ind(hybrid_mse, cbf_mse)
    
    # 输出结果
    print("\n混合算法性能分析")
    print("=" * 50)
    print(results)
    print("\n统计显著性检验 (MSE):")
    print(f"混合 vs 协同过滤: t={t_stat_cf:.4f}, p={p_val_cf:.4f} {'显著' if p_val_cf < 0.05 else '不显著'}")
    print(f"混合 vs 基于内容: t={t_stat_cbf:.4f}, p={p_val_cbf:.4f} {'显著' if p_val_cbf < 0.05 else '不显著'}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithm', y='mse', data=results)
    plt.title('不同推荐算法的MSE对比')
    plt.xlabel('算法')
    plt.ylabel('均方误差 (MSE)')
    plt.savefig('algorithm_comparison_mse.png')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='algorithm', y='mae', data=results)
    plt.title('不同推荐算法的MAE对比')
    plt.xlabel('算法')
    plt.ylabel('平均绝对误差 (MAE)')
    plt.savefig('algorithm_comparison_mae.png')
    
    return results, (t_stat_cf, p_val_cf), (t_stat_cbf, p_val_cbf)

def analyze_user_satisfaction():
    """分析用户满意度问卷结果"""
    eval_df = load_evaluation_results()
    
    if eval_df.empty:
        print("没有问卷评估数据可供分析")
        return
    
    # 计算每个问题的平均分和标准差
    question_stats = eval_df.groupby('question').agg({
        'score': ['mean', 'std', 'count']
    }).reset_index()
    
    question_stats.columns = ['question', 'mean_score', 'std_dev', 'responses']
    
    # 计算总体满意度
    overall_satisfaction = eval_df['score'].mean()
    
    # 计算可靠性 - Cronbach's Alpha
    # 首先需要将数据透视为每个用户的多个问题评分
    user_responses = eval_df.pivot(index='user_id', columns='question', values='score')
    
    # 如果有完整回答所有问题的用户
    if not user_responses.empty and not user_responses.isna().any().any():
        # 计算Cronbach's Alpha
        item_variances = user_responses.var(axis=0)
        total_variance = user_responses.sum(axis=1).var()
        k = user_responses.shape[1]
        
        cronbach_alpha = (k / (k - 1)) * (1 - item_variances.sum() / total_variance)
    else:
        cronbach_alpha = None
    
    # 输出结果
    print("\n用户满意度分析")
    print("=" * 50)
    print(f"总体满意度平均分: {overall_satisfaction:.2f}/5.0")
    if cronbach_alpha:
        print(f"问卷可靠性 (Cronbach's Alpha): {cronbach_alpha:.4f}")
    
    print("\n各问题统计:")
    for _, row in question_stats.iterrows():
        print(f"{row['question']}: {row['mean_score']:.2f} ± {row['std_dev']:.2f} (n={int(row['responses'])})")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    sns.barplot(x='question', y='mean_score', data=question_stats)
    plt.title('用户满意度评分 (各问题)')
    plt.ylim(0, 5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('user_satisfaction_scores.png')
    
    return question_stats, overall_satisfaction, cronbach_alpha

def generate_evaluation_report():
    """生成综合评估报告"""
    # 创建报告目录
    report_dir = 'evaluation_report'
    os.makedirs(report_dir, exist_ok=True)
    
    # 加载数据
    ratings_df, feedback_df, users_df = load_ratings_data()
    
    # 基本系统用量统计
    user_count = len(users_df)
    ratings_count = len(ratings_df)
    feedback_count = len(feedback_df)
    
    # 推荐算法性能分析
    algo_results, cf_stats, cbf_stats = analyze_hybrid_performance()
    
    # 用户满意度分析
    satisfaction_stats, overall_satisfaction, reliability = analyze_user_satisfaction()
    
    # 生成HTML报告
    html_report = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <title>音乐推荐系统评估报告</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #3273dc; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; }}
            img {{ max-width: 100%; height: auto; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>音乐推荐系统评估报告</h1>
        <p>生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        
        <h2>1. 系统使用统计</h2>
        <table>
            <tr><th>指标</th><th>数值</th></tr>
            <tr><td>用户总数</td><td>{user_count}</td></tr>
            <tr><td>评分总数</td><td>{ratings_count}</td></tr>
            <tr><td>反馈总数</td><td>{feedback_count}</td></tr>
            <tr><td>平均每用户评分数</td><td>{ratings_count/user_count if user_count > 0 else 0:.2f}</td></tr>
        </table>
        
        <h2>2. 推荐算法性能</h2>
        <p>比较了三种推荐算法的性能：协同过滤 (CF)、基于内容的推荐 (CBF) 和混合推荐算法。</p>
        
        <table>
            <tr><th>算法</th><th>均方误差 (MSE)</th><th>平均绝对误差 (MAE)</th></tr>
    '''
    
    # 添加算法性能数据到报告
    if algo_results is not None:
        for _, row in algo_results.iterrows():
            html_report += f'<tr><td>{row["algorithm"]}</td><td>{row["mse"]:.4f}</td><td>{row["mae"]:.4f}</td></tr>'
    
    html_report += f'''
        </table>
        
        <p><strong>统计显著性检验结果:</strong></p>
        <ul>
            <li>混合 vs 协同过滤: t={cf_stats[0]:.4f}, p={cf_stats[1]:.4f} {'<span style="color:green">显著</span>' if cf_stats[1] < 0.05 else '<span style="color:red">不显著</span>'}</li>
            <li>混合 vs 基于内容: t={cbf_stats[0]:.4f}, p={cbf_stats[1]:.4f} {'<span style="color:green">显著</span>' if cbf_stats[1] < 0.05 else '<span style="color:red">不显著</span>'}</li>
        </ul>
        
        <div>
            <img src="../algorithm_comparison_mse.png" alt="算法MSE对比">
            <img src="../algorithm_comparison_mae.png" alt="算法MAE对比">
        </div>
        
        <h2>3. 用户满意度分析</h2>
        <p>总体满意度: {overall_satisfaction:.2f}/5.0</p>
    '''
    
    if reliability:
        html_report += f'<p>问卷可靠性 (Cronbach\'s Alpha): {reliability:.4f} {"(良好)" if reliability > 0.7 else "(一般)"}</p>'
    
    html_report += '''
        <h3>各问题满意度评分:</h3>
        <table>
            <tr><th>问题</th><th>平均分</th><th>标准差</th><th>回答人数</th></tr>
    '''
    
    # 添加满意度问题数据到报告
    if satisfaction_stats is not None:
        for _, row in satisfaction_stats.iterrows():
            html_report += f'<tr><td>{row["question"]}</td><td>{row["mean_score"]:.2f}</td><td>{row["std_dev"]:.2f}</td><td>{int(row["responses"])}</td></tr>'
    
    html_report += f'''
        </table>
        
        <div>
            <img src="../user_satisfaction_scores.png" alt="用户满意度评分">
        </div>
        
        <h2>4. 结论与建议</h2>
        <p>基于上述分析，得出以下结论：</p>
        <ul>
            <li>{'混合推荐算法的性能优于单一算法，表明组合不同推荐方法能有效提高推荐质量。' if algo_results is not None and algo_results['mse'].min() == algo_results[algo_results['algorithm'] == 'hybrid']['mse'].values[0] else '推荐算法性能有待提高，可考虑调整参数或改进算法。'}</li>
            <li>{'用户对系统总体满意度良好，特别是' + satisfaction_stats.sort_values('mean_score', ascending=False)['question'].iloc[0] + '方面表现突出。' if satisfaction_stats is not None else '缺乏足够的用户评价数据，建议增加用户参与度。'}</li>
            <li>{'系统的用户参与度较高，平均每用户提供了' + f"{ratings_count/user_count:.1f}" + '个评分，为推荐系统提供了丰富的数据基础。' if user_count > 0 and ratings_count/user_count > 5 else '系统的用户参与度有待提高，可考虑增加用户互动和激励机制。'}</li>
        </ul>
        
        <p>改进建议：</p>
        <ul>
            <li>{'考虑调整混合算法的权重，进一步优化推荐性能。' if algo_results is not None else '收集更多用户数据，提高算法训练效果。'}</li>
            <li>{'改进' + satisfaction_stats.sort_values('mean_score', ascending=True)['question'].iloc[0] + '相关功能，提高用户满意度。' if satisfaction_stats is not None else '增加用户调研，了解用户需求和痛点。'}</li>
            <li>增强系统的社交功能，鼓励用户分享和推荐音乐，形成良性循环。</li>
        </ul>
    </body>
    </html>
    '''
    
    # 保存HTML报告
    with open(os.path.join(report_dir, 'evaluation_report.html'), 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"评估报告已生成: {os.path.join(report_dir, 'evaluation_report.html')}")

if __name__ == "__main__":
    print("运行音乐推荐系统评估工具...")
    
    # 分析混合算法性能
    print("\n正在分析推荐算法性能...")
    analyze_hybrid_performance()
    
    # 分析用户满意度
    print("\n正在分析用户满意度...")
    analyze_user_satisfaction()
    
    # 生成综合评估报告
    print("\n正在生成评估报告...")
    generate_evaluation_report()
    
    print("\n评估完成！") 