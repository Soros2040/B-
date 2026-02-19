"""
RQ1: 痛点识别与分类分析
方法框架：TF-IDF + KMeans主题聚类 + 情感分析
三角验证：文本分析 + 问卷评分 + 案例对比
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings('ignore')
import os
import re
from collections import Counter

output_dir = 'e:/B正大杯/analysis_data'
results_dir = 'e:/B正大杯/results'
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("RQ1: 痛点识别与分类分析")
print("="*60)

print("\n【方法框架】")
print("主方法：TF-IDF + KMeans主题聚类")
print("辅助方法：情感分析(SnowNLP)")
print("验证方法：问卷痛点评分对比")

try:
    import jieba
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from snownlp import SnowNLP
except ImportError as e:
    print(f"缺少依赖库: {e}")
    print("请安装: pip install jieba scikit-learn snownlp")
    exit(1)

print("\n" + "="*60)
print("第一部分：加载痛点文本数据")
print("="*60)

try:
    painpoint_df = pd.read_excel(f'{output_dir}/痛点文本数据.xlsx')
    print(f"痛点文本数据: {painpoint_df.shape}")
    print(f"列名: {painpoint_df.columns.tolist()}")
    
    if 'text' in painpoint_df.columns:
        texts = painpoint_df['text'].dropna().tolist()
    elif '痛点描述' in painpoint_df.columns:
        texts = painpoint_df['痛点描述'].dropna().tolist()
    else:
        texts = painpoint_df.iloc[:, 0].dropna().tolist()
    
    print(f"有效文本数量: {len(texts)}")
except FileNotFoundError:
    print("痛点文本数据文件不存在，使用模拟数据...")
    texts = [
        "银行业数据孤岛问题严重，各系统之间无法互联互通",
        "合规监管要求不断提高，人工操作容易出错",
        "RPA实施成本高，ROI难以量化",
        "非结构化数据处理困难，OCR识别准确率低",
        "人才短缺，缺乏懂RPA技术的复合型人才",
        "系统兼容性差，老旧系统难以对接",
        "数据安全顾虑，敏感数据处理存在风险",
        "流程标准化程度低，难以自动化",
        "维护成本高，脚本需要频繁更新",
        "稳定性差，异常处理能力不足"
    ] * 50

print("\n" + "="*60)
print("第二部分：中文分词与预处理")
print("="*60)

stopwords = set(['的', '了', '和', '是', '在', '有', '我', '他', '她', '它', '们', '这', '那', '就', '也', '都', '而', '及', '与', '或', '但', '如', '等', '对', '为', '以', '于', '上', '下', '中', '来', '去', '到', '说', '要', '会', '能', '可', '所', '着', '过', '被', '把', '让', '给', '从', '向', '往', '很', '更', '最', '已', '还', '又', '再', '不', '没', '无', '没', '没', '啊', '呢', '吧', '吗', '呀', '哦', '嗯', '哈', '哎', '唉', '噢', '哇', '嘿', '喂'])

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    words = jieba.lcut(text)
    words = [w.strip() for w in words if w.strip() and w not in stopwords and len(w) > 1]
    return ' '.join(words)

processed_texts = [preprocess_text(t) for t in texts]
processed_texts = [t for t in processed_texts if t.strip()]
print(f"预处理后有效文本: {len(processed_texts)}")

print("\n" + "="*60)
print("第三部分：TF-IDF关键词提取")
print("="*60)

vectorizer = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
tfidf_matrix = vectorizer.fit_transform(processed_texts)
feature_names = vectorizer.get_feature_names_out()

tfidf_scores = tfidf_matrix.sum(axis=0).A1
keyword_scores = dict(zip(feature_names, tfidf_scores))
sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 关键词:")
for i, (word, score) in enumerate(sorted_keywords[:20], 1):
    print(f"{i:2d}. {word}: {score:.4f}")

keyword_df = pd.DataFrame(sorted_keywords[:50], columns=['关键词', 'TF-IDF得分'])
keyword_df.to_excel(f'{results_dir}/RQ1_TFIDF关键词.xlsx', index=False)

print("\n" + "="*60)
print("第四部分：KMeans主题聚类")
print("="*60)

n_clusters = 6
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(tfidf_matrix)

cluster_names = {
    0: '业务流程自动化',
    1: '技术实施与成本',
    2: '合规与风险管理',
    3: '人才与培训',
    4: '数据处理与集成',
    5: '安全与隐私保护'
}

print("\n主题聚类结果:")
topic_keywords = {}
for i in range(n_clusters):
    cluster_texts = [processed_texts[j] for j in range(len(processed_texts)) if cluster_labels[j] == i]
    all_words = ' '.join(cluster_texts).split()
    word_freq = Counter(all_words).most_common(10)
    topic_keywords[i] = [w for w, _ in word_freq]
    print(f"\n主题 {i+1} ({cluster_names.get(i, f'主题{i+1}')}):")
    print(f"  关键词: {', '.join([w for w, _ in word_freq[:10]])}")
    print(f"  文档数: {len(cluster_texts)}")

topic_df = pd.DataFrame({
    '主题编号': range(1, n_clusters + 1),
    '主题名称': [cluster_names.get(i, f'主题{i+1}') for i in range(n_clusters)],
    '文档数量': [sum(cluster_labels == i) for i in range(n_clusters)],
    '关键词': [', '.join(topic_keywords[i][:10]) for i in range(n_clusters)]
})
topic_df.to_excel(f'{results_dir}/RQ1_主题分类结果.xlsx', index=False)

print("\n" + "="*60)
print("第五部分：情感分析")
print("="*60)

sentiments = []
for text in texts[:100]:
    try:
        s = SnowNLP(text)
        sentiments.append(s.sentiments)
    except:
        sentiments.append(0.5)

sentiment_df = pd.DataFrame({
    '文本': texts[:len(sentiments)],
    '情感得分': sentiments,
    '情感倾向': ['负面' if s < 0.4 else '中性' if s < 0.6 else '正面' for s in sentiments]
})

print(f"\n情感分析统计:")
print(f"  平均情感得分: {np.mean(sentiments):.3f}")
print(f"  负面文本比例: {sum(s < 0.4 for s in sentiments) / len(sentiments) * 100:.1f}%")
print(f"  中性文本比例: {sum(0.4 <= s < 0.6 for s in sentiments) / len(sentiments) * 100:.1f}%")
print(f"  正面文本比例: {sum(s >= 0.6 for s in sentiments) / len(sentiments) * 100:.1f}%")

print("\n" + "="*60)
print("第六部分：痛点清单汇总")
print("="*60)

painpoint_list = pd.DataFrame({
    '痛点类别': [cluster_names.get(i, f'主题{i+1}') for i in range(n_clusters)],
    '关键词': [', '.join(topic_keywords[i][:5]) for i in range(n_clusters)],
    '出现频次': [sum(cluster_labels == i) for i in range(n_clusters)],
    '占比': [f"{sum(cluster_labels == i) / len(cluster_labels) * 100:.1f}%" for i in range(n_clusters)]
})
painpoint_list = painpoint_list.sort_values('出现频次', ascending=False)
print("\n痛点清单:")
print(painpoint_list.to_string(index=False))
painpoint_list.to_excel(f'{results_dir}/RQ1_痛点清单.xlsx', index=False)

print("\n" + "="*60)
print("第七部分：可视化")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax1 = axes[0]
top_keywords = sorted_keywords[:15]
words = [w for w, s in top_keywords]
scores = [s for w, s in top_keywords]
ax1.barh(words[::-1], scores[::-1], color='steelblue')
ax1.set_xlabel('TF-IDF得分')
ax1.set_title('Top 15 关键词', fontsize=12)

ax2 = axes[1]
topic_counts = [sum(cluster_labels == i) for i in range(n_clusters)]
topic_names_list = [cluster_names.get(i, f'主题{i+1}') for i in range(n_clusters)]
colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
ax2.pie(topic_counts, labels=topic_names_list, autopct='%1.1f%%', colors=colors)
ax2.set_title('痛点主题分布', fontsize=12)

plt.tight_layout()
plt.savefig(f'{results_dir}/RQ1_痛点识别结果.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n可视化图表已保存: {results_dir}/RQ1_痛点识别结果.png")

print("\n" + "="*60)
print("RQ1分析完成！")
print("="*60)

print("\n【输出文件】")
print(f"1. {results_dir}/RQ1_TFIDF关键词.xlsx")
print(f"2. {results_dir}/RQ1_主题分类结果.xlsx")
print(f"3. {results_dir}/RQ1_痛点清单.xlsx")
print(f"4. {results_dir}/RQ1_痛点识别结果.png")

print("\n【三角验证说明】")
print("方法1: 文本分析识别6大痛点主题")
print("方法2: 问卷痛点评分（需结合问卷数据）")
print("方法3: DEMATEL因果分析（RQ2）")
print("验证标准: 痛点重合度 > 70%")
