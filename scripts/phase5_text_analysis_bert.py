# -*- coding: utf-8 -*-
"""
阶段五：文本分析（BERT版本 - 完整版）
任务5.1: 文本预处理
任务5.2: BERT主题建模（BERT嵌入 + KMeans）
任务5.3: BERT情感分析
任务5.4: 关键词提取
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)

print("=" * 80)
print("阶段五：文本分析（BERT完整版）")
print("=" * 80)

survey_df = pd.read_csv(r'e:\B正大杯\dataexample\survey_data_simulated.csv')

print("\n" + "=" * 80)
print("任务5.1: 文本预处理")
print("=" * 80)

pain_texts = survey_df['Q22_pain_point_text'].dropna().tolist()
demand_texts = survey_df['Q23_demand_text'].dropna().tolist()

print(f"\n【文本数据基本信息】")
print(f"  - 痛点文本数: {len(pain_texts)}")
print(f"  - 需求文本数: {len(demand_texts)}")

def preprocess_text(text):
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

pain_texts_clean = [preprocess_text(t) for t in pain_texts if len(preprocess_text(t)) > 50]
demand_texts_clean = [preprocess_text(t) for t in demand_texts if len(preprocess_text(t)) > 50]

print(f"  - 清洗后痛点文本数: {len(pain_texts_clean)}")
print(f"  - 清洗后需求文本数: {len(demand_texts_clean)}")

pain_lengths = [len(t) for t in pain_texts_clean]
demand_lengths = [len(t) for t in demand_texts_clean]
print(f"  痛点文本: 平均{np.mean(pain_lengths):.0f}字, 最短{min(pain_lengths)}字, 最长{max(pain_lengths)}字")
print(f"  需求文本: 平均{np.mean(demand_lengths):.0f}字, 最短{min(demand_lengths)}字, 最长{max(demand_lengths)}字")

print("\n" + "=" * 80)
print("任务5.2: BERT主题建模")
print("=" * 80)

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import jieba

print("\n正在加载BERT模型（paraphrase-multilingual-MiniLM-L12-v2）...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("BERT模型加载完成！")

print("\n正在生成痛点文本BERT嵌入...")
pain_embeddings = model.encode(pain_texts_clean, show_progress_bar=True, batch_size=32)
print(f"痛点文本嵌入维度: {pain_embeddings.shape}")

print("\n正在生成需求文本BERT嵌入...")
demand_embeddings = model.encode(demand_texts_clean, show_progress_bar=True, batch_size=32)
print(f"需求文本嵌入维度: {demand_embeddings.shape}")

def find_optimal_clusters(embeddings, max_k=15):
    """使用轮廓系数找最优聚类数"""
    silhouette_scores = []
    K_range = range(3, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
        print(f"  K={k}, 轮廓系数={score:.4f}")
    
    optimal_k = K_range[np.argmax(silhouette_scores)]
    return optimal_k, silhouette_scores

print("\n【痛点文本聚类分析】")
print("寻找最优聚类数...")
pain_optimal_k, pain_scores = find_optimal_clusters(pain_embeddings, max_k=12)
print(f"最优聚类数: {pain_optimal_k}")

pain_kmeans = KMeans(n_clusters=pain_optimal_k, random_state=42, n_init=10)
pain_labels = pain_kmeans.fit_predict(pain_embeddings)

print("\n【痛点文本主题分布】")
pain_topic_counts = pd.Series(pain_labels).value_counts().sort_index()
for topic_id, count in pain_topic_counts.items():
    pct = count / len(pain_labels) * 100
    bar = '█' * int(pct / 2)
    print(f"  主题{topic_id}: {count}条 ({pct:.1f}%) {bar}")

print("\n【需求文本聚类分析】")
print("寻找最优聚类数...")
demand_optimal_k, demand_scores = find_optimal_clusters(demand_embeddings, max_k=12)
print(f"最优聚类数: {demand_optimal_k}")

demand_kmeans = KMeans(n_clusters=demand_optimal_k, random_state=42, n_init=10)
demand_labels = demand_kmeans.fit_predict(demand_embeddings)

print("\n【需求文本主题分布】")
demand_topic_counts = pd.Series(demand_labels).value_counts().sort_index()
for topic_id, count in demand_topic_counts.items():
    pct = count / len(demand_labels) * 100
    bar = '█' * int(pct / 2)
    print(f"  主题{topic_id}: {count}条 ({pct:.1f}%) {bar}")

print("\n" + "=" * 80)
print("任务5.3: 主题关键词提取")
print("=" * 80)

from collections import Counter

stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', 
                 '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', 
                 '自己', '这', '那', '但是', '因为', '所以', '如果', '虽然', '可以', '这个', '那个',
                 '我们', '他们', '什么', '怎么', '如何', '为什么', '哪', '哪里', '哪个', '哪些',
                 '多少', '几', '非常', '特别', '更', '最', '还', '又', '再', '已经', '正在', '将',
                 '能', '可能', '应该', '必须', '需要', '希望', '期望', '期待', '建议', '认为',
                 '觉得', '感觉', '发现', '出现', '存在', '进行', '通过', '使用', '采用', '应用',
                 '实现', '完成', '达到', '提高', '提升', '增加', '减少', '降低', '改善', '优化',
                 '加强', '强化', '推动', '促进', '支持', '帮助', '解决', '处理', '管理', '控制',
                 '保证', '确保', '维护', '保护', '防止', '避免', '应对', '面对', '遇到', '面临'])

def extract_topic_keywords(texts, labels, top_n=10):
    """提取每个主题的关键词"""
    topic_keywords = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        topic_texts = [texts[i] for i in range(len(texts)) if labels[i] == label]
        all_words = []
        for text in topic_texts:
            words = list(jieba.cut(text))
            words = [w for w in words if w not in stopwords and len(w) >= 2]
            all_words.extend(words)
        word_counts = Counter(all_words)
        topic_keywords[label] = word_counts.most_common(top_n)
    
    return topic_keywords

print("\n【痛点文本各主题关键词】")
pain_topic_keywords = extract_topic_keywords(pain_texts_clean, pain_labels)
for topic_id, keywords in sorted(pain_topic_keywords.items()):
    kw_str = ', '.join([f"{w}({c})" for w, c in keywords[:5]])
    count = pain_topic_counts.get(topic_id, 0)
    print(f"  主题{topic_id} ({count}条): {kw_str}")

print("\n【需求文本各主题关键词】")
demand_topic_keywords = extract_topic_keywords(demand_texts_clean, demand_labels)
for topic_id, keywords in sorted(demand_topic_keywords.items()):
    kw_str = ', '.join([f"{w}({c})" for w, c in keywords[:5]])
    count = demand_topic_counts.get(topic_id, 0)
    print(f"  主题{topic_id} ({count}条): {kw_str}")

print("\n" + "=" * 80)
print("任务5.4: BERT情感分析")
print("=" * 80)

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

print("\n正在加载BERT情感分析模型...")
try:
    sentiment_model_name = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
    sentiment_pipeline = pipeline("sentiment-analysis", 
                                   model=sentiment_model,
                                   tokenizer=sentiment_tokenizer,
                                   device=-1)
    print("BERT情感分析模型加载完成！")
    BERT_SENTIMENT = True
except Exception as e:
    print(f"BERT情感模型加载失败: {e}")
    print("尝试使用备用模型...")
    try:
        sentiment_pipeline = pipeline("sentiment-analysis", 
                                       model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                                       device=-1)
        print("备用BERT情感模型加载完成！")
        BERT_SENTIMENT = True
    except Exception as e2:
        print(f"备用模型也失败: {e2}")
        BERT_SENTIMENT = False

if BERT_SENTIMENT:
    print("\n【痛点文本BERT情感分析】")
    pain_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    batch_size = 32
    
    for i in range(0, len(pain_texts_clean), batch_size):
        batch = pain_texts_clean[i:i+batch_size]
        batch = [t[:512] for t in batch]
        try:
            results = sentiment_pipeline(batch)
            for r in results:
                label = r['label'].lower()
                if 'pos' in label:
                    pain_sentiments['positive'] += 1
                elif 'neg' in label:
                    pain_sentiments['negative'] += 1
                else:
                    pain_sentiments['neutral'] += 1
        except Exception as e:
            print(f"批次处理错误: {e}")
            continue
    
    total = sum(pain_sentiments.values())
    if total > 0:
        print(f"  正面: {pain_sentiments['positive']} ({pain_sentiments['positive']/total*100:.1f}%)")
        print(f"  负面: {pain_sentiments['negative']} ({pain_sentiments['negative']/total*100:.1f}%)")
        print(f"  中性: {pain_sentiments['neutral']} ({pain_sentiments['neutral']/total*100:.1f}%)")
    
    print("\n【需求文本BERT情感分析】")
    demand_sentiments = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for i in range(0, len(demand_texts_clean), batch_size):
        batch = demand_texts_clean[i:i+batch_size]
        batch = [t[:512] for t in batch]
        try:
            results = sentiment_pipeline(batch)
            for r in results:
                label = r['label'].lower()
                if 'pos' in label:
                    demand_sentiments['positive'] += 1
                elif 'neg' in label:
                    demand_sentiments['negative'] += 1
                else:
                    demand_sentiments['neutral'] += 1
        except Exception as e:
            print(f"批次处理错误: {e}")
            continue
    
    total = sum(demand_sentiments.values())
    if total > 0:
        print(f"  正面: {demand_sentiments['positive']} ({demand_sentiments['positive']/total*100:.1f}%)")
        print(f"  负面: {demand_sentiments['negative']} ({demand_sentiments['negative']/total*100:.1f}%)")
        print(f"  中性: {demand_sentiments['neutral']} ({demand_sentiments['neutral']/total*100:.1f}%)")

print("\n" + "=" * 80)
print("任务5.5: 主题示例提取")
print("=" * 80)

def get_topic_examples(texts, labels, n_examples=2):
    """获取每个主题的示例文本"""
    examples = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        topic_indices = [i for i in range(len(texts)) if labels[i] == label]
        selected = np.random.choice(topic_indices, min(n_examples, len(topic_indices)), replace=False)
        examples[label] = [texts[i][:100] + '...' for i in selected]
    
    return examples

print("\n【痛点文本主题示例】")
pain_examples = get_topic_examples(pain_texts_clean, pain_labels)
for topic_id, examples in sorted(pain_examples.items()):
    print(f"\n  主题{topic_id}:")
    for i, ex in enumerate(examples, 1):
        print(f"    {i}. {ex}")

print("\n【需求文本主题示例】")
demand_examples = get_topic_examples(demand_texts_clean, demand_labels)
for topic_id, examples in sorted(demand_examples.items()):
    print(f"\n  主题{topic_id}:")
    for i, ex in enumerate(examples, 1):
        print(f"    {i}. {ex}")

print("\n" + "=" * 80)
print("阶段五完成！")
print("=" * 80)

print("\n【文本分析总结】\n")
print("1. 使用BERT模型生成文本嵌入")
print("2. 使用KMeans聚类进行主题发现")
print("3. 使用BERT进行情感分析")
print("4. 使用jieba进行中文分词和关键词提取")

results = {
    'pain_optimal_k': int(pain_optimal_k),
    'demand_optimal_k': int(demand_optimal_k),
    'pain_topic_counts': {int(k): int(v) for k, v in pain_topic_counts.to_dict().items()},
    'demand_topic_counts': {int(k): int(v) for k, v in demand_topic_counts.to_dict().items()},
    'pain_topic_keywords': {int(k): [(w, int(c)) for w, c in v] for k, v in pain_topic_keywords.items()},
    'demand_topic_keywords': {int(k): [(w, int(c)) for w, c in v] for k, v in demand_topic_keywords.items()},
}

if BERT_SENTIMENT:
    results['pain_sentiments'] = {k: int(v) for k, v in pain_sentiments.items()}
    results['demand_sentiments'] = {k: int(v) for k, v in demand_sentiments.items()}

import json
with open(r'e:\B正大杯\results\phase5_bert_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print("\n结果已保存到 results/phase5_bert_results.json")
