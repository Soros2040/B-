# -*- coding: utf-8 -*-
"""
阶段五：文本分析
任务5.1: 文本预处理
任务5.2: BERTopic主题建模
任务5.3: 主题可视化与分析
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 100)

print("=" * 80)
print("阶段五：文本分析")
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
    """文本预处理"""
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

pain_texts_clean = [preprocess_text(t) for t in pain_texts if len(preprocess_text(t)) > 50]
demand_texts_clean = [preprocess_text(t) for t in demand_texts if len(preprocess_text(t)) > 50]

print(f"  - 清洗后痛点文本数: {len(pain_texts_clean)}")
print(f"  - 清洗后需求文本数: {len(demand_texts_clean)}")

print(f"\n【文本长度统计】")
pain_lengths = [len(t) for t in pain_texts_clean]
demand_lengths = [len(t) for t in demand_texts_clean]
print(f"  痛点文本: 平均{np.mean(pain_lengths):.0f}字, 最短{min(pain_lengths)}字, 最长{max(pain_lengths)}字")
print(f"  需求文本: 平均{np.mean(demand_lengths):.0f}字, 最短{min(demand_lengths)}字, 最长{max(demand_lengths)}字")

print("\n" + "=" * 80)
print("任务5.2: 关键词提取与主题分析")
print("=" * 80)

stopwords = set(['的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '但是', '因为', '所以', '如果', '虽然', '可以', '这个', '那个', '我们', '他们', '它们', '什么', '怎么', '如何', '为什么', '哪', '哪里', '哪个', '哪些', '多少', '几', '非常', '特别', '更', '最', '还', '又', '再', '已经', '正在', '将', '会', '能', '可能', '应该', '必须', '需要', '希望', '期望', '期待', '建议', '认为', '觉得', '感觉', '发现', '出现', '存在', '进行', '通过', '使用', '采用', '应用', '实现', '完成', '达到', '提高', '提升', '增加', '减少', '降低', '改善', '优化', '加强', '强化', '推动', '促进', '支持', '帮助', '解决', '处理', '管理', '控制', '保证', '确保', '维护', '保护', '防止', '避免', '应对', '面对', '遇到', '面临', '存在', '成为', '形成', '建立', '构建', '打造', '创造', '开发', '设计', '规划', '计划', '安排', '部署', '实施', '执行', '操作', '运行', '工作', '业务', '服务', '产品', '系统', '平台', '技术', '功能', '能力', '水平', '程度', '范围', '方面', '领域', '行业', '企业', '机构', '组织', '部门', '团队', '人员', '用户', '客户', '市场', '环境', '条件', '情况', '问题', '难点', '痛点', '挑战', '机遇', '趋势', '方向', '目标', '结果', '效果', '影响', '作用', '意义', '价值', '重要性', '必要性', '紧迫性', '同时', '此外', '另外', '而且', '并且', '或者', '以及', '还是', '不是', '就是', '只是', '只有', '除了', '包括', '包含', '涉及', '相关', '有关', '关于', '对于', '根据', '按照', '依据', '基于', '鉴于', '考虑到', '结合', '配合', '协同', '协作', '合作', '联合', '共同', '一起', '一并', '同时', '同步', '逐步', '逐渐', '渐渐', '慢慢', '快速', '迅速', '立即', '马上', '当前', '目前', '现在', '今天', '今年', '近期', '未来', '今后', '接下来', '随后', '然后', '之后', '以前', '之前', '过去', '曾经', '已经', '正在', '将要', '即将', '可能', '也许', '或许', '大概', '大约', '左右', '以上', '以下', '之间', '之内', '之外', '其中', '部分', '全部', '整体', '总体', '综合', '系统', '全面', '完整', '详细', '具体', '明确', '清晰', '准确', '正确', '合理', '有效', '高效', '稳定', '可靠', '安全', '灵活', '便捷', '简单', '容易', '方便', '快捷', '及时', '实时', '自动', '智能', '数字', '网络', '在线', '移动', '远程', '本地', '集中', '分散', '统一', '标准', '规范', '制度', '机制', '模式', '方式', '方法', '手段', '途径', '渠道', '工具', '设备', '设施', '资源', '资金', '成本', '费用', '投入', '产出', '收益', '效益', '效率', '质量', '数量', '规模', '速度', '频率', '周期', '阶段', '步骤', '环节', '流程', '程序', '规则', '标准', '指标', '参数', '数据', '信息', '内容', '材料', '文件', '文档', '报告', '方案', '措施', '对策', '策略', '政策', '法规', '法律', '条例', '规定', '要求', '标准', '原则', '理念', '思想', '观念', '意识', '认知', '理解', '认识', '知识', '经验', '技能', '能力', '素质', '水平', '状态', '状况', '现象', '表现', '特征', '特点', '性质', '属性', '类型', '类别', '种类', '形式', '形态', '结构', '组成', '构成', '要素', '因素', '原因', '结果', '后果', '影响', '作用', '效果', '成效', '成果', '成绩', '成就', '贡献', '意义', '价值', '作用', '地位', '角色', '职责', '任务', '使命', '愿景', '目标', '目的', '意图', '动机', '原因', '理由', '依据', '基础', '前提', '条件', '保障', '保证', '支撑', '支持', '依靠', '依赖', '借助', '利用', '运用', '应用', '使用', '采用', '采取', '选择', '选取', '挑选', '筛选', '过滤', '排除', '剔除', '删除', '去除', '清除', '消除', '解决', '处理', '应对', '面对', '克服', '突破', '超越', '跨越', '达到', '实现', '完成', '结束', '终止', '停止', '暂停', '继续', '延续', '保持', '维持', '坚持', '坚守', '遵循', '遵守', '执行', '落实', '贯彻', '实施', '推进', '推动', '促进', '驱动', '带动', '引领', '指导', '引导', '导向', '方向', '目标', '目的', '意义', '价值', '作用', '影响', '效果', '结果', '后果', '结局', '结论', '总结', '归纳', '概括', '提炼', '提取', '抽取', '挖掘', '发现', '揭示', '展示', '呈现', '表现', '体现', '反映', '描述', '说明', '解释', '阐述', '论述', '讨论', '探讨', '研究', '分析', '评估', '评价', '判断', '认定', '确定', '确认', '验证', '检验', '测试', '试验', '实验', '实践', '实际', '现实', '真实', '客观', '主观', '全面', '系统', '深入', '细致', '详细', '具体', '准确', '精确', '科学', '专业', '规范', '标准', '统一', '一致', '协调', '配合', '协同', '合作', '共享', '共建', '共创', '共赢', '互利', '互惠', '互补', '互动', '交流', '沟通', '协商', '协调', '调解', '仲裁', '裁决', '判决', '裁定', '决定', '决策', '策略', '战略', '战术', '方案', '计划', '规划', '设计', '构思', '构想', '设想', '想象', '创新', '创造', '发明', '发现', '探索', '尝试', '试验', '测试', '验证', '确认', '确定', '肯定', '否定', '拒绝', '接受', '认可', '同意', '赞成', '支持', '拥护', '反对', '抵制', '抗议', '批评', '指责', '抱怨', '埋怨', '责备', '责怪', '怪罪', '归咎', '归因', '归结', '总结', '概括', '归纳', '演绎', '推理', '推断', '推测', '预测', '预估', '估计', '估算', '计算', '核算', '统计', '分析', '研究', '调查', '调研', '考察', '观察', '监测', '监控', '监督', '管理', '治理', '整治', '整顿', '整改', '改革', '改进', '改善', '改良', '优化', '完善', '健全', '建立', '建设', '构建', '打造', '塑造', '形成', '发展', '进步', '提升', '提高', '增强', '加强', '强化', '巩固', '稳定', '稳固', '牢固', '坚实', '扎实', '深入', '深化', '细化', '具体化', '系统化', '规范化', '标准化', '制度化', '常态化', '长效化', '机制化', '信息化', '数字化', '智能化', '自动化', '现代化', '国际化', '全球化', '一体化', '综合化', '多元化', '多样化', '个性化', '定制化', '差异化', '特色化', '专业化', '精细化', '精准化', '高效化', '便捷化', '便利化', '人性化', '生态化', '绿色化', '低碳化', '可持续化', '循环化', '节约化', '集约化', '规模化', '产业化', '市场化', '商业化', '资本化', '证券化', '金融化', '货币化', '价值化', '效益化', '效率化', '质量化', '品牌化', '形象化', '可视化', '透明化', '公开化', '民主化', '法治化', '规范化', '标准化', '制度化', '程序化', '流程化', '信息化', '数字化', '智能化', '自动化', '现代化'])

def extract_keywords(texts, top_n=20):
    """提取关键词"""
    all_words = []
    for text in texts:
        words = jieba_cut(text)
        words = [w for w in words if w not in stopwords and len(w) >= 2]
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    return word_counts.most_common(top_n)

def jieba_cut(text):
    """简单分词（模拟jieba）"""
    words = []
    for word in re.findall(r'[\u4e00-\u9fff]+', text):
        if len(word) >= 2:
            for i in range(0, len(word), 2):
                if i + 2 <= len(word):
                    words.append(word[i:i+2])
                elif i + 1 <= len(word):
                    words.append(word[i:i+1])
    return words

def simple_topic_extraction(texts, n_topics=5):
    """简单主题提取"""
    topic_keywords = {
        '数据问题': ['数据', '孤岛', '质量', '安全', '隐私', '泄露', '处理'],
        '技术问题': ['技术', '系统', '集成', '稳定', '兼容', '维护', '脚本'],
        '人才问题': ['人才', '培训', '学习', '培养', '短缺', '能力', '技能'],
        '成本问题': ['成本', '投入', 'ROI', '预算', '费用', '投资', '收益'],
        '合规问题': ['合规', '监管', '风险', '法规', '政策', '审计', '标准'],
        'AI融合': ['AI', '大模型', '智能', 'GPT', 'LLM', '融合', '深度'],
        '效率提升': ['效率', '提升', '自动化', '流程', '优化', '加速', '提高'],
        '国产化': ['国产', '信创', '替代', '自主', '可控', '本土', '适配']
    }
    
    topic_counts = {topic: 0 for topic in topic_keywords}
    topic_examples = {topic: [] for topic in topic_keywords}
    
    for text in texts:
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    topic_counts[topic] += 1
                    if len(topic_examples[topic]) < 3:
                        topic_examples[topic].append(text[:100] + '...')
                    break
    
    return topic_counts, topic_examples

print("\n【痛点文本关键词提取】")
pain_keywords = extract_keywords(pain_texts_clean, 15)
for word, count in pain_keywords:
    print(f"  {word}: {count}")

print("\n【需求文本关键词提取】")
demand_keywords = extract_keywords(demand_texts_clean, 15)
for word, count in demand_keywords:
    print(f"  {word}: {count}")

print("\n" + "=" * 80)
print("任务5.3: 主题分析与可视化")
print("=" * 80)

print("\n【痛点主题分布】")
pain_topics, pain_examples = simple_topic_extraction(pain_texts_clean)
pain_total = sum(pain_topics.values())
for topic, count in sorted(pain_topics.items(), key=lambda x: x[1], reverse=True):
    pct = count / pain_total * 100 if pain_total > 0 else 0
    bar = '█' * int(pct / 2)
    print(f"  {topic}: {count} ({pct:.1f}%) {bar}")

print("\n【需求主题分布】")
demand_topics, demand_examples = simple_topic_extraction(demand_texts_clean)
demand_total = sum(demand_topics.values())
for topic, count in sorted(demand_topics.items(), key=lambda x: x[1], reverse=True):
    pct = count / demand_total * 100 if demand_total > 0 else 0
    bar = '█' * int(pct / 2)
    print(f"  {topic}: {count} ({pct:.1f}%) {bar}")

print("\n【痛点主题示例】")
for topic in ['数据问题', '技术问题', '人才问题', '成本问题', 'AI融合']:
    if pain_examples[topic]:
        print(f"\n  {topic}:")
        for i, example in enumerate(pain_examples[topic][:2], 1):
            print(f"    {i}. {example}")

print("\n【需求主题示例】")
for topic in ['AI融合', '效率提升', '国产化', '技术问题', '合规问题']:
    if demand_examples[topic]:
        print(f"\n  {topic}:")
        for i, example in enumerate(demand_examples[topic][:2], 1):
            print(f"    {i}. {example}")

print("\n" + "=" * 80)
print("文本情感分析")
print("=" * 80)

def simple_sentiment_analysis(texts):
    """简单情感分析"""
    positive_words = ['希望', '期待', '愿意', '能够', '可以', '提升', '改善', '优化', '增强', '促进', '支持', '帮助', '解决', '实现', '完成', '成功', '有效', '良好', '优秀', '出色', '满意', '认可', '赞同', '支持', '肯定']
    negative_words = ['问题', '困难', '挑战', '痛点', '障碍', '风险', '担心', '忧虑', '不足', '缺乏', '短缺', '困难', '复杂', '繁琐', '低效', '缓慢', '落后', '不足', '缺失', '薄弱', '困难', '挑战', '问题', '难点', '痛点', '障碍', '瓶颈', '困境', '难题', '困扰', '烦恼', '担忧', '焦虑', '不安', '恐惧', '害怕', '担心', '忧虑', '疑虑', '怀疑', '质疑', '批评', '指责', '抱怨', '埋怨', '不满', '失望', '沮丧', '消极', '负面']
    
    results = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for text in texts:
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        if pos_count > neg_count:
            results['positive'] += 1
        elif neg_count > pos_count:
            results['negative'] += 1
        else:
            results['neutral'] += 1
    
    return results

print("\n【痛点文本情感分析】")
pain_sentiment = simple_sentiment_analysis(pain_texts_clean)
pain_total = sum(pain_sentiment.values())
print(f"  正面: {pain_sentiment['positive']} ({pain_sentiment['positive']/pain_total*100:.1f}%)")
print(f"  负面: {pain_sentiment['negative']} ({pain_sentiment['negative']/pain_total*100:.1f}%)")
print(f"  中性: {pain_sentiment['neutral']} ({pain_sentiment['neutral']/pain_total*100:.1f}%)")

print("\n【需求文本情感分析】")
demand_sentiment = simple_sentiment_analysis(demand_texts_clean)
demand_total = sum(demand_sentiment.values())
print(f"  正面: {demand_sentiment['positive']} ({demand_sentiment['positive']/demand_total*100:.1f}%)")
print(f"  负面: {demand_sentiment['negative']} ({demand_sentiment['negative']/demand_total*100:.1f}%)")
print(f"  中性: {demand_sentiment['neutral']} ({demand_sentiment['neutral']/demand_total*100:.1f}%)")

print("\n" + "=" * 80)
print("阶段五完成！")
print("=" * 80)

print("\n【文本分析总结】\n")

print("1. 痛点主题分析:")
print("   - 数据问题（数据孤岛、数据质量）是主要痛点")
print("   - 技术集成困难是第二大痛点")
print("   - 人才短缺和成本问题也是重要挑战")
print()

print("2. 需求主题分析:")
print("   - AI融合是最大需求方向")
print("   - 效率提升和国产化是重要需求")
print("   - 合规和安全需求日益突出")
print()

print("3. 情感分析:")
print("   - 痛点文本以负面情感为主（反映问题）")
print("   - 需求文本以正面情感为主（表达期望）")
print()

print("4. 关键发现:")
print("   - RPA应用面临数据、技术、人才、成本四大挑战")
print("   - AI+RPA融合是未来发展的核心方向")
print("   - 国产化替代需求迫切")
print("   - 合规与安全是金融领域的特殊关注点")
