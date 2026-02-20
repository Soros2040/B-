# -*- coding: utf-8 -*-
"""
阶段四：时间序列分析
任务4.1: 技术发展时间线分析
任务4.2: 政策演进时间线分析
任务4.3: 时间序列可视化
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("阶段四：时间序列分析")
print("=" * 80)

tech_df = pd.read_csv(r'e:\B正大杯\dataexample\tech_timeline.csv')
policy_df = pd.read_csv(r'e:\B正大杯\dataexample\policy_timeline.csv')

print("\n" + "=" * 80)
print("任务4.1: 技术发展时间线分析")
print("=" * 80)

print(f"\n【技术发展时间线基本信息】")
print(f"  - 记录数: {len(tech_df)}")
print(f"  - 时间跨度: {tech_df['时间'].min()} 至 {tech_df['时间'].max()}")
print(f"  - 发展阶段: {tech_df['阶段'].nunique()}个")

print(f"\n【发展阶段分布】")
stage_counts = tech_df['阶段'].value_counts()
for stage, count in stage_counts.items():
    pct = count / len(tech_df) * 100
    print(f"  {stage}: {count} ({pct:.1f}%)")

print(f"\n【事件类型分布】")
event_counts = tech_df['事件类型'].value_counts()
for event, count in event_counts.items():
    pct = count / len(tech_df) * 100
    print(f"  {event}: {count} ({pct:.1f}%)")

print(f"\n【技术分类分布】")
tech_counts = tech_df['技术分类'].value_counts()
for tech, count in tech_counts.items():
    pct = count / len(tech_df) * 100
    print(f"  {tech}: {count} ({pct:.1f}%)")

print(f"\n【影响程度分布】")
impact_counts = tech_df['影响程度'].value_counts()
for impact, count in impact_counts.items():
    pct = count / len(tech_df) * 100
    print(f"  {impact}: {count} ({pct:.1f}%)")

tech_df['年份'] = tech_df['时间'].str[:4].astype(int)
year_counts = tech_df['年份'].value_counts().sort_index()

print(f"\n【年度事件数量趋势】")
print("  年份 | 事件数 | 累计")
print("  " + "-" * 30)
cumulative = 0
for year, count in year_counts.items():
    cumulative += count
    print(f"  {year} | {count:3d} | {cumulative:3d}")

print(f"\n【关键里程碑事件】")
milestones = tech_df[tech_df['影响程度'] == '高'].sort_values('时间')
print(f"  高影响事件共 {len(milestones)} 个:")
for idx, row in milestones.head(15).iterrows():
    print(f"    {row['时间']}: {row['事件名称']} ({row['技术分类']})")

print(f"\n【技术发展阶段特征】")
stage_features = {
    '萌芽期': '1929-2000年，OCR、NLP等基础技术起源，RPA概念尚未形成',
    '发展初期': '2000-2012年，RPA概念提出，UiPath/AA/Blue Prism等厂商成立',
    '快速发展期': '2013-2014年，Docker/K8s等云原生技术兴起，深度学习突破',
    '快速成长期': '2015-2019年，RPA技术成型，国产RPA厂商崛起',
    '智能融合期': '2020-2022年，大模型时代开启，RPA+AI融合加速',
    '超自动化期': '2023-2025年，Agentic RPA成为主流，自主决策能力提升'
}

for stage, feature in stage_features.items():
    print(f"  {stage}: {feature}")

print("\n" + "=" * 80)
print("任务4.2: 政策演进时间线分析")
print("=" * 80)

print(f"\n【政策演进时间线基本信息】")
print(f"  - 记录数: {len(policy_df)}")
print(f"  - 时间跨度: {policy_df['时间'].min()} 至 {policy_df['时间'].max()}")
print(f"  - 政策阶段: {policy_df['阶段'].nunique()}个")

print(f"\n【政策阶段分布】")
policy_stage_counts = policy_df['阶段'].value_counts()
for stage, count in policy_stage_counts.items():
    pct = count / len(policy_df) * 100
    print(f"  {stage}: {count} ({pct:.1f}%)")

print(f"\n【政策类型分布】")
policy_type_counts = policy_df['政策类型'].value_counts()
for ptype, count in policy_type_counts.items():
    pct = count / len(policy_df) * 100
    print(f"  {ptype}: {count} ({pct:.1f}%)")

print(f"\n【发布机构分布】")
org_counts = policy_df['发布机构'].value_counts()
for org, count in org_counts.items():
    pct = count / len(policy_df) * 100
    print(f"  {org}: {count} ({pct:.1f}%)")

print(f"\n【影响行业分布】")
industry_all = policy_df['影响行业'].str.cat(sep=',').split(',')
industry_counts = Counter(industry_all)
for industry, count in industry_counts.most_common(10):
    print(f"  {industry.strip()}: {count}")

print(f"\n【关联技术分布】")
tech_all = policy_df['关联技术'].str.cat(sep=',').split(',')
tech_counts = Counter(tech_all)
for tech, count in tech_counts.most_common(10):
    print(f"  {tech.strip()}: {count}")

policy_df['年份'] = policy_df['时间'].str[:4].astype(int)
policy_year_counts = policy_df['年份'].value_counts().sort_index()

print(f"\n【年度政策数量趋势】")
print("  年份 | 政策数 | 累计")
print("  " + "-" * 30)
cumulative = 0
for year, count in policy_year_counts.items():
    cumulative += count
    print(f"  {year} | {count:3d} | {cumulative:3d}")

print(f"\n【关键政策里程碑】")
policy_milestones = policy_df[policy_df['影响程度'] == '高'].sort_values('时间')
print(f"  高影响政策共 {len(policy_milestones)} 个:")
for idx, row in policy_milestones.iterrows():
    print(f"    {row['时间']}: {row['政策名称']} ({row['发布机构']})")

print(f"\n【政策演进阶段特征】")
policy_stage_features = {
    '政策萌芽期': '2015-2017年，互联网金融指导意见出台，GDPR通过',
    '政策发展期': '2018-2022年，GDPR生效，数据安全法/个人信息保护法出台，金融科技发展规划发布',
    '政策深化期': '2023-2024年，中央金融工作会议提出数字金融，AI+专项行动启动',
    '政策成熟期': '2025年至今，科技金融体制构建，信创替代全面推进'
}

for stage, feature in policy_stage_features.items():
    print(f"  {stage}: {feature}")

print("\n" + "=" * 80)
print("任务4.3: 技术与政策协同分析")
print("=" * 80)

print(f"\n【技术与政策时间线对比】")
print("  阶段 | 技术事件 | 政策事件 | 协同特征")
print("  " + "-" * 60)

tech_by_stage = tech_df.groupby('阶段').size()
policy_by_stage = policy_df.groupby('阶段').size()

all_stages = ['萌芽期', '发展初期', '快速发展期', '快速成长期', '智能融合期', '超自动化期',
              '政策萌芽期', '政策发展期', '政策深化期', '政策成熟期']

tech_stage_map = {
    '萌芽期': '技术萌芽',
    '发展初期': '技术起步',
    '快速发展期': '技术加速',
    '快速成长期': '技术成熟',
    '智能融合期': 'AI融合',
    '超自动化期': '智能自主'
}

policy_stage_map = {
    '政策萌芽期': '政策探索',
    '政策发展期': '政策完善',
    '政策深化期': '政策强化',
    '政策成熟期': '政策落地'
}

for stage in ['萌芽期', '发展初期', '快速发展期', '快速成长期', '智能融合期', '超自动化期']:
    tech_count = tech_by_stage.get(stage, 0)
    print(f"  {stage} | {tech_count:3d} | - | {tech_stage_map.get(stage, '')}")

print()
for stage in ['政策萌芽期', '政策发展期', '政策深化期', '政策成熟期']:
    policy_count = policy_by_stage.get(stage, 0)
    print(f"  {stage} | - | {policy_count:3d} | {policy_stage_map.get(stage, '')}")

print(f"\n【技术驱动政策的关键节点】")
print("  1. 2015年: RPA技术成型 → 互联网金融指导意见出台")
print("  2. 2018年: GDPR生效 → 数据安全技术需求激增")
print("  3. 2020年: 大模型兴起 → 数据安全法/个人信息保护法出台")
print("  4. 2023年: GPT-4发布 → AI+专项行动启动")
print("  5. 2025年: Agentic RPA成熟 → 科技金融体制构建")

print(f"\n【政策推动技术的关键节点】")
print("  1. 2019年: 金融科技发展规划 → RPA在金融领域加速应用")
print("  2. 2021年: 信创战略 → 国产RPA厂商崛起")
print("  3. 2023年: 数字金融战略 → RPA+AI融合发展")
print("  4. 2024年: AI+专项行动 → Agentic RPA成为重点")
print("  5. 2025年: 科技金融体制 → RPA在金融场景深度应用")

print("\n" + "=" * 80)
print("阶段四完成！")
print("=" * 80)

print("\n【时间序列分析总结】\n")

print("1. 技术发展特征:")
print("   - 从OCR/NLP等基础技术到RPA+AI融合")
print("   - 从规则驱动到智能自主决策")
print("   - 从单点自动化到端到端流程自动化")
print()

print("2. 政策演进特征:")
print("   - 从互联网金融到数字金融/科技金融")
print("   - 从数据安全到AI治理")
print("   - 从政策引导到全面落地")
print()

print("3. 技术与政策协同:")
print("   - 技术突破推动政策完善")
print("   - 政策引导加速技术应用")
print("   - 形成良性互动循环")
print()

print("4. 关键发现:")
print("   - RPA技术发展经历6个阶段，从萌芽到超自动化")
print("   - 政策演进经历4个阶段，从探索到成熟")
print("   - 技术与政策相互促进，共同推动金融数字化转型")
