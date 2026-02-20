import pandas as pd
import numpy as np

print("="*80)
print("Agentic RPA金融领域数据深度分析")
print("="*80)

df1 = pd.read_excel('dataexample/金融经济领域 RPA 需求与技术.xlsx')
df2 = pd.read_excel('dataexample/人才端补充数据.xlsx')
df3 = pd.read_excel('dataexample/政策数据version2.xlsx')

print("\n【一、七大端生态分析】")
print("-"*40)
print("端分布:")
duan_dist = df1['端'].value_counts()
for duan, count in duan_dist.items():
    print(f"  {duan}: {count}条")

print("\n【二、十二大细分行业分析】")
print("-"*40)
industry_dist = df1['行业'].value_counts()
for ind, count in industry_dist.items():
    print(f"  {ind}: {count}条")

print("\n【三、需求分类分析】")
print("-"*40)
need_dist = df1['需求分类'].value_counts()
for need, count in need_dist.items():
    print(f"  {need}: {count}条 ({count/len(df1)*100:.1f}%)")

print("\n【四、人才角色层级分析】")
print("-"*40)
role_dist = df2['角色层级'].value_counts()
for role, count in role_dist.items():
    print(f"  {role}: {count}条")

print("\n【五、政策分类分析】")
print("-"*40)
policy_dist = df3['分类'].value_counts()
for pol, count in policy_dist.items():
    print(f"  {pol}: {count}条")

print("\n【六、重点二级部门分析】")
print("-"*40)
dept_dist = df1['重点二级部门'].value_counts().head(15)
for dept, count in dept_dist.items():
    print(f"  {dept}: {count}条")

print("\n【七、关联场景分析】")
print("-"*40)
scene_dist = df1['关联场景'].value_counts().head(10)
for scene, count in scene_dist.items():
    print(f"  {scene}: {count}条")

print("\n【八、人才培养周期分析】")
print("-"*40)
cycle_dist = df2['培养周期'].value_counts()
for cycle, count in cycle_dist.items():
    print(f"  {cycle}: {count}条")

print("\n【九、认证目标分析】")
print("-"*40)
cert_dist = df2['认证目标'].value_counts().head(10)
for cert, count in cert_dist.items():
    print(f"  {cert}: {count}条")

print("\n【十、行业数据支撑关键词提取】")
print("-"*40)
all_data = ' '.join(df2['行业数据支撑'].dropna().astype(str).tolist())
keywords = ['失败率', '需求', '企业', '项目', '认证', '淘汰', '增长', '渗透率']
for kw in keywords:
    count = all_data.count(kw)
    if count > 0:
        print(f"  '{kw}'出现{count}次")

print("\n分析完成!")
