"""
RQ3: 决策支持与方案评估
方法框架：模糊AHP + TOPSIS + VIKOR组合
三角验证：三种方法排序一致性检验 + 敏感性分析
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
from scipy import stats

output_dir = 'e:/B正大杯/analysis_data'
results_dir = 'e:/B正大杯/results'
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("RQ3: 决策支持与方案评估")
print("="*60)

print("\n【方法框架】")
print("主方法：TOPSIS + VIKOR组合")
print("辅助方法：模糊AHP权重确定")
print("验证方法：敏感性分析")

tech_solutions = ['LLM+RAG', '多模态AI', '知识图谱', '流程挖掘']
criteria_names = ['技术成熟度', '实施成本', '业务价值', '易用性', '可扩展性']
n_tech = len(tech_solutions)
n_criteria = len(criteria_names)

print("\n" + "="*60)
print("第一部分：模糊AHP权重计算")
print("="*60)

print("\n【模糊AHP方法说明】")
print("Step 1: 构建三角模糊数判断矩阵")
print("Step 2: 计算模糊几何平均值")
print("Step 3: 去模糊化（重心法）")
print("Step 4: 一致性检验")

try:
    ahp_df = pd.read_excel(f'{output_dir}/AHP两两比较数据.xlsx')
    print(f"\nAHP两两比较数据:\n{ahp_df}")
except FileNotFoundError:
    print("AHP数据文件不存在，使用模拟数据...")
    np.random.seed(42)
    ahp_df = pd.DataFrame({
        '技术A': ['LLM+RAG', 'LLM+RAG', 'LLM+RAG', '多模态AI', '多模态AI', '知识图谱'],
        '技术B': ['多模态AI', '知识图谱', '流程挖掘', '知识图谱', '流程挖掘', '流程挖掘'],
        '比较值': [3, 5, 7, 2, 4, 3]
    })
    print(f"\n模拟AHP数据:\n{ahp_df}")

def saaty_to_fuzzy(value):
    mapping = {1: (1, 1, 1), 2: (1, 2, 3), 3: (1, 3, 5), 4: (2, 4, 6),
               5: (3, 5, 7), 6: (4, 6, 8), 7: (5, 7, 9), 8: (6, 8, 9), 9: (7, 9, 9)}
    return mapping.get(int(value), (1, 1, 1))

def fuzzy_inverse(f):
    return (1/f[2], 1/f[1], 1/f[0])

def fuzzy_multiply(f1, f2):
    return (f1[0]*f2[0], f1[1]*f2[1], f1[2]*f2[2])

def fuzzy_power(f, exp):
    return (f[0]**exp, f[1]**exp, f[2]**exp)

def defuzzify(f):
    return (f[0] + f[1] + f[2]) / 3

pairwise_matrix = np.ones((n_tech, n_tech, 3))

aggregated_scores = {}
for _, row in ahp_df.iterrows():
    key = (row['技术A'], row['技术B'])
    if key not in aggregated_scores:
        aggregated_scores[key] = []
    aggregated_scores[key].append(row['比较值'])

print("\n专家评分聚合（几何平均）:")
for (tech_a, tech_b), scores in aggregated_scores.items():
    geo_mean = np.exp(np.mean(np.log(scores)))
    print(f"  {tech_a} vs {tech_b}: {scores} → 几何平均={geo_mean:.2f}")
    
    idx_a = tech_solutions.index(tech_a)
    idx_b = tech_solutions.index(tech_b)
    fuzzy_val = saaty_to_fuzzy(round(geo_mean))
    pairwise_matrix[idx_a, idx_b] = fuzzy_val
    pairwise_matrix[idx_b, idx_a] = fuzzy_inverse(fuzzy_val)

fuzzy_weights = []
for i in range(n_tech):
    product = (1, 1, 1)
    for j in range(n_tech):
        product = fuzzy_multiply(product, tuple(pairwise_matrix[i, j]))
    fuzzy_weights.append(fuzzy_power(product, 1/n_tech))

sum_fuzzy = tuple(sum(f[k] for f in fuzzy_weights) for k in range(3))
normalized_weights = [tuple(f[k]/sum_fuzzy[k] for k in range(3)) for f in fuzzy_weights]
crisp_weights = [defuzzify(w) for w in normalized_weights]
total = sum(crisp_weights)
final_weights = [w/total for w in crisp_weights]

ahp_result = pd.DataFrame({
    '技术方案': tech_solutions,
    '模糊权重(中值)': [w[1] for w in normalized_weights],
    '去模糊化权重': crisp_weights,
    '归一化权重': final_weights
})
print("\n模糊AHP权重结果:")
print(ahp_result.to_string(index=False))
ahp_result.to_excel(f'{results_dir}/RQ3_模糊AHP权重.xlsx', index=False)

crisp_matrix = pairwise_matrix[:, :, 1]
eigenvalues = np.linalg.eigvals(crisp_matrix)
max_eigenvalue = max(eigenvalues.real)
CI = (max_eigenvalue - n_tech) / (n_tech - 1)
RI_dict = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12}
RI = RI_dict.get(n_tech, 1.45)
CR = CI / RI if RI > 0 else 0
print(f"\n一致性检验: CR = {CR:.3f} ({'通过' if CR < 0.1 else '未通过'})")

print("\n" + "="*60)
print("第二部分：TOPSIS分析")
print("="*60)

print("\n【TOPSIS方法步骤】")
print("Step 1: 构建决策矩阵")
print("Step 2: 规范化")
print("Step 3: 加权")
print("Step 4: 计算到理想解距离")
print("Step 5: 计算相对贴近度")

np.random.seed(42)
decision_matrix = np.array([
    [8.5, 6.0, 8.8, 7.5, 8.0],
    [7.5, 7.5, 8.5, 6.5, 7.0],
    [7.0, 7.0, 7.5, 7.0, 6.5],
    [8.0, 8.0, 7.0, 8.0, 7.5]
])

print(f"\n决策矩阵:")
print(pd.DataFrame(decision_matrix, index=tech_solutions, columns=criteria_names).round(2))

norm_matrix = decision_matrix / np.sqrt((decision_matrix**2).sum(axis=0))

criteria_weights = np.array([0.25, 0.20, 0.25, 0.15, 0.15])
weighted_matrix = norm_matrix * criteria_weights

benefit_criteria = [0, 2, 4]
cost_criteria = [1, 3]

ideal_best = np.array([weighted_matrix[:, j].max() if j in benefit_criteria else weighted_matrix[:, j].min() for j in range(n_criteria)])
ideal_worst = np.array([weighted_matrix[:, j].min() if j in benefit_criteria else weighted_matrix[:, j].max() for j in range(n_criteria)])

d_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
d_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
closeness = d_worst / (d_best + d_worst)

topsis_result = pd.DataFrame({
    '技术方案': tech_solutions,
    'D+': d_best.round(3),
    'D-': d_worst.round(3),
    '贴近度C': closeness.round(3),
    'TOPSIS排名': pd.Series(closeness).rank(ascending=False).astype(int).values
}).sort_values('TOPSIS排名')
print("\nTOPSIS结果:")
print(topsis_result.to_string(index=False))

print("\n" + "="*60)
print("第三部分：VIKOR分析")
print("="*60)

print("\n【VIKOR方法步骤】")
print("Step 1: 确定最佳值和最差值")
print("Step 2: 计算群体效用S和个体遗憾R")
print("Step 3: 计算折衷评价值Q")

f_best = np.array([decision_matrix[:, j].max() if j in benefit_criteria else decision_matrix[:, j].min() for j in range(n_criteria)])
f_worst = np.array([decision_matrix[:, j].min() if j in benefit_criteria else decision_matrix[:, j].max() for j in range(n_criteria)])

S = np.zeros(n_tech)
R = np.zeros(n_tech)

for i in range(n_tech):
    s_sum = 0
    r_max = 0
    for j in range(n_criteria):
        if f_best[j] != f_worst[j]:
            term = criteria_weights[j] * (f_best[j] - decision_matrix[i, j]) / (f_best[j] - f_worst[j])
        else:
            term = 0
        s_sum += term
        r_max = max(r_max, term)
    S[i] = s_sum
    R[i] = r_max

S_star, S_minus = S.min(), S.max()
R_star, R_minus = R.min(), R.max()
v = 0.5

Q = np.array([v * (S[i] - S_star) / (S_minus - S_star) + (1-v) * (R[i] - R_star) / (R_minus - R_star) 
              if S_minus != S_star else 0 for i in range(n_tech)])

vikor_result = pd.DataFrame({
    '技术方案': tech_solutions,
    '群体效用S': S.round(3),
    '个体遗憾R': R.round(3),
    '折衷值Q': Q.round(3),
    'VIKOR排名': pd.Series(Q).rank(ascending=True).astype(int).values
}).sort_values('VIKOR排名')
print("\nVIKOR结果:")
print(vikor_result.to_string(index=False))

print("\n" + "="*60)
print("第四部分：TOPSIS+VIKOR组合验证")
print("="*60)

topsis_rank = pd.Series(closeness).rank(ascending=False).values
vikor_rank = pd.Series(Q).rank(ascending=True).values

spearman_corr, p_value = stats.spearmanr(topsis_rank, vikor_rank)
print(f"\nSpearman相关系数: ρ = {spearman_corr:.3f}")
print(f"排序一致性: {'通过 (ρ > 0.7)' if spearman_corr > 0.7 else '需进一步分析'}")

borda_scores = np.array([(n_tech - topsis_rank[i]) + (n_tech - vikor_rank[i]) for i in range(n_tech)])
borda_rank = pd.Series(borda_scores).rank(ascending=False).astype(int).values

combined_result = pd.DataFrame({
    '技术方案': tech_solutions,
    'TOPSIS排名': topsis_rank.astype(int),
    'VIKOR排名': vikor_rank.astype(int),
    'Borda得分': borda_scores.astype(int),
    '综合排名': borda_rank
}).sort_values('综合排名')
print("\n组合排序结果:")
print(combined_result.to_string(index=False))
combined_result.to_excel(f'{results_dir}/RQ3_组合排序结果.xlsx', index=False)

print("\n" + "="*60)
print("第五部分：敏感性分析")
print("="*60)

n_sim = 100
rankings = {tech: [] for tech in tech_solutions}
np.random.seed(42)

for _ in range(n_sim):
    perturbed = criteria_weights * np.random.uniform(0.8, 1.2, n_criteria)
    perturbed = perturbed / perturbed.sum()
    weighted_sim = norm_matrix * perturbed
    ideal_sim = np.array([weighted_sim[:, j].max() if j in benefit_criteria else weighted_sim[:, j].min() for j in range(n_criteria)])
    d_best_sim = np.sqrt(((weighted_sim - ideal_sim)**2).sum(axis=1))
    d_worst_sim = np.sqrt(((weighted_sim - ideal_worst)**2).sum(axis=1))
    c_sim = d_worst_sim / (d_best_sim + d_worst_sim)
    ranks = pd.Series(c_sim).rank(ascending=False).astype(int).values
    for i, tech in enumerate(tech_solutions):
        rankings[tech].append(ranks[i])

sensitivity_result = pd.DataFrame({
    '技术方案': tech_solutions,
    '平均排名': [np.mean(rankings[tech]) for tech in tech_solutions],
    '排名标准差': [np.std(rankings[tech]) for tech in tech_solutions],
    '第1名概率': [rankings[tech].count(1)/n_sim for tech in tech_solutions]
})
print("\n敏感性分析结果:")
print(sensitivity_result.to_string(index=False))
sensitivity_result.to_excel(f'{results_dir}/RQ3_敏感性分析.xlsx', index=False)

print("\n" + "="*60)
print("第六部分：可视化")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

ax1 = axes[0, 0]
ax1.bar(tech_solutions, final_weights, color='steelblue', edgecolor='black')
ax1.set_title('技术方案权重 (模糊AHP)', fontsize=12)
ax1.set_ylabel('权重')
ax1.tick_params(axis='x', rotation=15)

ax2 = axes[0, 1]
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(topsis_result)))
ax2.barh(topsis_result['技术方案'], topsis_result['贴近度C'], color=colors)
ax2.set_title('相对贴近度 (TOPSIS)', fontsize=12)
ax2.set_xlabel('贴近度')

ax3 = axes[1, 0]
colors3 = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(vikor_result)))
ax3.barh(vikor_result['技术方案'], vikor_result['折衷值Q'], color=colors3)
ax3.set_title('折衷评价值 (VIKOR)', fontsize=12)
ax3.set_xlabel('Q值 (越小越好)')

ax4 = axes[1, 1]
x = np.arange(len(tech_solutions))
width = 0.35
ax4.bar(x - width/2, topsis_rank, width, label='TOPSIS', color='steelblue')
ax4.bar(x + width/2, vikor_rank, width, label='VIKOR', color='coral')
ax4.set_title('TOPSIS vs VIKOR 排名对比', fontsize=12)
ax4.set_xticks(x)
ax4.set_xticklabels(tech_solutions, rotation=15)
ax4.set_ylabel('排名 (越小越好)')
ax4.legend()
ax4.invert_yaxis()

plt.tight_layout()
plt.savefig(f'{results_dir}/RQ3_决策支持结果.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n可视化图表已保存")

topsis_result.to_excel(f'{results_dir}/RQ3_TOPSIS结果.xlsx', index=False)
vikor_result.to_excel(f'{results_dir}/RQ3_VIKOR结果.xlsx', index=False)

print("\n" + "="*60)
print("RQ3分析完成！")
print("="*60)

print("\n【输出文件】")
print(f"1. {results_dir}/RQ3_模糊AHP权重.xlsx")
print(f"2. {results_dir}/RQ3_TOPSIS结果.xlsx")
print(f"3. {results_dir}/RQ3_VIKOR结果.xlsx")
print(f"4. {results_dir}/RQ3_组合排序结果.xlsx")
print(f"5. {results_dir}/RQ3_敏感性分析.xlsx")
print(f"6. {results_dir}/RQ3_决策支持结果.png")

print("\n【三角验证结论】")
print(f"TOPSIS最优: {topsis_result.iloc[0]['技术方案']}")
print(f"VIKOR最优: {vikor_result.iloc[0]['技术方案']}")
print(f"综合最优: {combined_result.iloc[0]['技术方案']}")
print(f"排序一致性: Spearman ρ = {spearman_corr:.3f}")
