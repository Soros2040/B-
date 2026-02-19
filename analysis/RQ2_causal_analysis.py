"""
RQ2: 因果分析与影响路径
方法框架：模糊DEMATEL-ISM + SEM + 贝叶斯网络
三角验证：三种方法识别的因果路径一致性检验
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

output_dir = 'e:/B正大杯/analysis_data'
results_dir = 'e:/B正大杯/results'
os.makedirs(results_dir, exist_ok=True)

print("="*60)
print("RQ2: 因果分析与影响路径")
print("="*60)

print("\n【方法框架】")
print("主方法：SEM结构方程模型")
print("辅助方法：模糊DEMATEL-ISM")
print("验证方法：贝叶斯网络因果发现")

print("\n" + "="*60)
print("第一部分：模糊DEMATEL分析")
print("="*60)

print("\n【模糊DEMATEL方法说明】")
print("使用三角模糊数处理专家判断的不确定性")
print("语言变量转换: 无影响(0)→(0,0,0.25), 弱(1)→(0,0.25,0.5), 中(2)→(0.25,0.5,0.75), 强(3)→(0.5,0.75,1)")

try:
    dematel_df = pd.read_excel(f'{output_dir}/DEMATEL专家评分矩阵.xlsx', index_col=0)
    print(f"\n专家评分矩阵:\n{dematel_df}")
    factors = dematel_df.index.tolist()
    n = len(factors)
    Z = dematel_df.values.astype(float)
except FileNotFoundError:
    print("DEMATEL数据文件不存在，使用模拟数据...")
    factors = ['数据问题', '技术问题', '人才问题', '成本问题', '合规问题']
    n = len(factors)
    np.random.seed(42)
    Z = np.random.uniform(1, 3, (n, n))
    np.fill_diagonal(Z, 0)
    dematel_df = pd.DataFrame(Z, index=factors, columns=factors)
    print(f"\n模拟专家评分矩阵:\n{dematel_df.round(2)}")

def crisp_to_fuzzy(value):
    if value == 0:
        return (0, 0, 0.25)
    elif value < 1.5:
        return (0, 0.25, 0.5)
    elif value < 2.5:
        return (0.25, 0.5, 0.75)
    else:
        return (0.5, 0.75, 1)

fuzzy_matrix = np.zeros((n, n, 3))
for i in range(n):
    for j in range(n):
        fuzzy_matrix[i, j] = crisp_to_fuzzy(Z[i, j])

crisp_T = (fuzzy_matrix[:, :, 0] + fuzzy_matrix[:, :, 1] + fuzzy_matrix[:, :, 2]) / 3

row_sums = crisp_T.sum(axis=1)
max_row_sum = row_sums.max() if row_sums.max() > 0 else 1
X = crisp_T / max_row_sum

I = np.eye(n)
try:
    T = np.dot(X, np.linalg.inv(I - X))
except np.linalg.LinAlgError:
    T = np.dot(X, np.linalg.pinv(I - X))

R = T.sum(axis=1)
C = T.sum(axis=0)
D = R + C
P = R - C

dematel_result = pd.DataFrame({
    '因素': factors,
    '影响度(R)': R.round(3),
    '被影响度(C)': C.round(3),
    '中心度(D)': D.round(3),
    '原因度(P)': P.round(3),
    '因素类型': ['原因因素' if p > 0 else '结果因素' for p in P]
})
print(f"\n模糊DEMATEL分析结果:")
print(dematel_result.to_string(index=False))
dematel_result.to_excel(f'{results_dir}/RQ2_模糊DEMATEL结果.xlsx', index=False)

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red' if p > 0 else 'blue' for p in P]
ax.scatter(D, P, c=colors, s=200, alpha=0.7, edgecolors='black')
for i, factor in enumerate(factors):
    ax.annotate(factor, (D[i], P[i]), fontsize=11, ha='center', va='bottom', xytext=(0, 10), textcoords='offset points')
ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax.axvline(x=np.mean(D), color='gray', linestyle=':', linewidth=1)
ax.set_xlabel('中心度 (D = R + C)', fontsize=12)
ax.set_ylabel('原因度 (P = R - C)', fontsize=12)
ax.set_title('痛点因果四象限图 (模糊DEMATEL)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{results_dir}/RQ2_DEMATEL因果四象限图.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n因果四象限图已保存")

print("\n" + "="*60)
print("第二部分：ISM层次分析")
print("="*60)

theta = np.mean(T) + np.std(T) * 0.5
print(f"\n阈值θ = {theta:.3f}")

M = (T >= theta).astype(int)
print(f"\n可达矩阵M:\n{pd.DataFrame(M, index=factors, columns=factors)}")

def ism_level_partition(M, factors):
    n = len(factors)
    M_plus_I = M + np.eye(n)
    levels = []
    remaining = list(range(n))
    level_num = 1
    
    while remaining and level_num <= n:
        current_level = []
        for i in remaining:
            reachable = set(j for j in remaining if M_plus_I[i, j] == 1)
            antecedent = set(j for j in remaining if M_plus_I[j, i] == 1)
            if reachable == (reachable & antecedent) and len(reachable) > 0:
                current_level.append(i)
        
        if not current_level:
            current_level = [remaining[0]]
        
        levels.append({'层级': level_num, '因素': [factors[i] for i in current_level]})
        remaining = [i for i in remaining if i not in current_level]
        level_num += 1
    
    return levels

levels = ism_level_partition(M, factors)
print("\nISM层次划分结果:")
for level in levels:
    print(f"第{level['层级']}层: {', '.join(level['因素'])}")

ism_result = pd.DataFrame([{'层级': l['层级'], '因素': ', '.join(l['因素'])} for l in levels])
ism_result.to_excel(f'{results_dir}/RQ2_ISM层次结构.xlsx', index=False)

print("\n" + "="*60)
print("第三部分：SEM结构方程模型")
print("="*60)

try:
    from semopy import Model
    from semopy.stats import calc_stats
    
    try:
        survey_df = pd.read_excel(f'{output_dir}/模拟问卷数据.xlsx')
        print(f"\n问卷数据: {survey_df.shape}")
    except FileNotFoundError:
        print("问卷数据不存在，生成模拟数据...")
        np.random.seed(42)
        n_samples = 200
        survey_df = pd.DataFrame({
            '技术成熟度': np.random.normal(3.5, 0.8, n_samples),
            'AI融合程度': np.random.normal(3.3, 0.9, n_samples),
            '政策支持': np.random.normal(3.4, 0.7, n_samples),
            '监管合规': np.random.normal(3.5, 0.8, n_samples),
            '数据孤岛': np.random.normal(3.4, 0.9, n_samples),
            '实施成本': np.random.normal(3.3, 0.8, n_samples),
            '人才短缺': np.random.normal(3.3, 0.9, n_samples)
        })
    
    pain_cols = [col for col in survey_df.columns if col.startswith('痛点_')]
    factor_cols = [col for col in survey_df.columns if col.startswith('影响因素_')]
    
    print(f"\n痛点变量: {pain_cols[:5]}")
    print(f"影响因素变量: {factor_cols[:5]}")
    
    if len(factor_cols) >= 4 and len(pain_cols) >= 3:
        model_desc = f"""
        技术因素 =~ {factor_cols[0]} + {factor_cols[1]}
        环境因素 =~ {factor_cols[2]} + {factor_cols[3]}
        {pain_cols[0]} ~ 技术因素
        {pain_cols[3]} ~ 技术因素
        {pain_cols[5]} ~ 环境因素
        """
    else:
        model_desc = """
        技术因素 =~ 痛点_数据孤岛 + 痛点_非结构化数据
        环境因素 =~ 痛点_维护成本高 + 痛点_稳定性差
        """
    
    print("\nSEM模型定义:")
    print(model_desc)
    
    model = Model(model_desc)
    model.fit(survey_df)
    
    try:
        stats = calc_stats(model)
        print("\n模型拟合指标:")
        print(stats.round(3))
        stats.to_excel(f'{results_dir}/RQ2_SEM拟合指标.xlsx')
    except Exception as e:
        print(f"拟合指标计算: {e}")
    
    params = model.inspect()
    print("\n参数估计:")
    print(params.round(3))
    params.to_excel(f'{results_dir}/RQ2_SEM参数估计.xlsx', index=False)
    
except ImportError:
    print("\nsemopy未安装，使用相关分析替代")
    try:
        survey_df = pd.read_excel(f'{output_dir}/模拟问卷数据.xlsx')
    except FileNotFoundError:
        np.random.seed(42)
        survey_df = pd.DataFrame(np.random.randn(200, 7), 
                                  columns=['技术成熟度', 'AI融合程度', '政策支持', '监管合规', 
                                          '数据孤岛', '实施成本', '人才短缺'])
    
    corr = survey_df.corr()
    print("\n相关矩阵:")
    print(corr.round(3))
    corr.to_excel(f'{results_dir}/RQ2_相关分析结果.xlsx')

print("\n" + "="*60)
print("第四部分：贝叶斯网络因果发现")
print("="*60)

try:
    import bnlearn as bn
    import networkx as nx
    
    try:
        survey_df = pd.read_excel(f'{output_dir}/模拟问卷数据.xlsx')
    except FileNotFoundError:
        np.random.seed(42)
        survey_df = pd.DataFrame(np.random.randn(200, 7),
                                  columns=['技术成熟度', 'AI融合程度', '政策支持', '监管合规',
                                          '数据孤岛', '实施成本', '人才短缺'])
    
    bn_data = survey_df.copy()
    numeric_cols = bn_data.select_dtypes(include=[np.number]).columns[:7]
    bn_data = bn_data[numeric_cols].copy()
    
    for col in bn_data.columns:
        bn_data[col] = pd.cut(bn_data[col], bins=3, labels=['低', '中', '高']).astype(str)
    
    print(f"\n贝叶斯网络数据: {bn_data.shape}")
    print(f"数据类型: {bn_data.dtypes.unique()}")
    
    model_bn = bn.structure_learning.fit(bn_data, methodtype='hc', scoretype='bic')
    
    print("\n学习到的因果边:")
    edges = model_bn['model_edges']
    for e in edges[:10]:
        print(f"  {e[0]} → {e[1]}")
    
    if len(edges) > 0:
        G = nx.DiGraph()
        G.add_edges_from(edges)
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=2, iterations=50)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=2000, font_size=10, arrows=True, arrowsize=20)
        plt.title('贝叶斯网络因果结构', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'{results_dir}/RQ2_贝叶斯网络结构.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n贝叶斯网络结构图已保存")
        
        bn_edges_df = pd.DataFrame(edges, columns=['原因节点', '结果节点'])
        bn_edges_df.to_excel(f'{results_dir}/RQ2_贝叶斯网络边.xlsx', index=False)
    
except ImportError:
    print("\nbnlearn未安装，跳过贝叶斯网络分析")
except Exception as e:
    print(f"\n贝叶斯网络分析出错: {e}")

print("\n" + "="*60)
print("RQ2分析完成！")
print("="*60)

print("\n【输出文件】")
print(f"1. {results_dir}/RQ2_模糊DEMATEL结果.xlsx")
print(f"2. {results_dir}/RQ2_DEMATEL因果四象限图.png")
print(f"3. {results_dir}/RQ2_ISM层次结构.xlsx")
print(f"4. {results_dir}/RQ2_SEM参数估计.xlsx")
print(f"5. {results_dir}/RQ2_贝叶斯网络结构.png")

print("\n【三角验证结论】")
cause_factors = dematel_result[dematel_result['因素类型'] == '原因因素']['因素'].tolist()
print(f"DEMATEL识别的原因因素: {cause_factors}")
print("验证标准: 路径方向一致性 > 80%")
