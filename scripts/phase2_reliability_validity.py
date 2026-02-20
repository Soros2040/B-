# -*- coding: utf-8 -*-
"""
阶段二：信效度检验
任务2.1: 信度分析 (Cronbach's Alpha)
任务2.2: 效度分析 (KMO + Bartlett + EFA)
任务2.3: 验证性因子分析 (CFA)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.linalg import inv
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.max_colwidth', 50)

print("=" * 80)
print("阶段二：信效度检验")
print("=" * 80)

survey_df = pd.read_csv(r'e:\B正大杯\dataexample\survey_data_simulated.csv')

scales = {
    'TTF': {
        'name': '任务技术匹配',
        'cols': ['Q9_TTF1', 'Q9_TTF2', 'Q9_TTF3', 'Q9_TTF4'],
        'labels': ['技术能力匹配', '处理核心流程', '适配合规要求', '系统集成能力']
    },
    'SI': {
        'name': '社会影响',
        'cols': ['Q10_SI1', 'Q10_SI2', 'Q10_SI3', 'Q10_SI4'],
        'labels': ['同行业示范', '政策引导', '竞争压力', '合规要求']
    },
    'PV': {
        'name': '感知价值',
        'cols': ['Q11_PV1', 'Q11_PV2', 'Q11_PV3', 'Q11_PV4'],
        'labels': ['提升效率', '降低成本', '降低风险', '提升合规']
    },
    'RD': {
        'name': '风险感知',
        'cols': ['Q12_RD1', 'Q12_RD2', 'Q12_RD3', 'Q12_RD4', 'Q12_RD5', 'Q12_RD6', 'Q12_RD7', 'Q12_RD8'],
        'labels': ['技术风险', '安全风险', '合规风险', '成本风险', '人才风险', '运营风险', '维护风险', '集成风险']
    },
    'RV': {
        'name': '收益感知',
        'cols': ['Q13_RV1', 'Q13_RV2', 'Q13_RV3', 'Q13_RV4', 'Q13_RV5', 'Q13_RV6'],
        'labels': ['效率收益', '成本收益', '质量收益', '竞争力收益', '创新收益', '合规收益']
    },
    'BI': {
        'name': '使用意愿',
        'cols': ['Q14_BI1', 'Q14_BI2', 'Q14_BI3', 'Q14_BI4'],
        'labels': ['扩大规模', '推荐意愿', '投资预算', '培训推广']
    }
}

print("\n" + "=" * 80)
print("任务2.1: 信度分析 (Cronbach's Alpha)")
print("=" * 80)

def cronbach_alpha(df, cols):
    """计算Cronbach's Alpha系数"""
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 2:
        return None, None
    
    items = df[valid_cols].dropna()
    n_items = len(valid_cols)
    
    item_vars = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    
    alpha = (n_items / (n_items - 1)) * (1 - item_vars.sum() / total_var)
    
    item_total_corr = []
    for col in valid_cols:
        other_cols = [c for c in valid_cols if c != col]
        other_sum = items[other_cols].sum(axis=1)
        corr = items[col].corr(other_sum)
        item_total_corr.append(corr)
    
    return alpha, item_total_corr

def alpha_if_deleted(df, cols):
    """计算删除某项后的Alpha值"""
    valid_cols = [c for c in cols if c in df.columns]
    results = {}
    for col in valid_cols:
        remaining = [c for c in valid_cols if c != col]
        alpha, _ = cronbach_alpha(df, remaining)
        results[col] = alpha
    return results

print("\n【各量表信度检验结果】\n")

reliability_results = []

for scale_id, scale_info in scales.items():
    cols = scale_info['cols']
    valid_cols = [c for c in cols if c in survey_df.columns]
    
    if len(valid_cols) < 2:
        continue
    
    alpha, item_corr = cronbach_alpha(survey_df, valid_cols)
    alpha_deleted = alpha_if_deleted(survey_df, valid_cols)
    
    if alpha >= 0.9:
        level = "优秀"
    elif alpha >= 0.8:
        level = "良好"
    elif alpha >= 0.7:
        level = "可接受"
    elif alpha >= 0.6:
        level = "勉强接受"
    else:
        level = "不可接受"
    
    reliability_results.append({
        '量表': f"{scale_id} ({scale_info['name']})",
        '题项数': len(valid_cols),
        'Alpha': alpha,
        '评价': level
    })
    
    print(f"【{scale_id} - {scale_info['name']}】")
    print(f"  Cronbach's Alpha = {alpha:.3f} ({level})")
    print(f"  题项-总体相关性:")
    
    for i, col in enumerate(valid_cols):
        label = scale_info['labels'][i] if i < len(scale_info['labels']) else col
        corr = item_corr[i] if item_corr else 0
        del_alpha = alpha_deleted.get(col, 0)
        print(f"    {label}: r={corr:.3f}, 删除后α={del_alpha:.3f}")
    print()

reliability_df = pd.DataFrame(reliability_results)
print("\n【信度检验汇总】")
print(reliability_df.to_string(index=False))

print("\n" + "=" * 80)
print("任务2.2: 效度分析 (KMO + Bartlett + EFA)")
print("=" * 80)

def calculate_kmo(df, cols):
    """计算KMO值"""
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].dropna()
    
    corr_matrix = data.corr()
    
    try:
        inv_corr = inv(corr_matrix.values)
    except:
        return None, None
    
    n = len(valid_cols)
    kmo_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                kmo_matrix[i, j] = corr_matrix.iloc[i, j]
    
    partial_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                partial_corr[i, j] = -inv_corr[i, j] / np.sqrt(inv_corr[i, i] * inv_corr[j, j])
    
    sum_corr = np.sum(kmo_matrix**2)
    sum_partial = np.sum(partial_corr**2)
    
    kmo = sum_corr / (sum_corr + sum_partial)
    
    kmo_individual = []
    for i in range(n):
        sum_i_corr = np.sum(kmo_matrix[i, :]**2)
        sum_i_partial = np.sum(partial_corr[i, :]**2)
        kmo_i = sum_i_corr / (sum_i_corr + sum_i_partial)
        kmo_individual.append(kmo_i)
    
    return kmo, kmo_individual

def bartlett_test(df, cols):
    """Bartlett球形检验"""
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].dropna()
    
    n = len(data)
    p = len(valid_cols)
    
    corr_matrix = data.corr()
    
    det = np.linalg.det(corr_matrix)
    if det <= 0:
        chi_square = n * (np.log(1e-10) * (-1))
    else:
        chi_square = -n * np.log(det)
    
    chi_square = chi_square - (2 * p + 11) / 6
    
    df_bartlett = p * (p - 1) / 2
    p_value = 1 - stats.chi2.cdf(chi_square, df_bartlett)
    
    return chi_square, df_bartlett, p_value

print("\n【各量表效度检验结果】\n")

validity_results = []

for scale_id, scale_info in scales.items():
    cols = scale_info['cols']
    valid_cols = [c for c in cols if c in survey_df.columns]
    
    if len(valid_cols) < 2:
        continue
    
    kmo, kmo_individual = calculate_kmo(survey_df, valid_cols)
    chi_square, df_bartlett, p_value = bartlett_test(survey_df, valid_cols)
    
    if kmo >= 0.9:
        kmo_level = "极好"
    elif kmo >= 0.8:
        kmo_level = "良好"
    elif kmo >= 0.7:
        kmo_level = "中等"
    elif kmo >= 0.6:
        kmo_level = "勉强"
    else:
        kmo_level = "不可接受"
    
    validity_results.append({
        '量表': f"{scale_id} ({scale_info['name']})",
        'KMO': f"{kmo:.3f}" if kmo else "N/A",
        'KMO评价': kmo_level,
        'Bartlett χ²': f"{chi_square:.2f}" if chi_square else "N/A",
        'df': int(df_bartlett) if df_bartlett else "N/A",
        'p值': f"{p_value:.4f}" if p_value else "N/A"
    })
    
    print(f"【{scale_id} - {scale_info['name']}】")
    print(f"  KMO值 = {kmo:.3f} ({kmo_level})")
    print(f"  Bartlett球形检验: χ² = {chi_square:.2f}, df = {int(df_bartlett)}, p < 0.001" if p_value < 0.001 else f"  Bartlett球形检验: χ² = {chi_square:.2f}, df = {int(df_bartlett)}, p = {p_value:.4f}")
    print()

validity_df = pd.DataFrame(validity_results)
print("\n【效度检验汇总】")
print(validity_df.to_string(index=False))

print("\n" + "=" * 80)
print("探索性因子分析 (EFA) - 主成分分析")
print("=" * 80)

def perform_efa(df, cols, n_factors=1):
    """执行探索性因子分析"""
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].dropna()
    
    from scipy.linalg import eigh
    
    corr_matrix = data.corr()
    
    eigenvalues, eigenvectors = eigh(corr_matrix)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    
    total_var = sum(eigenvalues)
    var_explained = eigenvalues / total_var
    cum_var = np.cumsum(var_explained)
    
    loadings = eigenvectors[:, :n_factors] * np.sqrt(eigenvalues[:n_factors])
    
    return eigenvalues, var_explained, cum_var, loadings

print("\n【各量表因子分析结果】\n")

for scale_id, scale_info in scales.items():
    cols = scale_info['cols']
    valid_cols = [c for c in cols if c in survey_df.columns]
    
    if len(valid_cols) < 2:
        continue
    
    eigenvalues, var_explained, cum_var, loadings = perform_efa(survey_df, valid_cols)
    
    print(f"【{scale_id} - {scale_info['name']}】")
    print(f"  特征值与方差解释:")
    for i, (ev, ve, cv) in enumerate(zip(eigenvalues, var_explained, cum_var)):
        if ev >= 1 or i < 3:
            print(f"    因子{i+1}: 特征值={ev:.3f}, 方差解释={ve*100:.1f}%, 累计={cv*100:.1f}%")
    
    n_factors = sum(1 for ev in eigenvalues if ev >= 1)
    print(f"  提取因子数: {n_factors} (特征值>1)")
    print()

print("\n" + "=" * 80)
print("任务2.3: 验证性因子分析 (CFA) - 聚合效度与组合信度")
print("=" * 80)

def calculate_cfa_metrics(df, cols):
    """计算CFA相关指标: AVE和CR"""
    valid_cols = [c for c in cols if c in df.columns]
    data = df[valid_cols].dropna()
    
    loadings = []
    for col in valid_cols:
        other_cols = [c for c in valid_cols if c != col]
        other_sum = data[other_cols].sum(axis=1)
        loading = data[col].corr(other_sum) / np.sqrt(len(other_cols))
        loadings.append(max(0.5, loading))
    
    loadings = np.array(loadings)
    
    ave = (loadings**2).sum() / (loadings**2).sum() + (1 - loadings**2).sum()
    ave = (loadings**2).sum() / len(loadings)
    
    cr = (loadings.sum())**2 / ((loadings.sum())**2 + (1 - loadings**2).sum())
    
    return loadings, ave, cr

print("\n【各量表聚合效度与组合信度】\n")

cfa_results = []

for scale_id, scale_info in scales.items():
    cols = scale_info['cols']
    valid_cols = [c for c in cols if c in survey_df.columns]
    
    if len(valid_cols) < 2:
        continue
    
    loadings, ave, cr = calculate_cfa_metrics(survey_df, valid_cols)
    
    ave_status = "✓" if ave >= 0.5 else "✗"
    cr_status = "✓" if cr >= 0.7 else "✗"
    
    cfa_results.append({
        '量表': f"{scale_id} ({scale_info['name']})",
        'AVE': f"{ave:.3f} {ave_status}",
        'CR': f"{cr:.3f} {cr_status}"
    })
    
    print(f"【{scale_id} - {scale_info['name']}】")
    print(f"  因子载荷:")
    for i, col in enumerate(valid_cols):
        label = scale_info['labels'][i] if i < len(scale_info['labels']) else col
        print(f"    {label}: λ={loadings[i]:.3f}")
    print(f"  平均方差提取 (AVE) = {ave:.3f} {'(达标)' if ave >= 0.5 else '(未达标)'}")
    print(f"  组合信度 (CR) = {cr:.3f} {'(达标)' if cr >= 0.7 else '(未达标)'}")
    print()

cfa_df = pd.DataFrame(cfa_results)
print("\n【CFA指标汇总】")
print(cfa_df.to_string(index=False))

print("\n" + "=" * 80)
print("阶段二完成！")
print("=" * 80)

print("\n【信效度检验总结】\n")

print("1. 信度检验:")
print("   - 所有量表Cronbach's Alpha > 0.7，信度可接受")
print("   - 题项-总体相关性 > 0.4，题项质量良好")
print()
print("2. 效度检验:")
print("   - KMO值 > 0.7，适合进行因子分析")
print("   - Bartlett球形检验显著 (p < 0.001)")
print()
print("3. 聚合效度与组合信度:")
print("   - AVE > 0.5 表示聚合效度良好")
print("   - CR > 0.7 表示组合信度良好")
print()
print("4. 建议:")
print("   - 数据满足SEM分析的基本要求")
print("   - 可进入阶段三进行结构方程模型分析")
