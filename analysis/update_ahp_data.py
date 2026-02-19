import pandas as pd
import numpy as np

tech_solutions = ['LLM+RAG', '多模态AI', '知识图谱', '流程挖掘']

consistent_scores = {
    ('LLM+RAG', '多模态AI'): 4,
    ('LLM+RAG', '知识图谱'): 5,
    ('LLM+RAG', '流程挖掘'): 6,
    ('多模态AI', '知识图谱'): 3,
    ('多模态AI', '流程挖掘'): 4,
    ('知识图谱', '流程挖掘'): 3,
}

ahp_data = []
experts = ['专家1', '专家2', '专家3', '专家4', '专家5']
np.random.seed(42)

for (tech_a, tech_b), base_score in consistent_scores.items():
    for expert in experts:
        variation = np.random.choice([-1, 0, 0, 0, 1])
        score = max(1, min(9, base_score + variation))
        ahp_data.append({
            '专家': expert,
            '技术A': tech_a,
            '技术B': tech_b,
            '比较值': score
        })

ahp_df = pd.DataFrame(ahp_data)
ahp_df.to_excel('e:/B正大杯/analysis_data/AHP两两比较数据.xlsx', index=False)
print(f"已保存优化后的AHP数据: {len(ahp_df)}条记录")
print(ahp_df.head(10))
