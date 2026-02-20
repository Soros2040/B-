import pandas as pd
df3 = pd.read_excel('dataexample/政策数据version2.xlsx')
print('政策数据详细内容:')
for i, row in df3.iterrows():
    print(f'--- 第{i+1}条 ---')
    print(f'分类: {row["分类"]}')
    print(f'文献名称: {row["文献名称"]}')
    print(f'发布部门: {row["发布部门/来源"]}')
    core = str(row["核心内容（与RPA相关）"])[:200] if pd.notna(row["核心内容（与RPA相关）"]) else "N/A"
    print(f'核心内容: {core}')
    scene = row["对应RPA场景"] if pd.notna(row["对应RPA场景"]) else "N/A"
    print(f'对应RPA场景: {scene}')
    print()
