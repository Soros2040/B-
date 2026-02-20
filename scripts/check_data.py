import pandas as pd

df = pd.read_excel('data/金融经济领域 RPA 需求与技术.xlsx')

print('='*80)
print('【七大端分布】')
print('='*80)
print(df['端'].value_counts())
print()

print('='*80)
print('【各端对应的行业】')
print('='*80)
for duan in df['端'].unique():
    industries = df[df['端']==duan]['行业'].unique()
    print(f'{duan}: {list(industries)}')
print()

print('='*80)
print('【需求分类分布】')
print('='*80)
print(df['需求分类'].value_counts())
