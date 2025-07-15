import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_generated')))
from BasicDes import cal_discriptors

# 路径设置
gen_path = os.path.join(os.path.dirname(__file__), '../results/generated_peptides/candidate_amps.csv')
cls_path = os.path.join(os.path.dirname(__file__), '../classify_data/classify.csv')

# 读取生成序列
gen_df = pd.read_csv(gen_path, header=None, names=['sequence'])
gen_df['label'] = 'Generated'

# 读取分类数据
df = pd.read_csv(cls_path)
if 'sequence' not in df.columns or 'label' not in df.columns:
    raise ValueError('classify.csv 必须包含 sequence 和 label 两列')
# 只保留label为1和0，并映射为GramPA和Negative
df = df[df['label'].isin([1, 0])].copy()
df['label'] = df['label'].map({1: 'GramPA', 0: 'Negative'})

# 合并三类
total_df = pd.concat([gen_df, df[['sequence', 'label']]], ignore_index=True)
groups = total_df.groupby('label')
labels = ['Generated', 'GramPA', 'Negative']

# 理化性质计算（使用BasicDes.py）
phy_dict = {label: {'length': [], 'mw': [], 'pi': [], 'charge': [], 'hydrophobicity': []} for label in labels}
for label in labels:
    seqs = groups.get_group(label)['sequence']
    for seq in seqs:
        seq = str(seq)
        try:
            desc = cal_discriptors(seq)
            phy_dict[label]['length'].append(len(seq))
            phy_dict[label]['mw'].append(desc.get('Mw', np.nan))
            phy_dict[label]['pi'].append(desc.get('ph_number', np.nan))
            phy_dict[label]['charge'].append(desc.get('charge of all', np.nan))
            phy_dict[label]['hydrophobicity'].append(desc.get('hydrophobicity', np.nan))
        except Exception:
            phy_dict[label]['length'].append(np.nan)
            phy_dict[label]['mw'].append(np.nan)
            phy_dict[label]['pi'].append(np.nan)
            phy_dict[label]['charge'].append(np.nan)
            phy_dict[label]['hydrophobicity'].append(np.nan)

# 理化性质分布图
for prop in ['length', 'mw', 'pi', 'charge', 'hydrophobicity']:
    plt.figure(figsize=(10,5))
    for label in labels:
        arr = np.array(phy_dict[label][prop])
        arr = arr[~np.isnan(arr)]
        plt.hist(arr, bins=30, alpha=0.5, label=label, density=True)
    plt.xlabel(prop)
    plt.ylabel('Density')
    plt.title(f'{prop.capitalize()} Distribution by Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), f'{prop}_hist.png'))
    plt.close()

# 散点图：净电荷 vs 疏水性
plt.figure(figsize=(8,6))
for label in labels:
    x = np.array(phy_dict[label]['charge'])
    y = np.array(phy_dict[label]['hydrophobicity'])
    mask = ~np.isnan(x) & ~np.isnan(y)
    plt.scatter(x[mask], y[mask], alpha=0.5, label=label, s=20)
plt.xlabel('Net Charge (pH=7)')
plt.ylabel('Hydrophobicity (mean)')
plt.title('Net Charge vs Hydrophobicity')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'charge_vs_hydro.png'))
plt.close()

print('理化性质分析完成，图片已保存在 analysis 目录下。')
