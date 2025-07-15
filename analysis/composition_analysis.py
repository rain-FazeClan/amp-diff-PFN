import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# 添加data_generated目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_generated'))
from AAComposition import CalculateAAComposition, CalculateDipeptideComposition

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

# 氨基酸组成分析
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')
def calc_aa_composition(seq_list):
    """使用data_generated/AAComposition.py计算氨基酸组成"""
    comp = {aa: [] for aa in AA_LIST}
    for seq in seq_list:
        aa_comp = CalculateAAComposition(str(seq))
        for aa in AA_LIST:
            comp[aa].append(aa_comp.get(aa, 0) / 100.0)  # 转换为小数形式
    avg_comp = {aa: np.mean(comp[aa]) for aa in AA_LIST}
    return avg_comp

# 双肽组成分析
def calc_dipeptide_composition(seq_list):
    """使用data_generated/AAComposition.py计算双肽组成"""
    dipeptides = []
    for i in AA_LIST:
        for j in AA_LIST:
            dipeptides.append(i + j)
    
    dipep_comp = {dipep: [] for dipep in dipeptides}
    for seq in seq_list:
        dipep_data = CalculateDipeptideComposition(str(seq))
        for dipep in dipeptides:
            dipep_comp[dipep].append(dipep_data.get(dipep, 0) / 100.0)  # 转换为小数形式
    avg_dipep_comp = {dipep: np.mean(dipep_comp[dipep]) for dipep in dipeptides}
    return avg_dipep_comp

comp_dict = {}
dipep_dict = {}
for label in labels:
    seqs = groups.get_group(label)['sequence']
    comp_dict[label] = calc_aa_composition(seqs)
    dipep_dict[label] = calc_dipeptide_composition(seqs)


# 创建分析目录

# 修改输出目录为 ./results/candidate_analysis
analysis_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results/candidate_analysis'))
os.makedirs(analysis_dir, exist_ok=True)

# 1. 氨基酸组成柱状图
plt.figure(figsize=(16, 8))
comp_df = pd.DataFrame(comp_dict).T[AA_LIST]
ax = comp_df.plot(kind='bar', figsize=(16, 8), colormap='viridis')
plt.ylabel('Average Percentage')
plt.title('Amino Acid Composition Comparison')
plt.xlabel('Class')
plt.legend(title='Amino Acid', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'aa_composition_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# 2. 氨基酸组成热力图
plt.figure(figsize=(12, 8))
comp_df_normalized = comp_df.div(comp_df.sum(axis=1), axis=0)
sns.heatmap(comp_df_normalized.T, annot=True, fmt='.3f', cmap='viridis',
            xticklabels=labels, yticklabels=AA_LIST)
plt.title('Amino Acid Composition Heatmap')
plt.xlabel('Class')
plt.ylabel('Amino Acid')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'aa_composition_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 3. Amino acid radar chart (key amino acids)
radar_aas = ['R', 'K', 'W', 'I', 'V', 'L', 'A', 'G']
angles = np.linspace(0, 2 * np.pi, len(radar_aas), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
for i, label in enumerate(labels):
    values = [comp_dict[label][aa] for aa in radar_aas]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_aas)
ax.set_ylim(0, max([max([comp_dict[label][aa] for aa in radar_aas]) for label in labels]) * 1.1)
plt.title('Key Amino Acid Composition (Radar)', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'aa_composition_radar.png'), dpi=300, bbox_inches='tight')
plt.close()

# 4. Dipeptide composition analysis (important dipeptides)
important_dipeptides = ['RK', 'KR', 'LL', 'AA', 'GG', 'RR', 'KK', 'WW', 'FF', 'YY']
dipep_subset = {}
for label in labels:
    dipep_subset[label] = {dipep: dipep_dict[label][dipep] for dipep in important_dipeptides}

plt.figure(figsize=(14, 8))
dipep_df = pd.DataFrame(dipep_subset).T[important_dipeptides]
ax = dipep_df.plot(kind='bar', figsize=(14, 8), colormap='plasma')
plt.ylabel('Average Percentage')
plt.title('Important Dipeptide Composition Comparison')
plt.xlabel('Class')
plt.legend(title='Dipeptide', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'dipeptide_composition_bar.png'), dpi=300, bbox_inches='tight')
plt.close()

# 5. 氨基酸组成差异分析
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 疏水性氨基酸
hydrophobic_aas = ['A', 'I', 'L', 'V', 'F', 'W', 'Y', 'M']
hydrophobic_comp = {}
for label in labels:
    hydrophobic_comp[label] = sum([comp_dict[label][aa] for aa in hydrophobic_aas])

axes[0, 0].bar(labels, [hydrophobic_comp[label] for label in labels], color=colors)
axes[0, 0].set_title('Total Hydrophobic Amino Acids')
axes[0, 0].set_ylabel('Percentage')

# 带电氨基酸
charged_aas = ['R', 'K', 'D', 'E', 'H']
charged_comp = {}
for label in labels:
    charged_comp[label] = sum([comp_dict[label][aa] for aa in charged_aas])

axes[0, 1].bar(labels, [charged_comp[label] for label in labels], color=colors)
axes[0, 1].set_title('Total Charged Amino Acids')
axes[0, 1].set_ylabel('Percentage')

# 极性氨基酸
polar_aas = ['S', 'T', 'N', 'Q', 'C']
polar_comp = {}
for label in labels:
    polar_comp[label] = sum([comp_dict[label][aa] for aa in polar_aas])

axes[1, 0].bar(labels, [polar_comp[label] for label in labels], color=colors)
axes[1, 0].set_title('Total Polar Amino Acids')
axes[1, 0].set_ylabel('Percentage')

# 芳香族氨基酸
aromatic_aas = ['F', 'W', 'Y']
aromatic_comp = {}
for label in labels:
    aromatic_comp[label] = sum([comp_dict[label][aa] for aa in aromatic_aas])

axes[1, 1].bar(labels, [aromatic_comp[label] for label in labels], color=colors)
axes[1, 1].set_title('Total Aromatic Amino Acids')
axes[1, 1].set_ylabel('Percentage')

plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'aa_property_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# 6. 双肽组成热力图（选择前20个最变化的双肽）
dipep_df_full = pd.DataFrame(dipep_dict).T
dipep_variance = dipep_df_full.var(axis=1).sort_values(ascending=False)
top_dipeptides = dipep_variance.head(20).index.tolist()

plt.figure(figsize=(12, 10))
dipep_subset_top = dipep_df_full[dipep_df_full.index.isin(top_dipeptides)]
sns.heatmap(dipep_subset_top.T, annot=True, fmt='.4f', cmap='coolwarm',
            xticklabels=top_dipeptides, yticklabels=labels)
plt.title('Top 20 Most Variable Dipeptide Heatmap')
plt.xlabel('Dipeptide')
plt.ylabel('Class')
plt.tight_layout()
plt.savefig(os.path.join(analysis_dir, 'dipeptide_composition_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# 7. 统计信息输出
print("=" * 60)
print("Amino acid and dipeptide composition analysis complete!")
print("=" * 60)
print(f"Analyzed {len(labels)} groups of sequences:")
for label in labels:
    seq_count = len(groups.get_group(label))
    print(f"- {label}: {seq_count} sequences")

print("\nGenerated plots:")
print("1. aa_composition_bar.png - Amino acid composition bar chart")
print("2. aa_composition_heatmap.png - Amino acid composition heatmap")
print("3. aa_composition_radar.png - Key amino acid radar chart")
print("4. dipeptide_composition_bar.png - Important dipeptide bar chart")
print("5. aa_property_analysis.png - Amino acid property analysis")
print("6. dipeptide_composition_heatmap.png - Dipeptide composition heatmap")
print("=" * 60)

# 8. Data summary
print("\nAmino acid composition summary:")
print("-" * 40)
comp_summary = pd.DataFrame(comp_dict).T
print(comp_summary.describe())

print("\nImportant dipeptide composition summary:")
print("-" * 40)
dipep_summary = pd.DataFrame(dipep_subset).T
print(dipep_summary.describe())
