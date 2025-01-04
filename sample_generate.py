import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('D:/amp/peptide_descriptors.csv')

# 定义菌群列名（注意：与你的CSV文件中的列名完全匹配，包括空格）
bacteria_cols = {
    'MRSA ': 'MRSA',
    'E. coil ': 'E.coli',
    'A. baannii ': 'A.baumannii',
    'P. aeruginosa': 'P.aeruginosa',
    'K. pneoniae': 'K.pneumoniae'
}

# 创建空的DataFrame来存储正负样本
positive_samples = []
negative_samples = []

# 获取特征列名（所有的理化特性列）
feature_cols = [
    'pro_dipole_moment', 'pro_hyd_moment', 'pro_volume', 'pro_app_charge',
    'pro_asa_hph', 'pro_asa_hyd', 'pro_asa_vdw', 'pro_coeff_280',
    'pro_debye', 'pro_eccen', 'pro_helicity', 'pro_mass', 'pro_mobility',
    'pro_net_charge', 'pro_patch_hyd', 'pro_pI_3D', 'pro_pI_seq',
    'pro_r_gyr', 'pro_r_solv', 'pro_zeta', 'pro_zquadrupole',
    'logP(o/w)', 'vol', 'Weight', 'b_rotN', 'a_acc', 'a_don',
    'b_1rotN', 'KierFlex', 'mr', 'solvent_accessibility', 'charge',
    'electrostatic_charge', 'hydrophobicity', 'side_chain_mass',
    'surface_area', 'isoelectric_point', 'alpha_helix_propensity',
    'beta_sheet_propensity', 'turn_propensity'
]

# 处理每种菌群的数据
for col, bacteria_name in bacteria_cols.items():
    # 获取当前菌群的数据
    current_data = df[['Sequence', col] + feature_cols].dropna()

    # 根据MIC值分类
    # 阳性样本 (MIC ≤ 8)
    positive = current_data[current_data[col] <= 8].copy()
    if not positive.empty:
        positive['bacteria'] = bacteria_name
        positive.rename(columns={col: 'MIC'}, inplace=True)
        positive_samples.append(positive)

    # 阴性样本 (MIC > 8)
    negative = current_data[current_data[col] > 8].copy()
    if not negative.empty:
        negative['bacteria'] = bacteria_name
        negative.rename(columns={col: 'MIC'}, inplace=True)
        negative_samples.append(negative)

# 合并所有样本
if positive_samples:
    positive_df = pd.concat(positive_samples, ignore_index=True)
    # 重新排列列顺序
    cols_order = ['Sequence', 'MIC', 'bacteria'] + feature_cols
    positive_df = positive_df[cols_order]
    # 保存阳性样本
    positive_df.to_csv('positive_new.csv', index=False)
    print(f"已保存阳性样本文件，共 {len(positive_df)} 条记录")

if negative_samples:
    negative_df = pd.concat(negative_samples, ignore_index=True)
    # 重新排列列顺序
    cols_order = ['Sequence', 'MIC', 'bacteria'] + feature_cols
    negative_df = negative_df[cols_order]
    # 保存阴性样本
    negative_df.to_csv('negative_new.csv', index=False)
    print(f"已保存阴性样本文件，共 {len(negative_df)} 条记录")

print("处理完成！")