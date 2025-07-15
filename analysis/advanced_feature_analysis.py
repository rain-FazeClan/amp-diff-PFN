import pandas as pd
import numpy as np
import os
import sys
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import pairwise_kernels

# 路径设置
gen_path = os.path.join(os.path.dirname(__file__), '../results/generated_peptides/candidate_amps.csv')
cls_path = os.path.join(os.path.dirname(__file__), '../classify_data/classify.csv')
output_dir = os.path.join(os.path.dirname(__file__), '../results/candidate_analysis')
os.makedirs(output_dir, exist_ok=True)

# 导入特征计算脚本
data_gen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data_generated'))
sys.path.append(data_gen_path)
from CTD import CalculateCTD
from Autocorrelation import CalculateAutoTotal
from QuasiSequenceOrder import GetQuasiSequenceOrder

# 读取生成序列
gen_df = pd.read_csv(gen_path, header=None, names=['sequence'])
gen_df['label'] = 'Generated'

# 读取分类数据
df = pd.read_csv(cls_path)
if 'sequence' not in df.columns or 'label' not in df.columns:
    raise ValueError('classify.csv 必须包含 sequence 和 label 两列')
df = df[df['label'].isin([1, 0])].copy()
df['label'] = df['label'].map({1: 'GramPA', 0: 'Negative'})

# 合并三类
total_df = pd.concat([gen_df, df[['sequence', 'label']]], ignore_index=True)
labels = ['Generated', 'GramPA', 'Negative']

# 特征计算函数
def extract_features(seq):
    """
    提取蛋白质序列的高阶特征
    
    参数:
    seq: 蛋白质序列字符串
    
    返回:
    feature_dict: 包含特征名称和数值的字典
    feature_array: 特征数组
    """
    try:
        # CTD (Composition, Transition, Distribution) 描述符
        # 基于7种氨基酸属性计算组成、转换和分布描述符，共147个特征
        ctd_dict = CalculateCTD(seq)
        
        # Autocorrelation 描述符 (lambda=3)
        # 基于8种氨基酸属性计算自相关描述符，包括Moreau-Broto、Moran、Geary三种类型
        auto_dict = CalculateAutoTotal(seq, lamba=3)
        
        # Quasi-Sequence Order 描述符 (maxlag=3)
        # 基于氨基酸序列顺序和距离矩阵计算准序列顺序描述符
        qso_dict = GetQuasiSequenceOrder(seq, maxlag=3)
        
        # 合并所有特征字典
        feature_dict = {}
        feature_dict.update(ctd_dict)
        feature_dict.update(auto_dict)
        feature_dict.update(qso_dict)
        
        # 转换为特征数组
        feature_array = np.array(list(feature_dict.values()))
        
        return feature_dict, feature_array
    except Exception as e:
        print(f"计算序列 {seq[:20]}... 的特征时出错: {str(e)}")
        return None, None

# 批量计算特征
print("开始计算特征...")
features = []
label_list = []
feature_info = []  # 存储特征信息
expected_len = None

for idx, row in total_df.iterrows():
    if idx % 1000 == 0:  # 每1000个序列打印一次进度
        print(f"处理序列 {idx+1}/{len(total_df)}: {row['sequence'][:20]}...")
    
    feature_dict, feature_array = extract_features(row['sequence'])
    
    if feature_dict is not None and feature_array is not None:
        # 第一个成功的序列时保存特征信息和预期长度
        if expected_len is None:
            feature_info = list(feature_dict.keys())
            expected_len = len(feature_info)
            print(f"特征模板设定完成，预期特征数量: {expected_len}")
        
        # 检查特征长度一致性
        if len(feature_array) == expected_len:
            features.append(feature_array)
            label_list.append(row['label'])
        else:
            print(f"跳过序列 {idx+1}: 特征长度不一致 ({len(feature_array)} vs {expected_len})")
    else:
        if idx < 10:  # 只为前10个失败的序列打印错误信息
            print(f"跳过序列 {idx+1}: 特征计算失败")

if len(features) == 0:
    raise ValueError("没有成功计算任何特征，请检查序列格式")

features = np.array(features)
label_list = np.array(label_list)
print(f"成功处理 {len(features)} 个序列，跳过 {len(total_df) - len(features)} 个序列")

print(f"特征计算完成！")
print(f"特征矩阵形状: {features.shape}")
print(f"特征类型统计:")

# 根据实际特征名称统计不同类型的特征数量
ctd_features = [f for f in feature_info if f.startswith('_')]  # CTD特征通常以下划线开头
auto_features = [f for f in feature_info if any(keyword in f for keyword in ['MoreauBrotoAuto', 'MoranAuto', 'GearyAuto'])]
qso_features = [f for f in feature_info if any(keyword in f for keyword in ['QSOSW', 'QSOgrant'])]
other_features = [f for f in feature_info if not any(keyword in f for keyword in ['_', 'MoreauBrotoAuto', 'MoranAuto', 'GearyAuto', 'QSOSW', 'QSOgrant'])]

print(f"  - CTD特征: {len(ctd_features)}个")
print(f"  - Autocorrelation特征: {len(auto_features)}个")
print(f"  - QuasiSequenceOrder特征: {len(qso_features)}个")
print(f"  - 其他特征: {len(other_features)}个")
print(f"  - 总特征数: {len(feature_info)}个")

# 验证特征数量一致性
assert len(feature_info) == features.shape[1], f"特征名数量({len(feature_info)})与特征矩阵列数({features.shape[1]})不一致"

# 保存特征信息
with open(os.path.join(output_dir, 'feature_info.txt'), 'w', encoding='utf-8') as f:
    f.write("特征信息:\n")
    f.write(f"特征矩阵形状: {features.shape}\n")
    f.write(f"样本数量: {len(label_list)}\n")
    f.write(f"总特征数: {len(feature_info)}\n\n")
    
    f.write("特征类型统计:\n")
    f.write(f"  - CTD特征: {len(ctd_features)}个\n")
    f.write(f"  - Autocorrelation特征: {len(auto_features)}个\n")
    f.write(f"  - QuasiSequenceOrder特征: {len(qso_features)}个\n")
    f.write(f"  - 其他特征: {len(other_features)}个\n\n")
    
    f.write("特征名称列表:\n")
    for i, feat_name in enumerate(feature_info):
        f.write(f"{i+1:4d}. {feat_name}\n")
    
    # 调试信息：显示前10个特征名称的类型判断
    f.write("\n调试信息 - 前10个特征名称类型判断:\n")
    for i, feat_name in enumerate(feature_info[:10]):
        feat_type = "未知"
        if feat_name.startswith('_'):
            feat_type = "CTD"
        elif any(keyword in feat_name for keyword in ['MoreauBrotoAuto', 'MoranAuto', 'GearyAuto']):
            feat_type = "Autocorrelation"
        elif any(keyword in feat_name for keyword in ['QSOSW', 'QSOgrant']):
            feat_type = "QuasiSequenceOrder"
        f.write(f"  {feat_name} -> {feat_type}\n")

# PCA降维分析
print("进行PCA降维分析...")
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features)

plt.figure(figsize=(12, 8))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 更好的颜色搭配
markers = ['o', 's', '^']

for i, lab in enumerate(labels):
    idx = label_list == lab
    plt.scatter(features_pca[idx, 0], features_pca[idx, 1], 
                label=f'{lab} (n={np.sum(idx)})', 
                alpha=0.7, s=50, c=colors[i], marker=markers[i])

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA Analysis of High-level Protein Descriptors')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pca_scatter.png'), dpi=300, bbox_inches='tight')
plt.close()

# 保存PCA分析结果
with open(os.path.join(output_dir, 'pca_analysis.txt'), 'w', encoding='utf-8') as f:
    f.write("PCA分析结果:\n")
    f.write(f"PC1解释方差比: {pca.explained_variance_ratio_[0]:.4f}\n")
    f.write(f"PC2解释方差比: {pca.explained_variance_ratio_[1]:.4f}\n")
    f.write(f"前两个主成分累计解释方差比: {sum(pca.explained_variance_ratio_):.4f}\n\n")
    
    f.write("各组样本在PCA空间中的统计信息:\n")
    for lab in labels:
        idx = label_list == lab
        if np.sum(idx) > 0:
            pc1_mean = np.mean(features_pca[idx, 0])
            pc1_std = np.std(features_pca[idx, 0])
            pc2_mean = np.mean(features_pca[idx, 1])
            pc2_std = np.std(features_pca[idx, 1])
            f.write(f"{lab}: PC1={pc1_mean:.3f}±{pc1_std:.3f}, PC2={pc2_mean:.3f}±{pc2_std:.3f}\n")

# t-SNE降维分析
print("进行t-SNE降维分析...")
try:
    # 根据样本数量调整perplexity
    n_samples = len(features)
    perplexity = min(30, max(5, n_samples // 4))
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
                max_iter=1000, learning_rate=200)
    features_tsne = tsne.fit_transform(features)
    
    plt.figure(figsize=(12, 8))
    for i, lab in enumerate(labels):
        idx = label_list == lab
        plt.scatter(features_tsne[idx, 0], features_tsne[idx, 1], 
                    label=f'{lab} (n={np.sum(idx)})', 
                    alpha=0.7, s=50, c=colors[i], marker=markers[i])
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title('t-SNE Analysis of High-level Protein Descriptors')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存t-SNE分析结果
    with open(os.path.join(output_dir, 'tsne_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write("t-SNE分析结果:\n")
        f.write("使用的perplexity: {perplexity}\n")
        f.write("迭代次数: 1000\n")
        f.write("学习率: 200\n\n")
        
        f.write("各组样本在t-SNE空间中的统计信息:\n")
        for lab in labels:
            idx = label_list == lab
            if np.sum(idx) > 0:
                tsne1_mean = np.mean(features_tsne[idx, 0])
                tsne1_std = np.std(features_tsne[idx, 0])
                tsne2_mean = np.mean(features_tsne[idx, 1])
                tsne2_std = np.std(features_tsne[idx, 1])
                f.write(f"{lab}: t-SNE1={tsne1_mean:.3f}±{tsne1_std:.3f}, t-SNE2={tsne2_mean:.3f}±{tsne2_std:.3f}\n")
    
    print("t-SNE分析完成")
    
except Exception as e:
    print(f't-SNE分析失败: {e}')
    with open(os.path.join(output_dir, 'tsne_analysis.txt'), 'w', encoding='utf-8') as f:
        f.write(f"t-SNE分析失败: {str(e)}\n")

# 量化分布距离分析
print("进行分布距离分析...")

# 分别获取三类数据的特征
X_gen = features[label_list == 'Generated']
X_grampa = features[label_list == 'GramPA']
X_neg = features[label_list == 'Negative']

print(f"Generated样本数: {len(X_gen)}")
print(f"GramPA样本数: {len(X_grampa)}")
print(f"Negative样本数: {len(X_neg)}")

# 计算距离函数
def compute_wasserstein_distance(X1, X2):
    """计算两个特征集合的Wasserstein距离"""
    distances = []
    for i in range(X1.shape[1]):
        wd = wasserstein_distance(X1[:, i], X2[:, i])
        distances.append(wd)
    return np.array(distances)

def compute_mmd(X, Y, kernel='rbf', gamma=1.0):
    """计算MMD距离"""
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# 计算Generated与GramPA的距离
if len(X_gen) > 0 and len(X_grampa) > 0:
    wd_gen_grampa = compute_wasserstein_distance(X_gen, X_grampa)
    wd_gen_grampa_mean = np.mean(wd_gen_grampa)
    wd_gen_grampa_std = np.std(wd_gen_grampa)
    
    mmd_gen_grampa = compute_mmd(X_gen, X_grampa, kernel='rbf')
    
    print(f"Generated vs GramPA - Wasserstein距离: {wd_gen_grampa_mean:.4f}±{wd_gen_grampa_std:.4f}")
    print(f"Generated vs GramPA - MMD距离: {mmd_gen_grampa:.4f}")
else:
    wd_gen_grampa_mean = wd_gen_grampa_std = mmd_gen_grampa = 0
    print("Generated或GramPA样本为空，无法计算距离")

# 计算Generated与Negative的距离
if len(X_gen) > 0 and len(X_neg) > 0:
    wd_gen_neg = compute_wasserstein_distance(X_gen, X_neg)
    wd_gen_neg_mean = np.mean(wd_gen_neg)
    wd_gen_neg_std = np.std(wd_gen_neg)
    
    mmd_gen_neg = compute_mmd(X_gen, X_neg, kernel='rbf')
    
    print(f"Generated vs Negative - Wasserstein距离: {wd_gen_neg_mean:.4f}±{wd_gen_neg_std:.4f}")
    print(f"Generated vs Negative - MMD距离: {mmd_gen_neg:.4f}")
else:
    wd_gen_neg_mean = wd_gen_neg_std = mmd_gen_neg = 0
    print("Generated或Negative样本为空，无法计算距离")

# 计算GramPA与Negative的距离
if len(X_grampa) > 0 and len(X_neg) > 0:
    wd_grampa_neg = compute_wasserstein_distance(X_grampa, X_neg)
    wd_grampa_neg_mean = np.mean(wd_grampa_neg)
    wd_grampa_neg_std = np.std(wd_grampa_neg)
    
    mmd_grampa_neg = compute_mmd(X_grampa, X_neg, kernel='rbf')
    
    print(f"GramPA vs Negative - Wasserstein距离: {wd_grampa_neg_mean:.4f}±{wd_grampa_neg_std:.4f}")
    print(f"GramPA vs Negative - MMD距离: {mmd_grampa_neg:.4f}")
else:
    wd_grampa_neg_mean = wd_grampa_neg_std = mmd_grampa_neg = 0
    print("GramPA或Negative样本为空，无法计算距离")

# 保存量化结果
with open(os.path.join(output_dir, 'distribution_distance.txt'), 'w', encoding='utf-8') as f:
    f.write("高阶特征分布距离分析结果\n")
    f.write("="*50 + "\n\n")
    
    f.write("样本统计:\n")
    f.write(f"Generated样本数: {len(X_gen)}\n")
    f.write(f"GramPA样本数: {len(X_grampa)}\n")
    f.write(f"Negative样本数: {len(X_neg)}\n\n")
    
    f.write("距离度量结果:\n")
    f.write(f"Generated vs GramPA:\n")
    f.write(f"  Wasserstein距离 (均值±标准差): {wd_gen_grampa_mean:.4f}±{wd_gen_grampa_std:.4f}\n")
    f.write(f"  MMD距离 (RBF核): {mmd_gen_grampa:.4f}\n\n")
    
    f.write(f"Generated vs Negative:\n")
    f.write(f"  Wasserstein距离 (均值±标准差): {wd_gen_neg_mean:.4f}±{wd_gen_neg_std:.4f}\n")
    f.write(f"  MMD距离 (RBF核): {mmd_gen_neg:.4f}\n\n")
    
    f.write(f"GramPA vs Negative:\n")
    f.write(f"  Wasserstein距离 (均值±标准差): {wd_grampa_neg_mean:.4f}±{wd_grampa_neg_std:.4f}\n")
    f.write(f"  MMD距离 (RBF核): {mmd_grampa_neg:.4f}\n\n")
    
    f.write("解释:\n")
    f.write("- Wasserstein距离: 衡量两个分布之间的差异，值越小表示分布越相似\n")
    f.write("- MMD距离: 最大平均差异，用于衡量两个分布的差异，值越小表示分布越相似\n")
    f.write("- 理想情况下，Generated与GramPA的距离应该较小，而与Negative的距离应该较大\n")

# 特征重要性分析
print("进行特征重要性分析...")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# 准备数据进行分类分析
if len(set(label_list)) > 1:
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, label_list, test_size=0.2, random_state=42, stratify=label_list
    )
    
    # 训练随机森林模型
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"随机森林分类准确率: {accuracy:.4f}")
    
    # 获取特征重要性
    importances = rf.feature_importances_
    
    # 找到最重要的特征，确保不超过特征总数
    n_top_features = min(20, len(feature_info))
    top_indices = np.argsort(importances)[::-1][:n_top_features]
    
    # 绘制特征重要性图
    plt.figure(figsize=(12, 8))
    top_features = [feature_info[i] for i in top_indices]
    top_importances = importances[top_indices]
    
    plt.barh(range(len(top_features)), top_importances)
    plt.yticks(range(len(top_features)), [f[:30] + '...' if len(f) > 30 else f for f in top_features])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n_top_features} Most Important Features')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存特征重要性结果
    with open(os.path.join(output_dir, 'feature_importance.txt'), 'w', encoding='utf-8') as f:
        f.write("特征重要性分析结果\n")
        f.write("="*50 + "\n\n")
        f.write(f"随机森林分类准确率: {accuracy:.4f}\n\n")
        
        f.write(f"前{n_top_features}个最重要的特征:\n")
        for i, (idx, importance) in enumerate(zip(top_indices, top_importances)):
            f.write(f"{i+1:2d}. {feature_info[idx]}: {importance:.6f}\n")
        
        f.write(f"\n分类报告:\n")
        f.write(classification_report(y_test, y_pred))
    
    print("特征重要性分析完成")
else:
    print("只有一类样本，无法进行特征重要性分析")

print('\n高阶特征分析完成！')
print('结果已保存在以下文件中：')
print(f'- PCA分析图: {os.path.join(output_dir, "pca_scatter.png")}')
print(f'- PCA分析结果: {os.path.join(output_dir, "pca_analysis.txt")}')
print(f'- t-SNE分析图: {os.path.join(output_dir, "tsne_scatter.png")}')
print(f'- t-SNE分析结果: {os.path.join(output_dir, "tsne_analysis.txt")}')
print(f'- 分布距离分析: {os.path.join(output_dir, "distribution_distance.txt")}')
print(f'- 特征信息: {os.path.join(output_dir, "feature_info.txt")}')
print(f'- 特征重要性图: {os.path.join(output_dir, "feature_importance.png")}')
print(f'- 特征重要性分析: {os.path.join(output_dir, "feature_importance.txt")}')
print(f'\n所有结果保存在目录: {output_dir}')
