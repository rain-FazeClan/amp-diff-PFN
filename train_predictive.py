import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from tabpfn import TabPFNClassifier


def load_data(data_path):
    """加载数据并处理特征和标签"""
    df = pd.read_csv(data_path)
    X = df.drop(['Sequence', 'label'], axis=1, errors='ignore').select_dtypes(include=np.number).fillna(method='ffill')
    y = df['label']
    return X, y


def save_plot(fig, path):
    """保存绘图"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(cm, results_dir):
    """绘制混淆矩阵"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg (0)', 'Pos (1)'], yticklabels=['Neg (0)', 'Pos (1)'], ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    save_plot(fig, os.path.join(results_dir, 'confusion_matrix.png'))


def plot_roc_curve(fpr, tpr, roc_auc, results_dir):
    """绘制ROC曲线"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    save_plot(fig, os.path.join(results_dir, 'roc_curve.png'))


def train_classifier(data_path, model_output_path, results_dir, test_sample_fraction=1.0):
    """
    训练分类器并评估。
    test_sample_fraction: 用于评估的测试集样本比例 (0.0 到 1.0)。
    """
    try:
        print("加载数据...")
        X, y = load_data(data_path) # 假设 load_data 函数能正确加载数据并返回 DataFrame/Series

        print(f"特征数量: {X.shape[1]}")

        # 确保数据是适合 TabPFN 的数值类型并处理缺失值
        # 添加处理缺失值的代码，这里使用均值填充作为示例
        X = X.select_dtypes(include=np.number)
        if X.isnull().sum().sum() > 0:
             print("检测到缺失值，正在使用均值填充...")
             # 计算训练集的均值来填充训练集和测试集的缺失值会更严谨
             # 这里简化处理，直接对整个 X 填充
             X = X.fillna(X.mean())
             print("缺失值填充完成。")


        X_train, X_test_orig, y_train, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 根据 test_sample_fraction 对测试集进行抽样
        if test_sample_fraction < 1.0 and test_sample_fraction > 0.0:
            print(f"对测试集进行抽样，比例为 {test_sample_fraction}...")
            n_test_samples = int(len(X_test_orig) * test_sample_fraction)
            # 确保抽样数量至少是1，除非原始测试集为空
            if n_test_samples == 0 and len(X_test_orig) > 0:
                n_test_samples = 1
                print("抽样比例过低，至少保留一个测试样本。")
            # 使用 sample 方法进行抽样
            test_indices = np.random.choice(X_test_orig.index, size=n_test_samples, replace=False)
            X_test = X_test_orig.loc[test_indices]
            y_test = y_test_orig.loc[test_indices]
            print(f"使用 {len(X_test)} 个样本进行评估。")
        else:
            X_test = X_test_orig
            y_test = y_test_orig
            print(f"使用 {len(X_test)} 个样本进行评估 (未抽样)。")


        print("初始化并训练分类器...")
        # 初始化 TabPFN 分类器
        # 保持 ignore_pretraining_limits=True
        # 移除 N_ensemble_configurations 参数
        # device='cpu' 可以保留，如果你确定不想使用 GPU，
        # 否则可以删除 device 参数让模型自动检测 GPU (如果可用)
        classifier = TabPFNClassifier(ignore_pretraining_limits=True)


        classifier.fit(X_train, y_train)

        print("评估模型...")

        # 添加调试打印，确认预测步骤是否开始
        print("开始预测...")
        # 核心：这一步是预测过程，内存消耗大
        y_pred = classifier.predict(X_test)
        print("预测完成。")

        # 添加调试打印，确认概率预测步骤是否开始
        print("开始预测概率...")
        # 确保 y 的类别是 0 和 1，并且在训练时模型学习到了这两个类别
        if len(classifier.classes_) == 2:
             # 核心：这一步是概率预测过程，内存消耗也很大
             y_proba = classifier.predict_proba(X_test)[:, 1]
             print("预测概率完成 (二分类)。")
        else:
             print("警告：分类器不是二分类，无法计算 ROC AUC。")
             y_proba = None # 或者根据实际情况处理
             print("预测概率跳过。")


        # 添加调试打印，确认评估指标计算步骤是否开始
        print("开始计算评估指标...")
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print("基本评估指标计算完成。")

        # 只有在 y_proba 存在且是二分类的情况下才计算 ROC AUC 和绘制曲线
        if y_proba is not None:
            print("开始计算 ROC AUC...")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            print(f"准确率: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
            print("开始绘制 ROC 曲线...")
            plot_roc_curve(fpr, tpr, roc_auc, results_dir) # 确保 plot_roc_curve 函数存在
            print("ROC 曲线绘制完成。")
        else:
             print(f"准确率: {accuracy:.4f}")


        print("开始绘制混淆矩阵...")
        plot_confusion_matrix(cm, results_dir) # 确保 plot_confusion_matrix 函数存在
        print("混淆矩阵绘制完成。")


        print("保存结果...")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)
        print("分类报告保存完成。")


        print("保存模型...")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(classifier, f)
        print("模型保存完成。")

        print("保存测试数据和标签...")
        # 保存原始的测试集数据和标签，而不是抽样后的
        X_test_orig.to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
        y_test_orig.to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
        print("测试数据和标签保存完成。")

        print("训练完成，结果已保存。")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        # 添加打印更详细的 traceback 信息
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    data_path = os.path.join('preprocessed_data/classify.csv')
    model_path = os.path.join('models/predictive_model.pkl')
    eval_results_dir = 'results/predictive_evaluation'

    train_classifier(data_path, model_path, eval_results_dir)