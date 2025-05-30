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


def train_classifier(data_path, model_output_path, results_dir):
    """训练分类器并评估"""
    try:
        print("加载数据...")
        X, y = load_data(data_path)

        print(f"特征数量: {X.shape[1]}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        print("初始化并训练分类器...")
        classifier = TabPFNClassifier(device='cpu')
        classifier.fit(X_train, y_train)

        print("评估模型...")
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        print(f"准确率: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
            f.write(report)

        plot_confusion_matrix(cm, results_dir)
        plot_roc_curve(fpr, tpr, roc_auc, results_dir)

        print("保存模型...")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(classifier, f)

        X_test.to_csv(os.path.join(results_dir, 'X_test.csv'), index=False)
        y_test.to_csv(os.path.join(results_dir, 'y_test.csv'), index=False)
        print("训练完成，结果已保存。")

    except Exception as e:
        print(f"训练过程中发生错误: {e}")


if __name__ == '__main__':
    data_path = os.path.join('classify_data/classify.csv')
    model_path = os.path.join('models/predictive_model.pkl')
    eval_results_dir = 'results/predictive_evaluation'

    train_classifier(data_path, model_path, eval_results_dir)