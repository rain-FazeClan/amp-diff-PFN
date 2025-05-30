import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from featured_generated import calculate_all_descriptors


MAX_FEATURE_LEN = 20 # Ensure this matches featured_data_generated.py


def calculate_features_for_sequences(sequences):
    """
    使用 featured_generated.py 中的 calculate_all_descriptors 方法
    为一组序列计算特征并返回 DataFrame。
    """
    all_descriptors = []  # 用于存储所有序列的特征
    count = 1  # 计数器

    for sequence in sequences:
        if len(sequence) < 6:  # 跳过长度小于6的序列
            print(f"跳过长度小于6的序列: {sequence}")
            continue
        if sequence and isinstance(sequence, str):
            try:
                # 调用 calculate_all_descriptors 计算特征
                descriptors = calculate_all_descriptors(sequence.upper(), count)
                descriptors['sequence'] = sequence  # 添加序列信息
                all_descriptors.append(descriptors)
                count += 1
            except Exception as e:
                print(f"计算序列特征时出错: {sequence[:10]}...: {e}")
        else:
            print(f"跳过无效序列: {sequence}")

    # 将所有特征合并为 DataFrame
    if all_descriptors:
        feature_df = pd.DataFrame(all_descriptors)

        # 处理可能的 NaN 值
        numeric_cols = feature_df.select_dtypes(include=np.number).columns.tolist()
        for col in numeric_cols:
            if feature_df[col].isnull().any():
                feature_df[col] = feature_df[col].fillna(0)  # 用0填充NaN值

        return feature_df
    else:
        print("未找到有效的序列特征。")
        return pd.DataFrame()  # 返回空的 DataFrame


def evaluate_classifier_vespa(model_path, vespa_data_path, results_dir):
    """
    加载训练好的分类器，加载Vespa数据，计算特征，
    评估性能并保存结果。
    假设Vespa数据仅包含已知的AMP（label = 1）。
    """
    try:
        # 加载模型
        print(f"从 {model_path} 加载分类器模型")
        with open(model_path, 'rb') as f:
            classifier = pickle.load(f)
        print("模型加载成功。")

        # 加载Vespa数据
        print(f"从 {vespa_data_path} 加载Vespa数据")
        vespa_df = pd.read_csv(vespa_data_path, sep=';')  # 指定分隔符为分号
        vespa_sequences = vespa_df['sequence'].tolist()  # 获取`sequence`列

        if not vespa_sequences:
            print("Vespa数据文件中未找到序列。")
            return

        # 计算Vespa序列的特征
        print(f"为 {len(vespa_sequences)} 条Vespa序列计算特征...")
        X_vespa = calculate_features_for_sequences(vespa_sequences)

        if X_vespa.empty:
            print("无法为Vespa序列计算有效特征。跳过评估。")
            return

        # 创建真实标签（所有为1，表示AMP）
        y_vespa_true = pd.Series([1] * len(X_vespa), name='label')

        # 评估
        print("在Vespa数据上评估分类器...")
        y_vespa_pred = classifier.predict(X_vespa)

        # 计算指标
        accuracy = accuracy_score(y_vespa_true, y_vespa_pred)
        report = classification_report(y_vespa_true, y_vespa_pred, zero_division=0)
        cm = confusion_matrix(y_vespa_true, y_vespa_pred)

        print(f"Vespa数据的准确率: {accuracy:.4f}")
        print("Vespa数据的分类报告:")
        print(report)
        print("Vespa数据的混淆矩阵:\n", cm)

        # 保存评估结果
        os.makedirs(results_dir, exist_ok=True)
        report_path = os.path.join(results_dir, 'classification_report_vespa.txt')
        cm_path = os.path.join(results_dir, 'confusion_matrix_vespa.png')

        with open(report_path, 'w') as f:
            f.write("Vespa数据评估结果（假设全为AMP）:\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write("分类报告:\n")
            f.write(report)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Neg (0)', 'Pos (1)'], yticklabels=['Neg (0)', 'Pos (1)'])
        plt.ylabel('实际值 (Vespa = 1)')
        plt.xlabel('预测值')
        plt.title('混淆矩阵 (Vespa数据)')
        plt.savefig(cm_path)

        print(f"Vespa评估结果已保存到 {results_dir}")

    except FileNotFoundError as e:
        print(f"文件加载错误: {e}。请确保模型和数据文件存在。")
    except Exception as e:
        print(f"分类器评估过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    model_path = os.path.join('models/predictive_model.pkl') # Path to your trained model
    vespa_data_path = os.path.join('origin_data/Vespa.csv') # Path to the Vespa data file
    eval_results_dir = 'results/predictive_evaluation_vespa'

    evaluate_classifier_vespa(model_path, vespa_data_path, eval_results_dir)