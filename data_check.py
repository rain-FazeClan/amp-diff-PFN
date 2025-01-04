import pandas as pd
import numpy as np
import argparse
from datetime import datetime


def analyze_column(df, column_name):
    """详细分析单个列的数据"""
    print(f"\n=== 分析列: {column_name} ===")
    print(f"数据类型: {df[column_name].dtype}")
    print(f"是否包含空值: {df[column_name].isnull().any()}")
    print(f"空值数量: {df[column_name].isnull().sum()}")

    # 对于非Sequence列，检查是否可以转换为浮点数
    if column_name != 'Sequence':
        try:
            df[column_name].astype(float)
            print("✅ 可以成功转换为浮点数")
        except Exception as e:
            print("❌ 无法转换为浮点数")
            # 找出问题行
            problematic_rows = []
            for idx, value in df[column_name].items():
                try:
                    float(value)
                except:
                    problematic_rows.append((idx, value))
            if problematic_rows:
                print("\n问题数据:")
                for idx, value in problematic_rows[:5]:  # 只显示前5个问题
                    print(f"行号 {idx}: 值 = {value}")


def check_dataset(file_path, dataset_name=""):
    """检查数据集文件"""
    print(f"\n{'=' * 20} 检查{dataset_name}数据集 {'=' * 20}")
    print(f"文件路径: {file_path}")

    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        print(f"\n基本信息:")
        print(f"行数: {len(df)}")
        print(f"列数: {len(df.columns)}")
        print(f"列名: {df.columns.tolist()}")

        # 检查必需的列是否存在
        required_columns = ['Sequence', 'MIC']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"\n❌ 缺少必需的列: {missing_columns}")
            return

        # 数据类型概览
        print("\n数据类型概览:")
        print(df.dtypes)

        # 检查每一列
        for column in df.columns:
            analyze_column(df, column)

        # 特别检查特征列（除Sequence和label列外的所有列）
        feature_columns = [col for col in df.columns if col not in ['Sequence', 'label']]
        print("\n=== 特征列检查 ===")
        feature_df = df[feature_columns]

        # 检查是否所有特征都可以转换为浮点数
        conversion_errors = []
        for column in feature_columns:
            try:
                feature_df[column].astype(float)
            except Exception as e:
                conversion_errors.append((column, str(e)))

        if conversion_errors:
            print("\n❌ 以下列无法转换为浮点数:")
            for column, error in conversion_errors:
                print(f"列 '{column}': {error}")
        else:
            print("\n✅ 所有特征列都可以成功转换为浮点数")

        # 检查数值范围
        print("\n数值范围检查:")
        for column in feature_columns:
            if df[column].dtype in [np.float64, np.float32, np.int64, np.int32]:
                print(f"{column}:")
                print(f"  最小值: {df[column].min()}")
                print(f"  最大值: {df[column].max()}")
                print(f"  平均值: {df[column].mean():.2f}")

        # 检查MIC列
        print("\n=== MIC值检查 ===")
        try:
            mic_values = df['MIC'].astype(float)
            print("✅ MIC列可以成功转换为浮点数")
            print(f"MIC范围: {mic_values.min()} - {mic_values.max()}")
            print(f"MIC平均值: {mic_values.mean():.2f}")
        except Exception as e:
            print(f"❌ MIC列转换错误: {str(e)}")
            # 找出问题行
            problematic_mics = []
            for idx, value in df['MIC'].items():
                try:
                    float(value)
                except:
                    problematic_mics.append((idx, value))
            if problematic_mics:
                print("\nMIC问题数据:")
                for idx, value in problematic_mics[:5]:
                    print(f"行号 {idx}: 值 = {value}")

    except Exception as e:
        print(f"❌ 读取文件时发生错误: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='检查抗菌肽数据集的数据类型')
    parser.add_argument('--train_data', type=str, required=True, help='训练数据CSV文件路径')
    parser.add_argument('--test_data', type=str, required=True, help='测试数据CSV文件路径')

    args = parser.parse_args()

    print(f"数据检查开始时间: {datetime.now()}")

    # 检查训练集
    check_dataset(args.train_data, "训练")

    # 检查测试集
    check_dataset(args.test_data, "测试")

    print(f"\n数据检查结束时间: {datetime.now()}")


if __name__ == '__main__':
    main()