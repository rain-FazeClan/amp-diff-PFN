import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# 设置随机种子，确保结果可重现
np.random.seed(42)


def prepare_dataset(test_size=0.2):
    """
    读取阳性和阴性数据，合并，随机打乱，并划分训练集和测试集

    Parameters:
    test_size : float, default=0.2
        测试集的比例，默认为20%
    """

    try:
        # 读取阳性和阴性数据
        print("正在读取数据文件...")
        positive_df = pd.read_csv('positive_new.csv')
        negative_df = pd.read_csv('negative_new.csv')

        # 添加标签列（label）：阳性为1，阴性为0
        print("添加标签...")
        positive_df['label'] = 1
        negative_df['label'] = 0

        # 合并数据集
        print("合并数据集...")
        combined_df = pd.concat([positive_df, negative_df], ignore_index=True)

        # 随机打乱数据集
        print("随机打乱数据集...")
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # 划分训练集和测试集
        print(f"划分训练集（{(1 - test_size) * 100}%）和测试集（{test_size * 100}%）...")
        train_data, test_data = train_test_split(
            combined_df,
            test_size=test_size,
            random_state=42,
            stratify=combined_df['label']  # 确保训练集和测试集中的正负样本比例一致
        )

        # 重置索引
        train_data = train_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

        # 保存为CSV文件
        print("保存数据集...")
        train_data.to_csv('train_data.csv', index=False)
        test_data.to_csv('test_data.csv', index=False)

        # 打印数据集信息
        print("\n数据集统计信息：")
        print(f"总样本数: {len(combined_df)}")
        print(f"训练集样本数: {len(train_data)} ({len(train_data) / len(combined_df) * 100:.1f}%)")
        print(f"测试集样本数: {len(test_data)} ({len(test_data) / len(combined_df) * 100:.1f}%)")
        print("\n训练集中:")
        print(f"阳性样本数: {len(train_data[train_data['label'] == 1])}")
        print(f"阴性样本数: {len(train_data[train_data['label'] == 0])}")
        print("\n测试集中:")
        print(f"阳性样本数: {len(test_data[test_data['label'] == 1])}")
        print(f"阴性样本数: {len(test_data[test_data['label'] == 0])}")

        return train_data, test_data

    except FileNotFoundError as e:
        print("错误：找不到数据文件，请确保文件存在于当前目录")
        raise e
    except Exception as e:
        print(f"发生错误：{str(e)}")
        raise e


if __name__ == "__main__":
    # 执行数据集准备，测试集占20%
    try:
        train_data, test_data = prepare_dataset(test_size=0.2)
        print("\n数据集准备完成！")
        print("训练集已保存为：train_data.csv")
        print("测试集已保存为：test_data.csv")
    except Exception as e:
        print(f"程序执行失败：{str(e)}")