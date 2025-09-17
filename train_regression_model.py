import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from tabpfn import TabPFNRegressor


def load_regression_data(data_path):
    """加载回归数据并处理特征和目标值"""
    df = pd.read_csv(data_path)
    # 假设数据中包含一个连续值目标列，如'activity'或'rank'
    X = df.drop(['Sequence', 'activity', 'rank'], axis=1, errors='ignore').select_dtypes(include=np.number).fillna(method='ffill')
    # 选择合适的连续目标变量列，这里以'activity'为例
    y = df['activity'] if 'activity' in df.columns else df['rank']
    return X, y


def save_plot(fig, path):
    """保存绘图"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_predicted_vs_actual(y_test, y_pred, results_dir):
    """绘制预测值vs实际值散点图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, y_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Predicted vs Actual Values')
    save_plot(fig, os.path.join(results_dir, 'predicted_vs_actual.png'))


def train_regressor(data_path, model_output_path, results_dir):
    """
    训练回归模型并评估
    """
    try:
        print("加载回归数据...")
        X, y = load_regression_data(data_path)

        print(f"特征数量: {X.shape[1]}")

        # 处理缺失值
        X = X.select_dtypes(include=np.number)
        if X.isnull().sum().sum() > 0:
            print("检测到缺失值，正在使用均值填充...")
            X = X.fillna(X.mean())
            print("缺失值填充完成。")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("初始化并训练回归模型...")
        # 使用TabPFNRegressor作为主要回归模型
        regressor = TabPFNRegressor(device='cpu', N_ensemble_configurations=3)

        regressor.fit(X_train, y_train)

        print("评估回归模型...")
        y_pred = regressor.predict(X_test)

        # 计算评估指标
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"R² Score: {r2:.4f}")

        # 绘制预测值vs实际值图
        plot_predicted_vs_actual(y_test, y_pred, results_dir)

        # 保存结果
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'regression_metrics.txt'), 'w') as f:
            f.write(f"Mean Squared Error: {mse:.4f}\n")
            f.write(f"R2 Score: {r2:.4f}\n")

        print("保存回归模型...")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(regressor, f)
        print("回归模型保存完成。")

        print("保存测试数据和预测结果...")
        test_results = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred
        })
        test_results.to_csv(os.path.join(results_dir, 'regression_test_results.csv'), index=False)
        print("测试数据和预测结果保存完成。")

        print("回归模型训练完成，结果已保存。")
        
        return regressor

    except Exception as e:
        print(f"回归模型训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 需要准备包含连续目标值的回归数据集
    data_path = os.path.join('regression_data/regression.csv')
    model_path = os.path.join('models/regression_model.pkl')
    eval_results_dir = 'results/regression_evaluation'

    train_regressor(data_path, model_path, eval_results_dir)