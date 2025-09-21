import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tabpfn import TabPFNRegressor


def load_regression_data(data_path):
    """加载回归数据并处理特征和目标值"""
    df = pd.read_csv(data_path)

    X = df.drop(['sequence', 'MIC'], axis=1, errors='ignore').select_dtypes(include=np.number)
    X = X.ffill()
    y = df['MIC']
    return X, y


def save_plot(fig, path):
    """保存绘图"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def plot_predicted_vs_actual(y_true, y_pred, results_dir):
    """绘制预测值vs实际值散点图，简化样式"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 绘制散点图
    scatter = ax.scatter(y_true, y_pred, alpha=0.7, c='red', edgecolors='black', s=30)
    
    # 添加对角线（黑色虚线）
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    
    # 设置坐标轴标签
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    
    # 添加 R² 值到图上
    r2 = r2_score(y_true, y_pred)
    ax.text(0.02, 0.98, f'R^2 = {r2:.4f}', transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    
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
        if X.isnull().sum().sum() > 0:
            print("检测到缺失值，正在使用均值填充...")
            X = X.fillna(X.mean())
            print("缺失值填充完成。")

        # 数据集划分为训练集和验证集（比例 80%:20%）
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        print("初始化并训练回归模型...")
        regressor = TabPFNRegressor(ignore_pretraining_limits=True)

        regressor.fit(X_train, y_train)

        print("评估回归模型...")
        # 验证集评估
        y_val_pred = regressor.predict(X_val)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(f"验证集 - 均方误差 (MSE): {val_mse:.4f}")
        print(f"验证集 - R² Score: {val_r2:.4f}")

        # 绘制预测值vs实际值图
        plot_predicted_vs_actual(y_val, y_val_pred, results_dir)

        # 保存结果
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, 'regression_metrics.txt'), 'w') as f:
            f.write(f"Validation MSE: {val_mse:.4f}\n")
            f.write(f"Validation R2: {val_r2:.4f}\n")

        print("保存回归模型...")
        os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
        with open(model_output_path, 'wb') as f:
            pickle.dump(regressor, f)
        print("回归模型保存完成。")

        print("保存测试数据和预测结果...")
        test_results = pd.DataFrame({
            'actual': y_val,
            'predicted': y_val_pred
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

    data_path = os.path.join('preprocessed_data', 'regression.csv')
    model_path = os.path.join('models', 'regression_model.pkl')
    eval_results_dir = 'results/regression_evaluation'

    train_regressor(data_path, model_path, eval_results_dir)