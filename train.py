import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from datetime import datetime
import os
from model import AntibacterialPeptidePredictor  # 直接导入模型类


class AntibacterialPeptideDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # 将bacteria转换为数值索引
        self.bacteria_map = {
            'MRSA': 0, 'E. coli': 1, 'A. baumannii': 2,
            'P. aeruginosa': 3, 'K. pneumoniae': 4
        }
        # 获取特征列（去掉前3列和最后1列）
        self.feature_columns = self.data.columns[3:-1].tolist()
        # 确保特征列为数值类型
        self.data[self.feature_columns] = self.data[self.feature_columns].apply(pd.to_numeric, errors='coerce')
        # 处理缺失值
        self.data = self.data.dropna(subset=self.feature_columns)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        features = torch.FloatTensor(row[self.feature_columns].values.astype(np.float32))
        bacteria_idx = self.bacteria_map[row['bacteria']]
        label = torch.FloatTensor([1.0 if row['mic'] <= 8 else 0.0])
        return features, bacteria_idx, label


def create_model(input_dim=40, hidden_dims=[128, 64], num_experts_per_task=3, num_shared_experts=2):
    """创建抗菌肽预测模型实例"""
    model = AntibacterialPeptidePredictor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_experts_per_task=num_experts_per_task,
        num_shared_experts=num_shared_experts
    )
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    batch_count = 0

    for batch_idx, (features, bacteria_idx, labels) in enumerate(train_loader):
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)

        # 计算特定菌群的损失
        loss = 0
        for i in range(5):
            mask = (bacteria_idx == i)
            if mask.any():
                loss += criterion(outputs[i][mask], labels[mask])

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        batch_count += 1

        if (batch_idx + 1) % 100 == 0:
            print(f'Batch {batch_idx + 1}, Average Loss: {total_loss / batch_count:.4f}')

    return total_loss / batch_count


def evaluate(model, data_loader, device):
    model.eval()
    all_predictions = {i: [] for i in range(5)}
    all_labels = {i: [] for i in range(5)}

    with torch.no_grad():
        for features, bacteria_idx, labels in data_loader:
            features = features.to(device)
            outputs = model(features)

            # 收集每种菌群的预测结果
            for i in range(5):
                mask = (bacteria_idx == i)
                if mask.any():
                    pred = (outputs[i][mask] >= 0.5).float().cpu().numpy()
                    all_predictions[i].extend(pred)
                    all_labels[i].extend(labels[mask].numpy())

    # 计算每种菌群的评估指标
    metrics = {}
    bacteria_names = ['MRSA', 'E. coli', 'A. baumannii', 'P. aeruginosa', 'K. pneumoniae']

    for i, name in enumerate(bacteria_names):
        if len(all_predictions[i]) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels[i], all_predictions[i], average='binary'
            )
            accuracy = accuracy_score(all_labels[i], all_predictions[i])
            try:
                auc = roc_auc_score(all_labels[i], all_predictions[i])
            except:
                auc = 0.0

            metrics[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train antibacterial peptide prediction model')
    parser.add_argument('--train_path', required=True, help='Path to training data CSV')
    parser.add_argument('--test_path', required=True, help='Path to test data CSV')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 64],
                        help='Hidden dimensions for the model')
    parser.add_argument('--num_experts_per_task', type=int, default=3,
                        help='Number of experts per task')
    parser.add_argument('--num_shared_experts', type=int, default=2,
                        help='Number of shared experts')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    args = parser.parse_args()

    # 创建保存模型的目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'models_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    # 保存训练配置
    config = {
        'hidden_dims': args.hidden_dims,
        'num_experts_per_task': args.num_experts_per_task,
        'num_shared_experts': args.num_shared_experts,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs
    }

    # 加载数据
    train_dataset = AntibacterialPeptideDataset(args.train_path)
    test_dataset = AntibacterialPeptideDataset(args.test_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 创建模型
    model = create_model(
        input_dim=40,
        hidden_dims=args.hidden_dims,
        num_experts_per_task=args.num_experts_per_task,
        num_shared_experts=args.num_shared_experts
    )
    model = model.to(args.device)

    # 打印模型配置
    print("\nModel Configuration:")
    print(f"Hidden Dimensions: {args.hidden_dims}")
    print(f"Experts per Task: {args.num_experts_per_task}")
    print(f"Shared Experts: {args.num_shared_experts}")
    print(f"Device: {args.device}")

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # 记录最佳性能
    best_metrics = {'avg_f1': 0.0}

    # 训练循环
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, args.device)
        print(f'Training Loss: {train_loss:.4f}')

        # 评估
        test_metrics = evaluate(model, test_loader, args.device)

        # 计算平均 F1 分数
        avg_f1 = np.mean([metrics['f1'] for metrics in test_metrics.values()])

        # 打印评估结果
        print('\nTest Results:')
        for bacteria, metrics in test_metrics.items():
            print(f'\n{bacteria}:')
            for metric_name, value in metrics.items():
                print(f'{metric_name}: {value:.4f}')

        # 更新最佳模型
        if avg_f1 > best_metrics['avg_f1']:
            best_metrics = {
                'epoch': epoch + 1,
                'avg_f1': avg_f1,
                'test_metrics': test_metrics
            }
            # 保存最佳模型和配置
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
                'config': config
            }, f'{save_dir}/best_model.pth')

    # 打印最佳性能
    print('\nBest Performance:')
    print(f'Epoch: {best_metrics["epoch"]}')
    print(f'Average F1: {best_metrics["avg_f1"]:.4f}')
    for bacteria, metrics in best_metrics['test_metrics'].items():
        print(f'\n{bacteria}:')
        for metric_name, value in metrics.items():
            print(f'{metric_name}: {value:.4f}')


if __name__ == '__main__':
    main()