# train_predictive_pt.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils import class_weight
import numpy as np
import os
from data_loader import get_predictive_dataloader, load_preprocessed_data # Import the PyTorch DataLoader helper
from model import PeptidePredictiveModel
from utils import NUM_AMINO_ACIDS, PADDING_VALUE # Get actual vocab size and padding value


def train_predictive(data_filepath='data/preprocessed_data.npz', model_save_path='models/weights/predictive_model_pt.pth', epochs=20, batch_size=64):
    """
    训练抗菌肽预测模型（PyTorch）。

    Args:
        data_filepath (str): 预处理数据文件路径。
        model_save_path (str): 模型权重保存路径。
        epochs (int): 训练轮数。
        batch_size (int): 训练批次大小。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, test_loader = get_predictive_dataloader(data_filepath, batch_size)

    if train_loader is None:
        print("加载预测模型训练数据失败，退出训练。")
        return

    # Get parameters from preprocessed data for model initialization
    _, _, y_train_np, _, max_sequence_length, num_amino_acids_one_hot = load_preprocessed_data(data_filepath)
    vocab_size = NUM_AMINO_ACIDS + 1 # Total tokens including padding

    # Build model
    # Example model parameters - adjust as needed
    embedding_dim = 128
    hidden_dim_lstm = 128
    num_filters = 128
    kernel_size = 5
    dropout_rate = 0.3 # Use the default in model definition

    model = PeptidePredictiveModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim_lstm=hidden_dim_lstm,
        num_filters=num_filters,
        kernel_size=kernel_size,
        max_sequence_length=max_sequence_length,
        dropout_rate=dropout_rate
    )
    model.to(device)

    # Define Loss and Optimizer
    # Calculate class weights for imbalanced dataset
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_np),
        y=y_train_np
    )
    # Convert to PyTorch tensor, move to device
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    # BCELoss needs weight per sample, or use weighted BCELoss if available per class
    # For nn.BCELoss, 'weight' argument weights the loss of each element in the output.
    # A better approach for per-class weighting with BCELoss is to compute it manually or use BCELoss with logits and a weight tensor
    # Let's use nn.BCEWithLogitsLoss for better numerical stability and use pos_weight for positive class weight
    # Note: BCEWithLogitsLoss combines Sigmoid and BCELoss. The model's last layer should *not* have Sigmoid if using this.
    # Let's modify the model's forward pass to output logits only, or adjust loss computation.
    # Option A: Remove sigmoid from model, use BCEWithLogitsLoss (Recommended)
    # Option B: Keep sigmoid, manually apply weight to loss per class

    # Option A (Recommended): Modify models/predictive_model_pt.py to remove sigmoid from forward()
    # Let's assume we modified it to output logits only.
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([class_weights_tensor[1]], device=device)) # Weight for the positive class

    # Option B (if you keep sigmoid): Use BCELoss and potentially manual weighting
    # criterion = nn.BCELoss()
    # In train loop, calculate weighted loss: loss = criterion(outputs, labels) * sample_weights (if you compute sample weights)
    # Or maybe just rely on pos_weight with BCEWithLogitsLoss. Let's use BCEWithLogitsLoss.

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print("开始训练预测模型（PyTorch）...")

    best_accuracy = 0.0
    for epoch in range(epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs) # outputs are logits if using BCEWithLogitsLoss

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy (optional during training)
            preds = torch.sigmoid(outputs) > 0.5 # If model outputs logits, apply sigmoid here for prediction
            correct_predictions += (preds.squeeze() == labels.squeeze()).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions

        # Evaluate on test set
        model.eval() # Set model to evaluation mode
        test_running_loss = 0.0
        test_correct_predictions = 0
        test_total_predictions = 0
        test_all_labels = []
        test_all_preds_prob = []

        with torch.no_grad(): # No gradients needed for evaluation
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs) # outputs are logits

                test_loss = criterion(outputs, labels)
                test_running_loss += test_loss.item()

                preds_prob = torch.sigmoid(outputs) # Get probabilities for metrics
                preds = preds_prob > 0.5 # Get binary predictions

                test_correct_predictions += (preds.squeeze() == labels.squeeze()).sum().item()
                test_total_predictions += labels.size(0)

                test_all_labels.extend(labels.cpu().numpy())
                test_all_preds_prob.extend(preds_prob.cpu().numpy())


        test_loss_avg = test_running_loss / len(test_loader)
        test_accuracy = test_correct_predictions / test_total_predictions

        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.4f}, Test Loss: {test_loss_avg:.4f}, Test Acc: {test_accuracy:.4f}')

        # Save best model based on test accuracy
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"保存最佳模型权重到 {model_save_path} (Acc: {test_accuracy:.4f})")


    print("预测模型训练完成。")

    # Optionally, load the best model and evaluate again with more detailed metrics
    print(f"加载最佳模型权重 {model_save_path} 进行最终评估...")
    model.load_state_dict(torch.load(model_save_path))
    # Re-evaluate on test set using the loaded model state
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    model.eval()
    final_test_labels = []
    final_test_preds_prob = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs) # Logits
            preds_prob = torch.sigmoid(outputs)
            final_test_labels.extend(labels.cpu().numpy())
            final_test_preds_prob.extend(preds_prob.cpu().numpy())

    final_test_labels = np.array(final_test_labels)
    final_test_preds_prob = np.array(final_test_preds_prob)
    final_test_preds = (final_test_preds_prob > 0.5).astype(int)

    print("\n最终测试集评估报告:")
    print(classification_report(final_test_labels, final_test_preds, target_names=['Negative', 'Positive']))
    print("\n混淆矩阵:")
    print(confusion_matrix(final_test_labels, final_test_preds))

    # ROC Curve (optional, needs matplotlib)
    # import matplotlib.pyplot as plt
    # fpr, tpr, thresholds = roc_curve(final_test_labels, final_test_preds_prob)
    # roc_auc = auc(fpr, tpr)
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show() # This might pause execution, consider saving figure instead


if __name__ == '__main__':
    os.makedirs('models/weights', exist_ok=True)
    train_predictive()