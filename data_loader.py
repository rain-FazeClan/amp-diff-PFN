# src/data_loader_pt.py
import os
import numpy as np
from Bio import SeqIO # Requires biopython
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
from utils import sequence_to_int_sequence, pad_sequences, PADDING_VALUE, sequences_to_one_hot, NUM_AMINO_ACIDS # Import the new utilities

def load_sequences_from_fasta(filepath):
    """从FASTA文件加载序列"""
    sequences = []
    if os.path.exists(filepath):
        print(f"Loading sequences from {filepath}...")
        for record in SeqIO.parse(filepath, "fasta"):
            sequences.append(str(record.seq).upper().strip()) # Ensure upper case and no leading/trailing spaces
        print(f"Loaded {len(sequences)} sequences.")
    else:
        print(f"Warning: File not found {filepath}")
    return sequences

def load_sequences_from_csv(filepath, sequence_column_name):
    """从CSV文件加载序列"""
    sequences = []
    if os.path.exists(filepath):
        print(f"Loading sequences from {filepath}...")
        df = pd.read_csv(filepath)
        if sequence_column_name in df.columns:
            sequences = [str(seq).upper().strip() for seq in df[sequence_column_name].tolist() if pd.notna(seq)] # Ensure upper case, strip, and handle NaNs
            print(f"Loaded {len(sequences)} sequences.")
        else:
             print(f"Error: Column '{sequence_column_name}' not found in {filepath}")
    else:
         print(f"Warning: File not found {filepath}")
    return sequences


def preprocess_data(grampa_filepath, negative_filepath, output_filepath, max_sequence_length=None):
    """
    加载、预处理和保存数据为 .npz 文件。

    Args:
        grampa_filepath (str): 正样本文件路径 (FASTA 或 CSV)。
        negative_filepath (str): 负样本文件路径 (FASTA 或 CSV)。
        output_filepath (str): 保存预处理数据的 .npz 文件路径。
        max_sequence_length (int, optional): 序列填充的最大长度。如果为None，
                                             则使用所有序列中的最大长度。
    """
    print("加载原始序列数据...")
    # Adjust based on your file format (FASTA recommended for sequence databases)
    grampa_sequences = load_sequences_from_fasta(grampa_filepath)
    negative_sequences = load_sequences_from_fasta(negative_filepath)

    if not grampa_sequences:
        print(f"错误：未能加载正样本序列数据从 {grampa_filepath}，请检查文件路径和格式。")
        return
    if not negative_sequences:
         print(f"警告：未能加载负样本序列数据从 {negative_filepath}，请检查文件路径和格式。如果负样本缺失，后续预测模型训练会受到影响。")
         # Optionally exit or handle lack of negative data if predictor is crucial

    print(f"加载完成: 正样本 {len(grampa_sequences)} 条, 负样本 {len(negative_sequences)} 条")

    # 将序列转换为整数序列
    print("将序列转换为整数...")
    grampa_int_sequences = [sequence_to_int_sequence(seq) for seq in grampa_sequences]
    negative_int_sequences = [sequence_to_int_sequence(seq) for seq in negative_sequences]

    # 确定最大序列长度
    all_lengths = [len(seq) for seq in grampa_int_sequences] + [len(seq) for seq in negative_int_sequences]
    if not all_lengths:
        print("错误：没有有效的序列长度信息。退出预处理。")
        return

    actual_max_length = max(all_lengths) if all_lengths else 0

    if max_sequence_length is None or max_sequence_length < actual_max_length:
        final_max_length = actual_max_length
        if max_sequence_length is not None and max_sequence_length < actual_max_length:
            print(f"警告: 设定的最大序列长度 {max_sequence_length} 小于实际数据最大长度 {actual_max_length}。将使用实际最大长度进行填充/截断。")
    else:
        final_max_length = max_sequence_length

    print(f"最终确定的序列处理长度为: {final_max_length}")

    # 填充序列 (NumPy array of integers)
    print(f"填充/截断序列到长度: {final_max_length}...")
    grampa_padded = pad_sequences(grampa_int_sequences, max_length=final_max_length, padding='post', value=PADDING_VALUE)
    negative_padded = pad_sequences(negative_int_sequences, max_length=final_max_length, padding='post', value=PADDING_VALUE)

    # 创建标签
    grampa_labels = np.ones(len(grampa_padded), dtype=np.float32)
    negative_labels = np.zeros(len(negative_padded), dtype=np.float32)

    # 合并数据和标签
    all_data = np.concatenate((grampa_padded, negative_padded), axis=0)
    all_labels = np.concatenate((grampa_labels, negative_labels), axis=0)

    # 打乱数据
    all_data, all_labels = shuffle(all_data, all_labels, random_state=42)

    # 划分训练集和测试集
    # 注意处理类别不平衡： Stratify by labels
    # Need at least 2 samples per class in each split for stratification
    min_samples_per_class = min(np.sum(all_labels == 1), np.sum(all_labels == 0))
    if min_samples_per_class < 2:
         print(f"错误：类别样本数量过少，无法进行分层抽样。正样本数: {np.sum(all_labels == 1)}, 负样本数: {np.sum(all_labels == 0)}")
         # Fallback to non-stratified split or exit
         X_train, X_test, y_train, y_test = train_test_split(all_data, all_labels, test_size=0.2, random_state=42)
         print("警告：已回退到非分层抽样。")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            all_data, all_labels, test_size=0.2, random_state=42, stratify=all_labels
        )

    # 保存预处理后的数据
    print(f"保存预处理数据到 {output_filepath}...")
    # Save with metadata if needed, e.g., actual_max_length
    np.savez(output_filepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, max_sequence_length=final_max_length, num_amino_acids=NUM_AMINO_ACIDS + 1) # Save vocab size
    print("数据预处理完成。")
    return final_max_length # Return the length used

def load_preprocessed_data(filepath):
    """加载预处理后的数据 (.npz)"""
    if not os.path.exists(filepath):
        print(f"错误：预处理数据文件 {filepath} 不存在。请先运行数据预处理。")
        return None, None, None, None, None, None # Return data arrays, max_len, vocab_size

    data = np.load(filepath)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    max_sequence_length = data['max_sequence_length'].item() if 'max_sequence_length' in data else X_train.shape[1]
    num_amino_acids_with_padding = data['num_amino_acids'].item() if 'num_amino_acids' in data else len(np.unique(X_train)) # Guess vocab size if not saved
    num_amino_acids_one_hot = num_amino_acids_with_padding - 1 # Excluding padding

    print(f"加载预处理数据成功: X_train={X_train.shape}, X_test={X_test.shape}, Max_len={max_sequence_length}, Vocab size (incl padding)={num_amino_acids_with_padding}")

    return X_train, X_test, y_train, y_test, max_sequence_length, num_amino_acids_one_hot


# --- PyTorch Dataset Classes ---

class PeptideDataset(Dataset):
    """PyTorch Dataset for peptide sequences and labels."""
    def __init__(self, sequences, labels):
        # Convert NumPy arrays to PyTorch tensors
        # Sequences are integer encoded and padded
        self.sequences = torch.from_numpy(sequences).long() # Use LongTensor for indices
        # Labels are floats for BCELoss
        self.labels = torch.from_numpy(labels).float().unsqueeze(1) # Add dimension for BCELoss input shape

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class GANPeptideDataset(Dataset):
    """
    PyTorch Dataset for GAN training, returning one-hot encoded real sequences.
    Only uses positive samples.
    """
    def __init__(self, positive_sequences_int, max_sequence_length):
        # Convert integer sequences to one-hot encoding upfront
        # sequences_to_one_hot returns NumPy, convert to Tensor
        self.one_hot_sequences = torch.from_numpy(
            sequences_to_one_hot(positive_sequences_int, max_sequence_length, NUM_AMINO_ACIDS) # Use NUM_AMINO_ACIDS as one_hot dim
        ).float() # Use FloatTensor for model input

    def __len__(self):
        return len(self.one_hot_sequences)

    def __getitem__(self, idx):
        return self.one_hot_sequences[idx] # Returns a single one-hot encoded sequence

# --- Data Loader Helper Functions ---

def get_predictive_dataloader(data_filepath, batch_size):
    """Helper to get DataLoader for predictive model training/evaluation."""
    X_train, X_test, y_train, y_test, _, _ = load_preprocessed_data(data_filepath)

    if X_train is None or X_test is None:
        return None, None # Indicate data loading failure

    train_dataset = PeptideDataset(X_train, y_train)
    test_dataset = PeptideDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def get_gan_dataloader(data_filepath, batch_size):
    """Helper to get DataLoader for GAN training (only positive samples)."""
    X_train, _, y_train, _, max_sequence_length, _ = load_preprocessed_data(data_filepath)

    if X_train is None:
        return None, None # Indicate data loading failure

    # Filter positive sequences from training data (integer sequences)
    positive_indices = np.where(y_train == 1)[0]
    positive_sequences_int = X_train[positive_indices]

    if len(positive_sequences_int) == 0:
         print("警告：GAN训练数据集中没有正样本。请检查数据。")
         return None, max_sequence_length # Return max_len even if no data

    gan_dataset = GANPeptideDataset(positive_sequences_int, max_sequence_length)

    gan_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True)

    return gan_loader, max_sequence_length