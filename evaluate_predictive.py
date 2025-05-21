# models/gan_pt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NUM_AMINO_ACIDS # Number of actual amino acid characters (for one-hot dim)

class Generator(nn.Module):
    """
    PyTorch GAN 生成器模型。
    从噪声向量生成序列概率。
    """
    def __init__(self, latent_dim, max_sequence_length, num_amino_acids_one_hot):
        """
        Args:
            latent_dim (int): 噪声向量的维度。
            max_sequence_length (int): 输出序列的最大长度。
            num_amino_acids_one_hot (int): One-Hot 向量的维度（实际氨基酸字符数）。
        """
        super(Generator, self).__init__()

        self.max_sequence_length = max_sequence_length
        self.num_amino_acids_one_hot = num_amino_acids_one_hot

        # Dense layer to expand noise, followed by reshape
        self.fc = nn.Linear(latent_dim, max_sequence_length * 64) # Example expansion factor
        self.bn1 = nn.BatchNorm1d(max_sequence_length) # BatchNorm after Linear and before Conv reshapes dimensions. Or apply after reshape.
                                                       # Let's reshape then BatchNorm1d applied across features dimension.
                                                       # Input (batch, max_len * 64), output (batch, max_len * 64)

        # Use ConvTranspose1d (Deconvolutional layers) for upsampling,
        # or stick with Conv1d applied on expanded representation
        # Let's stick with Conv1d on expanded representation as in the TF example for now.
        # The input will be reshaped to (batch, max_len, features)

        # Need layers to process (batch, max_len, features) -> (batch, max_len, features) -> ... -> (batch, max_len, num_amino_acids_one_hot)

        # Conv1D expects input (batch, channels, length). After reshape: (batch, max_len, features)
        # Need to permute to (batch, features, max_len)
        features_dim = 64 # Features after the first linear layer
        self.conv1d_1 = nn.Conv1d(in_channels=features_dim, out_channels=128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(128) # BatchNorm expects (batch, channels, length) or (batch, length) depending on layer
                                      # BatchNorm1d after Conv1d expects input (batch, channels, length)
        self.conv1d_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding='same')
        self.bn3 = nn.BatchNorm1d(128)

        # Final layer to output num_amino_acids_one_hot probabilities for each position
        self.conv1d_final = nn.Conv1d(in_channels=128, out_channels=num_amino_acids_one_hot, kernel_size=5, padding='same')


    def forward(self, noise):
        # noise is (batch_size, latent_dim)

        # Linear -> LeakyReLU
        x = self.fc(noise) # (batch_size, max_len * 64)
        # x = self.bn1(x) # BatchNorm before reshape? Not typical
        x = F.leaky_relu(x, 0.2)

        # Reshape for Conv1D input
        # (batch_size, max_len * 64) -> (batch_size, max_len, 64)
        features_dim = 64
        x = x.view(-1, self.max_sequence_length, features_dim) # Reshape to (batch, length, features)

        # Permute for Conv1D input (batch, features, length)
        x = x.permute(0, 2, 1) # (batch_size, features, max_len)

        # Conv1D -> BatchNorm -> LeakyReLU
        x = self.conv1d_1(x) # (batch_size, 128, max_len)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)

        x = self.conv1d_2(x) # (batch_size, 128, max_len)
        x = self.bn3(x)
        x = F.leaky_relu(x, 0.2)

        # Final Conv1D to get logits for each amino acid at each position
        # (batch_size, 128, max_len) -> (batch_size, num_amino_acids_one_hot, max_len)
        logits = self.conv1d_final(x)

        # Permute back to (batch_size, max_len, num_amino_acids_one_hot) to match desired output format
        logits = logits.permute(0, 2, 1)

        # Apply Softmax across the amino acid dimension (-1)
        output_probs = F.softmax(logits, dim=-1) # (batch_size, max_len, num_amino_acids_one_hot)

        return output_probs


class Discriminator(nn.Module):
    """
    PyTorch GAN 判别器模型。
    区分真实 One-Hot 序列和生成器生成的序列概率。
    """
    def __init__(self, max_sequence_length, num_amino_acids_one_hot):
        """
        Args:
            max_sequence_length (int): 输入序列的最大长度。
            num_amino_acids_one_hot (int): 输入序列的通道数（One-Hot 向量维度）。
        """
        super(Discriminator, self).__init__()

        # Conv1D layers (input: batch, channels, length)
        # Input is (batch_size, max_sequence_length, num_amino_acids_one_hot)
        # Permute to (batch_size, num_amino_acids_one_hot, max_sequence_length)
        self.conv1d_1 = nn.Conv1d(in_channels=num_amino_acids_one_hot, out_channels=128, kernel_size=5, stride=1, padding='same')
        self.dropout1 = nn.Dropout(0.3)

        self.conv1d_2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=2, padding='same')
        self.dropout2 = nn.Dropout(0.3)

        self.conv1d_3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding='same')
        self.dropout3 = nn.Dropout(0.3)

        self.conv1d_4 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding='same')
        self.dropout4 = nn.Dropout(0.3)

        # Need to calculate the flattened size after Conv and Pooling (if any)
        # Simple Conv1D with stride=2 halves the length roughly.
        # Initial length = max_sequence_length
        # After conv1: length ~ max_sequence_length (padding='same', stride=1)
        # After conv2: length ~ max_sequence_length / 2 (padding='same', stride=2)
        # After conv3: length ~ max_sequence_length / 2 (padding='same', stride=1)
        # After conv4: length ~ max_sequence_length / 4 (padding='same', stride=2)
        # Number of features = 256

        # To calculate flattened size precisely:
        # Use a dummy tensor or manual calculation
        dummy_input = torch.randn(1, num_amino_acids_one_hot, max_sequence_length)
        x = self.conv1d_1(dummy_input)
        x = self.conv1d_2(x)
        x = self.conv1d_3(x)
        x = self.conv1d_4(x)
        flattened_size = x.view(x.size(0), -1).size(1)
        print(f"Discriminator flattened size: {flattened_size}")


        self.fc1 = nn.Linear(flattened_size, 1) # Output 1 logit for real/fake

    def forward(self, x):
        # x is (batch_size, max_sequence_length, num_amino_acids_one_hot)
        # Permute for Conv1D input (batch, channels, length)
        x = x.permute(0, 2, 1) # (batch_size, num_amino_acids_one_hot, max_sequence_length)

        # Conv -> LeakyReLU -> Dropout
        x = F.leaky_relu(self.conv1d_1(x), 0.2)
        x = self.dropout1(x)

        x = F.leaky_relu(self.conv1d_2(x), 0.2)
        x = self.dropout2(x)

        x = F.leaky_relu(self.conv1d_3(x), 0.2)
        x = self.dropout3(x)

        x = F.leaky_relu(self.conv1d_4(x), 0.2)
        x = self.dropout4(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Dense layer
        output_logit = self.fc1(x)

        # Discriminator typically outputs a probability using Sigmoid
        output_prob = torch.sigmoid(output_logit)

        return output_prob