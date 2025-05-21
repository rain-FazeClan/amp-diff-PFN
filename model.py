# models/predictive_model_pt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import NUM_AMINO_ACIDS, PADDING_VALUE

class PeptidePredictiveModel(nn.Module):
    """
    PyTorch 模型用于抗菌肽预测。
    使用 Embedding, Conv1D, LSTM, Dense。
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim_lstm, num_filters, kernel_size, max_sequence_length, dropout_rate=0.5):
        """
        Args:
            vocab_size (int): 词汇表大小 (包括 padding)。
            embedding_dim (int): 嵌入向量的维度。
            hidden_dim_lstm (int): LSTM 层的隐藏状态维度。
            num_filters (int): 卷积层滤波器的数量。
            kernel_size (int): 卷积核的大小。
            max_sequence_length (int): 输入序列的最大长度。
            dropout_rate (float): Dropout 比率。
        """
        super(PeptidePredictiveModel, self).__init__()

        self.max_sequence_length = max_sequence_length

        # Embedding layer: vocab_size is the total number of possible tokens (including padding)
        # padding_idx prevents gradients from flowing back to padding token embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PADDING_VALUE)

        # Conv1D layers
        # Input to Conv1D is (batch_size, channels, sequence_length)
        # After embedding, sequence is (batch_size, sequence_length, embedding_dim)
        # Need to permute: (batch_size, embedding_dim, sequence_length)
        self.conv1d_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size, padding='same') # 'same' padding to preserve length
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv1d_2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters, kernel_size=kernel_size, padding='same')
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Calculate sequence length after pooling
        # MaxPool1d with kernel_size=2 and default stride=kernel_size halves the length
        # length = (input_length - kernel_size + 2*padding) / stride + 1  --> (input_length - 2)/2 + 1
        # With kernel=2, stride=2, length /= 2
        # For 'same' padding and odd kernel, output length = input length. With even kernel=2, length is also approx same or length/2
        # If stride=kernel_size=2: output length = input length / 2 (integer division)
        # Let's assume default stride=kernel_size=pool_size
        pooled_length = max_sequence_length // 2 // 2 # Length after two pooling layers

        # LSTM layers
        # Input to LSTM: (batch_size, sequence_length, input_size)
        # Output from Conv/Pool is (batch_size, filters, pooled_length) -> permute back (batch_size, pooled_length, filters)
        self.lstm1 = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim_lstm, batch_first=True, bidirectional=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        # Bidirectional LSTM doubles the output size (hidden_dim_lstm * 2)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_lstm * 2, hidden_size=hidden_dim_lstm, batch_first=True, bidirectional=True)
        self.dropout4 = nn.Dropout(dropout_rate)

        # Dense layers
        # Input to Dense: Flattened output from last LSTM (batch_size, pooled_length * hidden_dim_lstm * 2) if return_sequences=True,
        # or (batch_size, hidden_dim_lstm * 2) if return_sequences=False (last hidden state).
        # We'll use the last hidden state (return_sequences=False in the last LSTM)
        lstm_output_size = hidden_dim_lstm * 2 # For bidirectional LSTM
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 1) # Output 1 for binary classification

    def forward(self, x):
        # x is (batch_size, max_sequence_length) - integer indices

        # Embedding
        x = self.embedding(x) # x becomes (batch_size, max_sequence_length, embedding_dim)

        # Permute for Conv1D input
        x = x.permute(0, 2, 1) # x becomes (batch_size, embedding_dim, max_sequence_length)

        # Conv -> ReLU -> Pool -> Dropout
        x = F.relu(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = self.dropout1(x)

        x = F.relu(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = self.dropout2(x)

        # Permute back for LSTM input
        x = x.permute(0, 2, 1) # x becomes (batch_size, pooled_length, num_filters)

        # LSTM -> Dropout
        # lstm1 output: (batch_size, pooled_length, hidden_dim_lstm * 2), final_hidden_state, final_cell_state
        x, _ = self.lstm1(x)
        x = self.dropout3(x)

        # lstm2 input: (batch_size, pooled_length, hidden_dim_lstm * 2)
        # lstm2 output: (batch_size, pooled_length, hidden_dim_lstm * 2), final_hidden_state, final_cell_state
        # We only need the output of the *last* time step (or rather, the combined last states of both directions from the final hidden states).
        # The LSTM forward pass returns (output, (hidden_state, cell_state)).
        # For bidirectional LSTM, the hidden_state shape is (num_layers * 2, batch_size, hidden_size)
        # The final output 'x' from the last LSTM layer after batch_first=True is (batch_size, sequence_length, hidden_dim * num_directions).
        # For classification, we typically use the output corresponding to the last element *or* the final hidden state.
        # Using the output from the last time step:
        # x = x[:, -1, :] # Taking the last element output: (batch_size, hidden_dim_lstm * 2)
        # Or using the final hidden state which is more common with classification LSTMs
        # lstm2 returns (output, (hn, cn)) where hn is the final hidden state for each layer and direction
        # hn shape: (num_layers * num_directions, batch, hidden_size)
        # We need the last layer's combined final hidden state: layers 2 and 3 (for bidir), indices 2 and 3.
        # Let's redefine LSTM layers for simplicity and use the last hidden state.
        # For a single layer bidirectional LSTM: hn[0] is forward final, hn[1] is backward final. Concatenate them.
        # For multi-layer: hn[-2] is last layer forward, hn[-1] is last layer backward.

        # Let's revise the LSTM layers to make extraction of final hidden state clearer
        # Option 1: Keep return_sequences=True on all LSTMs and just take the last time step output
        # x = self.dropout4(x) # Apply dropout after lstm2
        # x = x[:, -1, :] # Take the last time step output for each sequence in the batch: (batch_size, hidden_dim_lstm * 2)

        # Option 2: Set return_sequences=False on the last LSTM layer.
        # Rerun with return_sequences=True for lstm1 and False for lstm2
        # self.lstm1 = nn.LSTM(...) return_sequences=True
        # self.lstm2 = nn.LSTM(input_size=hidden_dim_lstm*2, hidden_size=hidden_dim_lstm, batch_first=True, bidirectional=True, return_sequences=False)
        # x, _ = self.lstm1(x) # x is (batch_size, pooled_length, hidden_dim_lstm * 2)
        # x = self.dropout3(x)
        # x, (hn, cn) = self.lstm2(x) # x is now (batch_size, hidden_dim_lstm * 2) if batch_first=True, return_sequences=False
        # This seems simpler. Let's update __init__ and forward accordingly.

        # Revised __init__ (inside class definition):
        self.lstm1 = nn.LSTM(input_size=num_filters, hidden_size=hidden_dim_lstm, batch_first=True, bidirectional=True, return_sequences=True)
        self.dropout3 = nn.Dropout(dropout_rate)
        # Last LSTM returns only the final hidden state (or last timestep output if return_sequences=False)
        self.lstm2 = nn.LSTM(input_size=hidden_dim_lstm * 2, hidden_size=hidden_dim_lstm, batch_first=True, bidirectional=True, return_sequences=False) # <-- set to False
        self.dropout4 = nn.Dropout(dropout_rate) # Dropout will be before Dense

        # Revised forward (inside class definition):
        # ... (Embedding and Conv/Pool/Dropout layers as before)
        x = x.permute(0, 2, 1) # x becomes (batch_size, pooled_length, num_filters)

        # LSTM -> Dropout
        x, _ = self.lstm1(x) # x is (batch_size, pooled_length, hidden_dim_lstm * 2)
        x = self.dropout3(x)

        # Second LSTM: returns (output, (hn, cn)). Since return_sequences=False, output is (batch_size, hidden_dim_lstm*2)
        x, _ = self.lstm2(x) # x is (batch_size, hidden_dim_lstm * 2)
        x = self.dropout4(x) # Apply dropout after the last LSTM before Dense

        # Dense layers
        x = F.relu(self.fc1(x))
        # x = self.dropout5(x) # Optional dropout after first Dense
        x = self.fc2(x) # Output logits

        # Apply Sigmoid activation for binary classification probability
        output = torch.sigmoid(x)

        return output


# Helper function to get input size for the first Conv layer after embedding
# This needs max_sequence_length and embedding_dim, which we get from init.