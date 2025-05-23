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

class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len, pad_idx):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx

        # Project latent noise to initial hidden/cell state
        self.fc_initial_state = nn.Linear(latent_dim, hidden_dim * 2)  # For h_0 and c_0

        # Embedding layer for input tokens (e.g., previous token)
        # We'll use the padding token's embedding for the first input step
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Output layer: maps hidden state to vocabulary size
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

        # Gumbel-Softmax for differentiable sampling
        # Need to define temperature later

    def forward(self, noise, temperature):
        batch_size = noise.size(0)

        # Project noise to initial hidden and cell state
        initial_state = self.fc_initial_state(noise).view(batch_size, 2, -1)
        h_0 = initial_state[:, 0, :].unsqueeze(0).contiguous()  # shape (1, batch_size, hidden_dim)
        c_0 = initial_state[:, 1, :].unsqueeze(0).contiguous()  # shape (1, batch_size, hidden_dim)

        # Initial input: embedding of a start token or padding token
        # We'll feed a tensor of pad_idx indices as the "input" for all time steps
        # The LSTM state will be updated based on the noise initially, and subsequent steps depend on previous output.
        # A common way is to generate one token at a time in a loop.
        # Let's implement it as generating step-by-step.

        generated_sequence_onehot = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long,
                                   device=noise.device)  # Use pad_idx as start

        hidden = (h_0, c_0)

        for _ in range(self.max_len):
            # Get embedding for current input token(s)
            embedded_input = self.embedding(current_input)  # shape (batch_size, 1, embedding_dim)

            # Pass through LSTM
            output, hidden = self.lstm(embedded_input, hidden)  # output shape (batch_size, 1, hidden_dim)

            # Get logits for the next token
            logits = self.fc_out(output.squeeze(1))  # shape (batch_size, vocab_size)

            # Apply Gumbel-Softmax to get a differentiable approximation of one-hot
            # shape (batch_size, vocab_size)
            # Note: nn.functional.gumbel_softmax expects input (logits) shape (batch_size, num_classes)
            one_hot_token = torch.nn.functional.gumbel_softmax(logits, tau=temperature, hard=False)

            # For the *next* input to LSTM, we need an embedding.
            # We can get the embedding from the one-hot vector.
            # This simulates feeding the "sampled" token back into the embedding layer.
            # Note: This is a bit simplified; usually, the Gumbel-Softmax output *is* used directly
            # for calculating the loss, and the embedding of this approximated one-hot
            # is used as the input for the *next* step if generating sequentially.
            # Let's re-think: The standard way for training is to pass the Gumbel-Softmax output
            # as the input to the *next* step's embedding lookup.
            # However, LSTM expects fixed time steps. Let's use a simpler approach often seen in text GANs:
            # Generate all tokens in one go by feeding *something* (like initial state or start token embedding)
            # to the first step and processing sequentially without feeding back the "sampled" token during the *forward* pass in training.
            # This requires feeding the LSTM the sequence of embeddings *as if* they were sampled.
            # Let's stick to the step-by-step generation process which is more natural for sequence generation.

            # Resample for next step input: take the argmax from the Gumbel-Softmax for feeding *back*
            # This breaks differentiability slightly at the argmax step but the gradients flow through Gumbel-Softmax for the main loss.
            # Or, a simpler method is to use the *hard* one-hot from gumbel_softmax with straight-through estimator,
            # or directly use the continuous one-hot output. Let's use the continuous one-hot output.

            # Instead of getting the "index" and re-embedding, multiply one_hot_token by the embedding weights
            # This effectively gives an embedding corresponding to the weighted average of token embeddings
            # according to the Gumbel-Softmax probabilities.
            current_embedded_input = one_hot_token.unsqueeze(1) @ self.embedding.weight.unsqueeze(0)
            # shape (batch_size, 1, vocab_size) @ (1, vocab_size, embedding_dim) -> (batch_size, 1, embedding_dim)

            # We need to pass *something* at each timestep. A common approach for sequence generation with Gumbel-Softmax GANs:
            # Treat the LSTM as taking a fixed input (like the noise or a dummy token) at the *first* timestep,
            # and then using the Gumbel-Softmax output of the *previous* timestep as the input for the *current* timestep.

            # Let's refine: The input to the LSTM at time `t` is the embedding of the chosen token at time `t-1`.
            # The noise vector initializes the hidden state.
            # The token for t=1 is generated from the state after processing the initial state (potentially with a dummy input).

            # Revised Step-by-step generation:
            # t=0: Input is a dummy/start embedding (e.g., pad token embedding or just zeros). State is from noise.
            # Output for t=1 generated.
            # t=1: Input is embedding of (Gumbel-Softmax sampled) token from t=1. State is from t=0. Output for t=2 generated.
            # ... continues for MAX_LEN steps.

            if _ == 0:
                # First input: embedding of PAD token (or a dedicated START token)
                lstm_input = self.embedding(
                    torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device))
            else:
                # Input for subsequent steps: embedding based on the *previous* Gumbel-Softmax output
                # This is the differentiable connection allowing gradients to flow back
                lstm_input = one_hot_token.unsqueeze(1) @ self.embedding.weight.unsqueeze(0)
                # lstm_input = one_hot_token.unsqueeze(1) @ self.embedding.weight.t() # Check dimension match, should be batch_size, 1, embedding_dim

            # Pass through LSTM
            output, hidden = self.lstm(lstm_input, hidden)  # output shape (batch_size, 1, hidden_dim)

            # Get logits for the next token
            logits = self.fc_out(output.squeeze(1))  # shape (batch_size, vocab_size)

            # Apply Gumbel-Softmax
            one_hot_token = torch.nn.functional.gumbel_softmax(logits, tau=temperature,
                                                               hard=False)  # Use hard=True only potentially at the very end of training

            generated_sequence_onehot.append(one_hot_token.unsqueeze(1))  # shape (batch_size, 1, vocab_size)

        # Concatenate one-hot tokens along the sequence dimension
        generated_sequence_onehot = torch.cat(generated_sequence_onehot,
                                              dim=1)  # shape (batch_size, max_len, vocab_size)

        return generated_sequence_onehot

    def generate_discrete(self, noise, temperature=0.1):
        # This method is for actual sampling after training or for evaluation, not training G.
        # Here we use hard sampling (argmax)
        batch_size = noise.size(0)
        initial_state = self.fc_initial_state(noise).view(batch_size, 2, -1)
        h_0 = initial_state[:, 0, :].unsqueeze(0).contiguous()
        c_0 = initial_state[:, 1, :].unsqueeze(0).contiguous()

        generated_sequence_indices = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)

        hidden = (h_0, c_0)

        for _ in range(self.max_len):
            embedded_input = self.embedding(current_input)  # shape (batch_size, 1, embedding_dim)

            output, hidden = self.lstm(embedded_input, hidden)  # output shape (batch_size, 1, hidden_dim)

            logits = self.fc_out(output.squeeze(1))  # shape (batch_size, vocab_size)

            # Sample token using argmax for discrete generation
            # Using torch.multinomial would introduce more randomness
            probs = torch.softmax(logits, dim=1)  # Optional: see probabilities
            next_token_indices = torch.argmax(logits, dim=1)  # Sample hard

            generated_sequence_indices.append(next_token_indices.unsqueeze(1))  # shape (batch_size, 1)

            # For next input, use the embedding of the hard-sampled token
            current_input = next_token_indices.unsqueeze(1)

        generated_sequence_indices = torch.cat(generated_sequence_indices, dim=1)  # shape (batch_size, max_len)

        return generated_sequence_indices  # Returns token indices

class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # Linear layer to map LSTM output to class scores
        # Option 1: Use final hidden state
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Option 2: Use attention or pooling over time steps (more complex)
        # We'll use the final hidden state of the last layer as a representation of the sequence.

    def forward(self, sequence):
        # sequence is either integer indices (real data) or one-hot vectors (fake data)
        if sequence.dtype == torch.long:
             # Input is integer indices (real data)
             embedded = self.embedding(sequence) # shape (batch_size, seq_len, embedding_dim)
        else:
             # Input is Gumbel-Softmax one-hot vectors (fake data)
             # The embedding is a matrix multiplication of the one-hot vector and the embedding matrix
             embedded = torch.bmm(sequence, self.embedding.weight.unsqueeze(0).expand(sequence.size(0), -1, -1))
             # Check shapes: sequence (batch, len, vocab_size), embedding.weight (vocab_size, embed_dim)
             # @ (batch, len, vocab_size) x (vocab_size, embed_dim) -> (batch, len, embed_dim)
             # Use torch.matmul directly if sequence shape is (batch, vocab_size) -> (batch, embed_dim)
             # Wait, Gumbel-Softmax outputs (batch_size, max_len, vocab_size). So use bmm.
             embedded = torch.matmul(sequence, self.embedding.weight) # shape (batch_size, max_len, embedding_dim)
             # This essentially takes a weighted average of token embeddings based on the soft probabilities.

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded) # lstm_out shape (batch_size, seq_len, hidden_dim)

        # Get the output from the last time step (considering padding)
        # A common way to handle padding is to get lengths, pack, run LSTM, then unpack
        # But for a fixed max_len and using padding index, taking the output corresponding
        # to the *original* last token or simply taking the last output is simpler,
        # though maybe less robust to padding placement.
        # Let's use the output from the last time step after processing the full padded sequence.
        last_output = lstm_out[:, -1, :] # shape (batch_size, hidden_dim)

        # Pass through linear layer
        logits = self.fc(last_output) # shape (batch_size, num_classes)

        return logits