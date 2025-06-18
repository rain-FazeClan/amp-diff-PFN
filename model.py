import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES, PAD_TOKEN # Import shared utilities

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 256 # Keep this for Generator GRU and potentially for Transformer layers
LATENT_DIM = 100

# Regularization
DROPOUT_RATE = 0.5 # Keep for Generator and Discriminator
DISC_INPUT_NOISE_STDDEV = 0.1 # Keep for Discriminator input

# Transformer Discriminator Specific Hyperparameters
TRANSFORMER_NUM_HEADS = 4 # Number of attention heads
TRANSFORMER_NUM_LAYERS = 2 # Number of Transformer Encoder layers
TRANSFORMER_FFN_HIDDEN_DIM = HIDDEN_DIM * 4 # Feedforward network hidden dimension (often 4x hidden_dim)


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len, pad_idx, dropout_rate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim

        self.fc_initial_state = nn.Linear(latent_dim, hidden_dim)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, noise, temperature):
        batch_size = noise.size(0)
        device = noise.device
        h_0 = self.fc_initial_state(noise).unsqueeze(0).contiguous()
        hidden = h_0

        generated_sequence_onehot = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)

        for _ in range(self.max_len):
             if _ == 0:
                 lstm_input = self.embedding(current_input)
             else:
                 lstm_input = torch.matmul(previous_one_hot_token.unsqueeze(1), self.embedding.weight)

             output, hidden = self.gru(lstm_input, hidden)

             output = self.dropout(output)

             logits = self.fc_out(output.squeeze(1))

             one_hot_token = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1)
             generated_sequence_onehot.append(one_hot_token.unsqueeze(1))

             previous_one_hot_token = one_hot_token

        generated_sequence_onehot = torch.cat(generated_sequence_onehot, dim=1)

        return generated_sequence_onehot

    def generate_discrete(self, noise, temperature=0.1):
        batch_size = noise.size(0)
        device = noise.device
        h_0 = self.fc_initial_state(noise).unsqueeze(0).contiguous()
        hidden = h_0

        generated_sequence_indices = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=device)

        with torch.no_grad():
            for _ in range(self.max_len):
                embedded_input = self.embedding(current_input)
                output, hidden = self.gru(embedded_input, hidden)

                logits = self.fc_out(output.squeeze(1))

                next_token_indices = torch.argmax(logits, dim=1)

                generated_sequence_indices.append(next_token_indices.unsqueeze(1))
                current_input = next_token_indices.unsqueeze(1)

            generated_sequence_indices = torch.cat(generated_sequence_indices, dim=1)

        return generated_sequence_indices


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx, dropout_rate, input_noise_stddev,
                 transformer_num_heads, transformer_num_layers, transformer_ffn_hidden_dim):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        # Transformer Encoder
        # d_model must be equal to embedding_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=transformer_num_heads,
                                                   dim_feedforward=transformer_ffn_hidden_dim, dropout=dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)

        # Need a positional encoding for Transformer
        self.positional_encoding = nn.Parameter(torch.randn(1, MAX_LEN, embedding_dim)) # Learnable positional embedding


        # Pooling layer to get a fixed-size representation
        # Option 1: Mean Pooling over sequence dimension
        # Option 2: Take output of the first token (if using a start token conceptually)
        # Option 3: Learnable Attention Pooling (more complex)
        # Let's start with Mean Pooling over non-padded elements. Requires mask.
        # Or simpler: Mean Pooling over the entire sequence (including padding - effect is reduced but might work with padding mask implicitly in transformer layer)
        # Let's use Mean Pooling + Pad Masking in forward pass.
        # A simpler approach is just using the last token IF Transformer's positional encoding/self-attention makes the last token representative.
        # But sequence classification typically pools over the sequence. Mean pooling with mask is common.

        # Assuming after transformer we get (batch_size, seq_len, embedding_dim)
        # After pooling, we need a tensor of shape (batch_size, some_dimension)

        # Linear layer maps pooled representation to class scores
        # Pooling dimension depends on the pooling method. Mean Pooling over embed_dim results in embed_dim size.
        self.fc = nn.Linear(embedding_dim, num_classes) # Output dim matches embedding_dim after mean pooling


        self.dropout = nn.Dropout(p=dropout_rate) # Keep dropout for final layer? Or handled within transformer?
        # TransformerEncoderLayer has dropout. Let's remove final dropout after pooling.

        self.input_noise_stddev = input_noise_stddev # Standard deviation for input noise
        self.pad_idx = pad_idx # Need pad_idx for masking


    def forward(self, sequence):
        # sequence is either integer indices (real data: batch, len)
        # or one-hot vectors (fake data: batch, len, vocab_size)
        device = sequence.device

        # Get embedding
        if sequence.dtype == torch.long:
             # Input is integer indices (real data)
             embedded = self.embedding(sequence) # shape (batch_size, seq_len, embedding_dim)
             # Create padding mask for Transformer (True for padding, False otherwise)
             padding_mask = (sequence == self.pad_idx) # shape (batch_size, seq_len)
        elif sequence.dtype == torch.float: # Assuming soft one-hot will be float
             # Input is Gumbel-Softmax one-hot vectors (fake data)
             embedded = torch.matmul(sequence, self.embedding.weight) # shape (batch_size, seq_len, embedding_dim)
             # For soft one-hot, assuming sequences are of max_len and padding token has non-zero probability at padding positions?
             # This masking might be tricky with soft outputs. A simple approach might be to mask based on where the hard prediction *would be* the pad token
             # or simply apply mean pooling over everything. Let's assume we can create a mask based on which time steps should conceptually be padding.
             # If the generator is trained to put PAD_TOKEN one-hot at the end, maybe we can threshold the one-hot?
             # Or, simplify and use mean pooling over the entire sequence for fake data (as soft output makes strict masking hard).
             # For now, let's generate a mask assuming any step with the highest prob being pad_idx is padding (imperfect for soft input)
             # A better approach for soft fake data might be to pass them through the Generator's generate_discrete to get hard tokens just for masking, but that's computationally expensive.
             # Let's try simplest: for fake data, assume no padding mask is needed or approximate one based on hard prediction.
             # If Generator perfectly learns padding at the end, maybe mask based on hard prediction?
             # Simpler: Use mask based on where the hard prediction *would* be padding only for real data. For fake, no mask?
             # No, transformer requires masking to ignore padded positions in attention. Masking fake sequences is still needed.
             # Let's create a mask based on the hard token index prediction for the soft sequence.
             predicted_hard_indices = torch.argmax(sequence, dim=-1)
             padding_mask = (predicted_hard_indices == self.pad_idx) # shape (batch_size, seq_len)

        else:
             raise TypeError(f"Unsupported sequence dtype: {sequence.dtype}")


        # Add positional encoding
        embedded = embedded + self.positional_encoding[:, :embedded.size(1), :] # Add PE, slicing handles variable length if needed (though using MAX_LEN here)

        # Add Gaussian noise to embedded input (only during training)
        if self.training and self.input_noise_stddev > 0:
            noise = torch.randn_like(embedded, device=device) * self.input_noise_stddev
            embedded = embedded + noise

        # Pass through Transformer Encoder
        # padding_mask goes to src_key_padding_mask in TransformerEncoder.
        # src_key_padding_mask (batch_size, seq_len) has True for padding, False otherwise. This is what we have.
        # transformer expects (seq_len, batch_size, feature_dim) IF batch_first is False.
        # With batch_first=True, it expects (batch_size, seq_len, feature_dim). Our data is (batch, len, embed_dim). Correct.
        # The mask needs to be passed to .forward(src, src_key_padding_mask=padding_mask)
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=padding_mask) # shape (batch_size, seq_len, embedding_dim)


        # Pooling
        # Mean pooling over non-padded elements
        # The mask is (batch_size, seq_len), True at padding, False otherwise.
        # Inverted mask is (batch_size, seq_len), False at padding, True otherwise (for selecting non-padding)
        # Need to handle case where an entire sequence is padding (shouldn't happen with GAN output, but for real data filter might remove short seqs)
        non_pad_elements_mask = ~padding_mask # shape (batch_size, seq_len)
        sum_embeddings = (transformer_output * non_pad_elements_mask.unsqueeze(-1)).sum(dim=1) # Zero out padding, then sum
        # Calculate sequence lengths for proper mean
        seq_lengths = non_pad_elements_mask.sum(dim=1).float() # shape (batch_size,)
        # Add epsilon to prevent division by zero if a seq_length is 0 (shouldn't happen with current setup but safe)
        pooled_output = sum_embeddings / (seq_lengths.unsqueeze(-1) + 1e-5) # shape (batch_size, embedding_dim)

        # Alternative simple pooling (might not handle padding perfectly depending on transformer)
        # pooled_output = transformer_output.mean(dim=1) # Mean pool over sequence length regardless of padding


        # Pass through linear layer
        logits = self.fc(pooled_output) # shape (batch_size, num_classes)

        return logits


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    latent_dim = LATENT_DIM
    batch_size = 4
    noise = torch.randn(batch_size, latent_dim).to(device)

    generator = Generator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, MAX_LEN, vocab.pad_idx, DROPOUT_RATE).to(device)
    discriminator = Discriminator(vocab_size=vocab.vocab_size,
                               embedding_dim=EMBEDDING_DIM,
                               hidden_dim=HIDDEN_DIM,
                               num_classes=NUM_DISC_CLASSES,
                               pad_idx=vocab.pad_idx,
                               dropout_rate=DROPOUT_RATE,
                               input_noise_stddev=DISC_INPUT_NOISE_STDDEV,
                               transformer_num_heads=TRANSFORMER_NUM_HEADS,
                               transformer_num_layers=TRANSFORMER_NUM_LAYERS,
                               transformer_ffn_hidden_dim=TRANSFORMER_FFN_HIDDEN_DIM).to(device)

    print("Generator Architecture:", generator)
    print("Discriminator Architecture:", discriminator)

    # Test generator forward
    generator.train()
    soft_output = generator(noise, temperature=1.0)
    print(f"\nGenerator output (soft): {soft_output.shape}, dtype: {soft_output.dtype}")

    # Test discriminator with fake data (training模式测试)
    discriminator.train()
    disc_output_fake = discriminator(soft_output.detach())
    print(f"Discriminator output (fake, train mode): {disc_output_fake.shape}")

    # 构造真实数据进行测试
    dummy_real_data = torch.randint(0, vocab.vocab_size - 1, (batch_size, MAX_LEN - 5), dtype=torch.long)
    dummy_pad = torch.full((batch_size, 5), vocab.pad_idx, dtype=torch.long)
    dummy_real_data = torch.cat([dummy_real_data, dummy_pad], dim=1).to(device)
    disc_output_real = discriminator(dummy_real_data)
    print(f"Discriminator output (real, train mode): {disc_output_real.shape}")

    # Test discrete generation（eval模式测试）
    generator.eval()
    discriminator.eval()
    discrete_output = generator.generate_discrete(noise)
    print(f"Generator output (discrete, eval mode): {discrete_output.shape}, dtype: {discrete_output.dtype}")

    # 转换为 one-hot 并确保在相同设备上
    discrete_one_hot = F.one_hot(discrete_output, num_classes=vocab.vocab_size).float().to(device)
    disc_output_discrete_eval = discriminator(discrete_one_hot)
    print(f"Discriminator output (discrete, eval mode): {disc_output_discrete_eval.shape}")