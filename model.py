import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES, PAD_TOKEN # Import shared utilities

# Hyperparameters (can be moved to config or train.py)
EMBEDDING_DIM = 64
HIDDEN_DIM = 128
LATENT_DIM = 100


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len, pad_idx):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx

        self.fc_initial_state = nn.Linear(latent_dim, hidden_dim * 2)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise, temperature):
        batch_size = noise.size(0)

        # Project noise to initial hidden/cell state
        initial_state = self.fc_initial_state(noise).view(batch_size, 2, -1)
        h_0 = initial_state[:, 0, :].unsqueeze(0).contiguous()
        c_0 = initial_state[:, 1, :].unsqueeze(0).contiguous()
        hidden = (h_0, c_0)

        generated_sequence_onehot = []
        # Initial input for the first step (e.g., pad token embedding)
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)


        for _ in range(self.max_len):
             # Get embedding for current input token(s) or weighted average of embeddings
             if _ == 0:
                 # First input is PAD token embedding
                 lstm_input = self.embedding(current_input) # shape (batch_size, 1, embedding_dim)
             else:
                 # Input for subsequent steps is the continuous embedding from the *previous* Gumbel-Softmax output
                 # previous_one_hot_token was saved in the last iteration
                 lstm_input = torch.matmul(previous_one_hot_token.unsqueeze(1), self.embedding.weight)
                 # Check shape match: (batch, 1, vocab_size) @ (vocab_size, embed_dim) -> (batch, 1, embed_dim)


             output, hidden = self.lstm(lstm_input, hidden) # output shape (batch_size, 1, hidden_dim)

             logits = self.fc_out(output.squeeze(1)) # shape (batch_size, vocab_size)

             # Apply Gumbel-Softmax
             one_hot_token = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1) # shape (batch_size, vocab_size)
             generated_sequence_onehot.append(one_hot_token.unsqueeze(1)) # shape (batch_size, 1, vocab_size)

             # Save the current step's soft token for the next step's input embedding
             previous_one_hot_token = one_hot_token


        # Concatenate one-hot tokens along the sequence dimension
        generated_sequence_onehot = torch.cat(generated_sequence_onehot, dim=1) # shape (batch_size, max_len, vocab_size)

        return generated_sequence_onehot # Returns soft one-hot vectors

    def generate_discrete(self, noise, temperature=0.1):
        """
        Generates discrete token indices (argmax sampling).
        """
        batch_size = noise.size(0)
        initial_state = self.fc_initial_state(noise).view(batch_size, 2, -1)
        h_0 = initial_state[:, 0, :].unsqueeze(0).contiguous()
        c_0 = initial_state[:, 1, :].unsqueeze(0).contiguous()

        generated_sequence_indices = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)

        hidden = (h_0, c_0)

        for _ in range(self.max_len):
            embedded_input = self.embedding(current_input) # shape (batch_size, 1, embedding_dim)

            output, hidden = self.lstm(embedded_input, hidden) # output shape (batch_size, 1, hidden_dim)

            logits = self.fc_out(output.squeeze(1)) # shape (batch_size, vocab_size)

            # Sample token using argmax
            next_token_indices = torch.argmax(logits, dim=1) # shape (batch_size,)

            generated_sequence_indices.append(next_token_indices.unsqueeze(1)) # shape (batch_size, 1)

            # For next input, use the embedding of the hard-sampled token
            current_input = next_token_indices.unsqueeze(1) # shape (batch_size, 1)

        generated_sequence_indices = torch.cat(generated_sequence_indices, dim=1) # shape (batch_size, max_len)

        return generated_sequence_indices # Returns discrete indices


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, sequence):
        # sequence is either integer indices (real data: batch, len)
        # or one-hot vectors (fake data: batch, len, vocab_size)

        if sequence.dtype == torch.long:
             # Input is integer indices (real data)
             embedded = self.embedding(sequence) # shape (batch_size, seq_len, embedding_dim)
        elif sequence.dtype == torch.float: # Assuming soft one-hot will be float
             # Input is Gumbel-Softmax one-hot vectors (fake data)
             embedded = torch.matmul(sequence, self.embedding.weight) # shape (batch_size, seq_len, embedding_dim)
        else:
             raise TypeError(f"Unsupported sequence dtype: {sequence.dtype}")

        # Pass through LSTM
        lstm_out, _ = self.lstm(embedded) # lstm_out shape (batch_size, seq_len, hidden_dim)

        # Get the output from the last time step
        last_output = lstm_out[:, -1, :] # shape (batch_size, hidden_dim)

        # Pass through linear layer
        logits = self.fc(last_output) # shape (batch_size, num_classes)

        return logits

if __name__ == '__main__':
     # Example usage
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     latent_dim = LATENT_DIM
     batch_size = 4
     noise = torch.randn(batch_size, latent_dim).to(device)

     generator = Generator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, MAX_LEN, vocab.pad_idx).to(device)
     discriminator = Discriminator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_DISC_CLASSES, vocab.pad_idx).to(device)

     # Test generator forward
     soft_output = generator(noise, temperature=1.0)
     print(f"Generator output (soft): {soft_output.shape}, dtype: {soft_output.dtype}") # Should be (batch, len, vocab_size) float

     # Test discriminator with fake data
     disc_output_fake = discriminator(soft_output)
     print(f"Discriminator output (fake): {disc_output_fake.shape}") # Should be (batch, num_classes)

     # Create dummy real data for discriminator test
     dummy_real_data = torch.randint(0, vocab.vocab_size, (batch_size, MAX_LEN), dtype=torch.long).to(device)
     disc_output_real = discriminator(dummy_real_data)
     print(f"Discriminator output (real): {disc_output_real.shape}") # Should be (batch, num_classes)

     # Test discrete generation
     discrete_output = generator.generate_discrete(noise)
     print(f"Generator output (discrete): {discrete_output.shape}, dtype: {discrete_output.dtype}") # Should be (batch, len) long