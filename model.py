# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES, PAD_TOKEN # Import shared utilities

# Hyperparameters (Increased Hidden Dim)
EMBEDDING_DIM = 64
HIDDEN_DIM = 256 # Increased Hidden Dimension
LATENT_DIM = 100

# Regularization
DROPOUT_RATE = 0.4 # Dropout rate (adjust as needed, 0.2 to 0.5 is common)
DISC_INPUT_NOISE_STDDEV = 0.05 # Standard deviation of Gaussian noise added to discriminator input (adjust as needed, typically small)


class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, latent_dim, max_len, pad_idx, dropout_rate):
        super(Generator, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.hidden_dim = hidden_dim # Store hidden_dim for initial state calculation

        self.fc_initial_state = nn.Linear(latent_dim, hidden_dim) # GRU only has h_0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Use GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout_rate)


    def forward(self, noise, temperature):
        batch_size = noise.size(0)

        # Project noise to initial hidden state
        h_0 = self.fc_initial_state(noise).unsqueeze(0).contiguous() # shape (1, batch_size, hidden_dim)
        hidden = h_0 # GRU only has hidden state


        generated_sequence_onehot = []
        # Initial input for the first step (embedding of pad token)
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)


        for _ in range(self.max_len):
             # Get embedding for current input token(s) or weighted average of embeddings
             if _ == 0:
                 # First input is PAD token embedding
                 lstm_input = self.embedding(current_input) # shape (batch_size, 1, embedding_dim)
             else:
                 # Input for subsequent steps is the continuous embedding from the *previous* Gumbel-Softmax output
                 lstm_input = torch.matmul(previous_one_hot_token.unsqueeze(1), self.embedding.weight)
                 # shape (batch_size, 1, embed_dim)


             # Pass through GRU
             output, hidden = self.gru(lstm_input, hidden) # output shape (batch_size, 1, hidden_dim)

             # Apply Dropout
             output = self.dropout(output) # Dropout after GRU


             # Get logits for the next token
             logits = self.fc_out(output.squeeze(1)) # shape (batch_size, vocab_size)

             # Apply Gumbel-Softmax
             one_hot_token = F.gumbel_softmax(logits, tau=temperature, hard=False, dim=-1) # shape (batch_size, vocab_size)
             generated_sequence_onehot.append(one_hot_token.unsqueeze(1)) # shape (batch_size, 1, vocab_size)

             # Save the current step's soft token for the next step's input embedding
             previous_one_hot_token = one_hot_token


        # Concatenate one-hot tokens along the sequence dimension
        generated_sequence_onehot = torch.cat(generated_sequence_onehot, dim=1) # shape (batch_size, max_len, vocab_size)

        return generated_sequence_onehot # Returns soft one-hot vectors

    # Keep generate_discrete the same, using argmax for hard sampling
    def generate_discrete(self, noise, temperature=0.1):
        batch_size = noise.size(0)
        h_0 = self.fc_initial_state(noise).unsqueeze(0).contiguous()
        hidden = h_0

        generated_sequence_indices = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)

        with torch.no_grad(): # Ensure no gradients are computed during discrete generation
            for _ in range(self.max_len):
                embedded_input = self.embedding(current_input) # shape (batch_size, 1, embedding_dim)

                output, hidden = self.gru(embedded_input, hidden) # Pass through GRU

                # Dropout is off in eval mode
                # output = self.dropout(output)


                logits = self.fc_out(output.squeeze(1)) # shape (batch_size, vocab_size)

                # Sample token using argmax
                next_token_indices = torch.argmax(logits, dim=1) # shape (batch_size,)

                generated_sequence_indices.append(next_token_indices.unsqueeze(1)) # shape (batch_size, 1)

                # For next input, use the embedding of the hard-sampled token
                current_input = next_token_indices.unsqueeze(1) # shape (batch_size, 1)

            generated_sequence_indices = torch.cat(generated_sequence_indices, dim=1) # shape (batch_size, max_len)

        return generated_sequence_indices # Returns discrete indices


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx, dropout_rate, input_noise_stddev):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        # Use GRU
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout after GRU
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.input_noise_stddev = input_noise_stddev # Standard deviation for input noise

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

        # Add Gaussian noise to embedded input
        if self.training and self.input_noise_stddev > 0: # Only add noise during training
            noise = torch.randn_like(embedded) * self.input_noise_stddev
            embedded = embedded + noise
            # Optional: Consider clipping embedded values if noise makes them too large

        # Pass through GRU
        gru_out, _ = self.gru(embedded) # gru_out shape (batch_size, seq_len, hidden_dim)

        # Get the output from the last time step
        last_output = gru_out[:, -1, :] # shape (batch_size, hidden_dim)

        # Apply Dropout
        last_output = self.dropout(last_output) # Dropout after GRU

        # Pass through linear layer
        logits = self.fc(last_output) # shape (batch_size, num_classes)

        return logits

# Example Usage (for testing the model structure)
if __name__ == '__main__':
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     latent_dim = LATENT_DIM
     batch_size = 4
     noise = torch.randn(batch_size, latent_dim).to(device)

     # Pass dropout_rate and input_noise_stddev to model constructors
     generator = Generator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, MAX_LEN, vocab.pad_idx, DROPOUT_RATE).to(device)
     discriminator = Discriminator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_DISC_CLASSES, vocab.pad_idx, DROPOUT_RATE, DISC_INPUT_NOISE_STDDEV).to(device)

     print("Generator Architecture:", generator)
     print("Discriminator Architecture:", discriminator)


     # Test generator forward
     soft_output = generator(noise, temperature=1.0)
     print(f"\nGenerator output (soft): {soft_output.shape}, dtype: {soft_output.dtype}") # Should be (batch, len, vocab_size) float

     # Test discriminator with fake data (in training mode to test noise/dropout)
     discriminator.train() # Set D to training mode
     disc_output_fake = discriminator(soft_output.detach()) # Detach for simulating D training path
     print(f"Discriminator output (fake, train mode): {disc_output_fake.shape}")

     # Create dummy real data for discriminator test
     dummy_real_data = torch.randint(0, vocab.vocab_size, (batch_size, MAX_LEN), dtype=torch.long).to(device)
     disc_output_real = discriminator(dummy_real_data)
     print(f"Discriminator output (real, train mode): {disc_output_real.shape}")

     # Test discrete generation (in eval mode)
     generator.eval() # Set G to eval mode
     discrete_output = generator.generate_discrete(noise)
     print(f"Generator output (discrete, eval mode): {discrete_output.shape}, dtype: {discrete_output.dtype}") # Should be (batch, len) long