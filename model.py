import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vocab, MAX_LEN, NUM_DISC_CLASSES, PAD_TOKEN # Import shared utilities

# Hyperparameters
EMBEDDING_DIM = 64
HIDDEN_DIM = 256 # Increased Hidden Dimension
LATENT_DIM = 100

# Regularization (Updated)
DROPOUT_RATE = 0.5 # Increased Dropout rate
DISC_INPUT_NOISE_STDDEV = 0.1 # Increased standard deviation of Gaussian noise


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
        h_0 = self.fc_initial_state(noise).unsqueeze(0).contiguous()
        hidden = h_0

        generated_sequence_indices = []
        current_input = torch.full((batch_size, 1), self.pad_idx, dtype=torch.long, device=noise.device)

        with torch.no_grad():
            for _ in range(self.max_len):
                embedded_input = self.embedding(current_input)
                output, hidden = self.gru(embedded_input, hidden)
                # Dropout is off in eval mode

                logits = self.fc_out(output.squeeze(1))

                next_token_indices = torch.argmax(logits, dim=1)

                generated_sequence_indices.append(next_token_indices.unsqueeze(1))
                current_input = next_token_indices.unsqueeze(1)

            generated_sequence_indices = torch.cat(generated_sequence_indices, dim=1)

        return generated_sequence_indices


class Discriminator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, pad_idx, dropout_rate, input_noise_stddev):
        super(Discriminator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.input_noise_stddev = input_noise_stddev

    def forward(self, sequence):
        if sequence.dtype == torch.long:
             embedded = self.embedding(sequence)
        elif sequence.dtype == torch.float:
             embedded = torch.matmul(sequence, self.embedding.weight)
        else:
             raise TypeError(f"Unsupported sequence dtype: {sequence.dtype}")

        # Add Gaussian noise to embedded input
        # Ensure noise is only added during training and stddev > 0
        if self.training and self.input_noise_stddev > 0:
            # Add noise only if input came from Generator (float type) or potentially add to real too?
            # Adding only to fake makes sense for making D's job harder on fake samples
            # Let's add to any embedded input during training
            noise = torch.randn_like(embedded) * self.input_noise_stddev
            embedded = embedded + noise
            # Optional: Consider clipping if values explode

        gru_out, _ = self.gru(embedded)

        last_output = gru_out[:, -1, :]

        last_output = self.dropout(last_output)

        logits = self.fc(last_output)

        return logits

# Example Usage (for testing the model structure)
if __name__ == '__main__':
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     latent_dim = LATENT_DIM
     batch_size = 4
     noise = torch.randn(batch_size, latent_dim).to(device)

     generator = Generator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, LATENT_DIM, MAX_LEN, vocab.pad_idx, DROPOUT_RATE).to(device)
     discriminator = Discriminator(vocab.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_DISC_CLASSES, vocab.pad_idx, DROPOUT_RATE, DISC_INPUT_NOISE_STDDEV).to(device)

     print("Generator Architecture:", generator)
     print("Discriminator Architecture:", discriminator)


     # Test generator forward
     soft_output = generator(noise, temperature=1.0)
     print(f"\nGenerator output (soft): {soft_output.shape}, dtype: {soft_output.dtype}")

     # Test discriminator with fake data (in training mode to test noise/dropout)
     discriminator.train()
     disc_output_fake = discriminator(soft_output.detach())
     print(f"Discriminator output (fake, train mode): {disc_output_fake.shape}")

     # Create dummy real data for discriminator test
     dummy_real_data = torch.randint(0, vocab.vocab_size, (batch_size, MAX_LEN), dtype=torch.long).to(device)
     disc_output_real = discriminator(dummy_real_data)
     print(f"Discriminator output (real, train mode): {disc_output_real.shape}")

     # Test discrete generation (in eval mode)
     generator.eval()
     discrete_output = generator.generate_discrete(noise)
     print(f"Generator output (discrete, eval mode): {discrete_output.shape}, dtype: {discrete_output.dtype}")