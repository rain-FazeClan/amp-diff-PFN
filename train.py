import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from utils import vocab, MAX_LEN, PAD_TOKEN
from data_loader import create_gan_dataloader
from model import DiffusionModel, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE, get_diffusion_beta_schedule

# --- Configuration ---
BATCH_SIZE = 128
NUM_EPOCHS = 300
LR = 1e-4

# Model saving
MODELS_DIR = 'models'
DIFFUSION_MODEL_FILE = 'diffusion_model_transformer.pth'

# Diffusion parameters
NUM_DIFFUSION_STEPS = 1000
BETA_SCHEDULE = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS)
ALPHA = 1. - BETA_SCHEDULE
ALPHA_BAR = torch.cumprod(ALPHA, dim=0)

def to_onehot(x, vocab_size):
    # x: (batch, seq) int64 -> (batch, seq, vocab_size)
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()

def q_sample(x_start, t, noise):
    # x_start: (batch, seq, vocab_size) one-hot
    # t: (batch,) int64
    # noise: (batch, seq, vocab_size)
    # Returns the noised x_t
    batch, seq, vocab_size = x_start.shape
    a_bar = ALPHA_BAR[t].view(-1, 1, 1).to(x_start.device)
    return torch.sqrt(a_bar) * x_start + torch.sqrt(1 - a_bar) * noise

def train_diffusion(epochs, batch_size, lr, model_save_path):
    dataloader, _ = create_gan_dataloader(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_RATE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    print("Starting Diffusion Model training...")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (real_sequences, _) in enumerate(dataloader):
            real_sequences = real_sequences.to(device)  # (batch, seq)
            x_start = to_onehot(real_sequences, vocab.vocab_size)  # (batch, seq, vocab_size)
            batch_size = x_start.size(0)
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = q_sample(x_start, t, noise)
            pred_noise = model(x_noisy, t)
            loss = mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i+1) % 100 == 0:
                print(f"Epoch {epoch} Step {i+1} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss/(i+1):.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining finished. Model saved to {model_save_path}")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds.")

if __name__ == '__main__':
    model_path = os.path.join(MODELS_DIR, DIFFUSION_MODEL_FILE)
    train_diffusion(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, model_save_path=model_path)