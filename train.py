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
NUM_EPOCHS = 500  # 增加到500
LR = 1e-4
DROPOUT_REG = 0.4  # 适度正则化

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
    a_bar = ALPHA_BAR.to(t.device)[t].view(-1, 1, 1).to(x_start.device)
    return torch.sqrt(a_bar) * x_start + torch.sqrt(1 - a_bar) * noise

def train_diffusion(epochs, batch_size, lr, model_save_path):
    dataloader, _ = create_gan_dataloader(batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(vocab_size=vocab.vocab_size,
                          embedding_dim=EMBEDDING_DIM,
                          hidden_dim=HIDDEN_DIM,
                          max_len=MAX_LEN,
                          pad_idx=vocab.pad_idx,
                          dropout_rate=DROPOUT_REG).to(device)  # 使用更大dropout
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)  # 加权重衰减
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

        # === 每10个epoch做重建率和采样可视化 ===
        if (epoch+1) % 10 == 0:
            # 重建率
            real_batch, _ = next(iter(dataloader))
            real_batch = real_batch[:16].to(device)
            recon_acc = calc_recon_acc(model, real_batch, device)
            print(f"[Eval] Reconstruction Accuracy (t=0, batch=16): {recon_acc:.4f}")
            # 采样
            gen_x = sample_ddpm(model, NUM_DIFFUSION_STEPS, (8, MAX_LEN, vocab.vocab_size), device)
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=0.9)
            print("[Sample] Example generated sequences:")
            for idx in range(min(4, gen_tokens.size(0))):
                seq = gen_tokens[idx].cpu().tolist()
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                print(f"  {idx+1}: {aa_seq}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining finished. Model saved to {model_save_path}")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds.")

def sample_ddpm(model, num_steps, shape, device):
    """DDPM采样流程，返回(batch, seq, vocab_size)"""
    x = torch.randn(shape, device=device)
    for t_ in reversed(range(num_steps)):
        t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t)
        a_bar = ALPHA_BAR.to(device)[t_]
        a = ALPHA.to(device)[t_]
        if t_ > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) + torch.sqrt(BETA_SCHEDULE.to(device)[t_]) * noise
    return x

def onehot_to_token(x):
    # x: (batch, seq, vocab_size) -> (batch, seq)
    return torch.argmax(x, dim=-1)

def onehot_to_token_with_temperature(x, temperature=0.9):
    # x: (batch, seq, vocab_size)
    probs = torch.softmax(x / temperature, dim=-1)
    # 多项分布采样
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled

def trim_pad(seq, pad_idx):
    # 截断到第一个pad
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq

def calc_recon_acc(model, real_sequences, device):
    # 只做t=0的重建（即直接送入模型反向扩散）
    model.eval()
    with torch.no_grad():
        x_start = to_onehot(real_sequences, vocab.vocab_size).to(device)
        # 直接加极小噪声，t=0
        t = torch.zeros(real_sequences.size(0), dtype=torch.long, device=device)
        noise = torch.randn_like(x_start) * 1e-4
        x_noisy = q_sample(x_start, t, noise)
        pred_noise = model(x_noisy, t)
        # 反推x0
        a_bar = ALPHA_BAR.to(device)[0]
        x0_pred = (x_noisy - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)
        tokens_pred = onehot_to_token(x0_pred)
        acc = (tokens_pred == real_sequences).float().mean().item()
    model.train()
    return acc

if __name__ == '__main__':
    model_path = os.path.join(MODELS_DIR, DIFFUSION_MODEL_FILE)
    train_diffusion(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, lr=LR, model_save_path=model_path)