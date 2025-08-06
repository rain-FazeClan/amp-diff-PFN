import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import argparse
from utils import vocab, PAD_TOKEN, DEFAULT_MAX_LEN, DEFAULT_BATCH_SIZE
from data_loader import create_diffusion_dataloader
from model import DiffusionModel, EMBEDDING_DIM, HIDDEN_DIM, DROPOUT_RATE, get_diffusion_beta_schedule


def parse_args():
    parser = argparse.ArgumentParser(description='Train Diffusion Model for Peptide Generation')
    parser.add_argument('--max_len', type=int, default=DEFAULT_MAX_LEN,
                        help=f'Maximum sequence length (default: {DEFAULT_MAX_LEN})')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help=f'Batch size for training (default: {DEFAULT_BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='Dropout rate for regularization (default: 0.4)')
    return parser.parse_args()


# Model saving
MODELS_DIR = 'models'
DIFFUSION_MODEL_FILE = 'diffusion_model_transformer.pth'

# Diffusion parameters
NUM_DIFFUSION_STEPS = 1000


def to_onehot(x, vocab_size):
    return torch.nn.functional.one_hot(x, num_classes=vocab_size).float()


def q_sample(x_start, t, noise, alpha_bar):
    batch, seq, vocab_size = x_start.shape
    a_bar = alpha_bar.to(t.device)[t].view(-1, 1, 1).to(x_start.device)
    return torch.sqrt(a_bar) * x_start + torch.sqrt(1 - a_bar) * noise


def train_diffusion(epochs, batch_size, lr, dropout_reg, max_len, model_save_path):
    # 初始化扩散调度
    beta_schedule = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS)
    alpha = 1. - beta_schedule
    alpha_bar = torch.cumprod(alpha, dim=0)

    dataloader, _ = create_diffusion_dataloader(batch_size, max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiffusionModel(vocab_size=vocab.vocab_size,
                           embedding_dim=EMBEDDING_DIM,
                           hidden_dim=HIDDEN_DIM,
                           max_len=max_len,
                           pad_idx=vocab.pad_idx,
                           dropout_rate=dropout_reg).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    print(f"Starting Diffusion Model training...")
    print(f"Parameters: max_len={max_len}, batch_size={batch_size}, epochs={epochs}")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        for i, (real_sequences, _) in enumerate(dataloader):
            real_sequences = real_sequences.to(device)
            x_start = to_onehot(real_sequences, vocab.vocab_size)
            batch_size_actual = x_start.size(0)
            t = torch.randint(0, NUM_DIFFUSION_STEPS, (batch_size_actual,), device=device).long()
            noise = torch.randn_like(x_start)
            x_noisy = q_sample(x_start, t, noise, alpha_bar)
            pred_noise = model(x_noisy, t)
            loss = mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch} Step {i + 1} Loss: {loss.item():.4f}")

        print(f"Epoch {epoch} finished. Avg Loss: {epoch_loss / (i + 1):.4f}")

        if (epoch + 1) % 10 == 0:
            real_batch, _ = next(iter(dataloader))
            real_batch = real_batch[:16].to(device)
            recon_acc = calc_recon_acc(model, real_batch, device, alpha_bar)
            print(f"[Eval] Reconstruction Accuracy (t=0, batch=16): {recon_acc:.4f}")

            gen_x = sample_ddpm(model, NUM_DIFFUSION_STEPS, (8, max_len, vocab.vocab_size), device, alpha, alpha_bar,
                                beta_schedule)
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=0.9)
            print("[Sample] Example generated sequences:")
            for idx in range(min(4, gen_tokens.size(0))):
                seq = gen_tokens[idx].cpu().tolist()
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                print(f"  {idx + 1}: {aa_seq}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining finished. Model saved to {model_save_path}")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds.")


def sample_ddpm(model, num_steps, shape, device, alpha, alpha_bar, beta_schedule):
    x = torch.randn(shape, device=device)
    for t_ in reversed(range(num_steps)):
        t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
        with torch.no_grad():
            pred_noise = model(x, t)
        a_bar = alpha_bar.to(device)[t_]
        a = alpha.to(device)[t_]
        if t_ > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)
        x = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) + torch.sqrt(
            beta_schedule.to(device)[t_]) * noise
    return x


def onehot_to_token_with_temperature(x, temperature=0.9):
    probs = torch.softmax(x / temperature, dim=-1)
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled


def trim_pad(seq, pad_idx):
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq


def calc_recon_acc(model, real_sequences, device, alpha_bar):
    model.eval()
    with torch.no_grad():
        x_start = to_onehot(real_sequences, vocab.vocab_size).to(device)
        t = torch.zeros(real_sequences.size(0), dtype=torch.long, device=device)
        noise = torch.randn_like(x_start) * 1e-4
        x_noisy = q_sample(x_start, t, noise, alpha_bar)
        pred_noise = model(x_noisy, t)
        a_bar = alpha_bar.to(device)[0]
        x0_pred = (x_noisy - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)
        tokens_pred = torch.argmax(x0_pred, dim=-1)
        acc = (tokens_pred == real_sequences).float().mean().item()
    model.train()
    return acc


if __name__ == '__main__':
    args = parse_args()
    model_path = os.path.join(MODELS_DIR, DIFFUSION_MODEL_FILE)
    train_diffusion(epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout_reg=args.dropout,
                    max_len=args.max_len,
                    model_save_path=model_path)