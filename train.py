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
    parser.add_argument('--batch_size', type=int, default=64,  # 减小batch size
                        help=f'Batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200,  # 减少训练轮数
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=3e-5,  # 降低学习率
                        help='Learning rate (default: 3e-5)')
    parser.add_argument('--dropout', type=float, default=0.7,  # 大幅增加dropout
                        help='Dropout rate for regularization (default: 0.7)')
    parser.add_argument('--weight_decay', type=float, default=5e-3,  # 大幅增加权重衰减
                        help='Weight decay for regularization (default: 5e-3)')
    parser.add_argument('--early_stop_patience', type=int, default=15,  # 减少耐心
                        help='Early stopping patience (default: 15)')
    parser.add_argument('--min_loss_improvement', type=float, default=1e-3,  # 提高最小改善阈值
                        help='Minimum loss improvement for early stopping (default: 1e-3)')
    parser.add_argument('--eval_diversity', action='store_true',
                        help='Evaluate generation diversity')
    parser.add_argument('--label_smoothing', type=float, default=0.2,  # 添加标签平滑
                        help='Label smoothing factor (default: 0.2)')
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


def train_diffusion(epochs, batch_size, lr, dropout_reg, max_len, model_save_path, weight_decay=5e-3,
                    early_stop_patience=15, min_loss_improvement=1e-3, eval_diversity=False, label_smoothing=0.2):
    # 使用更强的噪声调度
    beta_schedule = get_diffusion_beta_schedule(NUM_DIFFUSION_STEPS, beta_start=1e-3, beta_end=0.05)
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

    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)
    mse_loss = nn.MSELoss()

    # 早停相关变量
    best_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    print(f"Starting Diffusion Model training...")
    print(f"Parameters: max_len={max_len}, batch_size={batch_size}, epochs={epochs}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"Early stopping: patience={early_stop_patience}, min_improvement={min_loss_improvement}")
    print(f"Regularization: dropout={dropout_reg}, weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for i, (real_sequences, _) in enumerate(dataloader):
            real_sequences = real_sequences.to(device)
            x_start = to_onehot(real_sequences, vocab.vocab_size)

            # 添加标签平滑，防止one-hot过于尖锐
            x_start = x_start * (1 - label_smoothing) + label_smoothing / vocab.vocab_size

            batch_size_actual = x_start.size(0)
            # 避免使用t=0，增加时间步采样的随机性
            t = torch.randint(20, NUM_DIFFUSION_STEPS, (batch_size_actual,), device=device).long()

            # 增加噪声强度和随机性
            noise_scale = torch.rand(batch_size_actual, 1, 1, device=device) * 1.5 + 0.5  # 0.5-2.0倍
            noise = torch.randn_like(x_start) * noise_scale
            x_noisy = q_sample(x_start, t, noise, alpha_bar)
            pred_noise = model(x_noisy, t)
            loss = mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 更严格的梯度裁剪
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / (i + 1)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # 早停检查
        if avg_loss < best_loss - min_loss_improvement:
            best_loss = avg_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"  → No improvement. Patience: {patience_counter}/{early_stop_patience}")

        # 早停触发
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"Best loss: {best_loss:.4f}")
            break

        # 每8轮评估一次，减少评估频率
        if (epoch + 1) % 8 == 0:
            real_batch, _ = next(iter(dataloader))
            real_batch = real_batch[:16].to(device)

            # 多时间步重构准确率评估
            recon_results = calc_recon_acc_multi_timesteps(model, real_batch, device, alpha_bar,
                                                           timesteps=[20, 50, 100, 200])  # 避免t=0和t=10
            print(f"[Eval] Reconstruction Accuracy:")
            for time_step, acc in recon_results.items():
                print(f"  {time_step}: {acc:.4f}")

            # 生成样本并评估多样性
            gen_x = sample_ddpm(model, NUM_DIFFUSION_STEPS, (32, max_len, vocab.vocab_size),
                                device, alpha, alpha_bar, beta_schedule)
            gen_tokens = onehot_to_token_with_temperature(gen_x, temperature=2.5)  # 更高温度

            generated_sequences = []
            valid_sequences = []
            print("[Sample] Example generated sequences:")

            for idx in range(gen_tokens.size(0)):
                seq = gen_tokens[idx].cpu().tolist()
                seq = trim_pad(seq, vocab.pad_idx)
                aa_seq = vocab.decode(seq)
                if len(aa_seq) >= 5:  # 只考虑长度>=5的序列
                    valid_sequences.append(aa_seq)
                    if len(generated_sequences) < 4:
                        generated_sequences.append(aa_seq)
                        print(f"  {len(generated_sequences)}: {aa_seq}")

            # 计算有效序列的多样性
            if len(valid_sequences) > 0:
                diversity_score = len(set(valid_sequences)) / len(valid_sequences)
                avg_length = sum(len(seq) for seq in valid_sequences) / len(valid_sequences)
                print(f"[Diversity] Valid sequences: {len(valid_sequences)}/{gen_tokens.size(0)}")
                print(f"[Diversity] Generation diversity: {diversity_score:.4f}, Avg length: {avg_length:.1f}")
            else:
                print(f"[Warning] No valid sequences generated!")

    # 使用最佳模型状态保存
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with loss: {best_loss:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTraining finished. Model saved to {model_save_path}")
    print(f"Total training time: {(time.time() - start_time):.2f} seconds.")


def sample_ddpm(model, num_steps, shape, device, alpha, alpha_bar, beta_schedule):
    model.eval()
    x = torch.randn(shape, device=device)

    with torch.no_grad():
        for t_ in reversed(range(num_steps)):
            t = torch.full((shape[0],), t_, device=device, dtype=torch.long)
            pred_noise = model(x, t)
            a_bar = alpha_bar.to(device)[t_]
            a = alpha.to(device)[t_]

            if t_ > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(a)) * (x - (1 - a) / torch.sqrt(1 - a_bar) * pred_noise) + torch.sqrt(
                beta_schedule.to(device)[t_]) * noise

    model.train()
    return x


def onehot_to_token_with_temperature(x, temperature=1.0):
    probs = torch.softmax(x / temperature, dim=-1)
    batch, seq, vocab_size = probs.shape
    probs_2d = probs.view(-1, vocab_size)
    sampled = torch.multinomial(probs_2d, 1).view(batch, seq)
    return sampled


def trim_pad(seq, pad_idx):
    return seq[:seq.index(pad_idx)] if pad_idx in seq else seq


def calc_recon_acc_multi_timesteps(model, real_sequences, device, alpha_bar, timesteps=[20, 50, 100, 200]):
    """在多个时间步评估重构准确率，避免t=0"""
    model.eval()
    results = {}

    with torch.no_grad():
        x_start = to_onehot(real_sequences, vocab.vocab_size).to(device)

        for t_val in timesteps:
            t = torch.full((real_sequences.size(0),), t_val, dtype=torch.long, device=device)
            noise = torch.randn_like(x_start)
            x_noisy = q_sample(x_start, t, noise, alpha_bar)
            pred_noise = model(x_noisy, t)

            a_bar = alpha_bar.to(device)[t_val]
            x0_pred = (x_noisy - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)
            tokens_pred = torch.argmax(x0_pred, dim=-1)
            acc = (tokens_pred == real_sequences).float().mean().item()
            results[f't={t_val}'] = acc

    model.train()
    return results


def calc_diversity_score(generated_tokens, vocab_obj):
    """计算生成序列的多样性分数"""
    sequences = []
    for i in range(generated_tokens.size(0)):
        seq = generated_tokens[i].cpu().tolist()
        seq = trim_pad(seq, vocab_obj.pad_idx)
        aa_seq = vocab_obj.decode(seq)
        if len(aa_seq) >= 5:  # 只考虑长度>=5的序列
            sequences.append(aa_seq)

    if len(sequences) == 0:
        return 0.0

    # 计算唯一序列比例
    unique_sequences = set(sequences)
    diversity = len(unique_sequences) / len(sequences)
    return diversity


if __name__ == '__main__':
    args = parse_args()
    model_path = os.path.join(MODELS_DIR, DIFFUSION_MODEL_FILE)
    train_diffusion(epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    dropout_reg=args.dropout,
                    max_len=args.max_len,
                    model_save_path=model_path,
                    weight_decay=args.weight_decay,
                    early_stop_patience=args.early_stop_patience,
                    min_loss_improvement=args.min_loss_improvement,
                    eval_diversity=args.eval_diversity,
                    label_smoothing=args.label_smoothing)