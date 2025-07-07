import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import vocab, MAX_LEN, PAD_TOKEN

# 超参数
EMBEDDING_DIM = 64
HIDDEN_DIM = 256
LATENT_DIM = 100
DROPOUT_RATE = 0.5

# Diffusion超参数
NUM_DIFFUSION_STEPS = 1000

class SinusoidalPosEmb(nn.Module):
    """扩散模型常用的正弦时间步编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim-1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class DiffusionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, max_len, pad_idx, dropout_rate):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim))
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dim_feedforward=hidden_dim*4, dropout=dropout_rate, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, t):
        # x: (batch, seq_len, vocab_size) 软one-hot
        # t: (batch,) int64, 时间步
        emb = torch.matmul(x, self.embedding.weight)  # (batch, seq, emb)
        emb = emb + self.positional_encoding[:, :emb.size(1), :]
        t_emb = self.time_mlp(t)  # (batch, emb)
        t_emb = t_emb.unsqueeze(1).repeat(1, emb.size(1), 1)
        h = emb + t_emb
        h = self.transformer(h)
        h = self.dropout(h)
        out = self.fc_out(h)  # (batch, seq, vocab_size)
        return out

# 扩散噪声调度器（beta/timestep等）
def get_diffusion_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

# 其余辅助函数可在train.py实现