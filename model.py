import torch
import torch.nn as nn

# 保持当前超参数，重点加强正则化
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
LATENT_DIM = 100
DROPOUT_RATE = 0.7  # 大幅增加dropout防止过拟合

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
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / (half_dim - 1)))
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
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, embedding_dim) * 0.02)  # 减小初始化方差

        # 时间编码层，增加更多dropout和正则化
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(embedding_dim * 2),  # 添加层归一化
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Dropout(dropout_rate * 0.5),  # 额外dropout
        )

        # Transformer结构，减少层数防止过拟合
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,  # 减少注意力头数
            dim_feedforward=hidden_dim // 2,  # 减小前馈网络维度
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)  # 减少层数

        # 输出层，增加更多正则化
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, vocab_size)
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """更保守的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=0.5)  # 更小的增益
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

    def forward(self, x, t):
        # x: (batch, seq_len, vocab_size) 软one-hot
        # t: (batch,) int64, 时间步
        emb = torch.matmul(x, self.embedding.weight)  # (batch, seq, emb)
        emb = emb + self.positional_encoding[:, :emb.size(1), :]

        t_emb = self.time_mlp(t)  # (batch, emb)
        t_emb = t_emb.unsqueeze(1).repeat(1, emb.size(1), 1)

        h = emb + t_emb
        h = self.layer_norm(h)
        h = self.dropout(h)
        h = self.transformer(h)

        h = self.dropout(h)  # 额外dropout
        out = self.fc_out(h)  # (batch, seq, vocab_size)
        return out


# 更强的扩散噪声调度器
def get_diffusion_beta_schedule(T, beta_start=1e-3, beta_end=0.05):
    """增加噪声强度的调度器"""
    return torch.linspace(beta_start, beta_end, T)