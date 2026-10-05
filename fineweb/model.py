"""
GPT model. Features:
- RoPE
- norm after QK
- norm after token embedding
- untied weights for embedding and head
- fixed residual scaling of 1/sqrt(2*L)
- no bias in linear layers
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def precompute_freqs_cis(dim, end, theta=10_000.0):
    """Precompute RoPE frequencies."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(end).float()
    freqs = torch.outer(t, freqs)
    return torch.stack([freqs.cos(), freqs.sin()], dim=-1)  # (end, dim//2, 2)


def apply_rotary_emb(x, freqs_cis):
    """Apply RoPE to tensor x of shape (B, n_heads, T, d_head)."""
    # x: (B, H, T, D) -> (B, H, T, D//2, 2)
    x_ = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = freqs_cis[:, :, 0].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D//2)
    sin = freqs_cis[:, :, 1].unsqueeze(0).unsqueeze(0)
    x_rot = torch.stack(
        [
            x_[..., 0] * cos - x_[..., 1] * sin,
            x_[..., 0] * sin + x_[..., 1] * cos,
        ],
        dim=-1,
    )
    return x_rot.flatten(-2).type_as(x)


class Attention(nn.Module):
    def __init__(self, dim, n_heads=8):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.d_head = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.norm_q = nn.RMSNorm(self.d_head, eps=1e-6)
        self.norm_k = nn.RMSNorm(self.d_head, eps=1e-6)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, freqs_cis):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.norm_q(q), self.norm_k(k)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)
        x = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x).square()
        x = self.fc2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.att = Attention(dim, n_heads=n_heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, freqs_cis):
        x = x + self.att(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size=50257,
        seq_len=1024,
        n_layers=6,
        n_heads=6,
        d_embds=384,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.d_embds = d_embds

        self.embd = nn.Embedding(vocab_size, d_embds)
        self.norm_embd = nn.RMSNorm(d_embds, eps=1e-6)
        self.blocks = nn.ModuleList(
            [Transformer(d_embds, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm_head = nn.RMSNorm(d_embds, eps=1e-6)
        self.head = nn.Linear(d_embds, vocab_size, bias=False)

        freqs_cis = precompute_freqs_cis(d_embds // n_heads, seq_len)
        self.register_buffer("freqs_cis", freqs_cis)
        self._init_weights()

    def forward(self, idx):
        B, T = idx.shape
        x = self.embd(idx)
        x = self.norm_embd(x)
        f = self.freqs_cis[:T]
        for block in self.blocks:
            x = block(x, f)
        x = self.norm_head(x)
        x = self.head(x)
        return x

    def _init_weights(self):
        nn.init.normal_(self.embd.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.001)
        scale = 1.0 / math.sqrt(2 * len(self.blocks))
        for block in self.blocks:
            nn.init.xavier_uniform_(block.att.qkv.weight)
            nn.init.xavier_uniform_(block.att.proj.weight)
            block.att.proj.weight.data *= scale
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            block.mlp.fc2.weight.data *= scale


def gpt_small(seq_len=1024, n_layers=6, n_heads=6, d_embds=384):
    return GPT(
        vocab_size=50257,
        seq_len=seq_len,
        n_layers=n_layers,
        n_heads=n_heads,
        d_embds=d_embds,
        mlp_ratio=4.0,
    )


if __name__ == "__main__":
    model = gpt_small(seq_len=256, n_layers=6)
    idx = torch.randint(0, 50257, (2, 256))
    logits = model(idx)
    print(f"Input shape: {idx.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
