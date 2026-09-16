from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        norm = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(norm + self.eps)
        return (self.weight * x_normed).to(x.dtype)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, activation="swiglu", dropout=0.0):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False) if activation == "swiglu" else None
    def forward(self, x):
        if self.activation == "swiglu":
            hidden = F.silu(self.w_gate(x)) * self.w1(x)
            return self.dropout(self.w2(hidden))
        return self.dropout(self.w2(F.gelu(self.w1(x))))

def build_rope_cache(seq_len, head_dim, device, dtype):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, theta)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)

def apply_rope(x, cos, sin):
    t = x.size(-2)
    cos, sin = cos[:t], sin[:t]
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot1 = x1 * cos.unsqueeze(0).unsqueeze(0) - x2 * sin.unsqueeze(0).unsqueeze(0)
    rot2 = x1 * sin.unsqueeze(0).unsqueeze(0) + x2 * cos.unsqueeze(0).unsqueeze(0)
    return torch.stack((rot1, rot2), dim=-1).flatten(-2)
