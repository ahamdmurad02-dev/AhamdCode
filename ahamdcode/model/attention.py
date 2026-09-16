from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import apply_rope, build_rope_cache

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, rope=True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = rope
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._rope_cache = None
    def _get_rope(self, seq_len, device, dtype):
        if self._rope_cache is None or self._rope_cache[0] < seq_len or self._rope_cache[1].device != device:
            cos, sin = build_rope_cache(seq_len, self.head_dim, device, dtype)
            self._rope_cache = (seq_len, cos, sin)
        _, cos, sin = self._rope_cache
        return cos.to(dtype=dtype), sin.to(dtype=dtype)
    def forward(self, x, attn_mask=None):
        b, t, _ = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.rope:
            cos, sin = self._get_rope(t, x.device, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scale = 1.0 / math.sqrt(self.head_dim)
        att = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(causal, float("-inf"))
        if attn_mask is not None:
            att = att.masked_fill(attn_mask == 0, float("-inf"))
        att = self.dropout(F.softmax(att, dim=-1))
        y = torch.matmul(att, v).transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.proj(y)
