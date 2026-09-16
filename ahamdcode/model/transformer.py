from __future__ import annotations
import torch
import torch.nn as nn
from .attention import CausalSelfAttention
from .config import ModelConfig
from .layers import FeedForward, RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout, cfg.rope)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.activation, cfg.dropout)
    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = None if cfg.rope else nn.Embedding(cfg.context_length, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_embed.weight
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids, labels=None):
        b, t = input_ids.shape
        x = self.tok_embed(input_ids)
        if self.pos_embed is not None:
            pos = torch.arange(t, device=input_ids.device)
            x = x + self.pos_embed(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            vocab = logits.size(-1)
            loss = torch.nn.functional.cross_entropy(logits[:, :-1].contiguous().view(-1, vocab), labels[:, 1:].contiguous().view(-1), ignore_index=self.cfg.pad_token_id)
        return {"logits": logits, "loss": loss}
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
