from __future__ import annotations
from dataclasses import asdict, dataclass
from typing import Any

@dataclass
class ModelConfig:
    name: str = "AhamdCode 0.1 Mini"
    variant: str = "mini_default"
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    context_length: int = 256
    vocab_size: int = 4096
    dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    tie_embeddings: bool = True
    activation: str = "swiglu"
    rope: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    @classmethod
    def mini_default(cls) -> "ModelConfig":
        return cls(variant="mini_default", n_layers=4, n_heads=4, d_model=128, d_ff=512, context_length=256)

    @classmethod
    def mini_fast(cls) -> "ModelConfig":
        return cls(name="AhamdCode 0.1 Mini Fast", variant="mini_fast", n_layers=3, n_heads=4, d_model=96, d_ff=384, context_length=256)

    @classmethod
    def mini_custom(cls, n_layers, n_heads, d_model, d_ff, context_length, vocab_size=4096) -> "ModelConfig":
        return cls(name="AhamdCode 0.1 Mini Custom", variant="mini_custom", n_layers=n_layers, n_heads=n_heads, d_model=d_model, d_ff=d_ff, context_length=context_length, vocab_size=vocab_size)

PRESETS = {"Mini Default": ModelConfig.mini_default, "Mini Fast": ModelConfig.mini_fast}
