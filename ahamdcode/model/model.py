from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import torch
from .config import ModelConfig
from .transformer import MiniTransformer
CHECKPOINT_VERSION = 1

def build_model(cfg: ModelConfig | None = None) -> MiniTransformer:
    return MiniTransformer(cfg or ModelConfig.mini_default())

def save_checkpoint(model: MiniTransformer, path: str | Path, optimizer=None, extra=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_name": model.cfg.name,
        "config": model.cfg.to_dict(),
        "state_dict": model.state_dict(),
        "parameter_count": model.count_parameters(),
        "precision": _infer_precision(model),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    path.with_suffix(".json").write_text(json.dumps({"model_name": payload["model_name"], "parameter_count": payload["parameter_count"], "precision": payload["precision"], "config": payload["config"]}, indent=2), encoding="utf-8")
    return path

def load_checkpoint(path, map_location="cpu", load_optimizer=False):
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig.from_dict(payload["config"])
    model = MiniTransformer(cfg)
    model.load_state_dict(payload["state_dict"], strict=True)
    extra = payload.get("extra") or {}
    extra["optimizer_state"] = payload.get("optimizer") if load_optimizer else None
    extra["precision"] = payload.get("precision", _infer_precision(model))
    extra["parameter_count"] = payload.get("parameter_count", model.count_parameters())
    return model, extra

def _infer_precision(model) -> str:
    dtypes = {p.dtype for p in model.parameters()}
    if torch.float16 in dtypes:
        return "FP16"
    return "FP32"

def quantize_dynamic_int8(model: MiniTransformer):
    model_cpu = model.cpu().eval()
    return torch.ao.quantization.quantize_dynamic(model_cpu, {torch.nn.Linear}, dtype=torch.qint8)

def cast_precision(model: MiniTransformer, precision: str) -> MiniTransformer:
    precision = precision.upper()
    if precision == "FP16":
        return model.half()
    if precision == "FP32":
        return model.float()
    if precision == "INT8":
        return quantize_dynamic_int8(model)
    raise ValueError(precision)
