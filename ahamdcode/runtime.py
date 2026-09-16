from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from .model.config import ModelConfig
from .model.model import build_model, load_checkpoint
from .model.transformer import MiniTransformer
from .tokenizer.tokenizer import CodeTokenizer, default_tokenizer_dir
from .utils.memory import process_rss_mb
from .utils.system import cpu_info, default_paths, device_info

@dataclass
class Runtime:
    model: MiniTransformer | None = None
    tokenizer: CodeTokenizer | None = None
    checkpoint_path: str = ""
    precision: str = "FP32"
    recent_projects: list[str] = field(default_factory=list)
    ram_safe: bool = True
    def load_tokenizer(self, directory=None):
        self.tokenizer = CodeTokenizer.load(directory or default_tokenizer_dir())
        return self.tokenizer
    def load_model(self, path):
        model, extra = load_checkpoint(path, map_location="cpu")
        self.model = model
        self.checkpoint_path = str(path)
        self.precision = str(extra.get("precision", "FP32"))
        tok = default_tokenizer_dir()
        if (Path(tok) / "vocab.json").exists():
            self.load_tokenizer(tok)
        return model
    def unload_model(self):
        self.model = None
        self.checkpoint_path = ""
        self.precision = "FP32"
    def discover_checkpoint(self):
        paths = default_paths()
        official = paths["models"] / "AhamdCode_0.1_Mini.pt"
        if official.exists():
            return official
        found = []
        for folder in (paths["models"], paths["checkpoints"]):
            if folder.exists():
                found.extend(folder.glob("*.pt"))
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)[0] if found else None
    def parameter_count(self):
        return 0 if self.model is None else self.model.count_parameters()
    def status_dict(self):
        cfg = self.model.cfg if self.model is not None else None
        return {
            "model_name": cfg.name if cfg else "—",
            "loaded": self.model is not None,
            "checkpoint": self.checkpoint_path or "—",
            "parameters": self.parameter_count(),
            "precision": self.precision if self.model is not None else "—",
            "cpu": cpu_info().get("processor"),
            "ram_mb": process_rss_mb(),
            "device": device_info()["preferred"],
            "tokenizer_ready": self.tokenizer is not None,
            "project_path": str(default_paths()["root"]),
        }
RUNTIME = Runtime()
