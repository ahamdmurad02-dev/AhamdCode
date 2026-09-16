from __future__ import annotations
import os, platform, sys
from pathlib import Path
import torch

def _looks_like_root(path: Path) -> bool:
    return (path / "app.py").is_file() and (path / "ahamdcode").is_dir()

def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents, Path.cwd(), *Path.cwd().parents]:
        if _looks_like_root(p):
            return p
        nested = p / "AhamdCode"
        if _looks_like_root(nested):
            return nested
    return here.parents[2]

def resolve_project_path(path=None, *parts):
    if path is None or str(path).strip() == "":
        return project_root().joinpath(*parts) if parts else project_root()
    p = Path(path)
    return p if p.is_absolute() else project_root() / p

def cpu_info():
    return {"source": "python", "processor": platform.processor() or platform.machine(), "machine": platform.machine(), "system": platform.system(), "cores_logical": os.cpu_count() or 1}

def device_info():
    cuda = bool(torch.cuda.is_available())
    return {"torch": torch.__version__, "cuda_available": cuda, "cuda_device": torch.cuda.get_device_name(0) if cuda else None, "preferred": "cuda" if cuda else "cpu"}

def default_paths():
    root = project_root()
    return {"root": root, "models": root / "models", "checkpoints": root / "checkpoints", "datasets": root / "datasets", "tokenizer": root / "tokenizer", "projects": root / "projects", "config": root / "config", "tests": root / "tests"}
