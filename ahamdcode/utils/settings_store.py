from __future__ import annotations
import json
from pathlib import Path
from .system import default_paths
DEFAULTS = {"ram_safe": True, "cpu_threads": 2, "checkpoint_dir": "", "dataset_dir": ""}

def load_settings():
    data = dict(DEFAULTS)
    paths = default_paths()
    data["checkpoint_dir"] = str(paths["checkpoints"])
    data["dataset_dir"] = str(paths["datasets"] / "seed")
    p = paths["root"] / "settings.json"
    if p.exists():
        try:
            saved = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                data.update(saved)
        except json.JSONDecodeError:
            pass
    return data
