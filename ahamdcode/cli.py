from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import MODEL_NAME, __version__
from .utils.system import default_paths, project_root


def _find_checkpoint() -> Path | None:
    paths = default_paths()
    official = paths["models"] / "AhamdCode_0.1_Mini.pt"
    if official.exists():
        return official
    checkpoints = sorted(paths["checkpoints"].glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints[0] if checkpoints else None


def cmd_status(_args):
    paths = default_paths()
    tok_dir = paths["tokenizer"]
    tok_ok = (tok_dir / "vocab.json").exists()
    checkpoint = _find_checkpoint()
    model_line = "Model: Not loaded"
    params_line = "Parameters: —"
    precision_line = "Precision: —"
    checkpoint_line = f"Checkpoint: {checkpoint}" if checkpoint else "Checkpoint: none"
    try:
        from .model.model import load_checkpoint
        from .utils.memory import system_ram
        from .utils.system import cpu_info, device_info
        if checkpoint:
            try:
                model, extra = load_checkpoint(checkpoint)
                model_line = f"Model: loaded ({model.cfg.name})"
                params_line = f"Parameters: {model.count_parameters():,}"
                precision_line = f"Precision: {extra.get('precision', 'FP32')}"
            except Exception as exc:
                print(f"Checkpoint found but could not be loaded: {exc}")
        cpu = cpu_info()
        ram = system_ram()
        device = device_info()
        torch_line = f"PyTorch: {device['torch']}  device={device['preferred']}"
        cpu_line = f"CPU: {cpu.get('processor')}  cores={cpu.get('cores_logical')}"
        ram_line = f"RAM: process {ram.get('rss_mb')} MB"
    except ImportError:
        torch_line = "PyTorch: not installed"
        cpu_line = f"CPU: {sys.platform}"
        ram_line = "RAM: unavailable"

    print(f"AhamdCode version: {__version__}")
    print(f"Model name: {MODEL_NAME}")
    print(model_line)
    print(params_line)
    print(precision_line)
    print(checkpoint_line)
    print(f"Tokenizer: {'ready' if tok_ok else 'missing'} ({tok_dir})")
    print(f"Python: {sys.version.split()[0]}")
    print(torch_line)
    print(cpu_line)
    print(ram_line)
    print(f"Project path: {project_root()}")
    return 0


def cmd_model(_args):
    checkpoint = _find_checkpoint()
    if checkpoint is None:
        print("Model: Not loaded")
        print("No checkpoint found.")
        return 1
    try:
        from .model.model import load_checkpoint
        model, extra = load_checkpoint(checkpoint)
    except Exception as exc:
        print("Model: Not loaded")
        print(exc)
        return 1
    cfg = model.cfg
    print(f"Name: {cfg.name}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {cfg.n_layers}")
    print(f"Heads: {cfg.n_heads}")
    print(f"Hidden size: {cfg.d_model}")
    print(f"FFN size: {cfg.d_ff}")
    print(f"Context: {cfg.context_length}")
    print(f"Precision: {extra.get('precision', 'FP32')}")
    print(f"Checkpoint: {checkpoint}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="AhamdCode", description="Native local AhamdCode 0.1 Mini")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("status", help="show truthful runtime and model status").set_defaults(func=cmd_status)
    sub.add_parser("help", help="show this help").set_defaults(func=lambda _args: (parser.print_help() or 0))
    sub.add_parser("model", help="show checkpoint model details").set_defaults(func=cmd_model)
    return parser


def run_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help()
        return 2
    return args.func(args)
