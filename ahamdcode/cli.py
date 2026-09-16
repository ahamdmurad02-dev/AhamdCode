from __future__ import annotations
import argparse, sys
from pathlib import Path
from . import MODEL_NAME, __version__
from .model.model import load_checkpoint
from .runtime import RUNTIME
from .tokenizer.tokenizer import default_tokenizer_dir
from .utils.memory import system_ram
from .utils.system import cpu_info, default_paths, device_info, project_root

def _find_checkpoint():
    official = default_paths()["models"] / "AhamdCode_0.1_Mini.pt"
    if official.exists():
        return official
    ckpts = sorted(default_paths()["checkpoints"].glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0] if ckpts else None

def cmd_status(_):
    root = project_root()
    paths = default_paths()
    tok_dir = paths["tokenizer"]
    tok_ok = (tok_dir / "vocab.json").exists()
    if tok_ok:
        try:
            RUNTIME.load_tokenizer(tok_dir)
        except Exception as exc:
            print("Tokenizer: found but failed to load", exc)
            tok_ok = False
    ckpt = _find_checkpoint()
    model_line = "Model: Not loaded"
    params_line = "Parameters: —"
    precision_line = "Precision: —"
    ckpt_line = "Checkpoint: none"
    if ckpt is not None:
        ckpt_line = f"Checkpoint: {ckpt}"
        try:
            model, extra = load_checkpoint(ckpt)
            model_line = f"Model: loaded ({model.cfg.name})"
            params_line = f"Parameters: {model.count_parameters():,}"
            precision_line = f"Precision: {extra.get('precision', 'FP32')}"
        except Exception as exc:
            print("Checkpoint found but could not be loaded:", exc)
    cpu = cpu_info(); ram = system_ram(); dev = device_info()
    print(f"AhamdCode version: {__version__}")
    print(f"Model name: {MODEL_NAME}")
    print(model_line); print(params_line); print(precision_line); print(ckpt_line)
    print(f"Tokenizer: {'ready' if tok_ok else 'missing'} ({tok_dir})")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {dev['torch']}  device={dev['preferred']}")
    print(f"CPU: {cpu.get('processor')}  cores={cpu.get('cores_logical')}")
    print(f"RAM: process {ram.get('rss_mb')} MB")
    print(f"Project path: {root}")
    return 0

def cmd_help(parser):
    def _inner(_):
        parser.print_help(); return 0
    return _inner

def cmd_model(_):
    ckpt = _find_checkpoint()
    if ckpt is None:
        print("Model: Not loaded"); print("No checkpoint found."); return 1
    try:
        model, extra = load_checkpoint(ckpt)
    except Exception as exc:
        print("Model: Not loaded"); print(exc); return 1
    cfg = model.cfg
    print(f"Name: {cfg.name}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Layers: {cfg.n_layers}")
    print(f"Heads: {cfg.n_heads}")
    print(f"Hidden size: {cfg.d_model}")
    print(f"FFN size: {cfg.d_ff}")
    print(f"Context: {cfg.context_length}")
    print(f"Precision: {extra.get('precision', 'FP32')}")
    print(f"Checkpoint: {ckpt}")
    return 0

def build_parser():
    p = argparse.ArgumentParser(prog="AhamdCode")
    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("status"); s.set_defaults(func=cmd_status)
    h = sub.add_parser("help"); h.set_defaults(func=cmd_help(p))
    m = sub.add_parser("model"); m.set_defaults(func=cmd_model)
    return p

def run_cli(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help(); return 2
    return args.func(args)
