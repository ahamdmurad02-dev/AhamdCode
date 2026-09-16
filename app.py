#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ahamdcode.runtime import RUNTIME
from ahamdcode.tokenizer.tokenizer import default_tokenizer_dir
from ahamdcode.utils.memory import apply_ram_safe_mode
from ahamdcode.utils.settings_store import load_settings
from ahamdcode.utils.system import default_paths

def ensure_dirs():
    for p in default_paths().values():
        Path(p).mkdir(parents=True, exist_ok=True)

def bootstrap_runtime():
    ensure_dirs()
    settings = load_settings()
    apply_ram_safe_mode(bool(settings.get('ram_safe', True)), int(settings.get('cpu_threads', 2)))
    tok = default_tokenizer_dir()
    if (tok / 'vocab.json').exists():
        RUNTIME.load_tokenizer(tok)
    official = default_paths()['models'] / 'AhamdCode_0.1_Mini.pt'
    if official.exists():
        try:
            RUNTIME.load_model(official)
            return
        except Exception:
            pass
    ckpt_dir = Path(settings.get('checkpoint_dir') or default_paths()['checkpoints'])
    latest = sorted(ckpt_dir.glob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest:
        try:
            RUNTIME.load_model(latest[0])
        except Exception:
            pass

def run_gui() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print('PySide6 is not installed. CLI: python app.py status')
        return 1
    from ui.main_window import MainWindow
    bootstrap_runtime()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()

def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in {'--gui', 'gui'}:
        from ahamdcode.cli import run_cli
        return run_cli(sys.argv[1:])
    return run_gui()

if __name__ == '__main__':
    raise SystemExit(main())
