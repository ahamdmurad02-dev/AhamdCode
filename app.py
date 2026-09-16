#!/usr/bin/env python3
"""AhamdCode desktop entry point. Native PySide6 window — not a web app.

CLI (no GUI):
  python app.py status
  python app.py train --data datasets/seed --steps 20
  python app.py generate --prompt "def add(a, b):"
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahamdcode.runtime import RUNTIME  # noqa: E402
from ahamdcode.tokenizer.tokenizer import default_tokenizer_dir  # noqa: E402
from ahamdcode.utils.memory import apply_ram_safe_mode  # noqa: E402
from ahamdcode.utils.settings_store import load_settings  # noqa: E402
from ahamdcode.utils.system import default_paths  # noqa: E402


STYLE = """
QWidget { font-family: "Segoe UI", "Noto Sans", sans-serif; font-size: 13px; color: #202124; background: #F8F9FA; }
QMainWindow, QWidget { background: #F8F9FA; }
QListWidget { background: #EEF0F2; border: none; padding: 8px; }
QListWidget::item { padding: 10px 8px; }
QListWidget::item:selected { background: #D2E3FC; color: #174EA6; }
QFrame#card { background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 6px; padding: 8px; }
QLabel#pageTitle { font-size: 22px; font-weight: 600; }
QLabel#muted { color: #5F6368; }
QPlainTextEdit, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 4px; padding: 4px;
}
QPushButton { background: #E8EAED; border: 1px solid #DADCE0; border-radius: 4px; padding: 6px 12px; }
QPushButton:hover { background: #D2E3FC; }
QMenuBar { background: #FFFFFF; }
"""


def ensure_dirs() -> None:
    for p in default_paths().values():
        Path(p).mkdir(parents=True, exist_ok=True)


def bootstrap_runtime() -> None:
    ensure_dirs()
    settings = load_settings()
    apply_ram_safe_mode(bool(settings.get("ram_safe", True)), int(settings.get("cpu_threads", 2)))
    tok = default_tokenizer_dir()
    if (tok / "vocab.json").exists():
        RUNTIME.load_tokenizer(tok)
    official = default_paths()["models"] / "AhamdCode_0.1_Mini.pt"
    if official.exists():
        try:
            RUNTIME.load_model(official)
            return
        except Exception:
            pass
    ckpt_dir = Path(settings.get("checkpoint_dir") or default_paths()["checkpoints"])
    latest = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if latest:
        try:
            RUNTIME.load_model(latest[0])
        except Exception:
            pass


def run_gui() -> int:
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed in this environment.")
        print("GUI:  pip install PySide6  &&  python app.py")
        print("CLI:  python app.py status")
        print("      python app.py generate --prompt \"def add(a, b):\"")
        return 1
    from ui.main_window import MainWindow

    bootstrap_runtime()
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow()
    win.show()
    return app.exec()


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] not in {"--gui", "gui"}:
        from ahamdcode.cli import run_cli
        return run_cli(sys.argv[1:])
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
