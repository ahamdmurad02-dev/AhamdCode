from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path as _P

block_cipher = None
hidden = collect_submodules("ahamdcode") + collect_submodules("ui")

datas = [
    ("tokenizer", "tokenizer"),
    ("docs", "docs"),
    ("datasets/seed", "datasets/seed"),
]
if _P("models/AhamdCode_0.1_Mini.pt").exists():
    datas.append(("models/AhamdCode_0.1_Mini.pt", "models"))
if _P("models/AhamdCode_0.1_Mini.json").exists():
    datas.append(("models/AhamdCode_0.1_Mini.json", "models"))

excludes = [
    "electron", "tkinter", "matplotlib", "scipy", "pandas", "IPython", "notebook",
    "nvidia", "triton", "cuda", "onnxruntime",
]

SKIP = ("nvidia", "triton", "cuda", "cudnn", "cublas")

def keep(item):
    name = str(item[0]).lower() + " " + str(item[1]).lower()
    return not any(s in name for s in SKIP)

a = Analysis(
    ["app.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden + ["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
a.binaries = [b for b in a.binaries if keep(b)]
a.datas = [d for d in a.datas if keep(d)]
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AhamdCode", debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="AhamdCode")
