from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path as P
block_cipher = None
hidden = []
try:
    hidden += collect_submodules("ahamdcode")
except Exception:
    hidden += ["ahamdcode"]
try:
    hidden += collect_submodules("ui")
except Exception:
    pass
datas = []
for src, dest in [("tokenizer", "tokenizer"), ("docs", "docs"), ("datasets/seed", "datasets/seed")]:
    if P(src).exists():
        datas.append((src, dest))
if P("models/AhamdCode_0.1_Mini.pt").exists():
    datas.append(("models/AhamdCode_0.1_Mini.pt", "models"))
a = Analysis(["app.py"], pathex=["."], binaries=[], datas=datas, hiddenimports=hidden + ["PySide6.QtCore", "PySide6.QtGui", "PySide6.QtWidgets"], excludes=["nvidia", "triton", "cuda"], noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="AhamdCode", debug=False, strip=False, upx=False, console=False)
coll = COLLECT(exe, a.binaries, a.zipfiles, a.datas, strip=False, upx=False, name="AhamdCode")
