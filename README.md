# AhamdCode 0.1 Mini

Local Windows desktop coding assistant.

- Native PySide6 application (not Electron, not a browser UI)
- Custom decoder-only Transformer trained and run on this machine
- No cloud LLM APIs

This is a **small** model (Mini Default is about 1.1M parameters).

## Clone

```
git clone https://github.com/ahamdmurad02-dev/AhamdCode.git
cd AhamdCode
```

## Run GUI

```
python -m pip install -r requirements.txt
python app.py
```

## Windows EXE

On Windows 10/11 x64 double-click `CREATE_EXE.bat`.
Output: `dist\\AhamdCode\\AhamdCode.exe`

## CLI

```
python app.py status
python app.py train --data datasets/seed --steps 30
python app.py generate --prompt "def add(a, b):" --max-tokens 32
```
