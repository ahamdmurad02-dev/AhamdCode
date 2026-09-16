# AhamdCode 0.1 Mini

Local Windows desktop coding assistant.

- Native PySide6 application (not Electron, not a browser UI)
- Custom decoder-only Transformer trained and run on this machine
- No cloud LLM APIs

This is a **small** model (Mini Default is about 1.1M parameters).

## Download the Windows EXE from GitHub

1. Open [Actions](https://github.com/ahamdmurad02-dev/AhamdCode/actions)
2. Open the latest **Build Windows EXE** run
3. Download artifact **AhamdCode-windows-x64**
4. Unzip and run `AhamdCode.exe`

You can also start the build yourself: Actions → Build Windows EXE → Run workflow. The artifact contains the EXE, its SHA-256 file, and BUILD_INFO.txt.

## Run from source (no EXE)

```
pip install -r requirements.txt
python app.py status
python app.py
```

## Windows local build

```
CREATE_EXE.bat
```

Output: `dist\AhamdCode\AhamdCode.exe`
