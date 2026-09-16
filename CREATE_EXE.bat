@echo off
title AhamdCode - Create Windows EXE
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Install Python 3.10+ and add it to PATH.
  pause
  exit /b 1
)
python -m pip install -r requirements.txt
if not exist build_cpp mkdir build_cpp
cmake -S . -B build_cpp -DPython3_EXECUTABLE=python
cmake --build build_cpp --config Release
python -m PyInstaller --noconfirm --clean AhamdCode.spec
if exist dist\AhamdCode\AhamdCode.exe echo Built %cd%\dist\AhamdCode\AhamdCode.exe
pause
