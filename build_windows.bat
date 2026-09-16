@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo [1/7] Checking Python
where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10+ x64 and retry.
  exit /b 1
)
python --version

echo [2/7] Installing dependencies
python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [3/7] Checking PyTorch
python -c "import torch; print('torch', torch.__version__)"
if errorlevel 1 exit /b 1

echo [4/7] Building C++ module with CMake
if not exist build_cpp mkdir build_cpp
cmake -S . -B build_cpp -DPYTHON_EXECUTABLE=python
if errorlevel 1 exit /b 1
cmake --build build_cpp --config Release
if errorlevel 1 exit /b 1

echo [5/7] Running tests
python -m pytest -q tests
if errorlevel 1 (
  echo Tests failed.
  exit /b 1
)

echo [6/7] Packaging with PyInstaller
python -m PyInstaller --noconfirm --clean AhamdCode.spec
if errorlevel 1 exit /b 1

echo [7/7] Done
echo Folder build: dist\AhamdCode\AhamdCode.exe
if exist dist\AhamdCode.exe echo One-file build: dist\AhamdCode.exe
exit /b 0
