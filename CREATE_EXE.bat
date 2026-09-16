@echo off
cd /d "%~dp0"
where python >nul 2>nul || (echo Python missing & pause & exit /b 1)
python -m pip install -r requirements.txt
python -m pytest -q tests
python -m PyInstaller --noconfirm --clean AhamdCode.spec
if exist dist\AhamdCode\AhamdCode.exe echo DONE %cd%\dist\AhamdCode\AhamdCode.exe
pause
