@echo off
setlocal
cd /d "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Create .venv and install requirements.txt using the README instructions.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m app.main
if errorlevel 1 (
    pause
    exit /b 1
)
