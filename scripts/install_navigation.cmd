@echo off
cd /d "%~dp0.."
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%TASS_PYTHON%" (
    "%TASS_PYTHON%" -m pip install --target "%~dp0..\.deps" -r "%~dp0..\requirements-navigation.txt"
) else (
    python -m pip install -r "%~dp0..\requirements-navigation.txt"
)
pause
