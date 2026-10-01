@echo off
cd /d "%~dp0.."
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
"%TASS_PYTHON%" "%~dp0..\src\download_connectome.py" --all-tables
if errorlevel 1 pause
