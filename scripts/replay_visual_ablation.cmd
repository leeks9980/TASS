@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0.."
set "TASS_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
"%TASS_PYTHON%" "%~dp0..\src\replay_visual_ablation.py" %*
exit /b %errorlevel%
