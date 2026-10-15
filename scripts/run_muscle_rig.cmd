@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
"%TASS_PYTHON%" "%~dp0..\src\mujoco_full_cns.py" %*
if errorlevel 1 pause
