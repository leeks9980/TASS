@echo off
chcp 65001 >nul
cd /d "%~dp0.."
set "PYTHONUTF8=1"
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
if "%~1"=="" (
    echo Usage: replay_fly_recording.cmd simulation_logs\full_fly_...
    pause
    exit /b 1
)
"%TASS_PYTHON%" "%~dp0..\src\replay_fly_recording.py" %*
if errorlevel 1 pause
