@echo off
cd /d "%~dp0.."
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
if "%~1"=="" (
    echo Drag a full_cns recording folder onto this file, or pass its path.
    pause
    exit /b 1
)
"%TASS_PYTHON%" "%~dp0..\src\analyze_full_cns.py" %*
if errorlevel 1 pause
