@echo off
cd /d "%~dp0.."
set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if exist "%TASS_PYTHON%" (
    "%TASS_PYTHON%" "%~dp0..\src\circuit_sim_gui.py"
) else (
    python "%~dp0..\src\circuit_sim_gui.py"
)
if errorlevel 1 pause
