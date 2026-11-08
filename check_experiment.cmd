@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "TASS_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
"%TASS_PYTHON%" "%~dp0src\check_experiment.py" %*
set "check_result=%errorlevel%"
echo.
if "%check_result%"=="0" (
    echo Experiment configuration is valid.
) else (
    echo Experiment configuration check failed with exit code %check_result%.
)
echo Press any key to close this window.
pause >nul
exit /b %check_result%
