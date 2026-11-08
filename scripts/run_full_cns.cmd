@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0.."
set "TASS_PYTHON=%~dp0..\.venv\Scripts\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not exist "%TASS_PYTHON%" set "TASS_PYTHON=python"
if not exist "%~dp0..\data\male-cns-v1.0\manifest.json" (
    "%TASS_PYTHON%" "%~dp0..\src\download_connectome.py" --all-tables
    if errorlevel 1 goto failed
)
set "TASS_VISUAL_MOTOR_CIRCUIT=%~dp0..\visual_motor_circuit"
if not exist "%TASS_VISUAL_MOTOR_CIRCUIT%\report.json" (
    "%TASS_PYTHON%" "%~dp0..\src\prepare_visual_motor_circuit.py" --output "%TASS_VISUAL_MOTOR_CIRCUIT%"
    if errorlevel 1 goto failed
)
if not exist "%TASS_VISUAL_MOTOR_CIRCUIT%\photoreceptor_hex_map.feather" (
    "%TASS_PYTHON%" "%~dp0..\src\build_photoreceptor_hex_map.py" --circuit "%TASS_VISUAL_MOTOR_CIRCUIT%"
    if errorlevel 1 goto failed
)
if not exist "%TASS_VISUAL_MOTOR_CIRCUIT%\photoreceptor_hex_map_report.json" (
    "%TASS_PYTHON%" "%~dp0..\src\build_photoreceptor_hex_map.py" --circuit "%TASS_VISUAL_MOTOR_CIRCUIT%"
    if errorlevel 1 goto failed
)
if not exist "%~dp0..\data\fly_body\manifest.json" (
    "%TASS_PYTHON%" "%~dp0..\src\prepare_fly_body.py"
    if errorlevel 1 goto failed
) else (
    "%TASS_PYTHON%" "%~dp0..\src\prepare_fly_body.py" --offline
    if errorlevel 1 goto failed
)
"%TASS_PYTHON%" "%~dp0..\src\mujoco_full_fly.py" %*
if errorlevel 1 goto failed
exit /b 0
:failed
pause
exit /b 1
