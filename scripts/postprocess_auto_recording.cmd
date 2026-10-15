@echo off
cd /d "%~dp0.."
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0postprocess_auto_recording.ps1" %*
exit /b %errorlevel%
