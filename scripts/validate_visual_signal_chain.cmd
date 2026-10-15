@echo off
cd /d "%~dp0.."
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found: .venv\Scripts\python.exe
  exit /b 1
)
".venv\Scripts\python.exe" "src\validate_visual_signal_chain.py" %*
exit /b %errorlevel%
