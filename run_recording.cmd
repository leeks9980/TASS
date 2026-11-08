@echo off
cd /d "%~dp0"
call "%~dp0scripts\run_full_cns.cmd" --no-auto-protocol --no-fixed-body --actuation-mode safe %*
exit /b %errorlevel%
