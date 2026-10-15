@echo off
cd /d "%~dp0"
call "%~dp0scripts\run_full_cns.cmd" %*
exit /b %errorlevel%
