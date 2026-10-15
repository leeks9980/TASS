@echo off
call "%~dp0scripts\validate_visual_signal_chain.cmd" %*
exit /b %errorlevel%
