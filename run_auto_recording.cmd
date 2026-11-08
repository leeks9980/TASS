@echo off
cd /d "%~dp0"
call "%~dp0scripts\run_full_cns.cmd" %*
if errorlevel 1 (
    set "run_result=%errorlevel%"
    goto :wait_for_escape
)
call "%~dp0scripts\postprocess_auto_recording.cmd"
set "run_result=%errorlevel%"

:wait_for_escape
echo.
echo Recording process finished. Press ESC to close this window.
powershell.exe -NoProfile -Command "$key = $null; while ($key.VirtualKeyCode -ne 27) { $key = $Host.UI.RawUI.ReadKey('NoEcho,IncludeKeyDown') }"
exit /b %run_result%
