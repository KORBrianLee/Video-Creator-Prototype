@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 pause
endlocal
