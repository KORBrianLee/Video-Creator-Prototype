@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0update.ps1" %*
if errorlevel 1 pause
endlocal
