@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0video.ps1" %*
exit /b %errorlevel%
