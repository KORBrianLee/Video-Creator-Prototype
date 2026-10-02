@echo off
setlocal
set "PYTHONDONTWRITEBYTECODE=1"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" -RuntimeDir "D:\CursorVideoLocal" -Profile neodragon -Backend auto
exit /b %errorlevel%
