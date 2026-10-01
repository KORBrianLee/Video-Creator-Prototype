@echo off
chcp 65001 >nul
cd /d "%~dp0"
where python >nul 2>nul && (python setup.py & goto :done)
where py >nul 2>nul && (py -3 setup.py & goto :done)
echo Python 3.8 이상이 필요합니다.
echo   winget install --id Python.Python.3.12 -e
echo 설치 후 이 파일을 다시 실행하세요.
:done
pause
