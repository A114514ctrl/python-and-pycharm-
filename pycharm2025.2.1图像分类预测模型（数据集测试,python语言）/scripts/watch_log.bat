@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set "ROOT=%~dp0..\"
if not exist "%ROOT%code\03_train.py" goto NOCODE
cd /d "%ROOT%code"
title Live Training Log
powershell -NoExit -Command "Get-Content -Path '%ROOT%results\train_log_improved.txt' -Encoding utf8 -Tail 60 -Wait"
echo.
pause
exit /b 0
:NOCODE
echo [ERROR] code folder not found. Please run this .bat inside the repository folder.
pause
exit /b 1
