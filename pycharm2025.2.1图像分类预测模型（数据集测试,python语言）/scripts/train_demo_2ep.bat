@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set "ROOT=%~dp0..\"
if not exist "%ROOT%code\03_train.py" goto NOCODE
cd /d "%ROOT%code"
title Short Training Demo (2 epochs, tag _demo)
python 03_train.py --model improved --epochs 2 --no-eval-test --tag _demo
echo.
pause
exit /b 0
:NOCODE
echo [ERROR] code folder not found. Please run this .bat inside the repository folder.
pause
exit /b 1
