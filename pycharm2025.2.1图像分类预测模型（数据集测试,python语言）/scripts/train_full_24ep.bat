@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set "ROOT=%~dp0..\"
if not exist "%ROOT%code\03_train.py" goto NOCODE
cd /d "%ROOT%code"
title Full Training (24 epochs)
if not exist "%ROOT%weights\backup" mkdir "%ROOT%weights\backup"
copy /y "%ROOT%weights\improved_best.pth" "%ROOT%weights\backup\improved_best_before_retrain.pth" >nul 2>nul
python 03_train.py --model improved --epochs 24 --no-eval-test
python 04_test_and_visualize.py --model improved
python 04_test_and_visualize.py --model improved --compare
python 06_make_report.py
echo.
pause
exit /b 0
:NOCODE
echo [ERROR] code folder not found. Please run this .bat inside the repository folder.
pause
exit /b 1
