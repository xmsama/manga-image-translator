@echo off
chcp 65001 >nul
cd /d "%~dp0"
set "PYW="
if exist "manga-image-translator\venv\Scripts\pythonw.exe" set "PYW=manga-image-translator\venv\Scripts\pythonw.exe"
if not defined PYW if exist "..\venv\Scripts\pythonw.exe" set "PYW=..\venv\Scripts\pythonw.exe"
if not defined PYW (
    echo 找不到 manga-image-translator 的虚拟环境 venv，请先按 README 安装
    pause
    exit /b 1
)
start "" "%PYW%" "%~dp0漫画翻译工具.pyw"
