@echo off
REM Script to run model with Miniconda environment

set MINICONDA_PATH=C:\Users\RAZER\miniconda3
set ENV_NAME=flood

REM Activate environment
call %MINICONDA_PATH%\Scripts\activate.bat %ENV_NAME%

REM Run script
cd /d "%~dp0"
python xgbmanningsv5.py

pause
