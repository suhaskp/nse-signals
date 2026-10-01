@echo off
title NSE signals dashboard
cd /d "%~dp0"
where python >nul 2>nul || goto :nopython
if not exist ".venv\Scripts\python.exe" (
  echo First start: setting up the environment. This happens only once and takes a few minutes.
  python -m venv .venv || goto :fail
)
fc /b requirements.txt ".venv\req_installed.txt" >nul 2>nul
if errorlevel 1 (
  echo Installing or updating packages...
  ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
  copy /y requirements.txt ".venv\req_installed.txt" >nul
)
if not exist "data\superstar" (
  mkdir "data\superstar"
  xcopy /e /i /q /y "data_template\superstar" "data\superstar" >nul
)
:loop
echo Starting the dashboard at http://localhost:8501 - keep this window open (minimised is fine).
".venv\Scripts\python.exe" serve.py
if errorlevel 3 if not errorlevel 4 goto :already
echo The dashboard stopped. Restarting in 15 seconds - close this window to stop it.
timeout /t 15 >nul
goto loop
:already
echo Another copy is already running at http://localhost:8501 - opened it in your browser. This window will close.
timeout /t 5 >nul
exit /b 0
:nopython
echo Python 3.10 or newer is required. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" during installation, then run this file again.
pause
exit /b 1
:fail
echo Setup failed - see the messages above. Check your internet connection and run this file again.
pause
exit /b 1
