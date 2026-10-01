@echo off
REM One-time: make the dashboard start automatically every time you log in to Windows.
set "APPDIR=%~dp0"
set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "LAUNCHER=%STARTUP%\NSE_Signals_Dashboard.bat"
> "%LAUNCHER%" echo @echo off
>> "%LAUNCHER%" echo cd /d "%APPDIR%"
>> "%LAUNCHER%" echo start "NSE signals dashboard" /min cmd /c start_dashboard.bat
echo Done. The dashboard will now start by itself whenever you log in to Windows.
echo To undo this, run uninstall_autostart.bat.
echo.
echo Starting it now as well...
start "NSE signals dashboard" /min cmd /c "%APPDIR%start_dashboard.bat"
pause
