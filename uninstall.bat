@echo off
title Sisyphean Engine Uninstaller
cd /d "%~dp0"

echo.
echo  Stopping Sisyphean Engine...

:: Kill tray + engine processes
taskkill /F /IM pythonw.exe /FI "WINDOWTITLE eq Sisyphean*" >nul 2>&1
for /f "tokens=2" %%p in ('tasklist /FI "IMAGENAME eq pythonw.exe" /FO CSV /NH 2^>nul') do (
    wmic process %%~p get commandline 2>nul | findstr /i "tray.py" >nul 2>&1
    if not errorlevel 1 taskkill /F /PID %%~p >nul 2>&1
)
taskkill /F /IM python.exe /FI "WINDOWTITLE eq Sisyphean*" >nul 2>&1

:: Remove startup shortcut
set "SHORTCUT=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\Sisyphean.lnk"
if exist "%SHORTCUT%" (
    del "%SHORTCUT%"
    echo  Startup shortcut removed.
)

echo.
echo  Sisyphean Engine uninstalled.
echo  (Python packages were not removed.)
echo.
pause
