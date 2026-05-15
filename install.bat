@echo off
setlocal enabledelayedexpansion
title Sisyphean Engine Installer / Updater
cd /d "%~dp0"

echo.
echo  +===========================================+
echo  ^|   Sisyphean Engine  --  Install/Update   ^|
echo  +===========================================+
echo.

:: ── Check Python ──────────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found.
    echo  Please install Python 3.10+ from https://python.org and re-run.
    pause
    exit /b 1
)
for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo  Found: %%v

:: ── Git pull if this is a git repo ────────────────────────────────────────────
git -C "%~dp0" rev-parse --git-dir >nul 2>&1
if not errorlevel 1 (
    echo.
    echo  Pulling latest code from git...
    git -C "%~dp0" pull
    echo  Code updated.
) else (
    echo  [INFO] Not a git repo - skipping code update.
)

:: ── Install / upgrade dependencies ────────────────────────────────────────────
echo.
echo  Installing/upgrading Python dependencies...
pip install -r requirements.txt --upgrade
pip install pystray Pillow --upgrade

if errorlevel 1 (
    echo.
    echo  [ERROR] pip install failed. Check your internet connection.
    pause
    exit /b 1
)
echo  Dependencies up to date.

:: ── Create startup shortcut (safe to re-run) ──────────────────────────────────
echo.
echo  Configuring Windows startup...

set "STARTUP=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"
set "SHORTCUT=%STARTUP%\Sisyphean.lnk"
set "ICON=%~dp0assets\favicon.ico"
set "TARGET=%~dp0tray.py"

powershell -NoProfile -NonInteractive -Command "$WS = New-Object -ComObject WScript.Shell; $S = $WS.CreateShortcut('%SHORTCUT%'); $S.TargetPath = 'pythonw.exe'; $S.Arguments = '\"%TARGET%\"'; $S.WorkingDirectory = '%~dp0'; $S.IconLocation = '%ICON%'; $S.Description = 'Sisyphean Engine tray'; $S.Save()" >nul 2>&1

if exist "%SHORTCUT%" (
    echo  Startup shortcut OK.
) else (
    echo  [WARN] Could not create startup shortcut.
)

:: ── Stop existing tray/engine ─────────────────────────────────────────────────
echo.
echo  Stopping existing instance (if running)...
powershell -NoProfile -NonInteractive -Command "Get-Process pythonw -ErrorAction SilentlyContinue | Where-Object { (Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.Id)\" -ErrorAction SilentlyContinue).CommandLine -like '*tray*' } | Stop-Process -Force" >nul 2>&1
powershell -NoProfile -NonInteractive -Command "Get-Process python -ErrorAction SilentlyContinue | Where-Object { (Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.Id)\" -ErrorAction SilentlyContinue).CommandLine -like '*main.py*' } | Stop-Process -Force" >nul 2>&1
timeout /t 2 >nul

:: ── Add project dir to user PATH (enables `sisyphean` command globally) ───────
echo.
echo  Registering sisyphean command...
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"
powershell -NoProfile -NonInteractive -Command "$p = [Environment]::GetEnvironmentVariable('PATH','User'); if ($p -notlike '*%PROJECT_DIR%*') { [Environment]::SetEnvironmentVariable('PATH', $p + ';%PROJECT_DIR%', 'User'); Write-Host '  Added to user PATH. Open a new terminal to use: sisyphean' } else { Write-Host '  Already on PATH.' }"

:: ── Launch tray app ───────────────────────────────────────────────────────────
echo.
echo  Launching Sisyphean...
start "" pythonw "%~dp0tray.py"

echo.
echo  +===========================================+
echo  ^|              All done!                    ^|
echo  ^|                                          ^|
echo  ^|  Sisyphean is running in your tray.      ^|
echo  ^|  It starts automatically on login.       ^|
echo  ^|                                          ^|
echo  ^|  Dashboard: localhost:47291/dashboard    ^|
echo  ^|  Command:   sisyphean (new terminal)     ^|
echo  ^|                                          ^|
echo  ^|  Right-click tray icon for options.      ^|
echo  +===========================================+
echo.
pause
