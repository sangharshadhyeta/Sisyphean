@echo off
:: Sisyphean CLI launcher
:: Usage:  sisyphean launch birdclaw  ? detect engine + open BirdClaw
::         sisyphean launch claude    ? detect engine + open Claude Code
::         sisyphean status   ? shows engine status
::         sisyphean start    ? starts the tray + engine
::         sisyphean stop     ? stops the engine
::         sisyphean logs     ? opens the log file

:: Derive project root from wherever this .bat file lives
set "SISYPHEAN_DIR=%~dp0"
if "%SISYPHEAN_DIR:~-1%"=="\" set "SISYPHEAN_DIR=%SISYPHEAN_DIR:~0,-1%"
set "ENGINE_PORT=47291"
set "ANTHROPIC_BASE_URL=http://127.0.0.1:%ENGINE_PORT%"
set "ANTHROPIC_API_KEY=sisyphean-local"
set "ANTHROPIC_MODEL=sisyphean"

if "%1"=="launch"  goto launch_sub
if "%1"=="status"  goto status
if "%1"=="start"   goto start
if "%1"=="stop"    goto stop
if "%1"=="logs"    goto logs
if "%1"=="help"    goto help
if "%1"=="setup"   goto setup

:: Default (no args): open Claude Code pointed at Sisyphean
:default_launch
  python "%SISYPHEAN_DIR%\main.py" launch claude
  goto end

:: launch <birdclaw|claude> — delegate entirely to main.py which probes the right port
:launch_sub
  python "%SISYPHEAN_DIR%\main.py" launch %2
  goto end

:status
  curl -sf http://127.0.0.1:%ENGINE_PORT%/health >nul 2>&1
  if errorlevel 1 (
    echo Sisyphean engine: STOPPED
  ) else (
    curl -sf http://127.0.0.1:%ENGINE_PORT%/api/status
    echo.
    echo Sisyphean engine: RUNNING at %ANTHROPIC_BASE_URL%
    echo Dashboard: http://127.0.0.1:%ENGINE_PORT%/dashboard
  )
  goto end

:start
  start "" pythonw "%SISYPHEAN_DIR%\tray.py"
  echo Sisyphean engine starting...
  goto end

:stop
  powershell -NoProfile -NonInteractive -Command "Get-Process python,pythonw -ErrorAction SilentlyContinue | Where-Object { (Get-CimInstance Win32_Process -Filter \"ProcessId=$($_.Id)\" -ErrorAction SilentlyContinue).CommandLine -match 'main\.py|tray\.py' } | Stop-Process -Force" >nul 2>&1
  echo Sisyphean engine stopped.
  goto end

:logs
  start "" notepad "%SISYPHEAN_DIR%\engine_live.txt"
  goto end

:setup
  python "%SISYPHEAN_DIR%\main.py" setup
  goto end

:help
  echo.
  echo  sisyphean                   - launch Claude Code with Sisyphean engine
  echo  sisyphean launch birdclaw   - detect engine + open BirdClaw
  echo  sisyphean launch claude     - detect engine + open Claude Code (TUI)
  echo  sisyphean status            - check if engine is running
  echo  sisyphean start             - start the engine + tray icon
  echo  sisyphean stop              - stop the engine
  echo  sisyphean logs              - view engine logs
  echo  sisyphean setup             - first-time configuration
  echo.
  goto end

:end