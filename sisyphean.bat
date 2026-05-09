@echo off
:: Sisyphean CLI launcher
:: Usage:  sisyphean          → opens Claude Code pointed at Sisyphean engine
::         sisyphean status   → shows engine status
::         sisyphean start    → starts the tray + engine
::         sisyphean stop     → stops the engine
::         sisyphean logs     → opens the log file

:: Derive project root from wherever this .bat file lives (works from any install path)
set "SISYPHEAN_DIR=%~dp0"
if "%SISYPHEAN_DIR:~-1%"=="\" set "SISYPHEAN_DIR=%SISYPHEAN_DIR:~0,-1%"
set "ANTHROPIC_BASE_URL=http://127.0.0.1:8000"
set "ANTHROPIC_API_KEY=sisyphean-local"
set "ANTHROPIC_MODEL=sisyphean"

if "%1"=="status" goto status
if "%1"=="start"  goto start
if "%1"=="stop"   goto stop
if "%1"=="logs"   goto logs
if "%1"=="help"   goto help

:: Default: open Claude Code pointed at Sisyphean
:launch
  :: Check engine is up
  curl -sf http://127.0.0.1:8000/health >nul 2>&1
  if errorlevel 1 (
    echo Sisyphean engine is not running. Starting it...
    start "" pythonw "%SISYPHEAN_DIR%\tray.py"
    echo Waiting for engine...
    :wait_loop
      timeout /t 2 >nul
      curl -sf http://127.0.0.1:8000/health >nul 2>&1
      if not errorlevel 1 goto engine_ready
    goto wait_loop
  )
  :engine_ready
  echo Connected to Sisyphean engine at %ANTHROPIC_BASE_URL%
  claude --model sisyphean
  goto end

:status
  curl -sf http://127.0.0.1:8000/health >nul 2>&1
  if errorlevel 1 (
    echo Sisyphean engine: STOPPED
  ) else (
    curl -sf http://127.0.0.1:8000/api/status
    echo.
    echo Sisyphean engine: RUNNING at %ANTHROPIC_BASE_URL%
    echo Dashboard: http://127.0.0.1:8000/dashboard
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

:help
  echo.
  echo  sisyphean          - launch Claude Code with Sisyphean engine
  echo  sisyphean status   - check if engine is running
  echo  sisyphean start    - start the engine + tray icon
  echo  sisyphean stop     - stop the engine
  echo  sisyphean logs     - view engine logs
  echo.
  goto end

:end
