@echo off
setlocal
cd /d "%~dp0"
title Read - Listen - Speak

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js is required but was not found.
  echo Download it from https://nodejs.org and run this file again.
  pause
  exit /b 1
)

if not exist "server\node_modules" (
  echo First run: installing dependencies...
  pushd server
  call npm install
  if errorlevel 1 (
    popd
    echo.
    echo npm install failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
  popd
)

rem Alpha: a previous instance may still be holding port 8787 (its window left
rem open, or a crash), which would make this launch serve stale server.js. Kill
rem whatever is LISTENING on 8787 first so every run starts a fresh server.
echo Freeing port 8787 if a previous instance is still running...
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8787" ^| findstr "LISTENING"') do taskkill /f /pid %%p >nul 2>nul
rem Brief pause so the socket is fully released before we rebind to it.
timeout /t 1 /nobreak >nul

echo.
echo Starting the app with neural voices at http://127.0.0.1:8787
echo Alpha: diagnostic logging is ON; entries are written to logs\debug.log
echo Keep this window open while using the app. Close it to stop.
echo.

rem Open the browser once the server has had a moment to start.
rem Alpha: launch with ?debug=1 so recognition diagnostics are captured to
rem logs\debug.log by default. Drop the query string before any public release.
start /b "" cmd /c "timeout /t 2 /nobreak >nul && start http://127.0.0.1:8787/?debug=1"

node server\server.js
