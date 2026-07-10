@echo off
title Autonomous Data Science Co-Pilot

echo ============================================================
echo   AUTONOMOUS DATA SCIENCE CO-PILOT
echo ============================================================
echo.

REM ── Check Python ────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Install Python 3.9+ from https://python.org
    pause & exit /b 1
)

REM ── Check Node / npm ────────────────────────────────────────
npm --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] npm not found. Install Node.js from https://nodejs.org
    pause & exit /b 1
)

REM ── Backend: create venv if missing ─────────────────────────
if not exist "backend\.venv\Scripts\python.exe" (
    echo [1/4] Creating Python virtual environment...
    python -m venv backend\.venv
)

REM ── Backend: install requirements ───────────────────────────
echo [2/4] Installing Python dependencies...
backend\.venv\Scripts\pip install -r requirements.txt --quiet

REM ── Frontend: install node_modules if missing ───────────────
if not exist "frontend\node_modules" (
    echo [3/4] Installing frontend npm packages...
    cd frontend
    npm install --silent
    cd ..
) else (
    echo [3/4] Frontend packages already installed.
)

REM ── Frontend: build ─────────────────────────────────────────
echo [4/4] Building frontend...
cd frontend
call npm run build --silent
cd ..

echo.
echo ============================================================
echo   Starting backend at http://localhost:5000
echo   Open http://localhost:5000 in your browser
echo   Press Ctrl+C to stop
echo ============================================================
echo.

REM ── Start Flask ─────────────────────────────────────────────
backend\.venv\Scripts\python backend\app.py
pause
