@echo off
title Data Co-Pilot (Dev Mode)

echo ============================================================
echo   DATA CO-PILOT  --  DEV MODE
echo   Backend  : http://localhost:5000
echo   Frontend : http://localhost:5173  (hot-reload)
echo ============================================================
echo.

REM Start Flask backend in a new window
start "Backend (Flask :5000)" cmd /k "backend\.venv\Scripts\python backend\app.py"

REM Start Vite dev server (proxies /api to :5000)
cd frontend
npm run dev
