@echo off
title RoyaleRL AI Bot Launcher
cd /d "%~dp0RoyaleRL"
call "..\venv\Scripts\activate.bat"
set PYTHONIOENCODING=utf-8

if not "%1"=="" (
    python runbot.py %*
    goto end
)

echo ===================================================
echo        CLASH ROYALE AI - 24/7 LAUNCHER
echo ===================================================
echo  [1] Autonomous 24/7 Bot (Auto Self-Play)
echo  [2] Human Teacher Mode (Record your games to buffer)
echo  [3] Offline Train on Replay Buffer (50 epochs)
echo ===================================================
set /p choice="Select an option (1-3) [default: 1]: "

if "%choice%"=="2" (
    echo Starting Human Teacher Recording Mode...
    python runbot.py --mode record
) else if "%choice%"=="3" (
    echo Starting Offline Training...
    python runbot.py --mode train --epochs 50
) else (
    echo Starting Autonomous 24/7 Bot...
    python runbot.py --mode auto
)

:end
pause
