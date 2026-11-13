@echo off
rem Daily data update for the Windows Task Scheduler (see reports/DATA_SOURCES.md, "Daily updates").
rem Adds newly published HARTI and CBSL prices, weather and fuel prices to data\raw, with backups.
rem Output is appended to logs\daily_update.log. Exit code 2 means a new HARTI bulletin couldn't
rem be read (its layout may have changed); 3 means another update was already running.
rem
rem Register it once (runs every day at 18:30), from a Command Prompt in this folder:
rem   schtasks /Create /TN "VegetablePriceDailyUpdate" /TR "\"%CD%\scripts\run_daily_update.bat\"" /SC DAILY /ST 18:30 /F
rem Remove it with:  schtasks /Delete /TN "VegetablePriceDailyUpdate" /F

setlocal
cd /d "%~dp0.."
if not exist logs mkdir logs
if not exist ".venv312\Scripts\python.exe" (
    echo [%date% %time%] .venv312 is missing; run setup first.>> logs\daily_update.log
    exit /b 1
)
echo ===== %date% %time% =====>> logs\daily_update.log
".venv312\Scripts\python.exe" scripts\daily_update.py %* >> logs\daily_update.log 2>&1
set RESULT=%ERRORLEVEL%
rem Redirection first: "echo exit code 0>> file" would read "0>>" as redirecting handle 0.
>> logs\daily_update.log echo exit code %RESULT%
exit /b %RESULT%
