@echo off
rem Starts the Sri Lankan Vegetable Price Prediction dashboard. Double-click this file.
rem It runs the documented command: .venv312\Scripts\python.exe -m streamlit run app/Home.py
title Sri Lankan Vegetable Price Prediction
cd /d "%~dp0"

rem The app needs the Python 3.12 environment .venv312. The older .venv folder uses Python 3.15, which
rem can't run this project: pymupdf 2.1 requires onnxruntime, which has no Python 3.15 build.
rem The entry point is app/Home.py (there is no app.py).
if not exist ".venv312\Scripts\python.exe" (
    echo The Python environment .venv312 was not found in this folder.
    echo Create it with Python 3.12 as described in README.md, section 7.
    pause
    exit /b 1
)

rem Check the packages the app imports before starting, so a broken environment shows a clear message.
".venv312\Scripts\python.exe" -c "import streamlit, pandas, numpy, sklearn, joblib, yaml, requests, pymupdf, bs4, altair" 2>nul
if errorlevel 1 (
    echo .venv312 is missing packages the app needs. Install them with:
    echo     .venv312\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    ".venv312\Scripts\python.exe" -c "import streamlit, pandas, numpy, sklearn, joblib, yaml, requests, pymupdf, bs4, altair"
    pause
    exit /b 1
)

echo Starting the app. Your browser will open at http://localhost:8501
echo Keep this window open while you use the app. Close it, or press Ctrl+C, to stop.
echo.
".venv312\Scripts\python.exe" -m streamlit run app/Home.py

rem Keep the window open if the app stopped with an error, so the message can be read.
if errorlevel 1 pause
