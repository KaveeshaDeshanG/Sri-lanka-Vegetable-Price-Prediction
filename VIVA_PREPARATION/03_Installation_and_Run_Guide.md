# 03 — Installation and Run Guide (Windows PowerShell)

**The short version (on your own PC, where everything is already installed):**
double-click `start_app.bat` in the project folder → your browser opens http://localhost:8501.

> සිංහල: ඔබේ පරිගණකයේ දැනටමත් සියල්ල install කර ඇත. `start_app.bat` එක double-click කරන්න — browser එකේ
> http://localhost:8501 විවෘත වේ. පහත පියවර "මුල සිට" (නව පරිගණකයක) කරන ආකාරයයි.

---

## Step 1 — Open the project folder

```powershell
cd "D:\ECU\Sri-lanka-Vegetable-Price-Prediction-main\Sri-lanka-Vegetable-Price-Prediction-main"
```

## Step 2 — Check Python versions

```powershell
py -0p                       # lists installed Pythons; you need 3.12
py -3.12 --version           # should print Python 3.12.x
```

**Why 3.12?** `pyproject.toml` says `requires-python = ">=3.12,<3.15"`. PyMuPDF 2.1 (used to read the PDF
bulletins) needs `onnxruntime`, which has **no Python 3.15 build**. That is why the folder `.venv` (Python 3.15) is
broken — its numpy/pandas imports fail — and `.venv312` (Python 3.12.10) is the one to use.

> සිංහල: Python 3.12 අනිවාර්යයි. 3.15 සමඟ PDF කියවන පුස්තකාලය (PyMuPDF) install කළ නොහැක.

## Step 3 — Create the virtual environment (only if `.venv312` does not exist)

```powershell
Test-Path .venv312\Scripts\python.exe        # True on your PC → skip to Step 5
py -3.12 -m venv .venv312
```

## Step 4 — Activate it (optional)

```powershell
.\.venv312\Scripts\Activate.ps1
```
If PowerShell says *"running scripts is disabled on this system"*:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv312\Scripts\Activate.ps1
```
After activation the prompt starts with `(.venv312)`. **You can skip activation** by always typing the full path
`.\.venv312\Scripts\python.exe` — this is the safest way.

## Step 5 — Install dependencies

```powershell
.\.venv312\Scripts\python.exe -m pip install -r requirements.txt
.\.venv312\Scripts\python.exe -m pip install -e ".[dev]"      # only needed to run the tests (adds pytest)
```

## Step 6 — Environment variables / API keys

**None are needed.** All sources are public: HARTI and CBSL are public web pages, Open-Meteo needs no key for
non-commercial use (`config/config.yaml`, `weather_api:` comment), CEYPETCO is a public page. There is no `.env` file
and no `st.secrets` usage.

## Step 7 — Check datasets and models exist

```powershell
Test-Path data\raw\harti_prices.csv, data\raw\harti_all_markets.csv, data\raw\weather_data_by_region.csv, data\raw\fuel_prices.csv, models\direct_forecaster.joblib
Get-Content data\raw\data_manifest.json | Select-String "latest_observation"
```
All should be `True`. If a file is missing, the Home page shows an error naming it (`app/services.py`
`historical_dataset_problems`, line 114; `forecaster_problems`, line 121).

**Optional before the viva — refresh the data** (adds newly published bulletins, takes about a minute, makes a backup):
```powershell
.\.venv312\Scripts\python.exe scripts\daily_update.py
```

## Step 8 — Check port 8501 is free

```powershell
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
```
No output = free. If something is listed, the app may already be running — just open http://localhost:8501, or
close the old window first (Streamlit would otherwise start on 8502).

## Step 9 — Start the application (choose one)

**A. Double-click `start_app.bat`** (recommended). What it does (`start_app.bat`):
- line 5: `cd /d "%~dp0"` — goes to the project folder;
- lines 10–14: stops with a message if `.venv312` is missing;
- lines 18–25: test-imports streamlit, pandas, numpy, sklearn, joblib, yaml, requests, pymupdf, bs4, altair; shows the error if any is missing;
- line 31: `".venv312\Scripts\python.exe" -m streamlit run app/Home.py`;
- line 34: keeps the window open if it crashed.

**B. From PowerShell:**
```powershell
.\.venv312\Scripts\python.exe -m streamlit run app/Home.py
```

The console prints `Local URL: http://localhost:8501`. **Never** use `python -m streamlit run app.py` — that file
does not exist.

## Step 10 — Open the app

Your browser opens automatically. If not, go to **http://localhost:8501**.
Health check: `Invoke-WebRequest -UseBasicParsing http://localhost:8501/_stcore/health` → `ok`.

## Step 11 — Stop and restart

- **Stop:** press **Ctrl + C** in the console window, or close that window.
- **Restart:** double-click `start_app.bat` again (or run the command in Step 9).
- After a data update the running app picks up new files automatically on the next click (caches are keyed on
  file modification times — `app/catalogue.py` `data_version`, line 162). No restart needed.

## Step 12 — Run the tests (optional, 3–9 minutes)

```powershell
.\.venv312\Scripts\python.exe -m pytest -q
```
Expected: `1060 passed` (last run 13 Nov 2026).

---

## Common startup errors and fixes

| Error you see | Cause | Fix |
|---|---|---|
| `Error: Invalid value: File does not exist: app.py` | Wrong entry point | Use `app/Home.py` |
| `No module named streamlit` (path shows `pythoncore-3.15`) | Plain `python` is the global 3.15, not `.venv312` | Use `.\.venv312\Scripts\python.exe -m streamlit run app/Home.py` |
| `numpy … _multiarray_umath … cp312 … not compatible with cpython-315` | You used the broken `.venv` (3.15) | Use `.venv312` |
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or skip activation |
| App opens on port **8502** | 8501 already used (an old copy is running) | Close the old window, or use the 8502 URL |
| "The forecaster's model file could not be read" / "missing file" on Home | Model or data file missing/damaged | Restore from `D:\ECU\Sri-lanka-Vegetable-Price-Backups\…` or rebuild with `scripts\update_data.py` and `scripts\train_forecaster.py` |
| "Couldn't fetch the latest HARTI bulletin" | No internet / HARTI site down | Use "Historical dataset (real)" instead |
| Browser shows nothing / spinner | First load builds features (~10 s) | Wait; the next clicks are fast (cached) |
| `.venv312 is missing packages the app needs` (from start_app.bat) | Incomplete install | `.\.venv312\Scripts\python.exe -m pip install -r requirements.txt` |

> සිංහල: වැඩිපුරම වන වැරැද්ද — වැරදි Python එක භාවිතා කිරීම. සෑම විටම `.venv312\Scripts\python.exe` හෝ `start_app.bat` භාවිතා කරන්න.
