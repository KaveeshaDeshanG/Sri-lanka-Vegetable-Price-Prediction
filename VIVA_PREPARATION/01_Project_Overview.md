# 01 — Project Overview

**Project:** AI-Powered Sri Lankan Vegetable Price Prediction System
**Module:** CS6P05ES Project (London Metropolitan University) · Student: Kaveesha Deshan (E-290292)
**Prepared:** 13 November 2026, from the actual source code, data files and test runs in this folder.

> How to read these documents: every technical claim names the file (and usually the line) where you can see it
> yourself. Sinhala notes (සිංහල) explain the idea in simple words so you understand what you are saying.

---

## 1. What the project does in one sentence

It reads the official daily vegetable price bulletins of **HARTI** (Hector Kobbekaduwa Agrarian Research and Training
Institute), turns them into clean data, and uses a machine learning model to **forecast the wholesale price of a
vegetable at the Dambulla market 1 to 30 days ahead**, shown in a **Streamlit** web dashboard.

> සිංහල: HARTI ආයතනය දිනපතා PDF ලෙස නිකුත් කරන එළවළු මිල වාර්තා කියවා, ඒවා පිරිසිදු දත්ත බවට පත් කර,
> දඹුල්ල තොග වෙළඳපොළේ එළවළු මිල දින 1 සිට 30 දක්වා ඉදිරියට අනාවැකි කීමට Machine Learning ආකෘතියක් භාවිතා කරයි.
> ප්‍රතිඵල Streamlit වෙබ් dashboard එකක පෙන්වයි.

## 2. The problem it solves

| Problem | How the project answers it |
|---|---|
| HARTI and CBSL publish prices only as **PDF** files — no API, no CSV | A PDF parser reads every bulletin since 22 June 2015 (`src/data_collection/harti_live_client.py`, `scripts/build_harti_prices.py`) |
| Bulletins show **past** prices only | A forecaster predicts future prices (`src/models/direct_forecaster.py`) |
| Forecast accuracy is easy to overstate | Every model is compared with the **naive baseline** "tomorrow's price = today's price" (`reports/metrics/direct_forecaster_metrics.json`) |
| Planting dates and prices are not connected | The **Cultivation Planner** links Department of Agriculture harvest durations to the forecast (`app/cultivation.py`) |

> සිංහල: ප්‍රධාන ප්‍රශ්නය — "Machine Learning ආකෘතියකට, 'හෙට මිල අද මිලම වේ' යන සරලම අනුමානයට වඩා හොඳින් අනාවැකි කිව හැකිද?"
> පිළිතුර: ඔව්, නමුත් සුළු ප්‍රමාණයකින් (MAE 4–8% කින් අඩුයි).

## 3. Key facts to remember (verified from files on 13 Nov 2026)

| Fact | Value | Evidence |
|---|---|---|
| Entry point | `app/Home.py` (there is **no** `app.py`) | `start_app.bat` line 31 |
| Python | **3.12** (environment `.venv312`) | `pyproject.toml` `requires-python = ">=3.12,<3.15"` |
| UI framework | Streamlit 1.66.1 | `requirements.txt` |
| Pages | Home + Predict, Model Performance, Price History, Cultivation Planner | `app/Home.py`, `app/pages/1_…4_*.py` |
| Main model | **Ridge Regression direct forecaster** — one model per day ahead (30 models) | `src/models/direct_forecaster.py` lines 34–52, 103–124 |
| Earlier models | Linear Regression, Random Forest, XGBoost (51 features, one day ahead) | `src/models/trainer.py`, `config/config.yaml` `model:` |
| Prices | 76,719 Dambulla readings, 22 Jun 2015 – 9 Nov 2026; 526,905 readings in 11 markets | `data/raw/data_manifest.json` |
| Model trained on | 25 Apr 2024 – 6 Nov 2026, 31 vegetable series | `models/direct_forecaster.joblib` (`trained_through`) |
| Accuracy (test 14 May–6 Nov 2026) | next-day MAE Rs 17.47/kg (7.5%), **4.3% better than the baseline**; 7.8% better at 22–30 days | `reports/metrics/direct_forecaster_metrics.json` |
| Storage | CSV/JSON files + joblib model files — **no database** | `data/raw`, `models/` |
| Tests | 1,060 automated test cases (565 functions) + 20 practical tests | `tests/`, `11_Project_Testing_Report.md` |

## 4. Implemented vs not implemented (be honest in the viva)

**Implemented and working (tested):**
- Next-day and 1–30-day forecasts for Dambulla wholesale prices (Predict page).
- Two data sources on the Predict page (since 15 Nov 2026): **Latest Real Data Prediction** (source status for HARTI, CBSL and DCS, each source's collected records, the exact starting record, forecasts at 1/7/14/30 days) and **Upload CSV**.
- Automatic checks of HARTI, CBSL and DCS every 3 hours (Windows task `VegetablePriceUpdate`), each source logged separately.
- Model Performance page comparing the model with the naive baseline.
- Price History for HARTI (11 markets), CBSL (wholesale and retail) and DCS (retail), always kept separate.
- Cultivation Planner (planting date → harvest window → forecast price where possible).
- Daily data update script with backups (`scripts/daily_update.py`) and a full rebuild script (`scripts/update_data.py`).

**Not implemented / limitations:**
- No real-time price feed (none exists publicly in Sri Lanka). Data is as current as the latest bulletin.
- Forecasts only for **Dambulla** (other markets are inputs only).
- No database, no user login, not deployed online (runs on your own PC).
- The update runs every 3 hours only while the laptop is on and you are logged in (Windows task `VegetablePriceUpdate`, registered 15 Nov 2026).
- The model has **not** been retrained on the full 2015–2026 history (`scripts/retrain_if_better.py` exists but was never run).
- No ARIMA / LSTM / SHAP. A direct production-region-vs-market-weather comparison was not run.

> සිංහල: Viva එකේදී "සිදු කළ දේ" සහ "සිදු නොකළ දේ" අවංකව වෙන් කර කියන්න. උදා: "Real-time API එකක් නැහැ;
> දිනපතා නිකුත් වන bulletin එකෙන් update වෙනවා."

## 5. Document map

| File | Use it for |
|---|---|
| 02 | Where everything is in the code |
| 03 | Installing and starting the app from zero |
| 04 | Every page, button and what runs behind it |
| 05 | Technologies and why they were chosen |
| 06 | The machine learning explained step by step |
| 07 | Data sources, live data and APIs |
| 08 | Architecture and data-flow diagrams (Mermaid) |
| 09 | The 15–20 minute live demo script |
| 10 | 70+ viva questions with answers |
| 11 | Test cases and actual results |
| 12 | What to say, word for word |
| 13 | One-page cheat sheet for the last hour |
