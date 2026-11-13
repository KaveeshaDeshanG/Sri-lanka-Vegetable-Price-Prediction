# 13 — Last-Minute Viva Cheat Sheet

## Start / stop
```powershell
cd "D:\ECU\Sri-lanka-Vegetable-Price-Prediction-main\Sri-lanka-Vegetable-Price-Prediction-main"
.\start_app.bat                      # or double-click it
# = .\.venv312\Scripts\python.exe -m streamlit run app/Home.py
```
URL **http://localhost:8501** · Stop **Ctrl + C** · Entry point **app/Home.py** (NO app.py) · Python **3.12** (`.venv312`)
Port check: `Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue`
Tests: `.\.venv312\Scripts\python.exe -m pytest -q` → **1,060 passed**

## The project in 3 lines
1. Parses **HARTI** daily PDF bulletins (2015-06-22 → 2026-11-09): **76,719** Dambulla prices, **526,905** across 11 markets.
2. **Ridge direct forecaster**: 30 models (one per day ahead), predicts **log(future/today)**, price = last × exp(change).
3. **Streamlit**: Home, Predict, Model Performance, Price History, Cultivation Planner.

## Numbers to remember
| | Value |
|---|---|
| Next-day MAE | **Rs 17.47/kg** (MAPE **7.51 %**, R² 0.969) vs hold 18.26 |
| Gain vs hold, next day | **4.3 %** (95 % CI 2.6–6.0) |
| Gain at 22–30 days | **7.8 %** (CI 4.4–11.2); MAPE 25.1 % vs 29.4 % |
| Vegetables improved next day | **21 of 30** |
| Earlier models next-day MAE | LR 18.56 · RF 18.75 · XGB 19.84 · yesterday's price **18.17** (none win) |
| Recursive vs direct MAPE (15–30 d) | 36.5 % vs **25.8 %** |
| Test period | **14 May – 6 Nov 2026** (fixed in config) |
| Model trained | 25 Apr 2024 – 6 Nov 2026, **31** series |
| Max horizon | **30 days** (beyond → refused) |
| Bootstrap | 2,000 resamples of whole dates |
| Tests | 1,060 pytest + 18 UI + independent check → all pass |

## Bands (`direct_forecaster.py` 47–52)
1 day α1000: momentum, level, market, fuel, weather_lag_14, other_markets ·
2–10 α1000: momentum, rain, weather_lag_14, level, other_markets ·
11–21 α1000: rain, market, level, other_markets ·
22–30 α100: momentum, calendar, rain, market, level
Pipeline: **SimpleImputer(median) → StandardScaler → Ridge(α)** (line 118)

## Key files
`app/Home.py` · `app/pages/1_Predict.py` · `app/forecasting.py` (`direct_forecast` 116) · `app/cultivation.py` (`build_plan` 276) ·
`src/models/direct_forecaster.py` · `src/feature_engineering/forecast_features.py` · `src/data_collection/harti_live_client.py` ·
`scripts/daily_update.py` · `config/config.yaml` · `models/direct_forecaster.joblib` · `reports/metrics/direct_forecaster_metrics.json`

## Definitions (one line each)
- **MAE** average absolute error (Rs) · **RMSE** penalises big errors · **MAPE** error as % · **R²** variance explained
- **Baseline / naive** price stays the same · **Ridge** linear regression + L2 penalty · **Direct** one model per horizon
- **Data leakage** using future info · **Overfitting** memorising noise · **Bootstrap** resampling to get an interval
- **Feature** input number · **Horizon** days ahead · **Imputation** filling missing values · **Scaling** mean 0, std 1

## Data sources
HARTI (wholesale, PDF, training) · CBSL (wholesale+retail, display only) · DCS (retail weekly, display only) ·
Open-Meteo (weather, no key) · CEYPETCO (fuel). **No real-time API exists** — latest bulletin on demand.

## Honest limitations (say them before they ask)
Modest gain (4–8 %) · RMSE slightly worse than hold at 8–14 and 22–30 days · Dambulla only · no supply data ·
not retrained on full history · daily update not scheduled · no database/login/deployment · no user study · no LSTM/ARIMA/SHAP.

## Demo values (13 Nov 2026 data)
Beans: last Rs 425 → **10 Nov Rs 453.74** · **25 Nov Rs 507.96** (16 d) · **25 Dec refused** (46 d) ·
Planner Beans bush planted 20 Oct → harvest 4 Dec → **Rs 450/kg** · Latest HARTI bulletin 12 Nov (30 prices)

## සිංහල — මතක තබා ගන්න
- App එක: `start_app.bat` → localhost:8501 → නැවැත්වීමට Ctrl + C.
- ආකෘතිය: Ridge, දින 1–30 සඳහා ආකෘති 30ක්; අද මිලෙන් % වෙනස අනාවැකි කියයි.
- ප්‍රතිඵලය: "මිල වෙනස් නොවේ" යන baseline එකට වඩා 4.3 % (හෙට) – 7.8 % (දින 22–30) හොඳයි.
- සැබෑ දත්ත පමණි; synthetic sample එක පැහැදිලිව ලේබල් කර ඇත.
- Real-time API නැත — නවතම HARTI bulletin එක.
- නොදන්නා දෙයක් ඇසුවොත්: "මම එය පරීක්ෂා කර නැත, නමුත් මගේ ප්‍රවේශය …"

## If the app fails
Wrong Python → use `start_app.bat` · Port busy → use 8502 or close old window · No internet → skip Latest HARTI ·
Slow first load → "building features, then cached".
