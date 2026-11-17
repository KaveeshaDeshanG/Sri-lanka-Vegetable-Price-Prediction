# Beans price prediction — verification report (15 November 2026)

Everything below was executed on 15 Nov 2026 against the real datasets, the saved model and the running app.
No code, dataset or model was changed during this investigation.

## 1. Files and functions involved

| Step | File | Function / line |
|---|---|---|
| Data source radio (default "Historical dataset (real)") | `app/pages/1_Predict.py` | line 354 |
| Load real history (HARTI Dambulla + weather + fuel) | `app/services.py` | `load_real_market_history` (cached by `data_version`) |
| Vegetable / market dropdowns (only forecastable series) | `app/pages/1_Predict.py` | ~578 / 596; `app/catalogue.py` `series_status` |
| Latest actual date for the chosen series | `app/pages/1_Predict.py` | line 657–659 (`actual_group_history["date"].max()`) |
| Feature building (cached, key includes data version) | `app/pages/1_Predict.py` 249–252, 696–697 | `ForecastFeatureBuilder.build_origin_features` (`src/feature_engineering/forecast_features.py` 65) |
| Forecast request | `app/pages/1_Predict.py` 703–713 | `_show_direct_forecast` → `direct_forecast` (`app/forecasting.py` 116) |
| Horizon = target date − latest actual date; refuses > 30 | `app/forecasting.py` | 140–149 |
| Model per horizon, price = last price × exp(predicted log change) | `src/models/direct_forecaster.py` | `forecast` 134, `predict_rows` 126 |
| Display | `app/pages/1_Predict.py` | `_render_direct_result` 255 |

Model: `models/direct_forecaster.joblib` — Ridge direct forecaster, 30 models (horizons 1–30), trained through
2026-11-06, includes Beans. No hardcoded dates or prices; the synthetic sample is used only when "Bundled sample data"
is chosen explicitly. The displayed price is the model output (independently recomputed below).

## 2. Latest official Beans data (checked live on 15 Nov 2026, 11:47)

`scripts/daily_update.py` run on 15 Nov: HARTI newest listed bulletin **2026-11-12**, CBSL nothing after **2026-11-13**,
DCS nothing after **November 2026 week 1** → no newer official price exists; nothing was invented.

| Source | Date | Commodity | Market | Price type | Min | Max | Value used | Unit | Document |
|---|---|---|---|---|---|---|---|---|---|
| HARTI daily bulletin | 2026-11-12 | Beans | Dambulla | Wholesale | 380 | 480 | **430** (midpoint) | Rs/kg | `Vegetable Pricenew ex1(2026.11.12).pdf` (harti.gov.lk) |
| CBSL Daily Price Report | 2026-11-13 | Beans | Dambulla | Wholesale | – | – | 375 | Rs/kg | `price_report_20261113_e.pdf` |
| CBSL Daily Price Report | 2026-11-13 | Beans | Dambulla / Pettah / Narahenpita | Retail | – | – | 405 / 500 / 700 | Rs/kg | same |
| DCS weekly retail | Nov 2026 week 1 | Beans – Green | Colombo District | Retail weekly avg | 640 | 800 | 731.03 | Rs/kg | `DCSB-WRP-2026-11-W1` |

The model predicts **HARTI Dambulla wholesale** prices only. CBSL and DCS are separate measurements (CBSL collects its
own prices; DCS is retail) and are not model inputs.

Recent HARTI Dambulla Beans (midpoints): 11-05 415 · 11-06 440 · 11-07 465 · 11-08 450 · 11-09 425 · 11-10 340 · 11-11 330 · 11-12 430.

## 3. Does the prediction use the latest data? — Yes
- 2,918 Beans prices; newest 2026-11-12 (Rs 430) is the forecast origin (`last_price` = 430, `days_since_reading` = 0).
- Leakage test: rebuilding features with every row after the origin removed gives identical features (max difference 0.0), at 2026-11-12 and at 2026-10-13.
- The label column `target_price` is never among the model's inputs.

## 4. Beans predictions (origin 2026-11-12, last actual Rs 430/kg)

| Horizon | Forecast date | Script result | Independent recompute | Streamlit UI |
|---|---|---|---|---|
| 1 day | 2026-11-13 | Rs 413.04 | Rs 413.04 | **Rs 413.04 / kg** |
| 7 days | 2026-11-19 | Rs 431.62 | Rs 431.62 | **Rs 431.62 / kg** |
| 14 days | 2026-11-26 | Rs 459.86 | Rs 459.86 | **Rs 459.86 / kg** |
| 30 days | 2026-12-12 | Rs 447.14 | Rs 447.14 | **Rs 447.14 / kg** |

UI: Beans and Dambulla available; 0 exceptions; price in Rs/kg; correct date and horizon; model named; chart shows
actual (green) and forecast (orange dashed) separately; a "FORECAST, not a guaranteed price" warning is shown.

## 5. Beans accuracy (chronological; models fitted only on targets before the test window; history from 2024-04-25 as in production)

| Window | Horizon | MAE | Hold MAE | RMSE | Hold RMSE | MAPE % | Hold MAPE % | R² | MAE vs hold |
|---|---|---|---|---|---|---|---|---|---|
| Test 2026-05-14 → 11-06 (168 per h) | 1 | 34.55 | 39.32 | 44.39 | 49.90 | 9.35 | 10.49 | 0.737 | 12.1 % lower |
| | 7 | 60.36 | 64.94 | 74.51 | 82.58 | 15.67 | 17.06 | 0.259 | 7.1 % lower |
| | 14 | 73.56 | 84.97 | 92.47 | 106.03 | 19.40 | 22.30 | −0.142 | 13.4 % lower |
| | 30 | 84.43 | 108.10 | 103.21 | 130.68 | 21.95 | 28.00 | −0.423 | 21.9 % lower |
| Recent 2026-09-14 → 11-12 (57 per h) | 1 | 35.39 | 40.61 | 47.42 | 52.27 | 8.67 | 9.73 | 0.456 | 12.9 % lower |
| | 7 | 63.46 | 68.25 | 82.99 | 90.35 | 14.79 | 15.81 | −0.667 | 7.0 % lower |
| | 14 | 58.33 | 75.09 | 78.37 | 96.42 | 13.42 | 17.22 | −0.486 | 22.3 % lower |
| | 30 | 52.61 | 69.30 | 67.21 | 80.36 | 12.35 | 16.13 | −0.093 | 24.1 % lower |

MAPE is safe: the lowest actual Beans price in these windows is Rs 200 (no zeros). Negative R² at 7–30 days means the
forecasts track the window's average level worse than a hindsight constant would; the errors are still lower than
holding the price. Typical next-day error ≈ Rs 35/kg (≈ 9 %); at 30 days ≈ Rs 50–85/kg (12–22 %).

## 6. Findings
**No bug in the prediction pipeline.** Data, features, model, horizons and display are correct and current.

Gaps against the requested result display (not errors):
1. The latest actual **price** (Rs 430) is not shown as a number — only its date (the chart shows it).
2. The **last data update** time is shown on Home, not on the Predict result.
3. **Accuracy for the chosen horizon** is not shown on the result — only a pointer to Model Performance.

Observation: in automated testing, typing into the date box needed a click elsewhere before Streamlit registered it;
picking from the calendar works normally.

## 7. Proposed minimal fix (not yet applied)
In `_render_direct_result` (`app/pages/1_Predict.py`) only: add a line with the latest actual price and its date, the
last update time (from `data/raw/update_log.jsonl`, via the existing `latest_update`), and the measured MAE / MAPE for
the forecast's horizon band from `reports/metrics/direct_forecaster_metrics.json` (all vegetables, test period).
No model, data or other page changes.
