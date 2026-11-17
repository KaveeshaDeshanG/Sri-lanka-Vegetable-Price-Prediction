# 06 — Machine Learning Explanation

There are **two generations** of models. Know both and which one the app uses.

| | Earlier models (phase 1) | **Direct forecaster (used by the app for real data)** |
|---|---|---|
| Code | `src/models/trainer.py`, `scripts/train.py` | `src/models/direct_forecaster.py`, `scripts/train_forecaster.py` |
| Algorithms | Linear Regression, Random Forest (200 trees, depth 10), XGBoost (200 trees, lr 0.05, depth 6) | **Ridge Regression**, one model per day ahead (30 models) |
| Predicts | Price tomorrow (1 day) | Price 1–30 days ahead |
| Target | Price in rupees | **log(future price ÷ today's price)** — the % change |
| Features | 51 (date, lags 1/7/14/30, rolling 7/14/30 mean/std, 14-day weather lag, fuel, one-hot vegetable, market) | Groups chosen per horizon (below) |
| Split | Chronological 80/20 (train 13,749, test 3,915 rows) | Fixed date: test from **2026-05-14** (`config.yaml` `forecasting.test_start`) |
| Used in app | Only for "Upload CSV" | For "Latest Real Data Prediction" |

> සිංහල: පළමුව Linear Regression, Random Forest, XGBoost ආකෘති තුනක් "හෙට මිල" අනාවැකි කීමට පුහුණු කළා. ඒවා
> කිසිවක් "ඊයේ මිලම හෙටත්" (naive baseline) යන සරල ක්‍රමයට වඩා හොඳ වුණේ නැහැ. එම නිසා දෙවනුව **Ridge direct
> forecaster** එක සෑදුවා — එය baseline එකට වඩා හොඳයි.

---

## 1. Step-by-step: from raw data to a prediction

### Step 1 — Raw data
`data/raw/harti_prices.csv` — 76,719 Dambulla wholesale prices (Rs/kg), 22 Jun 2015 – 9 Nov 2026, from HARTI PDFs.
Plus `harti_all_markets.csv` (other markets), `weather_data_by_region.csv` (Open-Meteo), `fuel_prices.csv` (CEYPETCO).

### Step 2 — Preprocessing (`src/preprocessing/`)
- `cleaner.py` `clean_market_data` (77), `clean_weather_data` (119), `clean_fuel_price_data` (150): drops exact
  duplicates (`_drop_exact_duplicates`, 183), rows missing critical fields (191), invalid prices (207), impossible
  weather (temperature 0–45 °C, humidity 0–100 %, line 217; limits in `config.yaml` `preprocessing`), dates before 2000;
  interpolates small weather gaps (311) and forward-fills fuel prices (347).
- `transformer.py` `standardize_vegetable_names` (76): one name per vegetable (e.g. HARTI "Eggplant" → "Brinjal").
- `merger.py` `merge` (19) and `merge_fuel_prices` (66): joins prices with weather and fuel by date.
- `pipeline.py` `PreprocessingPipeline.run` (line 133) runs them in order.

> සිංහල: Preprocessing = දත්ත පිරිසිදු කිරීම. වැරදි දින, අසාමාන්‍ය උෂ්ණත්ව අගයන්, එකම පේළිය දෙවරක් — ඉවත් කරයි.

### Step 3 — Feature engineering (`src/feature_engineering/forecast_features.py`)
A **feature** is an input number the model learns from. `ForecastFeatureBuilder.build_origin_features` (line 65)
makes one row per vegetable per day:
- A missing day's price is carried forward at most **7 days** (`MAX_CARRY_DAYS`, line 24); a vegetable needs ≥ 60 prices (`MIN_ROWS_PER_VEGETABLE`).
- Everything is on a **log scale** (`np.log`, line 70) so a Rs 20 change on Rs 100 and Rs 200 on Rs 1,000 look the same (both 20 %).

Feature groups (`FEATURE_GROUPS`, lines 29–45):

| Group | Columns | Meaning | සිංහල |
|---|---|---|---|
| level | `log_last` | Today's price (log) | අද මිල |
| momentum | `log_change_1/3/7/14/30`, `vs_mean_7/14/30`, `volatility_7/30`, `vs_max_30`, `vs_min_30`, `days_since_reading` | How the price has moved recently | මෑත මිල වෙනස්වීම |
| calendar | `target_day_of_week`, `target_year_sin/cos` | Day of week and season of the target date | සතියේ දිනය, කන්නය |
| rain_windows | `rain_sum_3/7/14/30`, `temp_mean_7/30`, `humidity_mean_7` | Recent rain/temperature in the **growing region** | වගා කරන ප්‍රදේශයේ වර්ෂාව |
| weather_lag_14 | `rain/temp/humidity_lag_14` | Weather 14 days before the target date | දින 14කට පෙර කාලගුණය |
| market | `market_change_1/7`, `market_vs_mean_30` | Is the whole market rising? (median of all vegetables) | මුළු වෙළඳපොළේ ප්‍රවණතාව |
| fuel | `diesel_change_30`, `petrol_change_30` | Fuel price change over 30 days | ඉන්ධන මිල වෙනස |
| other_markets | `other_premium`, `other_change_1/3`, `other_dispersion`, `premium_<market>` for Peliyagoda, Kandy, Keppetipola, Nuwaraeliya, Bandarawela, Meegoda | How other markets' prices compare with Dambulla | අනෙක් වෙළඳපොළවල් සමඟ සංසන්දනය |
| vegetable | `veg_<name>` one-hot | Which vegetable | කුමන එළවළුව |

Weather per vegetable comes from its production region (`config/crop_production_regions.yaml`, resolved in
`_vegetable_frame`, line 97); a vegetable with no mapped region uses Dambulla weather.

**Avoiding data leakage** (using information that would not be known on the day): other markets are keyed by the
date the bulletin was **published** (line 58, 136), not the trading date; weather lags use `max(14 − horizon, 0)` so they
never look past today (line 175); the target price is shifted **into the future** only as the label (line 179).

> සිංහල: "Data leakage" = අනාවැකි කීමේ දිනයේ නොදන්නා අනාගත දත්ත වැරදීමකින් පුහුණුවට ඇතුළු වීම. එය වැළැක්වීමට,
> අනෙක් වෙළඳපොළවල මිල bulletin එක **ප්‍රකාශයට පත් කළ දිනයට** අනුව පමණක් භාවිතා කරයි.

### Step 4 — Training (`DirectForecaster.fit`, `direct_forecaster.py` lines 102–124)
For each horizon h = 1 … 30:
1. `add_horizon_features` adds the target date, its calendar, and `target_price` = the real price h days later.
2. Keep rows where both today's price and the target price exist (`usable_rows`, line 167).
3. Pick that horizon's **band** (lines 47–52):

| Days ahead | Ridge α | Feature groups |
|---|---|---|
| 1 | 1000 | momentum, level, market, fuel, weather_lag_14, other_markets |
| 2–10 | 1000 | momentum, rain_windows, weather_lag_14, level, other_markets |
| 11–21 | 1000 | rain_windows, market, level, other_markets |
| 22–30 | 100 | momentum, calendar, rain_windows, market, level |

4. Model = `make_pipeline(SimpleImputer(median), StandardScaler(), Ridge(alpha))` (line 118):
   - **SimpleImputer** fills a missing feature with the median.
   - **StandardScaler** puts every feature on the same scale (mean 0, std 1).
   - **Ridge** = linear regression with a penalty (α) that keeps coefficients small → less overfitting.
5. It learns **y = log(target_price / last_price)** (line 121).

Trained on 25 Apr 2024 – 6 Nov 2026, 31 vegetable series; saved to `models/direct_forecaster.joblib`.

> සිංහල: **Ridge Regression** = සාමාන්‍ය රේඛීය regression එකට "දඬුවමක්" (penalty, α) එක් කළ ආකාරයයි. එයින්
> ආකෘතිය අධික ලෙස පුහුණු දත්තවලට ගැළපීම (overfitting) අඩු කරයි. **Direct** යනු එක් එක් දින ගණනට (1, 2 … 30)
> වෙනම ආකෘතියක් තිබීමයි — අනාවැකිය නැවත input එකක් ලෙස යොදන්නේ නැත, එබැවින් වැරදි එකතු නොවේ.

**Why the bands and α values?** Chosen on four 3-month **validation windows** (14 May 2025 – 13 May 2026), before the
test period, by `scripts/model_experiments.py`; then scored **once** on the test period (docstring lines 8–10).

### Step 5 — Prediction (`predict_rows`, line 126; `forecast`, line 134)
```
change = model[h].predict(features of today)
predicted price = last_price × exp(change)
```
Example (tested): Beans, last price Rs 425 → 1 day ahead Rs 453.74; 16 days ahead Rs 507.96.

### Step 6 — Evaluation (`score_predictions`, line 59; `mae_gain_vs_holding_price`, line 70)
| Metric | Formula idea | Meaning | සිංහල |
|---|---|---|---|
| **MAE** | average of \|actual − predicted\| | Average error in Rs/kg | සාමාන්‍ය දෝෂය (රුපියල්) |
| **RMSE** | √(average of error²) | Punishes big errors more | විශාල දෝෂවලට වැඩි බරක් |
| **MAPE** | average of \|error\| ÷ actual × 100 | Error as % of price | ප්‍රතිශතයක් ලෙස දෝෂය |
| **R²** | 1 − SSE/SST | Share of price variation explained (1 = perfect) | ආකෘතිය පැහැදිලි කරන ප්‍රමාණය |
| **Gain vs hold** | 1 − MAE_model / MAE_hold | % lower error than "price stays the same" | baseline එකට වඩා කොපමණ හොඳද |
| **95 % interval** | 2,000 bootstrap resamples of whole **dates** (line 84) | How sure the gain is real | විශ්වාසනීය පරාසය |

**Baseline ("hold the last price")**: tomorrow's (or day-h) price = today's price. Every result is compared to it.

## 2. Results (test 14 May – 6 Nov 2026, `reports/metrics/direct_forecaster_metrics.json`)

| Days ahead | Model MAE (Rs/kg) | Hold-price MAE | MAE lower by (95 % interval) |
|---|---|---|---|
| 1 | 17.47 (MAPE 7.51 %, R² 0.969) | 18.26 | **4.3 %** (2.6–6.0) |
| 2–3 | 23.78 | 25.10 | 5.3 % (3.5–6.9) |
| 4–7 | 32.13 | 33.53 | 4.2 % (2.5–5.9) |
| 8–14 | 42.76 | 44.70 | 4.3 % |
| 15–21 | 53.53 | 57.70 | 7.2 % |
| 22–30 | 65.66 (MAPE 25.1 %, R² 0.502) | 71.20 (MAPE 29.4 %) | 7.8 % (4.4–11.2) |

- Next-day, the model beats the baseline for **21 of 30** vegetables.
- Earlier models (next-day): Linear Regression MAE 18.56, Random Forest 18.75, XGBoost 19.84; yesterday's price 18.17 →
  **none beat the baseline**. On the same 3,836 rows: direct 17.56, earlier best 18.63, hold 18.27.
- Recursive (feed predictions back) vs direct at 15–30 days: MAPE 36.5 % vs 25.8 % → why "direct" was chosen.
- Ablation: other-market prices help most at 1 day; the price level matters most at long horizons.
- **Honest caveat:** at 8–14 and 22–30 days the model's **RMSE is slightly higher** than holding (73.34 vs 72.82;
  115.01 vs 114.33) — it is better on average but makes a few bigger misses.

> සිංහල: ආකෘතිය baseline එකට වඩා **4–8 %** කින් හොඳයි — විශාල වෙනසක් නොවේ, නමුත් සංඛ්‍යානමය වශයෙන් සත්‍යයි
> (95 % පරාසය ශුන්‍යයට ඉහළින්). දින 30ක් ඉදිරියට දෝෂය ~25 % යි — දිගු කාලීන අනාවැකි අවිනිශ්චිතයි.

## 3. Why Ridge and not a complex model?
- Random Forest and XGBoost did **not** beat the naive baseline next-day; prices are close to a random walk.
- With ~2.5 years of training rows and many correlated features, a penalised linear model generalises better.
- Fast to train (seconds), easy to explain, coefficients can be inspected.

## 4. Overfitting / underfitting
- **Overfitting** = learns noise in training data, fails on new data. Prevented by: Ridge penalty, chronological split
  (no shuffling), choosing settings only on validation windows, scoring the test period once.
- **Underfitting** = too simple. R² at 1 day is 0.969 (mostly because today's price explains tomorrow's), but the gain
  over the baseline is small — the honest conclusion is that short-term price changes are hard to predict.

## 5. Cultivation Planner's "likely range"
`reports/metrics/forecast_error_bands.json` (made by `scripts/forecast_error_bands.py`) stores the 10th and 90th
percentile of past forecast errors per horizon; `app/cultivation.py` `forecast_with_range` applies them to the
forecast. It is **not** a formal prediction interval from the model.

## 6. What was NOT done (be honest)
No ARIMA/SARIMA, Prophet, LSTM or SHAP. No hyperparameter grid search beyond the band/α validation. The model
has not been retrained on the full 2015–2026 history (`scripts/retrain_if_better.py` exists, never run).
