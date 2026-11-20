> **Superseded.** This is an earlier project report, kept for its audit history. The final report is
> `reports/final_submission/Final_Report.docx` (PDF alongside).

# AI-Powered Sri Lankan Vegetable Price Prediction: Project Report

**Version:** 1.0.0
**Report date:** 2026-11-07
**Data covered:** 2024-04-25 to 2026-11-06

## Summary

This project predicts wholesale vegetable prices at Sri Lanka's Dambulla market, from 1 to 30 days ahead. It uses real data: 870 HARTI price bulletins, Open-Meteo weather and CEYPETCO fuel prices. The system covers the whole path from reading source PDFs to a Streamlit dashboard.

The final model is a **direct forecaster**: a separate Ridge regression for each number of days ahead. On a test period of nearly six months, its next-day average error was Rs. 17.47 (7.51%), which is **4.3% lower** than simply repeating the last known price. The reduction ranges from 4.2% to 7.8% across all forecast ranges, and its 95% interval stays above zero in each one.

The gain is real but modest. Vegetable prices stay hard to predict, the gain varies by month and vegetable, and from 8 days ahead the model doesn't reduce the largest errors. A notable finding is that the earlier Linear Regression, Random Forest and XGBoost models had high R² (up to 0.965) but didn't beat yesterday's price. This is why every result here is compared with a simple baseline.

## 1. Problem and objectives

Vegetable prices in Sri Lankan markets move sharply from day to day. A useful estimate of future prices could help producers, traders and analysts plan. A prediction system is only useful, however, if it does better than the obvious alternative: assuming tomorrow's price will equal today's.

The objectives were to:

1. Build a validated dataset from real published sources, with no invented values.
2. Predict Dambulla wholesale prices up to 30 days ahead, using only information available on the forecast date.
3. Measure accuracy honestly on later data, against a simple baseline.
4. Provide a dashboard that explains its results and says when it can't produce one.
5. Back the system with automated tests and browser checks.

## 2. Data

All project datasets are real. A small synthetic sample is bundled only for trying the dashboard, and is labelled as synthetic everywhere it appears.

| Dataset | Rows | Range | Source |
|---|---|---|---|
| `harti_prices.csv` | 23,809 | 2024-04-25 to 2026-11-06 | HARTI daily bulletins, Dambulla wholesale |
| `harti_all_markets.csv` | 184,996 | 2024-04-25 to 2026-11-06 | The same bulletins, all 10 markets |
| `fuel_prices.csv` | 926 | 2024-04-25 to 2026-11-06 | CEYPETCO diesel and petrol revisions, per day |
| `weather_data.csv` | 924 | 2024-04-25 to 2026-11-04 | Open-Meteo archive, Dambulla |
| `weather_data_by_region.csv` | 7,392 | 2024-04-25 to 2026-11-04 | Open-Meteo archive, 8 locations |

**How values are derived.** HARTI publishes a daily price range, so each price is the range's midpoint. A fuel price is the latest published revision, carried forward until the next one. No value is random, estimated or interpolated in these files.

**Scope.** Predictions cover Dambulla and 31 vegetable labels. The other 9 markets are inputs only. `config/crop_production_regions.yaml` maps 14 vegetables to an evidence-based production region with a cited source, so Carrot, for example, uses Nuwara Eliya's weather. Other vegetables use the market's weather rather than a guessed location.

### 2.1 Validation

`scripts/validate_real_datasets.py` writes the full results to `DATA_QUALITY_REPORT.md`.

**Comparison with source PDFs.** All 870 PDFs were read again by a separate extraction route (`Table.extract()`, where the dataset was built by word position). Of 184,996 cells, **184,996 matched**, with none differing and none found on only one side. Both routes share the same label and market name mapping, so a mapping error wouldn't show up there. To cover this, 40 random readings (4 per market) were also checked by eye against page images, and all 40 matched.

| Check on `harti_all_markets.csv` | Result |
|---|---|
| Duplicate (bulletin, market, vegetable) rows | 0 |
| Missing values | 0 |
| Non-positive prices | 0 |
| Unknown market or vegetable names | 0 |
| Reported after its bulletin, or more than 7 days before it | 0 |
| Bulletins with no PDF, or PDFs with no readings | 0 |
| Dambulla rows or prices differing from `harti_prices.csv` | 0 |
| Dates outside the fuel data | 0 |
| Dates outside the weather data | 2 (2026-11-05 and 06, weather archive lag) |
| Same reading repeated in a later bulletin | 53 |
| At least 5× from the median (kept, needs review) | 275 |
| At least 3× day-over-day change (kept, needs review) | 26 |

Most of the unusual values are Lime in October 2025, seen across several markets. They are kept as published, because removing them would mean replacing real records with a judgement call. `DATA_SOURCES.md` lists the most notable cases.

### 2.2 Known gaps

Dambulla has no price on 63 of 926 calendar days (6.8%):

- **50 days** have no bulletin listed.
- **7 bulletins** exist, but with no Dambulla price.
- **6 listings** link to a folder instead of a PDF.

Four of those six (2025-06-29, 07-02, 07-03 and 07-04) were later located on HARTI's server as valid PDFs, each with 28 Dambulla prices. They are **not** in the datasets or the model, because adding them would change the validated data and require retraining and re-evaluation. Missing records were never filled in the datasets.

Some vegetables are sparse. Cabbage (Nuwara Eliya) has 103 rows and no reading since 2025-07-12, and Drumstick has 181.

## 3. System design

```
config/                  Settings, and vegetable-to-region mapping
src/
  data_collection/       Raw data loaders; live HARTI, Open-Meteo and CEYPETCO clients
  preprocessing/         Clean, standardize and merge
  feature_engineering/   Features for both model families; chronological split; encoding
  models/                Training, evaluation, selection, persistence; direct_forecaster.py
  prediction/            Inference for the earlier models
scripts/                 Rebuild datasets, validate, experiment, train and backtest
app/                     Streamlit dashboard
tests/                   Automated tests
reports/                 This report, data documentation, saved metrics
```

Stack: Python 3.12, pandas 3.1, NumPy 2.5, scikit-learn 1.9, XGBoost 3.4, Streamlit 1.66 and PyMuPDF 2.1. Paths and settings live in `config/config.yaml`, and exact versions are pinned in `requirements.txt`.

## 4. Models

### 4.1 Earlier models, kept for comparison

Linear Regression, Random Forest (`n_estimators=200`, `max_depth=10`) and XGBoost (`n_estimators=200`, `learning_rate=0.05`, `max_depth=6`) were trained on 51 features: date, exact-day price lags, rolling price statistics, 14-day-old weather, fuel lags and categorical identity. Forecasting beyond one day feeds each prediction back in as the next day's input, so errors build up.

Because these models didn't beat yesterday's price (section 6.2), a different design was developed.

### 4.2 The direct forecaster

`src/models/direct_forecaster.py` trains one model for each horizon from 1 to 30 days. Each one is a median imputer, a standard scaler and a Ridge regression, and predicts the **log change** from the last known price to the price that many days later. A forecast therefore uses only real data, never an earlier forecast, and needs no future weather.

| Days ahead | Ridge α | Feature groups |
|---|---|---|
| 1 | 1000 | momentum, level, market, fuel, 14-day-old weather, other markets |
| 2–10 | 1000 | momentum, rain totals, 14-day-old weather, level, other markets |
| 11–21 | 1000 | rain totals, market, level, other markets |
| 22–30 | 100 | momentum, calendar, rain totals, market, level |

All models also receive each vegetable's identity.

**Gaps.** A reading carries forward for up to 7 days, and the model is told how old it is. After that, the forecaster refuses rather than guesses.

### 4.3 How the configuration was chosen

Model type, regularization and feature groups were chosen on four consecutive 3-month validation windows before the test period (`scripts/model_experiments.py`), then scored on the test period once.

Validation MAE reduction compared with holding the last price:

| Days ahead | Linear | Ridge α=100 | Ridge α=1000 | Random Forest | XGBoost |
|---|---|---|---|---|---|
| 1 | 3.18% | 3.28% | **3.74%** | 3.59% | 3.26% |
| 7 | 5.36% | 5.47% | **6.09%** | 5.05% | 5.44% |
| 14 | 8.13% | 8.55% | **9.22%** | 7.65% | 8.96% |
| 30 | 16.71% | **16.85%** | 15.58% | 10.33% | 12.09% |

Percentage points of gain lost when each feature group is removed, on validation:

| Days ahead | Price level | Momentum | Other markets | Rain totals | 14-day-old weather | Fuel |
|---|---|---|---|---|---|---|
| 1 | 0.12 | 1.23 | **3.81** | -0.18 | 0.05 | -0.04 |
| 7 | **3.86** | 0.63 | 2.02 | 0.42 | 0.19 | -0.02 |
| 14 | **8.38** | -0.56 | 0.22 | 1.18 | -0.04 | -0.59 |
| 30 | **17.53** | 0.38 | -0.40 | 1.83 | 0.11 | -1.68 |

Two findings stand out. Other markets' prices drive the next-day gain, and a vegetable's price level drives the longer-range gain, because prices tend to drift back toward their usual range. Weather and fuel added little, which is a useful result in itself: rainfall totals help slightly at longer ranges, while fuel changes didn't help at any range tested.

Gains on the test period were lower than on validation at longer ranges, about 4–8% rather than 9–17%. The test scores in section 6 are the ones to rely on.

## 5. Preventing data leakage

A forecast is only meaningful if it uses what was actually known on the forecast date.

- **Time-ordered evaluation.** Test scores come from models trained only on prices dated before the test period. Data is never shuffled.
- **Backward-looking features.** Lag and rolling features look only at earlier dates. Tests change every later value and confirm that no earlier feature moves.
- **Publication dates.** Meegoda and Veyangoda are reported 1 to 3 days late, so a reading dated one day may not be public until later. Other markets' prices are therefore keyed to the bulletin that published them, and a test confirms that a reading can't affect features before its bulletin appears.
- **Training-only fitting.** The earlier models' categorical encoder is fitted once, on the training split, and never refitted at prediction time.

## 6. Results

The test period is 2026-05-14 to 2026-11-06. MAE, RMSE and MAPE are measured against the actual recorded price. "Hold" means repeating the last known price.

### 6.1 Direct forecaster

| Days ahead | Rows | MAE (Rs) | RMSE (Rs) | MAPE | R² | Hold MAE | Hold RMSE | Hold MAPE | MAE reduction (95% interval) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 4,773 | 17.47 | 28.62 | 7.51% | 0.969 | 18.26 | 30.83 | 7.81% | 4.3% (2.6 to 6.0) |
| 2–3 | 9,537 | 23.78 | 39.88 | 9.92% | 0.940 | 25.10 | 42.22 | 10.60% | 5.3% (3.5 to 6.9) |
| 4–7 | 19,029 | 32.13 | 54.12 | 13.01% | 0.889 | 33.53 | 55.42 | 13.98% | 4.2% (2.5 to 5.9) |
| 8–14 | 33,187 | 42.76 | 73.34 | 16.86% | 0.796 | 44.70 | 72.82 | 18.67% | 4.3% (2.0 to 6.4) |
| 15–21 | 33,067 | 53.53 | 93.93 | 20.72% | 0.667 | 57.70 | 93.71 | 23.91% | 7.2% (4.5 to 9.8) |
| 22–30 | 42,374 | 65.66 | 115.01 | 25.10% | 0.502 | 71.20 | 114.33 | 29.36% | 7.8% (4.4 to 11.2) |

The intervals come from resampling whole dates, because vegetables on one date tend to move together.

**Reading these results:**

- **Average errors improve at every range,** with intervals above zero.
- **The largest errors don't.** From 8 days ahead, RMSE is about the same as holding the price.
- **Error grows with distance.** At 22–30 days it is about 25%, so distant forecasts are rough guides.

### 6.2 Comparison with the earlier models

Next-day, on the 3,836 rows every method can score:

| Method | MAE (Rs) | RMSE (Rs) | MAPE | R² |
|---|---|---|---|---|
| **Direct forecaster** | **17.56** | **28.89** | **7.42%** | **0.9693** |
| Hold last price | 18.27 | 30.97 | 7.69% | 0.9647 |
| Earlier Linear Regression | 18.63 | 30.98 | 7.82% | 0.9647 |

Earlier models on their own 3,915 test rows:

| Method | MAE (Rs) | RMSE (Rs) | R² |
|---|---|---|---|
| Yesterday's price (no model) | 18.17 | 30.77 | 0.9649 |
| Linear Regression (selected) | 18.56 | 30.82 | 0.9648 |
| Random Forest | 18.75 | 31.23 | 0.9639 |
| XGBoost | 19.84 | 37.74 | 0.9472 |

None of these beats yesterday's price. Their high R² mostly reflects how closely today's price follows yesterday's and how much average price differs between vegetables, so **R² alone would have been misleading here**.

For 15–30 days ahead, on the same sample of 8 vegetables and 6 start dates, error was 25.8% for the direct forecaster, 27.8% for holding the price, and 36.5% for forecasting one day at a time.

### 6.3 Consistency

**By vegetable.** Next-day, the model beats holding the price for 21 of 30 vegetables with test rows. It does worse for Big Onion (Local), Big Onion (Imported), Manioc, Beetroot, Beetroot (Nuwara Eliya), Brinjal, Lime, Ladies Fingers and Drumstick. Cabbage (Nuwara Eliya) has no test rows.

**By month.** MAE reduction compared with holding the price, by the month of the predicted date:

| Days ahead | May | Jun | Jul | Aug | Sep | Oct | Nov |
|---|---|---|---|---|---|---|---|
| 1 | 6.2% | 7.2% | 3.6% | 3.5% | 3.3% | 1.7% | 6.1% |
| 7 | 11.0% | 10.4% | -1.8% | 8.5% | -3.0% | -4.8% | 11.2% |
| 14 | 18.9% | 12.7% | 2.5% | 24.4% | -10.1% | -14.2% | 7.8% |
| 30 | 37.8% | 18.4% | 18.4% | 14.0% | -6.3% | -15.5% | -17.9% |

May and November are part months. Next-day gains stayed positive throughout, but longer-range gains turned negative in several months, particularly September and October. Longer-range forecasts should not be assumed to beat a simple baseline in any given month.

## 7. Dashboard

The Streamlit dashboard (`app/Home.py`) has five pages:

- **Home:** the approach, and whether the forecaster is ready.
- **Predict:** next-day and multi-day forecasts, with a chart, the weather used, the sources and the model behind each result.
- **Model Performance:** the results above, read directly from the saved reports.
- **Price History:** each vegetable's prices and weather. The line breaks across gaps longer than 7 days rather than implying prices were recorded.
- **Cultivation Planner:** from a planting date, the expected harvest window, the price around it and a suggested selling period, for 77 crop and variety entries (24 crops, 17 with a HARTI price series).
  - **Harvest timing** is the Department of Agriculture's published guidance (`data/reference/doa_harvest_durations_v2.csv`), shown with its source. It is not a model prediction: no farm-level planting and harvest records exist to train one.
  - **Prices** are the direct forecaster's output, shown only when the harvest falls within its 30-day reach, and only when the price series is recent. Each forecast carries a likely range: the middle 80% of the forecaster's errors on the test period for all vegetables together (not per crop), from `scripts/forecast_error_bands.py`. The page states the latest start date that still gets a forecast, and names the HARTI record the forecast starts from and the weather region the model uses.
  - **A selling day** is suggested only when the forecast covers the whole harvest window and that day's low estimate beats the first harvest day's forecast.
  - **Otherwise**, the page leads with a labelled price reference, never a forecast: the prices recorded in a harvest window that has passed; else a Historical Seasonal Price Reference (median of the same weeks, ± 7 days, of the 5 most recent earlier years, needing at least 10 prices in 3 years); else the latest actual price with its age. A crop with no HARTI series gets an explanation and no price (`select_price` in `app/cultivation.py`).
  - **Further reading:** `reports/CULTIVATION_DATA_PLAN.md` sets out the data a trained harvest-duration model, and longer price forecasts, would need.

| Data source (since 2026-11-15) | Model |
|---|---|
| Latest Real Data Prediction: the stored official records, with HARTI, CBSL and DCS status and each source's records shown separately | Direct forecaster (HARTI data, weather and fuel only) |
| Upload CSV | Earlier comparison model, or a temporary sandbox model |

Earlier versions also offered "Bundled sample (synthetic)" and a live "Latest HARTI Data" download; they were removed from the page, and the scheduled update (every 3 hours) keeps the stored records current instead.

**No silent model changes.** The forecaster needs its model file, the all-markets dataset and the regional weather file. Final verification found that removing one of those datasets could change a forecast or switch model with no warning. This was corrected. The page now names the problem and gives no real-data prediction, and the earlier model is used only if the user explicitly asks for it. A sandbox model is saved separately and can't overwrite the project's models.

**Uploaded data.** Uploads use the earlier model because the forecaster needs other markets' prices and regional weather for the same dates, and it has never been evaluated on uploads. No accuracy score applies to an upload, and the page says so. Section 9 of the `README.md` lists these limits in full.

## 8. Testing and verification

**Automated tests:** 554 passed, 0 failed, 0 skipped.

| Area | Tests |
|---|---|
| Dashboard and services (`tests/app`) | 177 |
| Data collection | 125 |
| Feature engineering | 120 |
| Models | 57 |
| Prediction | 35 |
| Preprocessing | 29 |
| Scripts | 11 |

Tests cover correctness rather than just whether code runs: leakage, error handling, damaged files and model selection on the Predict page. Hashes confirmed that running the suite leaves datasets, models and saved metrics unchanged. The suite still reports 345 warnings, all the same pandas deprecation in `date_features.py`, which doesn't affect current results.

**Real-browser testing.** Every page was opened and used in a real browser against a running server. This covered several vegetables, forecast ranges of 1, 7 and 30 days, refusal of day 31, invalid uploads, and date ranges with no data. Damaged-file scenarios ran on a temporary copy of the project, including missing datasets, unreadable model files and corrupt reports. Each gave a clear message, with no crash.

**End-to-end testing.** Every vegetable at 1, 7, 14 and 30 days ahead (124 cases) was run in the browser. Each displayed price matched one calculated independently from the saved model, and each named the Ridge forecaster. Latest HARTI mode matched for all 31 vegetables, and the 31 vegetables' weather inputs matched values recalculated from the raw files. Model Performance and Price History matched their source files on 475 values. 44 kinds of uploaded file and 63 damaged-file scenarios gave clear messages, and simulated HARTI, CEYPETCO and Open-Meteo outages were handled visibly.

**Defects found by this verification, and fixed:**

- A live bulletin's "Eggplant" rows didn't reach "Brinjal" in the forecaster's features.
- Models could change silently when a dataset was missing (section 7).
- Messages about missing files were misleading or contradicted each other.
- A damaged regional weather file crashed the Predict and Price History pages.
- Exact duplicate rows in an upload silently changed its prediction, and a day with two different prices wasn't flagged.
- Home reported the forecaster as ready when a price file was damaged.

**Not verified:**

- Other browsers, mobile layouts and light theme.
- Every vegetable at every horizon.
- Behaviour during real outages of HARTI, Open-Meteo or CEYPETCO.
- Very large uploads.
- Whether future forecasts prove correct, which can't be known yet.

## 9. Limitations

- **Modest gain.** The improvement over the simple baseline is real but small, and it is inconsistent across months and vegetables.
- **Evaluation.** There is one fixed test period and one training cut-off, covering under six months, so seasonal effects aren't fully tested. Earlier work had already reported other models' results on this period, so it isn't untouched in the strictest sense, although its results didn't influence the configuration.
- **Saved model.** The saved models were refit on all data after scoring, so that exact version has no separate test score. The scores above come from models trained before the test period.
- **Publication timing.** The model assumes a bulletin is available by the next day's forecast. HARTI sometimes publishes later.
- **Scope.** Predictions cover Dambulla only.
- **Data.** Known bulletins are missing, some vegetables are sparse, and some published values look unusual.
- **Weather.** Each region uses one town's coordinates, not a district average, and only a fixed 14-day lag and recent totals are used.
- **Uploaded data.** It uses an older model with no accuracy score for that data.
- **Project setup.** The project isn't under version control, and `src/visualization/` is an unused empty package. `.gitignore` excludes the datasets, model files and metrics, so they need separate backups.

## 10. Future work

- Add the four located bulletins, then rebuild, revalidate, retrain and re-evaluate.
- Evaluate over several test periods, including a full year, using rolling evaluation.
- Investigate why longer-range gains fell in September and October 2026.
- Add prediction intervals, so that each forecast shows its uncertainty.
- Evaluate whether the direct forecaster can serve uploaded data.
- Extend forecasts to other markets, which would need their own evaluation.
- Adopt version control and automated test runs.

## 11. Conclusion

The project delivers a complete, tested system built on validated real data, from source PDFs to a working dashboard. Its Ridge direct forecaster beats the simple baseline on average at every range tested, by about 4–8%, with its largest measured benefit coming from other markets' published prices next-day and from price levels further ahead.

The honest conclusion is that this is a useful but limited improvement. Prices follow their recent values closely, so a simple baseline is hard to beat, and the earlier models' high R² hid that they weren't beating it. Measuring against that baseline, keeping evaluation strictly time-ordered, and stating limits plainly are as important to this project's results as the model itself.

## Appendix: reproducing the results

Requires Python 3.12. Python 3.15 isn't supported, because PyMuPDF 2.1 depends on onnxruntime, which has no Python 3.15 build. The commands below are for Windows.

```bash
py -3.12 -m venv .venv312
```

```bash
.venv312\Scripts\python.exe -m pip install -e ".[dev]"
```

```bash
.venv312\Scripts\python.exe -m pytest
```

```bash
.venv312\Scripts\python.exe -m streamlit run app/Home.py
```

Running `scripts/train_forecaster.py` retrains the model and overwrites the saved report, so back up `models/` and `reports/metrics/` first.

| Evidence | Location |
|---|---|
| Direct forecaster results | `reports/metrics/direct_forecaster_metrics.json` |
| Earlier model results | `reports/metrics/model_metrics.json` |
| One-day-at-a-time backtest | `reports/metrics/forecast_backtest.json` |
| Validation experiments | `reports/metrics/model_experiments_validation.json` |
| Data sources and known issues | `reports/DATA_SOURCES.md` |
| Measured data quality | `reports/DATA_QUALITY_REPORT.md` |
| Monthly gains (section 6.3) | Recomputed from `score_on_test_period()` in `scripts/train_forecaster.py`; not stored in a saved report |
