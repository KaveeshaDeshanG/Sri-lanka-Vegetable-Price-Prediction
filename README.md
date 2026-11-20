# Sri Lankan Vegetable Price Prediction

Vegetable price prediction for the Dambulla wholesale market, built on real HARTI bulletin prices,
Open-Meteo weather and CEYPETCO fuel prices. It is a modular, configuration-driven Python pipeline
(`src/`) covering preprocessing, feature engineering, multi-model training and evaluation, and
inference, with a Streamlit dashboard (`app/`).

**Key result:** next-day error is 4.3% lower than holding the last price, and 4–8% lower up to 30 days
ahead, all statistically clear. Its largest gains come from other markets' published prices (section 6).

## 1. Architecture

```
config/                  ConfigLoader, config.yaml, crop_production_regions.yaml
src/
  data_collection/       Raw CSV loaders and live HARTI, Open-Meteo and CEYPETCO clients
  preprocessing/         Clean, standardize, and merge market, weather and fuel data
  feature_engineering/   Date, lag, rolling, weather-lag and fuel-lag features; split; encoding;
                         forecast_features.py for the direct forecaster
  models/                Baseline, trainer, evaluator, selection, persistence, metrics report;
                         direct_forecaster.py
  prediction/            Inference artifacts and prediction pipeline
scripts/
  build_harti_prices.py                  Rebuild harti_prices.csv from HARTI's PDF archive
  build_market_prices.py                 Rebuild harti_all_markets.csv from the cached PDFs
  build_fuel_prices.py                   Rebuild fuel_prices.csv from CEYPETCO's price archive
  build_weather_data.py                  Rebuild weather_data.csv from Open-Meteo's archive
  build_historical_weather_by_region.py  Rebuild weather_data_by_region.csv
  validate_real_datasets.py              Write reports/DATA_QUALITY_REPORT.md
  train_forecaster.py                    Test and save the direct forecaster the app uses
  model_experiments.py                   Compare ideas on validation windows only
  train.py                               Train and compare the earlier 51-feature models
  backtest_forecast.py                   Measure the earlier one-day-at-a-time forecasting
app/
  Home.py, services.py, forecasting.py, cultivation.py, catalogue.py, i18n.py, translations/
  pages/  1_Predict.py, 2_Model_Performance.py, 3_Price_History.py, 4_Cultivation_Planner.py
tests/                   One pytest subpackage per src/ package, plus tests/app (1,320 test cases)
reports/
  final_submission/      Final report (DOCX, PDF), screenshots, diagrams and test evidence
  DATA_SOURCES.md, DATA_QUALITY_REPORT.md, PROJECT_REPORT.md (earlier report, superseded)
  metrics/direct_forecaster_metrics.json, metrics/model_metrics.json, metrics/forecast_backtest.json
```

The app uses the **direct forecaster** for real data (section 6), and the data flow below for the
earlier comparison models and for sandbox models on sample or uploaded data:

```
data/raw/*.csv
  -> PreprocessingPipeline: load -> clean -> transform -> merge
  -> apply_production_region_weather()   (weather from each vegetable's production region)
  -> FeatureEngineer.build_features()    (date, lag, rolling, weather-lag, fuel-lag)
  -> chronological_train_test_split()    (by date, never shuffled)
  -> CategoricalEncoder                  (fit on the training split only)
  -> ModelTrainer -> ModelEvaluator -> select_best_model()
  -> models/best_model.joblib, models/feature_pipeline_state.joblib,
     models/artifact_metadata.json, reports/metrics/model_metrics.json
  -> PredictionPipeline.predict() -> dashboard
```

Leakage prevention is structural: the split is chronological, every lag and rolling feature looks
only backward, and the encoder is fit once on the training split and never refit at inference.

## 2. Datasets

All datasets are real. Sources, derivation and known issues are in
[`reports/DATA_SOURCES.md`](reports/DATA_SOURCES.md); measured quality is in
[`reports/DATA_QUALITY_REPORT.md`](reports/DATA_QUALITY_REPORT.md).

| File | Rows | Range | Source |
|---|---|---|---|
| `harti_prices.csv` | 77,011 | 2015-06-22 to 2026-11-19 | HARTI bulletins, Dambulla wholesale (35 series) |
| `harti_all_markets.csv` | 529,020 | 2015-06-22 to 2026-11-19 | HARTI bulletins, all 11 markets (Pettah until 2022) |
| `cbsl_daily_prices.csv` | 10,551 | 2026-09-01 to 2026-11-19 | CBSL Daily Price Report, wholesale and retail, kept apart |
| `dcs_weekly_retail_prices.csv` | 11,439 | 2019-12 week 4 to 2026-11 week 1 | DCS weekly retail prices, Colombo District |
| `fuel_prices.csv` | 4,169 | 2015-06-22 to 2026-11-19 | CEYPETCO price revisions, per day |
| `weather_data.csv` | 4,166 | 2015-06-22 to 2026-11-16 | Open-Meteo archive, Dambulla |
| `weather_data_by_region.csv` | 33,328 | 2015-06-22 to 2026-11-16 | Open-Meteo archive, 8 locations |

Figures as of 2026-11-19; the datasets are kept current by the automatic update every 3 hours (section 10), so
the live numbers grow. The latest dates are recorded in `data/raw/data_manifest.json` and shown on the Home
page. 2,925 of the 4,169 calendar days have a Dambulla price. Gaps include days with no bulletin and 224 bulletins that couldn't be read. 151 of those
are from 2022-2023, mostly the one-page "Min/Max" layout, and 41 can't be downloaded. The reasons are
in `harti_prices_build_report.json`.

**The saved forecaster is still the one trained on 2024-04-25 to 2026-11-06.** Its accuracy below was
measured on that data. Retraining on the longer history goes through `scripts/retrain_if_better.py`,
which keeps a new model only if it beats the current approach on recent, unseen prices. Its last run
(2026-11-13) trained a candidate on 2015-2026 and rejected it: on the most recent 60 days it clearly beat
holding the last price up to 7 days ahead, was level at 8-14 days and worse at 15-30 days. The decision is in `reports/metrics/retrain_decision.json`.

The datasets (`data/raw/`), trained models (`models/*.joblib`) and evaluation results (`reports/metrics/`)
are committed to Git, so a fresh clone runs the app without rebuilding. The source PDF bulletins are not
(about 700 MB); keep them backed up separately.

Predictions are for one market (Dambulla) and the 31 series the forecaster was trained on. The other
markets are inputs only. Four series that HARTI stopped reporting in 2019-2023 (Bitter Gourd and
Brinjals "Other" and "Village") have history but no forecast, as does Cabbage (Nuwara Eliya), whose
last Dambulla price is 2025-07-12.
HARTI's "Eggplant" row is called "Brinjal" throughout the project, and is separate from HARTI's
"Brinjals" row.

The app also bundles a small **synthetic** sample (`app/sample_data/sample_history.csv`), for trying
it with a temporary sandbox model.

## 3. Crop-specific production-region weather

[`config/crop_production_regions.yaml`](config/crop_production_regions.yaml) maps 14 of the
vegetables to one evidence-based production region (for example, Carrot to Nuwara Eliya), each with
a cited source. Other vegetables use the market's own weather, never a guessed location.

`apply_production_region_weather()` replaces a row's same-day weather with its region's reading
before lag features are computed. `WeatherFeatureGenerator` groups by vegetable, so each vegetable's
lag feature comes from its own series. `market` still identifies where the price was observed.

When a prediction needs a reading that isn't in the loaded rows, `build_live_prediction_input()`
looks in this order:

1. `weather_data_by_region.csv`, which also works during a network outage.
2. Open-Meteo's archive, for dates over 7 days old.
3. Open-Meteo's forecast endpoint, up to 16 days ahead.
4. Open-Meteo's seasonal endpoint, up to 46 days ahead.

If none has it, the prediction is refused rather than invented.

## 4. Earlier model features (51)

- **Date (5):** `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`
- **Price lag (4):** `price_lag_{1,7,14,30}d`, exact calendar day, per vegetable and market
- **Price rolling (6):** `price_rolling_{mean,std}_{7,14,30}d`, excluding the current row
- **Weather lag (3):** `{rainfall_mm,temperature_c,humidity_pct}_lag14d`
- **Fuel lag (2):** `diesel_price_lag_1d`, `petrol_price_lag_1d`
- **Categorical (31):** 30 `vegetable_*` columns and `market_Dambulla`

## 5. Earlier models

These are kept as the comparison for the direct forecaster, and for sandbox models.

| Model | Hyperparameters (`config/config.yaml`) |
|---|---|
| Linear Regression | None |
| Random Forest | `n_estimators=200`, `max_depth=10`, `random_state=42` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`, `random_state=42` |

The split is chronological at 80/20 of 23,809 rows, with the boundary at 2026-05-14. Dropping rows
with incomplete features leaves 13,749 training rows and 3,915 test rows.

## 6. Results

### Direct forecaster (used by the app)

`src/models/direct_forecaster.py` has one Ridge regression for each day up to 30 days ahead. Each
predicts the log change from today's price to the price that many days later, so a forecast never
feeds on an earlier forecast or on future weather.

Held-out test period, 2026-05-14 to 2026-11-06, from `reports/metrics/direct_forecaster_metrics.json`:

| Days ahead | Model MAE (Rs) | Model error | Hold last price MAE (Rs) | Hold last price error | MAE reduction (95% interval) |
|---|---|---|---|---|---|
| 1 | 17.47 | 7.51% | 18.26 | 7.81% | 4.3% (2.6 to 6.0) |
| 2-3 | 23.78 | 9.92% | 25.10 | 10.60% | 5.3% (3.5 to 6.9) |
| 4-7 | 32.13 | 13.01% | 33.53 | 13.98% | 4.2% (2.5 to 5.9) |
| 8-14 | 42.76 | 16.86% | 44.70 | 18.67% | 4.3% (2.0 to 6.4) |
| 15-21 | 53.53 | 20.72% | 57.70 | 23.91% | 7.2% (4.5 to 9.8) |
| 22-30 | 65.66 | 25.10% | 71.20 | 29.36% | 7.8% (4.4 to 11.2) |

The interval comes from resampling whole dates. From 8 days ahead, RMSE is about the same as
holding the last price, so the model reduces typical errors rather than the largest ones. Next-day,
it beats holding the last price for 21 of 30 vegetables, and it isn't consistently better every
month.

**How the configuration was chosen.** Model type, regularization and feature groups were chosen on
four validation windows before the test period (`scripts/model_experiments.py`), then scored on the
test period once. Earlier work had reported other models' results on this same test period, so it
isn't untouched in the strictest sense, but no result from it influenced this configuration.

**What helped, on validation:**
- Direct forecasting and each vegetable's price level: prices tend to drift back toward their usual
  range. This drives most of the longer-range gain.
- Other markets' prices: this made next-day gain clear for the first time (0.6% to 3.9%). They're keyed
  to the bulletin that published them, because Meegoda and Veyangoda are reported 1 to 3 days late.
  Using their reported date instead inflated the gain to 5.8%, which would have been leakage.
- Rainfall totals over recent days: a small gain at 7 days and beyond.
- Weather and fuel features had little effect on next-day prediction.

### Earlier models

One-day-ahead predictions on the test period, from `reports/metrics/model_metrics.json`:

| Method | MAE (Rs) | RMSE (Rs) | R² |
|---|---|---|---|
| Linear Regression | 18.56 | 30.82 | 0.9648 |
| Random Forest | 18.75 | 31.23 | 0.9639 |
| XGBoost | 19.84 | 37.74 | 0.9472 |
| Yesterday's price (no model) | 18.17 | 30.77 | 0.9649 |

None of these beats yesterday's price: their high R² mostly reflects how closely today's price
follows yesterday's, and how much average price differs between vegetables.

On the 3,836 next-day rows that every method can score, MAE is 17.56 for the direct forecaster, 18.63
for the earlier Linear Regression, and 18.27 for holding the last price. Compared with forecasting one
day at a time on 8 vegetables and 6 start dates, its error at 15-30 days is 25.8% rather than 36.5%.

## 7. Installation

Use **Python 3.12**, which is what the project is tested with. Python 3.15 can't install it:
`pymupdf==2.1` requires `onnxruntime`, and no `onnxruntime` release has a Python 3.15 build
(checked 2026-11-08). Python 3.13 and 3.14 have builds of every pinned package but are untested.
Plain `python` may start a newer version, so name 3.12 explicitly when creating the environment:

```bash
py -3.12 -m venv .venv312
```

```bash
.venv312\Scripts\python.exe -m pip install -r requirements.txt
```

To run the tests, also install the project and the development tools:

```bash
.venv312\Scripts\python.exe -m pip install -e ".[dev]"
```

`onnxruntime`, `pymupdf_layout` and `pymupdf4llm` are installed as requirements of `pymupdf` 2.1. The
project's code doesn't use them.

An environment created with one Python version can't be switched to another: its installed
packages are built for the first version. Create a new one instead.

## 8. Running

Rebuild the datasets if `data/raw/` is empty. HARTI's download is about 700 MB, so point
`--cache-dir` somewhere with space; an interrupted run resumes from it. Use today's date for `--end`
on fuel, and about two days earlier on weather, because the archive runs behind.

```bash
.venv312\Scripts\python.exe scripts\build_harti_prices.py --cache-dir C:\harti_pdfs
```

```bash
.venv312\Scripts\python.exe scripts\build_market_prices.py --cache-dir C:\harti_pdfs
```

```bash
.venv312\Scripts\python.exe scripts\build_fuel_prices.py --end 2026-11-06
```

```bash
.venv312\Scripts\python.exe scripts\build_weather_data.py --end 2026-11-04
```

```bash
.venv312\Scripts\python.exe scripts\build_historical_weather_by_region.py
```

```bash
.venv312\Scripts\python.exe scripts\validate_real_datasets.py --cache-dir C:\harti_pdfs
```

Train and measure. Run `train.py` first, since `train_forecaster.py` compares with its model.

```bash
.venv312\Scripts\python.exe scripts\train.py --config config\config.yaml
```

```bash
.venv312\Scripts\python.exe scripts\backtest_forecast.py
```

```bash
.venv312\Scripts\python.exe scripts\train_forecaster.py
```

Run the tests and the dashboard:

```bash
.venv312\Scripts\python.exe -m pytest
```

```bash
.venv312\Scripts\python.exe -m streamlit run app/Home.py
```

On Windows, double-clicking `start_app.bat` in the project folder runs the same command and opens
http://localhost:8501. Keep its window open while using the app, and close it to stop.

The dashboard's dark theme is set in `.streamlit/config.toml`, which Streamlit reads when started from
the project folder. The shared cards, header, sidebar and chart styling are in `app/ui.py`.
Current screenshots (18 Nov 2026, in English, Sinhala and Tamil) are in `reports/final_submission/assets/screenshots/`;
`docs/screenshots/` keeps earlier ones from 9 Nov 2026, before the language switcher was added.

## 9. Dashboard

- **Predict:** choose a data source, a vegetable and a date, then request a next-day prediction or a
  forecast up to 30 days ahead.
- **Model Performance:** the direct forecaster's accuracy, the earlier models, and comparisons with
  the baseline, all read from the saved reports.
- **Price History:** each vegetable's recorded prices and weather, with a date range and table view.
- **Cultivation Planner:** for 77 crop and variety entries (24 crops, 17 of them with a HARTI price series), the expected
  harvest window from a start date and, where the harvest falls within 30 days of the latest HARTI Dambulla price, the
  forecast price around it. Before you generate a plan it states the latest start date that still gets a forecast.
  The plan names the HARTI record the forecast starts from and the weather region the model uses. The weather
  selector only changes the weather panel. The likely range is measured on all vegetables together. A selling day is
  suggested only when the forecast covers the whole harvest window. Harvest
  timing is the Department of Agriculture's published guidance (`data/reference/`), not a model:
  no planting and harvest records exist to train one. Every plan has a section 3, "Price at harvest", that leads
  with one labelled price, by priority: a **Model Forecast** (harvest within the forecaster's 30-day reach); for a
  harvest window that has passed, the **actual prices recorded** in it; beyond the reach (45, 90, 180, 365 days…), a
  **Historical Seasonal Price Reference – Not a Forecast** (the median of HARTI Dambulla prices in the same weeks,
  harvest dates ± 7 days, of the 5 most recent earlier years, with the count, years, range and median/average;
  at least 10 prices in 3 years); otherwise the **Latest Available Actual Price – Not a Future Prediction**, with its
  date and age. A crop with no HARTI series gets **No Verified Price Available** and the missing source, never Rs 0
  or another vegetable's price. The forecaster itself is unchanged. `reports/CULTIVATION_DATA_PLAN.md` explains
  the data a trained model would need.

The Predict page offers two data sources:

| Data source | What it uses | Model |
|---|---|---|
| **Latest Real Data Prediction** (the default) | The official records in `data/raw/`, kept current by `scripts/daily_update.py` (every 3 hours, or **Check official sources now**) | Direct forecaster |
| **Upload CSV** | Only your file: `date`, `vegetable`, `market`, `price`; weather and fuel columns are optional. Never mixed with official data. | Earlier model, or a sandbox model |

Latest Real Data Prediction has five sections:
1. **Official source status:** for HARTI, CBSL and DCS, the newest publication found, the last check,
   whether new data arrived, a failed check or import problem, and whether the source is behind schedule.
2. **Latest collected data:** the newest HARTI Dambulla price of every vegetable, then the selected
   vegetable's records source by source: HARTI Dambulla (the forecaster's series), HARTI's other markets
   (forecaster inputs), CBSL and DCS (reference only, matched through `data/reference/source_item_map.csv`).
   Each record shows its observation and publication date, market, price type, unit, minimum, maximum,
   average or midpoint, document link and collection time. The exact record the forecast starts from is named.
3. **Vegetable and market:** Dambulla, wholesale, Rs per kg.
4. **Prediction:** 1, 7, 14 or 30 days ahead (or another date within 30 days), plus a table of all four
   with the accuracy measured for each range.
5. **Data freshness and warnings.**

CBSL and DCS records are shown, never used by the model: their collectors, price types and periods
differ from HARTI's. The earlier "Bundled sample data" and live "Latest HARTI Data" options are no
longer offered; their code remains in `app/services.py` and is still tested.

A sandbox model never overwrites a production model, and `Predictor.load()` refuses a model, encoder
and feature list from different training runs.

The direct forecaster needs no future weather and no network for Historical data. If weather data
ends slightly before a price date, the rainfall totals use the days available.

**If the forecaster can't run:** it needs `models/direct_forecaster.joblib`,
`data/raw/harti_all_markets.csv` and `data/raw/weather_data_by_region.csv`. If any is missing or
unreadable, the Predict page names the problem and gives no real-data prediction. The earlier
comparison model is used only if you tick the box asking for it, and every result states which model
produced it. The dashboard never switches model on its own.

### Languages: English, Sinhala and Tamil

Every page has a language switcher at the top right: **English · සිංහල · தமிழ்** (English is the
default). The choice applies at once, stays through reruns and page changes, and is kept in the address
(`?lang=si`, `?lang=ta`), so a reload or a bookmark opens in the same language. Everything the app writes
is translated: page and section titles, the sidebar, buttons, help text, messages, tables, chart titles,
axes, legends and tooltips, the data freshness cards and the planner's price comparison. Dates keep their
numbers (month names are translated; chart axes show 2026-11-17-style dates), and prices stay in Rs.

- **How it works:** `app/i18n.py` provides `t("English text", **values)`. The English text is the key,
  so English output is exactly what the code says and a missing translation falls back to English.
  Translations are in `app/translations/si.json` and `ta.json`; `en.json` lists every key.
- **Vegetable names** keep the familiar English name and add the local one ("Beans · බෝංචි"), taken from
  the Department of Agriculture's names in `data/reference/vegetable_names.csv`.
- **Not translated, on purpose:** stored data (market, item and cultivar names, DCS's "week n" labels),
  source names (HARTI, CBSL, DCS, CEYPETCO, Open-Meteo), the Department of Agriculture's quoted timing
  wording (marked as quoted in English), file names and code identifiers. Model inputs, predictions and
  datasets are the same in every language.
- **Adding or changing text:** wrap it in `t()`, then run
  `.venv312/Scripts/python.exe scripts/i18n_texts.py --run-pages --write-en` to list it and see what
  needs a Sinhala and Tamil translation. `tests/app/test_i18n.py` fails if any text lacks one.
- The Sinhala and Tamil wording was written for this project and should be reviewed by a native
  speaker, especially the agricultural terms, before wider use.

### Uploaded data: current limitations

Uploaded data uses the earlier comparison model (Linear Regression, 51 features, trained on real
Dambulla prices), not the direct forecaster. This hasn't been changed because the forecaster has
never been evaluated on uploaded data, and:

- Its features include other markets' published prices and regional weather for the same dates, which
  an upload doesn't have. Without them it still returns a number, filled in from training medians, so
  the absence would go unnoticed.
- It only covers the vegetables it was trained on, and only Dambulla.

Switching would first need a compatibility check for these inputs and a separate evaluation on data of
that kind. Until then:

- **Accuracy:** no score applies to an upload. The results on Model Performance describe real Dambulla
  prices, and that model doesn't beat holding the last price next-day even there.
- **Distant dates:** these are reached one day at a time. On real data, this has about 36.5% error at
  15-30 days, against 27.8% for holding the last price and 25.8% for the direct forecaster.
- **Fuel columns:** without `diesel_price` and `petrol_price`, the saved model can't run. The page
  offers a temporary sandbox model instead, trained and scored only on the uploaded rows.
- **Other names:** a vegetable or market the model never saw gets a warning, and its result rests only
  on the recent prices supplied.
- **File checks:** the upload must be a comma-separated UTF-8 CSV with lowercase headers. Exact
  duplicate rows are dropped, as in training. Rows dated after today, or a day with two different
  prices, are kept but flagged on the page.
- **Gaps:** the features need prices on exact earlier dates. Missing days in the last 30 are estimated
  from surrounding prices and listed on the page.
- **Weather:** for a vegetable with a mapped production region, the saved regional weather replaces the
  upload's weather on dates it covers. Other missing readings are fetched from Open-Meteo, which
  needs a network connection unless that option is unticked.

### Using the prediction pipeline directly

```python
from src.prediction.pipeline import PredictionPipeline

# input_df: history in the merged schema, plus one row for the target date with price=NaN
pipeline = PredictionPipeline(config_path="config/config.yaml")
result = pipeline.predict(input_df, target_date=target_date, vegetable="Tomato", market="Dambulla")
```

A missing feature raises `FeatureAlignmentError` naming it, and the model never receives an
invented value.

## 10. Latest HARTI Data

HARTI has no API, only a [daily archive](https://www.harti.gov.lk/daily-price.php) of PDF bulletins
organized in one table per year. `HartiLiveClient` reads every year table, downloads the bulletin
and finds the Dambulla column. Each cell is read by word position, because PyMuPDF's
`Table.extract()` scrambles this PDF's dates and prices and removes spaces from labels.

This is HARTI's most recently *published* bulletin, not a real-time feed. Each price is the midpoint
of the published "min - max" range. A vegetable with no reading ("-") has no row, and the app falls
back to historical data for it.

Live fuel prices come from the public endpoint behind CEYPETCO's own page. If that fails, the last
known fuel price is carried forward and the page says so.

### Daily updates

The stored datasets are kept current by `scripts/daily_update.py`, which:
- adds every newly published HARTI bulletin, CBSL Daily Price Report and DCS weekly bulletin;
- extends the weather and fuel prices, and the other markets' min-max ranges (once
  `scripts/build_market_price_ranges.py` has built them);
- checks each source separately: one that can't be reached is recorded as failed and the others
  are still checked (exit code 4);
- backs up each file before changing it, and never changes stored rows.

The Windows task `VegetablePriceUpdate` runs it every 3 hours (`scripts/run_daily_update.bat`):

```bat
schtasks /Create /TN "VegetablePriceUpdate" /TR "\"<project folder>\scripts\run_daily_update.bat\"" /SC HOURLY /MO 3 /ST 07:00 /F
schtasks /Query /TN "VegetablePriceUpdate" /V /FO LIST
schtasks /Delete /TN "VegetablePriceUpdate" /F
```

Output goes to `logs/daily_update.log`; each run, with every source's status, is a line in
`data/raw/update_log.jsonl`, which the Home and Predict pages read. A full rebuild of every bulletin
since 2015 is `scripts/update_data.py`. Details: `reports/DATA_SOURCES.md`, "Daily updates".

## 11. Gaps in the data

The direct forecaster carries the last reading forward for up to 7 days and tells the model how old
it is. After that, it refuses to forecast rather than guess.

The earlier models' lag features need a price on an exact earlier date, but HARTI has no Dambulla
reading on some days. For them, a missing day in the trailing 30 days is filled in this order:

1. A real archived bulletin, when the live bulletin is used (the services behind the former
   "Latest HARTI Data" option).
2. A linear estimate between surrounding real prices, tagged `IMPUTED_HISTORICAL`.

Estimates exist only in memory, are never written to disk, and are listed on the page each time they
are used. The warning becomes an error-styled notice above 7 estimated days, as with Potato (Nuwara
Eliya), whose result then rests mostly on estimates.

## 12. Future date forecasts

**Real data:** the direct forecaster reaches up to 30 days after the latest price. Error still grows
with distance (about 25% at 22-30 days), so treat distant forecasts as rough guides.

**Uploaded data:** a distant date is reached by predicting one day at a time and feeding
each prediction back in. Forecast rows are tagged `FORECAST` and exist only in memory. The furthest
reachable date is today + 45 days + the 14-day weather lag, and beyond that the forecast is refused.

## 13. Known limitations

- **Size of the gain:** the improvement over holding the last price is real but modest. Prices remain
  hard to predict, especially around shocks such as Cyclone Ditwah.
- **Consistency:** gains vary by month and vegetable. Next-day, 9 of 30 vegetables do worse than holding
  the last price, and 14-day gains were negative in September and October 2026.
- **Large errors:** from 8 days ahead, RMSE is about the same as holding the last price, so the gain
  comes from typical errors, not the largest ones.
- **Publication timing:** the model assumes a bulletin is available by the next day's forecast. HARTI
  sometimes publishes later.
- **Evaluation:** one fixed test period and one training cut-off. The saved model is then refit on all
  data, so that exact version has no separate test score.
- **Data:** 6 bulletins have broken links on HARTI's archive, so the datasets and model don't include
  them, although 4 have since been located. Some published values look unusual and are kept as
  published. Both are covered in `reports/DATA_SOURCES.md`.
- **Uploaded data:** uses an older model with no accuracy score for that data (section 9).
- **Sparse vegetables:** Cabbage (Nuwara Eliya) has had no reading since 2025-07-12. Others have limited data.
- **Market:** predictions are only for Dambulla.
- **Location proxy:** each mapped region uses one town's coordinates, not a district average.
- **Weather lag:** only one fixed 14-day lag is used.
