# Sri Lankan Vegetable Price Prediction

AI-powered price prediction for Sri Lankan vegetable markets, built on real historical
market-price, weather, and fuel-price data. The system is a modular, configuration-driven
Python pipeline (`src/`) covering data preprocessing, feature engineering, multi-model
training/evaluation, and inference, plus a Streamlit application (`app/`) that exposes it
to an end user -- including a **Latest HARTI Data** prediction option that fetches HARTI's
own most recently published daily bulletin (see §10) alongside the original bundled-sample
and upload-your-own-CSV options.

For an exhaustive, code-traced technical write-up of the pipeline's internals (leakage
prevention, error-handling contracts, module-by-module design rationale), see
[`reports/PROJECT_REPORT.md`](reports/PROJECT_REPORT.md). That report was written before
real data was acquired, so its "no real data yet" / metrics-not-persisted statements are
superseded by this README and by `reports/metrics/model_metrics.json`.

## 1. Architecture

```
config/                  ConfigLoader -- dotted-path access to config/config.yaml
  crop_production_regions.yaml   Vegetable -> production-region mapping (see §3)
src/
  data_collection/       Load + validate raw market/weather/fuel-price CSVs, live weather
                         client, production-region resolver
  preprocessing/         Clean, standardize, and merge market + weather + fuel data
  feature_engineering/   Date/lag/rolling/weather-lag/fuel-lag features, production-region
                         weather correction, chronological split, categorical encoding,
                         dataset building
  models/                Baseline, trainer, evaluator, model selection, persistence,
                         metrics report
  prediction/            Inference artifact persistence + prediction pipeline
scripts/
  train.py                                  Production training entry point (this file's
                                             own orchestration only -- every step is a call
                                             into an already-tested component)
  build_historical_weather_by_region.py      One-time real Open-Meteo archive fetch per region
app/                     Streamlit UI (Home.py, pages/1_Predict.py) + services.py
tests/                   pytest suite, one subpackage per src/ package (357 tests)
reports/
  metrics/model_metrics.json   Persisted MAE/RMSE/R2 for every trained model + best-model pick
  PROJECT_REPORT.md            Full technical deep-dive (see caveat above)
```

Data flow:

```
data/raw/*.csv (real HARTI/weather/fuel data)
  -> PreprocessingPipeline: load -> clean -> transform -> merge
  -> apply_production_region_weather()           (corrects same-day weather per vegetable's
                                                   production region, see §3 -- data/raw/
                                                   weather_data_by_region.csv)
  -> FeatureEngineer.build_features()            (date/lag/rolling/weather-lag/fuel-lag)
  -> chronological_train_test_split()            (by date, never shuffled)
  -> CategoricalEncoder.fit_transform(X_train) / .transform(X_test)
  -> ModelTrainer.train_all()                    (Linear Regression, Random Forest, XGBoost)
  -> ModelEvaluator.evaluate_all() on the held-out test set
  -> select_best_model() by config.evaluation.best_model_metric
  -> save_model() + save_inference_artifact() + save_metrics_report() + save_artifact_metadata()
        -> models/best_model.joblib
        -> models/feature_pipeline_state.joblib
        -> reports/metrics/model_metrics.json
        -> models/artifact_metadata.json      (ties the three above to one training_run_id, see §6)
  -> PredictionPipeline.predict(new_row) -> Predictor.load() -> model.predict()
        -> app/pages/1_Predict.py renders the result or a typed error
```

Leakage prevention is structural, not just convention: the train/test split is
chronological (oldest 80% train, newest 20% test, per `model.test_size`); every lag/
rolling/weather-lag/fuel-lag feature only ever looks backward in time; the categorical
encoder is fit once, on the training split only, and reused (never refit) at inference.

## 2. Datasets (real data)

All three raw files under `data/raw/` are real, not synthetic:

| File | Rows | Source |
|---|---|---|
| `harti_prices.csv` | 24,459 | HARTI daily market-price bulletin, Dambulla (DEC) wholesale market |
| `weather_data.csv` | 855 | National daily weather series (rainfall, temperature, humidity) |
| `fuel_prices.csv` | 841 | National daily diesel/petrol retail prices (CEYPETCO) |

After cleaning (dropping 3,000 rows with a missing critical field) and merging on date,
the working dataset is **21,459 rows**, covering **one real market (Dambulla)** and
**30 real vegetables**: Ash Plantain, Beans, Beetroot, Beetroot (Nuwara Eliya), Big Onion
(Imported), Big Onion (Local), Bitter Gourd, Brinjal, Brinjals, Cabbage (Kandy), Cabbage
(Nuwara Eliya), Capsicum, Carrot, Cucumber, Green Chillies, Knol Khol, Ladies Fingers,
Leeks, Lime, Long Beans, Luffa, Manioc, Potato (Imported), Potato (Nuwara Eliya), Potato
(Welimada), Pumpkin, Radish, Snake Gourd, Sweet Potato, Tomato.

There is no `Pettah` market and no Carrot/Tomato-only vocabulary in this dataset --
those only appear in the Streamlit app's small bundled *synthetic* demo dataset
(`app/sample_data/sample_history.csv`, `source="SAMPLE"`), used purely so the app is
demonstrable without a user's own data.

## 3. Crop-Specific Production-Region Weather

By default, weather for a price prediction came from the market's own location
(Dambulla). This project now retrieves weather from a vegetable's evidence-based
**production region** instead, where the evidence supports one -- e.g. a Carrot
prediction uses Nuwara Eliya's weather, not Dambulla's, since 60% of national carrot
production is in Nuwara Eliya (HARTI Research Report No. 102, L.P. Rupasena,
*Production and Marketing of Vegetables*).

**Mapping**: [`config/crop_production_regions.yaml`](config/crop_production_regions.yaml)
maps 14 of the 30 real vegetables to a single primary production region, each with a
cited source and a `high`/`medium` confidence level (see the file's comments for the
full evidence list and the 16 vegetables deliberately left unmapped for insufficient
evidence -- e.g. Tomato, grown both up-country and in the Dambulla belt itself, or
Sweet Potato, documented as grown nationwide). A vegetable absent from the mapping
falls back to the market's own weather (`weather_api` section), identical to this
project's original behavior -- never a guessed location.

**How it reaches the model**: `src/feature_engineering/production_region_weather.py`'s
`apply_production_region_weather()` overwrites a row's same-day `rainfall_mm`/
`temperature_c`/`humidity_pct` with its vegetable's production region's reading for
that *exact* date (from `data/raw/weather_data_by_region.csv`, real Open-Meteo archive
data for 7 regions, built by `scripts/build_historical_weather_by_region.py`), before
`FeatureEngineer.build_features()` computes `rainfall_mm_lag14d`/etc. -- so those
lag features still mean exactly what they always meant ("the reading 14 days before
this row's date"), just sourced from the right location. This requires
`WeatherFeatureGenerator` to compute that lag lookup grouped by vegetable (see
`feature_engineer.py`'s `group_columns=[vegetable_column]`), not by date alone --
without that grouping, a date-only lookup would average different vegetables'
now-different same-day readings back together before the lag feature is even
computed, silently erasing this whole correction (a bug this project fixed after
initially building the mapping/merge layer above without also updating the lag
generator to match). Called from `scripts/train.py` (training) and
`app/pages/1_Predict.py` (prediction) -- both explicit, opt-in call sites, not baked
into `FeatureEngineer`/`PredictionPipeline` themselves.

**Live weather**: `app/services.py::build_live_prediction_input()` resolves the same
production region and fetches its *live* current weather from Open-Meteo
(`OpenMeteoWeatherClient.for_coordinates()`, the same client class used everywhere else
in this project -- no second API client) when a required historical date is missing.

**Market vs. weather location stay separate**: `market` still identifies where the
*price* was observed (Dambulla); the production-region mapping is keyed by vegetable
only and never reads `market` -- resolving Carrot's weather region is identical
whichever market it's sold in.

## 4. Features (51 total)

Built by `FeatureEngineer.build_features()` per `config/config.yaml`'s
`feature_engineering` section:

- **Date** (5): `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`
- **Price lag** (4): `price_lag_{1,7,14,30}d` -- exact-calendar-day self-join per
  (vegetable, market) group
- **Price rolling** (6): `price_rolling_{mean,std}_{7,14,30}d`, window `[date-n, date)`,
  excluding the current row
- **Weather lag** (3): `{rainfall_mm,temperature_c,humidity_pct}_lag14d`
- **Fuel-price lag** (2): `diesel_price_lag_1d`, `petrol_price_lag_1d`
- **Categorical** (31): one-hot `vegetable_*` (30 categories) + `market_Dambulla` (1
  category -- only one real market currently exists)

Same-day weather/fuel columns are excluded from the model-facing feature set (they'd
assume the current day is already observed, which doesn't hold for a genuinely future
prediction date) in favor of their lagged equivalents.

## 5. Models

All three trained via `ModelTrainer.train_all()`, hyperparameters read from
`config/config.yaml`:

| Model | Hyperparameters |
|---|---|
| Linear Regression | none (`sklearn.linear_model.LinearRegression`) |
| Random Forest | `n_estimators=200`, `max_depth=10`, `random_state=42` |
| XGBoost | `n_estimators=200`, `learning_rate=0.05`, `max_depth=6`, `random_state=42` |

Split: chronological, `test_size=0.2`, on the full 21,459-row merged dataset (2024-04-25
to 2026-08-13) -> 17,167 train-period / 4,292 test-period rows before dropping rows with
incomplete lag/rolling/weather-lag history. After that drop, the model-ready sets are
**12,383 training rows (2024-05-26 to 2026-02-27)** and **3,287 test rows (2026-02-27 to
2026-08-13)** -- the training set's effective start date is later than the raw data's
first date because the first ~30 days lack the full backward history `price_lag_30d`
needs.

## 6. Evaluation results (real test set, 3,287 rows)

Persisted in [`reports/metrics/model_metrics.json`](reports/metrics/model_metrics.json).
Trained WITH crop-specific production-region weather (§3):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | 18.4297 | 29.5057 | 0.9536 |
| **Random Forest** | 17.8510 | **28.7613** | 0.9559 |
| XGBoost | 17.7655 | 29.3845 | 0.9540 |

**Selected best model: Random Forest**, by `evaluation.best_model_metric: rmse` (lowest
RMSE), even though XGBoost has the lowest MAE -- selection is always by the single
configured metric, not a manual pick across metrics.

**Versus market-only weather** (the prior model, backed up as
`models/backups/*_MARKET_WEATHER_ONLY_backup_*.joblib`): Random Forest's RMSE improved
from 28.7676 to 28.7613 (~0.02%) -- a small, genuine improvement, not a dramatic one.
Linear Regression and XGBoost's RMSE moved slightly the other way. This is an honest,
expected result: weather-lag is 3 of 51 features, and only 14 of 30 vegetables (8,870
of 21,459 rows) actually got a different weather reading at all -- no larger effect
should be expected or was found.

Saved artifacts (all produced by one `scripts/train.py` run, sharing one `training_run_id`):
- `models/best_model.joblib` -- the selected Random Forest model
- `models/feature_pipeline_state.joblib` -- the fitted `CategoricalEncoder` + the exact
  51-column feature order the model expects, plus that run's `training_run_id`
- `models/artifact_metadata.json` -- `src/models/artifact_metadata.py`'s durable record of
  which training run produced the two files above: `training_run_id`, `trained_at`,
  `model_type`, the real training dataset's date range/row counts, `feature_columns`, and
  that run's metrics. `Predictor.load()` (`src/prediction/predictor.py`) validates the
  loaded model + `feature_pipeline_state.joblib` against this file whenever it's present,
  raising `ArtifactConsistencyError` on any mismatch -- see "Demo/production artifact
  separation" below for why this exists.
- `models/backups/*_SYNTHETIC_DEMO_backup_*.joblib` -- the prior demo-era artifacts
  (trained on synthetic Carrot/Tomato/Dambulla/Pettah data), kept for reference/rollback
- `models/backups/*_MARKET_WEATHER_ONLY_backup_*.joblib` -- the real-data model trained
  just before crop-specific production-region weather was added, kept for comparison/rollback
- `models/backups/*_SYNTHETIC_DEMO_OVERWRITE_backup_20260820.joblib` -- see below: the
  synthetic demo model that briefly, incorrectly, ended up at `models/best_model.joblib`
  itself, kept for the same reason

### Demo/production artifact separation

`app/services.py::train_demo_model()` backs the Predict page's "train a sandbox model on
the data loaded above" action, letting a user try the app on the bundled synthetic sample
or their own uploaded CSV without any real HARTI data. On 2026-08-20 this action was used
while the synthetic sample was loaded and it overwrote the real production
`models/best_model.joblib`/`feature_pipeline_state.joblib` with a `LinearRegression` fit on
that tiny 2-vegetable/2-market synthetic dataset -- `train_demo_model()` had no way to tell
"real production reload" apart from "synthetic sample" (both are just a `DataFrame`) and
saved to the same configured `paths.models_dir` either way.

Fixed by construction, not by convention: `train_demo_model()` now always persists under
`paths.demo_models_dir` (defaults to `<paths.models_dir>/demo` if unset; see
`app/services.py::resolve_demo_models_dir()`) and never accepts the production directory
implicitly. The Predict page loads a separate, independently cached "sandbox" `Predictor`
from that directory and only ever falls back to it when no production model exists at all
-- the production `Predictor` (loaded from `paths.models_dir`) is otherwise always what
predictions use. `models/artifact_metadata.json` (above) is the second, independent layer:
even if some future code path did write a mismatched model/encoder/feature-list into
`paths.models_dir`, `Predictor.load()` would refuse to use it rather than predict from an
inconsistent artifact set.

## 7. Installation

Requires Python >= 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\python.exe -m pip install -r requirements.txt
# macOS/Linux
.venv/bin/python -m pip install -r requirements.txt
```

To also run the test suite (pytest is a dev/test dependency, not a runtime one -- see
`pyproject.toml`'s `[project.optional-dependencies].dev`):

```bash
.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 8. Running

**(Optional, one-time) rebuild region weather** if `config/crop_production_regions.yaml`
changes or needs refreshing -- fetches real Open-Meteo archive data, does not touch
`harti_prices.csv`/`fuel_prices.csv`/`weather_data.csv`:

```bash
.venv\Scripts\python.exe scripts\build_historical_weather_by_region.py
```

**Train** (real data, crop-specific production-region weather applied, all 3 configured
models, best one persisted):

```bash
.venv\Scripts\python.exe scripts\train.py --config config\config.yaml
```

Add `--dry-run` to run preprocessing + feature engineering + the chronological split
and report shapes/feature names without fitting or saving any model.

**Run the test suite:**

```bash
.venv\Scripts\python.exe -m pytest
```

357 tests currently pass (`tests/data_collection` 101 incl. 36 in
`test_harti_live_client.py`, `tests/feature_engineering` 106, `tests/models` 41 incl. 9 in
`test_artifact_metadata.py`, `tests/preprocessing` 29, `tests/prediction` 33 incl. 6 covering
`Predictor.load()`'s artifact-metadata consistency validation, `tests/app` 36 incl. 11 in
`test_harti_live_services.py`, `tests/scripts` 11).

**Run the Streamlit app:**

```bash
.venv\Scripts\python.exe -m streamlit run app/Home.py
```

Navigate to the **Predict** page. If a trained model exists under `models/` (it does,
after step 1 above), it's loaded directly -- no retraining needed to make a prediction.

## 9. Prediction workflow

`src/prediction/pipeline.py`'s `PredictionPipeline.predict(input_df)` is the single
inference entry point, reused identically by the Streamlit app:

1. Provide a DataFrame with the same columns as the merged training data (`date`,
   `vegetable`, `market`, `price`, weather columns, and -- for this real, fuel-price-
   trained model -- `diesel_price`/`petrol_price`), with enough backward history for
   the vegetable/market group being predicted (up to 30 days for `price_lag_30d`, 14
   days for the weather/fuel lag, per `feature_engineering` config).
2. Add one row for the date being predicted, with `price` left `NaN` -- that is the row
   scored. Weather/fuel columns for that same row are also left `NaN`: no forecast is
   needed, since these are only ever looked up from *earlier* rows.
3. Call `PredictionPipeline(config_path=...).predict(input_df)`. It reuses the exact
   fitted encoder and feature-column order from `feature_pipeline_state.joblib` --
   never refits anything.
4. If historical context is insufficient for a required lag/rolling/weather-lag/fuel-lag
   feature, `Predictor` raises `FeatureAlignmentError` naming the exact missing
   feature(s) rather than fabricating a value. A missing required identifier column
   raises `FeatureEngineeringError`; a malformed date or "nothing to predict" raises
   `PredictionInputError`.

Minimal example (mirrors what `scripts/train.py`'s verification and the app both do):

```python
import pandas as pd
from src.prediction.pipeline import PredictionPipeline

# input_df: real historical rows for one (vegetable, market) group, in the same
# schema as data/processed's merged output, plus one target row with price=NaN
pipeline = PredictionPipeline(config_path="config/config.yaml")
result = pipeline.predict(input_df, target_date=target_date, vegetable="Tomato", market="Dambulla")
print(result)  # date, vegetable, market, predicted_price
```

**Streamlit app note:** the file uploader on the Predict page accepts a CSV with
`date, vegetable, market, price, rainfall_mm, temperature_c, humidity_pct`, and now also
documents `diesel_price`/`petrol_price` as required if the loaded model needs fuel-price
lag features (this real model does). The page applies production-region weather (§3) to
whatever's uploaded, then displays "Weather region: X" (or the market-fallback message)
and "Weather source: Open-Meteo" before predicting. The bundled synthetic sample dataset
has no fuel-price columns at all, so it cannot produce a prediction against this real
model -- it remains useful only as a UI walkthrough, not a working demo of real
predictions; uploading your own CSV with real historical data (including fuel prices)
is required to get a real prediction through the app.

## 10. Latest HARTI Data (live prediction option)

Alongside the bundled synthetic sample and uploading your own CSV, the Predict page
offers a third data source: **Latest HARTI Data (real)** -- predicts using the real
historical HARTI/Dambulla dataset (`data/raw/harti_prices.csv`, the same data the model
is trained on) plus the single most recently published HARTI daily bulletin price,
fetched live.

**What "latest" actually means:** HARTI (harti.gov.lk) has no API, CSV, or Excel export
-- investigated directly against the real site, not assumed. What exists is a daily
archive page (`harti.archive_url`, default `daily-price.php`) listing one PDF bulletin
per calendar day, going back to 2015. `src/data_collection/harti_live_client.py`'s
`HartiLiveClient` fetches that page, finds the most recent (or a specific) date's PDF
link, downloads it, and searches every page for whichever table has a "Dambulla Market"
column (`harti.target_market`) -- never assuming a fixed page index or filename
pattern, since HARTI's own filename convention has already changed historically. This
is HARTI's own most recently *published* bulletin -- typically 1-2 days behind today,
and not every market reports every day (a market that didn't report keeps showing its
last reported date, which this client reads from that column specifically, not the
bulletin's cover date). It is never described as real-time, because it is not.

**Schema differences handled:** HARTI's live PDF uses different vegetable-name
spellings than the project's historical CSV (e.g. "Raddish" vs "Radish", "Knolkhol" vs
"Knol Khol", "Beet root (N Eliya)" vs "Beetroot (Nuwara Eliya)") and reports prices as
"min - max" ranges rather than a single value. `HARTI_LIVE_VEGETABLE_ALIASES` maps every
live label to this project's canonical vocabulary (verified against a real bulletin --
every one of the 31 real vegetables in `harti_prices.csv` has a corresponding live-PDF
entry), and each range is converted to its midpoint. A vegetable with no reading that
day ("-" in the PDF) is simply absent from the fetched snapshot, never fabricated.

**How it reaches the model:** before merging the live bulletin itself, any gap between
the historical dataset's last known date for the selected (vegetable, market) and the
live bulletin's date is backfilled with real HARTI bulletins for the days in between
(`app/services.py::backfill_harti_price_gap()` -- see "Known limitations" below). The
live bulletin's own newest price for the selected vegetable is then merged onto the
real historical dataset as one additional row
(`app/services.py::merge_harti_live_price()`), and the whole result flows through the
exact same, unchanged `build_prediction_input()`/`build_live_prediction_input()` ->
`PredictionPipeline.predict()` path every other data source already uses -- no new
feature engineering, no new model, no new categorical encoder, and the persisted
encoder/model are still only ever loaded, never refit. HARTI's own bulletin doesn't
carry fuel prices, so each new row's `diesel_price`/`petrol_price` instead prefers
CEYPETCO's current retail price (`app/services.py::fetch_live_fuel_price()` ->
`src/data_collection/live_fuel_price_client.py`, fetched live from CEYPETCO's own
"Marketing & Sales" page) whenever that live reading's own effective date actually
covers the row's date; otherwise (live fuel unavailable, or the row predates the
current reading's effective date -- e.g. a backfilled gap day before the latest
revision) it falls back to forward-filling the historical dataset's own last known
diesel/petrol price, the same domain rule `DataCleaner` already applies elsewhere for
fuel-price gaps (a step-valued administrative price, constant between CEYPETCO
revisions) -- never fabricated either way.

**Caching:** the live HARTI bulletin is fetched at most once every `harti.cache_minutes`
(default 60), and the live CEYPETCO fuel price at most once every
`fuel_api.cache_minutes` (default 60), each via Streamlit's own cache -- HARTI
publishes at most once a day and CEYPETCO revises prices far less often than that. A
"Refresh" button forces an immediate HARTI re-fetch. A failed fetch always raises (HARTI)
or degrades to the historical forward-fill (fuel) rather than silently serving stale
data as if it were current; the UI always shows both the bulletin's own date and the
time it was fetched, and captions the resulting prediction with "Data source: Latest
HARTI Data -- bulletin dated X, not real-time."

**Configuration:** `config.yaml`'s `harti:` section (`latest_data_enabled`,
`archive_url`, `request_timeout_seconds`, `cache_minutes`, `target_market`) and
`fuel_api:` section (`enabled`, `url`, `timeout_seconds`, `cache_minutes`). Setting
`harti.latest_data_enabled: false` removes the "Latest HARTI Data" option from the UI
entirely; setting `fuel_api.enabled: false` just disables the live fuel-price lookup
within it (falling back to the historical forward-fill) without touching anything
else. The other two price data sources, and the training pipeline, are completely
unaffected either way.

## 11. Known limitations

- The Streamlit app's bundled synthetic sample dataset (Carrot/Tomato x
  Dambulla/Pettah, no fuel-price columns) cannot produce a prediction against this
  real, fuel-price-trained model -- see §9.
- 16 of 30 real vegetables have no production-region mapping (insufficient/mixed
  evidence found -- e.g. Tomato, Capsicum, Bitter Gourd) and fall back to market
  (Dambulla) weather, same as before this feature existed -- see §3's cited file.
- Each mapped vegetable uses one town/city's coordinates as a stand-in for its whole
  production district, not a district-wide average.
- `DataTransformer`'s vegetable/market alias tables were written before real data was
  available and have not been re-validated against this real HARTI/Dambulla vocabulary.
- `src/visualization/` is an empty package -- no exploratory/model-comparison charts
  beyond the single inline matplotlib plot in `app/pages/1_Predict.py`.
- `LagFeatureGenerator` groups only by `(vegetable, market)`, not by data source; only
  one source (HARTI) is configured today, so this hasn't been exercised in practice.
- No CI pipeline or hyperparameter search; hyperparameters are fixed, config-set values.
- **Latest HARTI Data** backfills real HARTI bulletins (`app/services.py::
  backfill_harti_price_gap()`) for any calendar day between the historical CSV's last
  date and the live bulletin's date, up to `max(feature_engineering.lag_periods_days +
  rolling_windows_days)` (default 30) days back, so `price_lag_{1,7,14,30}d` and the
  7/14/30-day rolling mean/std stay backed by real observations even if the CSV has
  gone stale. A day HARTI itself didn't publish (or had no reading for the requested
  vegetable) is skipped, never fabricated, so a large enough or holey-enough gap can
  still leave a required feature genuinely unavailable. Predicting beyond what real
  data (historical CSV + backfilled + live bulletin) can support will raise
  `FeatureAlignmentError` rather than fabricate a value -- this is deliberate, not a
  bug.
- `harti.target_market` is fixed to "Dambulla" (the only market the current model
  recognizes); HARTI's live bulletin itself covers 10 markets, but nothing downstream
  of `HartiLiveClient` supports more than one yet.
- **Live fuel price** (`CeypetcoLiveFuelPriceClient`) reads CEYPETCO's plain "Lanka
  Auto Diesel" and "Lanka Petrol 92 Octane" grades only -- matching
  `fuel_prices.csv`'s own historical vocabulary -- not the premium "Super Diesel"/"95
  Octane" grades CEYPETCO's page also lists. It is one national retail price, same as
  the historical CSV; there is no per-region fuel price to resolve.
