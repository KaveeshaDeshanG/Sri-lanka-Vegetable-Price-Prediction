# Sri Lankan Vegetable Price Prediction — Project Report

**Version:** 1.0.0 (per `pyproject.toml`)
**Generated:** 2026-08-13
**Scope of this report:** describes the system strictly as currently implemented and verified in this repository. No results, datasets, or experiments beyond what is directly traceable to the current code and test suite are included.

---

> ⚠️ **Historical snapshot — read this before relying on any claim below.**
> This report was generated on 2026-08-13, **before real HARTI/Dambulla market data was
> acquired and before the model was trained on it.** Every statement below claiming "no
> real data," describing `data/raw/` as empty/absent, giving a test count of 158, stating
> metrics are "never persisted," or describing `models/` as containing only `.gitkeep`
> reflects that earlier state and is now **superseded**.
>
> **For current, accurate status, see [`../README.md`](../README.md) and
> [`metrics/model_metrics.json`](metrics/model_metrics.json)** — real datasets, the real
> trained Random Forest model, 264 passing tests, and persisted MAE/RMSE/R² are all in
> place as of 2026-08-19.
>
> This document is kept as-is otherwise: its description of the pipeline's architecture,
> leakage-prevention design, and error-handling contracts remains accurate and is not
> restated in the README. Only its data-availability/results/test-count claims (§1, §5,
> §15, §20–24) are dated.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Complete Data Pipeline](#5-complete-data-pipeline)
6. [Data Preprocessing](#6-data-preprocessing)
7. [Feature Engineering](#7-feature-engineering)
8. [Chronological Train/Test Splitting](#8-chronological-traintest-splitting)
9. [Data Leakage Prevention](#9-data-leakage-prevention)
10. [Categorical Encoding and Train-Only Fitting](#10-categorical-encoding-and-train-only-fitting)
11. [Machine Learning Models](#11-machine-learning-models)
12. [Model Configuration and Hyperparameters](#12-model-configuration-and-hyperparameters)
13. [Evaluation Metrics](#13-evaluation-metrics)
14. [Best-Model Selection](#14-best-model-selection)
15. [Model and Inference-Artifact Persistence](#15-model-and-inference-artifact-persistence)
16. [Prediction/Inference Pipeline](#16-predictioninference-pipeline)
17. [Historical-Context Requirement](#17-historical-context-requirement)
18. [Error Handling](#18-error-handling)
19. [Streamlit Application](#19-streamlit-application)
20. [Testing and Validation](#20-testing-and-validation)
21. [Current Results](#21-current-results)
22. [Limitations](#22-limitations)
23. [Future Improvements](#23-future-improvements)
24. [Conclusion](#24-conclusion)

---

## 1. Project Overview

This project is an AI-powered Sri Lankan vegetable price prediction system built on historical market and weather data (`pyproject.toml`'s own description). It is implemented as a modular Python pipeline (`src/`) covering data preprocessing, feature engineering, model training, evaluation, and inference, plus a Streamlit application (`app/`) that exposes the pipeline to an end user. The stack is pandas, NumPy, scikit-learn, XGBoost, Streamlit, and PyYAML, targeting Python ≥ 3.12.

The system was built incrementally, with each stage backed by an automated pytest suite before the next stage was started. At the time of this report, **158 automated tests pass**.

## 2. Problem Statement

Vegetable prices in Sri Lankan markets (e.g. Dambulla, Pettah) vary day to day due to seasonal cycles, market-specific dynamics, and weather conditions. A system that can learn historical price patterns — combined with lagged weather signals — and produce a price estimate for a near-future date would be useful to producers, traders, and market analysts. This project builds the data and modelling infrastructure for that estimate, along with an inference pipeline and a demonstration UI.

## 3. Objectives

- Build a reproducible pipeline from raw market/weather data through to a trained, evaluated model.
- Keep every stage configuration-driven (`config/config.yaml`) rather than hardcoded, so the system is not tied to specific vegetables, markets, or file paths.
- Prevent data leakage at every boundary: chronological (not random) splitting, train-only categorical fitting, and backward-only lag/rolling/weather features.
- Compare multiple model families (Linear Regression, Random Forest, XGBoost) and select the best one by a configurable metric.
- Provide a prediction/inference pipeline that reuses training-time fitted state exactly, with no silent fallback behavior.
- Provide an accessible Streamlit interface over that pipeline.
- Maintain thorough automated test coverage validating correctness properties (not just "it runs").

## 4. System Architecture

The codebase is organized by pipeline stage:

| Package | Responsibility | Status |
|---|---|---|
| `config/` | `ConfigLoader` — typed, dotted-path access to `config.yaml` | Complete |
| `src/data_collection/` | Load and validate raw market/weather CSVs | Complete |
| `src/preprocessing/` | Clean, standardize, and merge market + weather data | Complete |
| `src/feature_engineering/` | Date/lag/rolling/weather features, chronological split, categorical encoding, dataset building | Complete |
| `src/models/` | Baseline, training, evaluation, model selection, persistence | Complete |
| `src/prediction/` | Inference artifact persistence and the prediction pipeline | Complete |
| `src/visualization/` | — | **Not implemented** (empty package, 0 lines beyond the module docstring-less `__init__.py`) |
| `app/` | Streamlit UI (`Home.py`, `pages/1_Predict.py`, `services.py`) and bundled sample data | Complete |
| `tests/` | pytest suite, one subpackage per `src/` package except `data_collection` | Present for every package **except `data_collection`** (see §20, §22) |

Within each package, the project consistently follows a pattern of small, single-responsibility modules composed by one orchestrator class per stage (`PreprocessingPipeline`, `TrainTestDatasetBuilder`, `ModelTrainingPipeline`, `PredictionPipeline`), so each transformation is independently testable and the orchestrators contain no transformation logic of their own — only sequencing.

## 5. Complete Data Pipeline

The production flow, traced through the actual current code:

```
raw CSVs (data/raw/*.csv — currently absent, see §22)
 → DataLoader.load_all()                              [src/data_collection/data_loader.py]
 → DataCleaner.clean_market_data()/clean_weather_data()[src/preprocessing/cleaner.py]
 → DataTransformer.transform_market_data()/            [src/preprocessing/transformer.py]
   transform_weather_data()
 → DataMerger.merge()                                  [src/preprocessing/merger.py]
   (the above four steps = PreprocessingPipeline.run() in src/preprocessing/pipeline.py)
 → FeatureEngineer.build_features()                    [src/feature_engineering/feature_engineer.py]
 → chronological_train_test_split()                    [src/feature_engineering/train_test_split.py]
 → FeatureEngineer.split_features_target()              (once per split)
 → CategoricalEncoder.fit_transform(X_train) /
   CategoricalEncoder.transform(X_test)
   (the above three steps = TrainTestDatasetBuilder.build() in dataset_builder.py)
 → ModelTrainer.train_all(X_train, y_train)             [src/models/trainer.py]
 → ModelEvaluator.evaluate_all(models, X_test, y_test)  [src/models/evaluator.py]
 → select_best_model(metrics)                           [src/models/model_selection.py]
 → save_model(best_model, "best_model.joblib")          [src/models/persistence.py]
 → save_inference_artifact(encoder, feature_columns,
   "feature_pipeline_state.joblib")                     [src/prediction/predictor.py]
   (training orchestration above = ModelTrainingPipeline.run() + app/services.train_demo_model())
 → new prediction input (a row with price = NaN)
 → PredictionPipeline._normalize() → build_features() → split_features_target(require_target=False)
 → Predictor.load() → Predictor.predict()               [src/prediction/pipeline.py, predictor.py]
 → app/pages/1_Predict.py renders the result or a typed error
```

Every stage in this diagram corresponds to code that exists and is exercised by the test suite today. The pipeline has never been run against real raw data (see §22); it has been run end to end against synthetic fixtures, including via the Streamlit app's own demo-training path.

## 6. Data Preprocessing

`src/preprocessing/cleaner.py` — `DataCleaner`:
- `clean_market_data()`: drops exact duplicate rows; drops rows missing any of date/vegetable/market/price (no imputation of these); removes non-positive or non-numeric prices; applies `clean_text_columns()` (whitespace/casing hygiene, title-casing); drops unparseable or implausible dates (`preprocessing.min_valid_year`, no future dates).
- `clean_weather_data()`: drops exact duplicates and missing dates; linearly interpolates missing rainfall/temperature/humidity readings in chronological order (edge gaps that can't be interpolated are dropped); removes physically implausible values (`preprocessing.temperature_min_c/max_c`, `preprocessing.humidity_min_pct/max_pct`).

`src/preprocessing/transformer.py` — `DataTransformer`: converts the date column to a real datetime dtype, standardizes vegetable/market names via `DEFAULT_VEGETABLE_ALIASES`/`DEFAULT_MARKET_ALIASES` (explicitly documented in the code as **illustrative alias sets, not yet validated against real HARTI/Dambulla exports**, since no real raw file has ever been loaded), coerces price/weather columns to numeric, and sorts chronologically.

`src/preprocessing/merger.py` — `DataMerger`: left-joins market data with weather data on date (market rows are the base — a price observation is always kept even if its weather is missing, which becomes NaN); duplicate weather rows for the same date are averaged before the join to avoid fan-out.

`src/preprocessing/pipeline.py` — `PreprocessingPipeline.run()` orchestrates load → clean → transform → merge → save as one call, with each step also individually callable.

## 7. Feature Engineering

All feature generators are in `src/feature_engineering/` and are composed by `FeatureEngineer.build_features()`.

**Date features** (`date_features.py`, `DateFeatureGenerator`): `day_of_week`, `day_of_month`, `week_of_year`, `month`, `quarter`, derived from the date column. No Sri Lanka-specific season boundaries are hardcoded, since no such rule is present in `config.yaml`.

**Lag features** (`lag_features.py`, `LagFeatureGenerator`): `price_lag_{n}d` for `n` in config's `feature_engineering.lag_periods_days` (default `[1, 7, 14, 30]`), computed via an exact-calendar-day **self-join** per `(vegetable, market)` group — not a positional `.shift()`, because the data is not guaranteed to be sampled on a strict daily cadence. A lag value is populated only when a row for that exact earlier date exists in the same group; otherwise it is left `NaN`, never fabricated.

**Rolling features** (`rolling_features.py`, `RollingFeatureGenerator`): `price_rolling_mean_{n}d` / `price_rolling_std_{n}d` for `n` in `feature_engineering.rolling_windows_days` (default `[7, 14, 30]`), computed via pandas time-based rolling with `closed="left"` — the window is `[date − n, date)`, explicitly excluding the current row's own price from its own rolling statistic.

**Weather features** (`weather_features.py`, `WeatherFeatureGenerator`): `{rainfall_mm,temperature_c,humidity_pct}_lag{n}d` for `n = feature_engineering.weather_lag_days` (default `14`), looked up by date only (weather is one national/regional series, not per group). The same-day weather columns present after the merge are deliberately **excluded** from the final feature set by `split_features_target()` — they are a fine training-time feature but assume the current day's weather is already observed, which does not hold for predicting a genuinely future date. The lagged version is always already-observed by construction, which is the documented reason it is used instead.

## 8. Chronological Train/Test Splitting

`src/feature_engineering/train_test_split.py` — `chronological_train_test_split()` splits by the **date column's actual values** (via a stable `np.argsort`), not by row position, and not with `sklearn.model_selection.train_test_split`'s shuffling. The oldest `1 − test_size` fraction of rows (by date rank) becomes train, the newest `test_size` fraction becomes test. `test_size` defaults to `model.test_size` in config (`0.2`). Rows within each resulting split keep their original relative order. Ties on the same date are resolved deterministically (stable sort), not randomly.

## 9. Data Leakage Prevention

Leakage prevention is enforced structurally, not just by convention:

- **Chronological, not random, split** — see §8.
- **Feature computation is backward-only** — lag features require an exact *earlier* date; rolling features use `closed="left"`; weather-lag features reference an earlier date. None of the three generators can reference a value from a later date. This is why feature engineering runs on the full historical dataset *before* the split (test-period rows may legitimately need train-period history) without leaking train-period information forward, since nothing looks forward.
- **Train-only categorical fitting** — see §10.
- **Training never sees the test set** — `ModelTrainer.train_all(X_train, y_train)` has no `X_test`/`y_test` parameter at all; it is structurally impossible for it to use test data.
- **Inference never fits anything** — grepping `src/prediction/` for `.fit(`/`.fit_transform(` returns zero matches. `Predictor.predict()` only ever calls `.transform()` on the persisted, already-fitted encoder.
- **No train/test split during inference** — `PredictionPipeline.predict()` never calls `chronological_train_test_split()`; a single new-input DataFrame goes straight to feature engineering and prediction.

These properties are additionally regression-tested (e.g. `test_no_future_information_leaks_into_training_features`, `test_lag_feature_does_not_leak_future_prices`, `test_predictor_never_calls_fit_on_the_encoder`, `test_pipeline_training_is_unaffected_by_test_split_contents`).

## 10. Categorical Encoding and Train-Only Fitting

`src/feature_engineering/categorical_features.py` — `CategoricalEncoder` wraps sklearn's `OneHotEncoder` with `handle_unknown="ignore"`, chosen specifically because the real vegetable/market vocabulary has never been observed (no real data has been loaded) and a category seen only at prediction time must not crash the pipeline — it is encoded as all-zero across that category's columns instead.

The encoder is fit **exactly once in the entire codebase**: inside `TrainTestDatasetBuilder.build()` (`src/feature_engineering/dataset_builder.py`), on `X_train` only, then reused via `.transform()` for `X_test`. The fitted instance is exposed afterward via the builder's `fitted_encoder` property specifically so it can be persisted for inference — inference reuses this exact object and never creates or fits a new one.

## 11. Machine Learning Models

All three models are built and trained through one config-driven entry point, `ModelTrainer.train_all()` in `src/models/trainer.py`.

- **Linear Regression** (`src/models/baseline.py`): plain `sklearn.linear_model.LinearRegression`, no hyperparameters. Used as the project's baseline — its closed-form OLS solution is deterministic, so it needs no random seed.
- **Random Forest**: `sklearn.ensemble.RandomForestRegressor`, hyperparameters and `random_state` read from config (see §12).
- **XGBoost**: `xgboost.XGBRegressor`, hyperparameters and `random_state` read from config (see §12).

`ModelTrainer` uses a small factory registry (`_MODEL_FACTORIES`) keyed by the model names in `config.model.models_to_train`; the `"linear_regression"` entry delegates to `baseline.py` rather than re-implementing model construction.

## 12. Model Configuration and Hyperparameters

Current values in `config/config.yaml`, all confirmed consumed by code (see also §5 of the prior audit turn, re-verified this session by grep against every call site):

| Key | Value | Consumed at |
|---|---|---|
| `project.random_seed` | `42` | `trainer.py` → `random_state` for Random Forest and XGBoost |
| `model.test_size` | `0.2` | `train_test_split.py` (default when not passed explicitly) |
| `model.target_column` | `"price"` | `feature_engineer.py` |
| `model.models_to_train` | `["linear_regression", "random_forest", "xgboost"]` | `trainer.py` |
| `model.random_forest.n_estimators` | `200` | `trainer.py` |
| `model.random_forest.max_depth` | `10` | `trainer.py` |
| `model.xgboost.n_estimators` | `200` | `trainer.py` |
| `model.xgboost.learning_rate` | `0.05` | `trainer.py` |
| `model.xgboost.max_depth` | `6` | `trainer.py` |

Linear Regression intentionally receives no seed, since OLS has no stochastic component. Hyperparameter values are read via `ConfigLoader.get()` with in-code fallback defaults that mirror `config.yaml`'s own values; a dedicated test (`test_random_forest_uses_configured_hyperparameters`, `test_xgboost_uses_configured_hyperparameters`) verifies against a fixture config with deliberately *different* values, so this is confirmed to be genuine config consumption rather than a coincidental match between the fallback and the real file.

## 13. Evaluation Metrics

`src/models/evaluator.py` — `ModelEvaluator` computes exactly the metrics listed in `config.evaluation.metrics` (`["mae", "rmse", "r2"]`):

- **MAE** — `sklearn.metrics.mean_absolute_error`
- **RMSE** — `mean_squared_error(y_true, y_pred) ** 0.5` (computed manually rather than via a version-specific `squared=False`/`root_mean_squared_error` API, for compatibility across sklearn versions)
- **R²** — `sklearn.metrics.r2_score`

Evaluation is always performed with `model.predict(X_test)` against `y_test` — the held-out chronological test set — never against the training set.

## 14. Best-Model Selection

`src/models/model_selection.py` — `select_best_model()` reads `config.evaluation.best_model_metric` (default `"rmse"`) and picks the model with the best score for that metric, using a direction lookup (`mae`/`rmse` → lower is better, `r2` → higher is better) rather than assuming "lower always wins." It raises a clear `ModelSelectionError` if the configured metric is absent from a model's results or if no models were evaluated at all.

## 15. Model and Inference-Artifact Persistence

Two separate joblib files are written under `config.paths.models_dir` (`"models"`):

- **`best_model.joblib`** — the selected best model, written by `ModelTrainingPipeline.run()` via `src/models/persistence.py`'s `save_model()`.
- **`feature_pipeline_state.joblib`** — an `InferenceArtifact` (the fitted `CategoricalEncoder` + the exact ordered `feature_columns` list from `X_train`), written by `src/prediction/predictor.py`'s `save_inference_artifact()`.

These are deliberately kept as two independent files so that "model missing" and "transformation state missing" are distinguishable failure modes at load time (`ModelArtifactNotFoundError` vs. `TransformationArtifactNotFoundError`), each independently tested. `persistence.py`'s `save_model`/`load_model` are generic joblib wrappers (not model-type-specific), reused by both the models and prediction layers rather than duplicated.

As of this report, the real `models/` directory contains only a `.gitkeep` placeholder — these artifacts only exist when created inside a test's `tmp_path`, or if the Streamlit app's training flow has been run locally by a user.

## 16. Prediction/Inference Pipeline

`src/prediction/pipeline.py` — `PredictionPipeline.predict(input_df)`:

1. **Normalizes** the new input using the same non-destructive steps training data went through — `DataCleaner.clean_text_columns()` and `DataTransformer`'s alias-standardization/dtype-conversion methods — so that category casing (e.g. `"carrot"` vs. the trained `"Carrot"`) matches what the encoder learned. The row-*dropping* cleaning rules (missing/invalid price, implausible dates) are deliberately **not** reused here, since they would delete the very row being predicted (its price is unknown by definition).
2. Runs `FeatureEngineer.build_features()` and `split_features_target(drop_incomplete_rows=False, require_target=False)` — identical feature computation to training. `require_target` is a parameter added specifically to support this: it lets a row with an unknown (`NaN`) target survive, which `split_features_target()` would otherwise always drop.
3. Treats every row whose price is `NaN` as "to predict" — no separate target-date parameter exists in the API.
4. Delegates to `src/prediction/predictor.py`'s `Predictor`, which loads the persisted model and `InferenceArtifact`, encodes with the reused fitted encoder, reindexes to the persisted `feature_columns`, and calls `model.predict()`.

## 17. Historical-Context Requirement

Because lag/rolling/weather features are defined exactly as in §7, a prediction for a given date is only fully computable if the input's historical rows satisfy those same backward-looking requirements — specifically, an exact row at `date − n` for each configured lag/weather-lag window (up to 30 days back by default), and at least one to two points within each configured rolling window. A single new observation with no history is **not sufficient** to produce a real prediction.

This is a deliberate design choice, not an oversight: `Predictor.predict()` does not substitute zero or any other fabricated value for a missing engineered feature. If the aligned feature matrix contains `NaN` in any column the model was trained on, it raises `FeatureAlignmentError`, naming the exact row and feature(s) responsible. This matters concretely for XGBoost, which has native missing-value handling and would otherwise silently produce a plausible-looking prediction from incomplete input rather than failing — the explicit pre-check exists specifically to prevent that.

## 18. Error Handling

Every stage defines its own specific exception type(s) rather than relying on generic exceptions, and none of them are swallowed silently:

| Exception | Raised in | Condition |
|---|---|---|
| `ConfigKeyError` | `config/config_loader.py` | required config key missing/null |
| `DataValidationError` | `src/data_collection/` | schema/type violation in raw data |
| `FeatureEngineeringError` | `feature_engineer.py` | date/vegetable/market/price missing |
| `ChronologicalSplitError` | `train_test_split.py` | bad `test_size`, missing/unparseable dates, split would be empty |
| `CategoricalFeatureEncodingError` | `categorical_features.py` | vegetable/market column missing |
| `ModelTrainingError` | `trainer.py` | unrecognized entry in `models_to_train` |
| `EvaluationError` | `evaluator.py` | unrecognized entry in `evaluation.metrics` |
| `ModelSelectionError` | `model_selection.py` | empty metrics, missing metric, unknown comparison direction |
| `ModelArtifactNotFoundError` / `TransformationArtifactNotFoundError` | `predictor.py` | persisted model / inference artifact file absent |
| `FeatureAlignmentError` | `predictor.py` | required feature missing/NaN after alignment (see §17) |
| `PredictionInputError` | `prediction/pipeline.py` | malformed date, nothing to predict |

In the Streamlit app, each of these is caught explicitly and rendered as a specific `st.error()` message rather than an unhandled stack trace.

## 19. Streamlit Application

`app/Home.py`: overview page. States the real-data limitation directly to the user, describes the pipeline, and links to the Predict page.

`app/pages/1_Predict.py`: the interactive page. Lets the user choose between the bundled sample dataset (`app/sample_data/sample_history.csv` — 360 rows, 4 synthetic vegetable/market groups over 90 days, `source="SAMPLE"`) or uploading their own CSV of the same schema. If no trained model is found, it offers a "train model on loaded data" action, which is the first place in the project that actually wires `TrainTestDatasetBuilder` → `ModelTrainingPipeline` → `save_inference_artifact` together outside of a test. Once a model is available, the user selects a vegetable, market, and target date, and the page calls `PredictionPipeline`, displaying the result as a metric plus a matplotlib chart, or a specific error message per §18.

`app/services.py`: all of the above's non-widget logic (`load_sample_history()`, `build_prediction_input()`, `train_demo_model()`), deliberately kept free of any Streamlit import so it is independently unit-testable.

## 20. Testing and Validation

**Automated (verified this session by running the suite directly):**

```
158 passed, 0 failed
```

| Package | Test count |
|---|---|
| `tests/preprocessing/` | 19 |
| `tests/feature_engineering/` | 85 |
| `tests/models/` | 27 |
| `tests/prediction/` | 20 |
| `tests/app/` | 7 |
| **Total** | **158** |

After the run, `models/`, `data/processed/`, and `reports/` were checked directly and contain only `.gitkeep` files — no test leaves artifacts in the real project directories; all tests that write files use pytest's `tmp_path`.

**`tests/data_collection/` does not exist.** `src/data_collection/` has zero direct unit tests; it is only exercised indirectly through `tests/preprocessing/test_pipeline.py`'s fixture-based integration test.

**Not automated — manual/browser verification only, and not performed in this environment:** clicking through the Streamlit app's widgets, the train → rerun → predict interaction flow, uploading a real CSV file through the file uploader, `st.cache_resource` invalidation behavior after training, and whether the matplotlib chart renders as intended. What *was* verified automatically: both page scripts execute standalone with exit code 0 (no exceptions), and `streamlit run app/Home.py` was started headlessly and confirmed to serve HTTP 200 with a clean startup log before being shut down. This confirms the app boots and its Python executes without error; it does not confirm interactive correctness.

## 21. Current Results

No real-world results are reported, and none should be inferred from this project. Specifically:

- **No real HARTI/Dambulla/weather data has been used anywhere in this project.** `data/raw/` contains only a `.gitkeep` file.
- All 158 passing tests validate **correctness properties** (leakage-safety, column alignment, error handling, deterministic behavior, config consumption) against small, deterministic **synthetic** fixtures constructed for that purpose — they are not a sample of real market behavior and were never intended to be.
- Model comparison metrics (MAE/RMSE/R²) are produced only transiently, in memory, during a training run (`ModelTrainingResult`), and are never written to `reports/metrics/` (which remains empty except its `.gitkeep`). No such values were captured or recorded as part of this report, and none are quoted here.
- If a user runs the Streamlit app's demo-training action locally, it will produce metric numbers for models fit to the bundled **synthetic** sample dataset. Those numbers would describe how well a model fits synthetic data generated by a fixed formula, not real-world vegetable pricing, and are not reported here since doing so would risk being misread as a real accuracy claim.

## 22. Limitations

- **No real source data.** `data/raw/harti_prices.csv`, `dambulla_prices.csv`, and `weather_data.csv` are all absent. This is the project's central open blocker and has been true since the first investigation of this repository.
- **No real-world accuracy claim is possible** as a direct consequence of the above.
- **`src/visualization/` is unimplemented** — the package exists but contains no code beyond an empty `__init__.py`. The only chart in the project is the single matplotlib plot inline in `app/pages/1_Predict.py`.
- **Evaluation metrics are not persisted** — `reports/metrics/` and `reports/figures/` exist as empty directories; comparison results only ever exist in memory during a single training run.
- **`src/data_collection/` has no direct unit tests** (see §20).
- **No version control.** This repository is not a git repository; there is no commit history to audit or reference.
- **Streamlit interactivity is not machine-verified**, only its startup (see §20).
- **`DataTransformer`'s vegetable/market alias tables are explicitly marked illustrative** in the code — built from general knowledge of Sri Lankan market naming variants, not validated against any real HARTI/Dambulla export, since none has been loaded.
- **`LagFeatureGenerator` groups only by `(vegetable, market)`, not by data source.** The code documents that if HARTI and Dambulla DEC both report the same vegetable/market/date, lag lookups could draw from either source's series; this is flagged as an open question in the code, not resolved, pending real multi-source data to know how much it matters in practice.

## 23. Future Improvements

- Acquire and integrate real HARTI/Dambulla price data and real weather data (data sources previously researched include data.gov.lk, HARTI, meteo.gov.lk, Open-Meteo, and NASA POWER).
- Add a `tests/data_collection/` suite.
- Implement `src/visualization/` for exploratory data analysis and model-comparison charts.
- Persist evaluation metrics and figures to `reports/metrics/` and `reports/figures/` rather than keeping them in memory only.
- Add browser-based or Streamlit-native (`streamlit.testing.v1.AppTest`) automated UI tests.
- Add hyperparameter search; hyperparameters are currently fixed, config-set values with no tuning process.
- Revisit `LagFeatureGenerator`'s grouping to include `source` once real multi-source overlap is understood.
- Adopt version control and a CI pipeline.

## 24. Conclusion

The project implements a complete, modular, and tested pipeline from raw-data ingestion through preprocessing, feature engineering, chronological splitting, leakage-safe categorical encoding, multi-model training and evaluation, best-model selection, artifact persistence, inference, and a Streamlit interface. Every stage is configuration-driven and backed by automated tests; 158 tests currently pass with no known failures. The system has not yet been validated against real Sri Lankan market or weather data — every result produced so far comes from deterministic synthetic fixtures built specifically to test correctness, not from real observations — so no claim of real-world predictive accuracy is made. The architecture is designed to accept real data without structural changes once it is acquired.
