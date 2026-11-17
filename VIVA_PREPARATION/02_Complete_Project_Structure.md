# 02 — Complete Project Structure

Project root: `D:\ECU\Sri-lanka-Vegetable-Price-Prediction-main\Sri-lanka-Vegetable-Price-Prediction-main`

## 1. Top-level folders and files

```
├── app/                    Streamlit web application (the user interface)          ← ENTRY POINT: app/Home.py
│   ├── Home.py             Home page
│   ├── pages/              Streamlit multipage folder: 1_Predict, 2_Model_Performance, 3_Price_History, 4_Cultivation_Planner
│   ├── ui.py               Shared theme, header, stat cards, data-source panel
│   ├── services.py         Loading data, live HARTI/fuel/weather helpers, upload validation, sandbox training
│   ├── forecasting.py      direct_forecast() and the older recursive forecast_future_price()
│   ├── cultivation.py      Cultivation Planner logic (harvest dates, plan building)
│   ├── catalogue.py        Vegetable names, freshness checks, data_version() cache key
│   └── sample_data/        sample_history.csv — SYNTHETIC demo data (Carrot, Tomato)
├── src/                    Reusable Python packages (no Streamlit inside)
│   ├── data_collection/    Loaders, validators, HARTI/CBSL/DCS PDF parsers, Open-Meteo and CEYPETCO clients
│   ├── preprocessing/      Cleaning, transforming, merging (Load → Clean → Transform → Merge)
│   ├── feature_engineering/ 51 earlier features + the direct forecaster's as-of-origin features
│   ├── models/             Trainer, evaluator, model selection, persistence, DirectForecaster
│   └── prediction/         Predictor and PredictionPipeline for the earlier models
├── scripts/                Command-line jobs: build/update data, train, evaluate, validate
├── config/                 config.yaml (all settings), config_loader.py, crop_production_regions.yaml
├── data/
│   ├── raw/                The datasets (CSV/JSON) — see section 4
│   └── reference/          DOA harvest durations, Sinhala/Tamil names, evidence copies
├── models/                 Saved models (.joblib) + metadata
├── reports/                PROJECT_REPORT.md, DATA_SOURCES.md, DATA_QUALITY_REPORT.md, metrics/*.json, figures/
├── tests/                  pytest test suite (565 test functions, 1,060 cases) + real PDF fixtures
├── docs/screenshots/       Older screenshots
├── notebooks/              Empty (only .gitkeep) — not used
├── Final_Project_Report/   Your Word report, PDF, figures, testing_evidence
├── VIVA_PREPARATION/       These documents
├── logs/                   Output of run_daily_update.bat (created when it runs)
├── .streamlit/config.toml  Dark theme colours; toolbarMode = "minimal"
├── .venv312/               ✅ Python 3.12 environment — the one that works
├── .venv/                  ❌ Python 3.15 environment — broken for this project (see 03)
├── start_app.bat           Double-click starter (checks packages, then runs app/Home.py)
├── requirements.txt        Pinned package versions
├── pyproject.toml          Package metadata; requires-python >=3.12,<3.15; dev extras (pytest)
└── README.md               Project documentation
```

> සිංහල: `app` = පරිශීලකයා දකින වෙබ් පිටු. `src` = දත්ත සකසන සහ ML කේත. `scripts` = command line එකෙන් ධාවනය කරන කාර්යයන්
> (දත්ත update, train). `data/raw` = දත්ත ගොනු. `models` = පුහුණු කළ ආකෘති. `tests` = ස්වයංක්‍රීය පරීක්ෂණ.

## 2. The entry point (verified)

- There is **no root `app.py`** (checked: `Test-Path app.py` → False).
- `start_app.bat` line 31: `".venv312\Scripts\python.exe" -m streamlit run app/Home.py`
- Streamlit automatically turns every file in `app/pages/` into a sidebar page. The number prefix sets the order
  (`1_Predict.py`, `2_Model_Performance.py`, `3_Price_History.py`, `4_Cultivation_Planner.py`).
- `app/Home.py` lines 11–14 add the project root to `sys.path` and `chdir` to it, so `import src…` and relative data
  paths work.

## 3. Important Python files and their responsibilities

### 3.1 Application (`app/`)
| File | Key functions (line) | Responsibility |
|---|---|---|
| `Home.py` | `_load_forecaster` (31), `_saved_accuracy` (36) | Model summary cards, "Data in use" table from the manifest, last update, page links (190–196) |
| `pages/1_Predict.py` | `_forecast_origin_rows` (271), `_render_direct_result` (276), `_render_source_status` (386), `_load_source_records` (438), `_render_vegetable_records` (468), `_render_real_prediction` (542), `_render_freshness` (604), data source radio (637: Upload CSV / Latest Real Data Prediction), `_show_direct_forecast` (963) | The prediction page (two options since 15 Nov 2026) |
| `source_records.py` | `harti_dambulla_records`, `harti_other_market_records`, `cbsl_records`, `dcs_records`, `records_for_series`, `latest_dambulla_overview` | Each official source's records with traceability fields, kept apart |
| `data_refresh.py` | `run_data_update`, `interpret_exit_code` | Runs `scripts/daily_update.py` from the page buttons |
| `pages/2_Model_Performance.py` | `_show_direct` (81), `_show_earlier_models` (172), `_show_backtest` (238) | Reads `reports/metrics/*.json` and shows accuracy vs baseline |
| `pages/3_Price_History.py` | `_load_history` (40), `_load_markets` (45), `_load_dcs` (56), `_load_cbsl` (67); source radio (97) | Charts of recorded prices |
| `pages/4_Cultivation_Planner.py` | selectboxes (133, 136, 146), forecast-reach box (191), date (200), weather to display (209), form button (220) | The planner page |
| `services.py` | `load_real_market_history` (264), `prepare_uploaded_history` (148), `fetch_harti_latest_snapshot` (552), `fetch_live_fuel_price` (617), `backfill_harti_price_gap` (705), `train_demo_model` (1042) | Glue between pages and `src/` |
| `forecasting.py` | `direct_forecast` (116), `forecast_future_price` (195, recursive) | Turns a request into a forecast |
| `cultivation.py` | `add_duration` (110), `load_crop_timings` (138), `latest_start_in_reach` (198), `build_plan` (305), `suggest_selling_day` (270) | Planner logic |
| `catalogue.py` | `freshness` (96), `past_date_notice` (108), `series_status` (122), `data_version` (162) | Names, freshness, which series can be forecast |
| `ui.py` | `apply_theme` (198), `available_sources` (228), `data_source_panel` (265) | Look and feel |

### 3.2 Data collection (`src/data_collection/`)
| File | Key items | Responsibility |
|---|---|---|
| `harti_live_client.py` | `parse_price_range` (138), `split_merged_label_rows` (235), `HartiLiveClient` (323), `fetch_latest_prices` (379), `_extract_vegetable_table` (535) | Reads HARTI archive page + PDF bulletins |
| `cbsl_price_report_parser.py` | `parse_cbsl_price_report` (73) | Reads CBSL Daily Price Report PDFs |
| `dcs_retail_parser.py` | `parse_dcs_weekly_pdf` (107) | Reads DCS weekly retail PDFs |
| `live_weather_client.py` | `OpenMeteoWeatherClient` (78), `fetch_recent_daily_weather` (180), `fetch_historical_daily_weather` (215) | Open-Meteo weather API |
| `live_fuel_price_client.py` | `CeypetcoLiveFuelPriceClient` (95), `fetch_latest_prices` (131) | CEYPETCO fuel prices |
| `data_validator.py` | `DataValidator` (40) | Column/type/date/range checks |
| `production_region_resolver.py` | `ProductionRegionResolver` (45) | Which region's weather a vegetable uses |
| `market_loader.py`, `weather_loader.py`, `fuel_price_loader.py`, `data_loader.py` | `load` methods | Load the CSV files |

### 3.3 Preprocessing, features, models, prediction (`src/`)
| File | Key items | Responsibility |
|---|---|---|
| `preprocessing/pipeline.py` | `PreprocessingPipeline` (18), `run` (133) | Load → Clean → Transform → Merge |
| `preprocessing/cleaner.py` | `clean_market_data` (77), `clean_weather_data` (119) | Removes bad rows, interpolates weather, forward-fills fuel |
| `preprocessing/transformer.py` | `transform_market_data` (107) | Dates, sorting, name standardisation |
| `preprocessing/merger.py` | `merge` (19) | Joins prices with weather (left join) |
| `feature_engineering/feature_engineer.py` | `build_features` (104), `split_features_target` (119) | 51 features for the earlier models |
| `feature_engineering/forecast_features.py` | constants (23–27), `ForecastFeatureBuilder` (48), `build_origin_features` (65), `add_horizon_features` (164) | Features for the direct forecaster |
| `feature_engineering/train_test_split.py` | `chronological_train_test_split` (17) | 80/20 split by date |
| `models/direct_forecaster.py` | `MAX_HORIZON_DAYS` (34), `FORECAST_BANDS` (47), `score_predictions` (59), `DirectForecaster.fit` (103), `predict_rows` (126), `forecast` (134) | **The main model** |
| `models/trainer.py` | `_build_random_forest` (22), `_build_xgboost` (30), `ModelTrainer.train_all` (66) | Trains earlier models |
| `models/evaluator.py`, `model_selection.py` | `evaluate_all` (56), `select_best_model` (17) | MAE/RMSE/R², pick best by RMSE |
| `models/persistence.py` | `save_model` (21), `load_model` (44) | joblib save/load |
| `prediction/predictor.py`, `pipeline.py` | `Predictor` (117), `PredictionPipeline.predict` (82) | Predict with earlier models (sample/upload data) |

### 3.4 Scripts (`scripts/`) — run from PowerShell, not from the web app
| Script | What it does |
|---|---|
| `daily_update.py` (`run` 419) | Adds newly published HARTI/CBSL/DCS data, weather, fuel; backups; manifest |
| `run_daily_update.bat` | Runs the daily update for Windows Task Scheduler, logs to `logs/` |
| `update_data.py` (`main` 374) | Full rebuild of all datasets since 2015 (with backup) |
| `train_forecaster.py` (`main` 142) | Scores the direct forecaster on the test period, then trains on all data and saves it |
| `train.py` (`run_training` 81) | Trains the earlier LR/RF/XGBoost models |
| `model_experiments.py` | Validation-window comparison of models and feature groups |
| `backtest_forecast.py` | Recursive multi-day backtest |
| `evaluate_forecaster.py`, `forecast_error_bands.py` | Evaluation report; error bands for the planner |
| `retrain_if_better.py` | Retrain only if a held-out check passes (**never run yet**) |
| `validate_real_datasets.py` | Cross-checks data against the original PDFs |
| `build_*.py` | Older single-purpose dataset builders |

## 4. Data, models and reports (what exists on disk)

| Path | Content |
|---|---|
| `data/raw/harti_prices.csv` | Dambulla wholesale, 76,719 rows, 2015-06-22 → 2026-11-09 |
| `data/raw/harti_prices_detail.csv` | Same + low, high, raw cell, bulletin date, source PDF URL |
| `data/raw/harti_all_markets.csv` | 11 markets, 526,905 rows |
| `data/raw/cbsl_daily_prices.csv` | CBSL wholesale & retail, 9,277 rows, 2026-09-01 → 2026-11-10 |
| `data/raw/dcs_weekly_retail_prices.csv` | DCS retail, 11,439 rows |
| `data/raw/weather_data_by_region.csv`, `weather_data.csv` | Open-Meteo weather, 8 locations, → 2026-11-08 |
| `data/raw/fuel_prices.csv` | CEYPETCO diesel/petrol, → 2026-11-11 |
| `data/raw/data_manifest.json` | Source, rows, hash, first/latest date of every dataset |
| `data/raw/update_log.jsonl`, `harti_import_log.csv` | Daily update history and per-bulletin provenance |
| `data/reference/doa_harvest_durations_v2.csv` | 77 published harvest timings (DOA HORDI) |
| `models/direct_forecaster.joblib` | **Main model** (30 Ridge pipelines) |
| `models/best_model.joblib`, `feature_pipeline_state.joblib`, `artifact_metadata.json` | Earlier Linear Regression model + encoder + metadata |
| `reports/metrics/direct_forecaster_metrics.json` | Test results used by Home and Model Performance |
| `reports/metrics/model_metrics.json` | Earlier model results |
| `reports/metrics/forecast_error_bands.json` | Likely range used by the planner |

> සිංහල: Database එකක් නැහැ. සියලු දත්ත CSV ගොනු ලෙස `data/raw` තුළ, ආකෘති `.joblib` ගොනු ලෙස `models` තුළ ගබඩා වේ.

## 5. Dependencies and actual usage (`requirements.txt`)

| Package | Version | Used for (where) |
|---|---|---|
| streamlit | 1.66.1 | Web UI (`app/`) |
| pandas / numpy | 3.1.0 / 2.5.4 | All data handling |
| scikit-learn | 1.9.1 | Ridge, LinearRegression, RandomForest, imputer, scaler, OneHotEncoder |
| xgboost | 3.4.1 | XGBoost in `src/models/trainer.py` (earlier models) |
| joblib | 1.6.0 | Saving/loading models (`src/models/persistence.py`) |
| pymupdf | 2.1 | Reading PDF bulletins |
| beautifulsoup4 | 4.15.0 | Reading the HARTI archive HTML page |
| requests | 2.34.2 | HTTP downloads (HARTI, CBSL, DCS, Open-Meteo, CEYPETCO) |
| pyyaml | 6.0.3 | `config/*.yaml` |
| matplotlib | 3.11.2 | Some charts / figures |
| pillow | 13.0.0 | Image crops in `scripts/validate_real_datasets.py` |
| altair | (installed with Streamlit) | Charts on Price History and Model Performance — **not listed in requirements.txt** but installed as a Streamlit dependency |
| pytest | (dev extra in pyproject.toml) | Tests |

## 6. Unused or incomplete items (honest list)

- `notebooks/` — empty.
- `src/visualization/` — empty package (only `__init__.py`).
- `data/reference/doa_harvest_durations.csv` — old version, replaced by `_v2`.
- `app/catalogue.py` `load_catalogue` (152) — function exists, its CSV was never built (`scripts/build_vegetable_catalogue.py` not run).
- `scripts/retrain_if_better.py` — implemented, never run.
- `.venv` (Python 3.15) — cannot run the project.
