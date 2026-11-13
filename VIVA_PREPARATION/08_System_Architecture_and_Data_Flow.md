# 08 — System Architecture and Data Flow

The system is a **single-machine, layered Python application**: data-collection scripts → CSV files → feature
builder → saved models → Streamlit pages. There is no separate backend server and no database.

> සිංහල: පද්ධතිය ස්ථර (layers) 4කින් සමන්විතයි: (1) දත්ත එකතු කිරීම, (2) CSV ගොනු ලෙස ගබඩා කිරීම,
> (3) ML ආකෘතිය, (4) Streamlit අතුරුමුහුණත. සියල්ල එකම පරිගණකයේ ක්‍රියා කරයි.

## 1. Overall architecture

```mermaid
flowchart LR
    subgraph EXT[External public sources]
        H[HARTI daily PDF bulletins<br/>harti.gov.lk]
        C[CBSL Daily Price Report PDFs]
        D[DCS weekly retail PDFs]
        W[Open-Meteo weather API<br/>no key]
        F[CEYPETCO fuel prices]
    end
    subgraph COL[Data collection - src/data_collection, scripts]
        HC[harti_live_client.py<br/>requests + BeautifulSoup + PyMuPDF]
        CP[cbsl_price_report_parser.py]
        DP[dcs_retail_parser.py]
        WC[live_weather_client.py]
        FC[live_fuel_price_client.py]
        UPD[scripts/daily_update.py<br/>scripts/update_data.py]
    end
    subgraph STORE[Storage - files only]
        RAW[(data/raw/*.csv<br/>data_manifest.json)]
        MOD[(models/direct_forecaster.joblib<br/>models/best_model.joblib)]
        REP[(reports/metrics/*.json)]
    end
    subgraph ML[Machine learning - src]
        FE[forecast_features.py<br/>ForecastFeatureBuilder]
        DF[direct_forecaster.py<br/>30 Ridge models]
        TR[scripts/train_forecaster.py<br/>evaluate_forecaster.py]
    end
    subgraph UI[Streamlit app - app/]
        HOME[Home.py]
        P1[1_Predict.py]
        P2[2_Model_Performance.py]
        P3[3_Price_History.py]
        P4[4_Cultivation_Planner.py]
        SV[services.py / forecasting.py<br/>cultivation.py / catalogue.py / ui.py]
    end
    H --> HC
    C --> CP
    D --> DP
    W --> WC
    F --> FC
    HC & CP & DP & WC & FC --> UPD --> RAW
    RAW --> FE --> TR --> DF --> MOD
    TR --> REP
    RAW & MOD & REP --> SV
    SV --> HOME & P1 & P2 & P3 & P4
    HC -. live on demand .-> P1
    WC -. live .-> SV
    FC -. live .-> SV
    USER((User browser<br/>localhost:8501)) <--> UI
```

## 2. Data collection and preprocessing

```mermaid
flowchart TD
    A[daily_update.py starts] --> L{Lock file free?}
    L -- no --> X3[Exit code 3: already running]
    L -- yes --> B[Back up files that will change<br/>Sri-lanka-Vegetable-Price-Backups/daily]
    B --> H1[List HARTI archive page<br/>daily-price.php]
    H1 --> H2[Download new PDF bulletins]
    H2 --> H3[PyMuPDF: words + x/y positions]
    H3 --> H4[_extract_vegetable_table<br/>split_merged_label_rows]
    H4 --> H5[parse_price_range: 400-450 -> 425]
    H5 --> H6{Readable?}
    H6 -- no --> LOG[Log it, exit code 2 later]
    H6 -- yes --> H7[Append to harti_prices.csv<br/>and harti_all_markets.csv]
    H7 --> CB[Import CBSL and DCS reports]
    CB --> WX[Top up Open-Meteo weather + CEYPETCO fuel]
    WX --> MAN[Rewrite data_manifest.json<br/>append update_log.jsonl]
    MAN --> APP[App sees new files via data_version]
```

Preprocessing for the earlier models (`src/preprocessing/pipeline.py` `run`, line 133):
load → `cleaner` (duplicates, invalid prices/weather, missing fields) → `transformer` (names, types, sort) →
`merger` (join weather and fuel by date) → `data/processed/processed_prices.csv`.

## 3. Machine learning training

```mermaid
flowchart TD
    R[(harti_prices + all_markets + weather_by_region + fuel)] --> O[build_origin_features<br/>one row per vegetable per day<br/>log prices, carry max 7 days]
    O --> FG[Feature groups: level, momentum, calendar,<br/>rain_windows, weather_lag_14, market, fuel, other_markets]
    FG --> LOOP{{For each horizon h = 1..30}}
    LOOP --> HF[add_horizon_features:<br/>target_date, target_price = price h days later]
    HF --> BAND[band_for h -> feature groups + alpha]
    BAND --> PIPE[SimpleImputer median -> StandardScaler -> Ridge alpha]
    PIPE --> FIT[fit on y = log target_price / last_price]
    FIT --> LOOP
    LOOP --> SAVE[(models/direct_forecaster.joblib)]
    SAVE --> EVAL[evaluate_forecaster.py on test period<br/>from 2026-05-14]
    EVAL --> MET[(reports/metrics/direct_forecaster_metrics.json<br/>MAE RMSE MAPE R2, gain vs hold, bootstrap CI)]
    VAL[model_experiments.py:<br/>4 validation windows before test] -. chose bands and alpha .-> BAND
```

## 4. Prediction workflow (Predict page, real data)

```mermaid
sequenceDiagram
    actor U as User
    participant P as 1_Predict.py
    participant S as services.py
    participant FB as ForecastFeatureBuilder
    participant F as forecasting.py direct_forecast
    participant M as DirectForecaster
    U->>P: choose vegetable + date, click Predict
    P->>S: load_real_market_history (cached)
    S-->>P: prices + weather + fuel
    P->>FB: build_origin_features (cached _forecast_origin_rows)
    FB-->>P: feature rows
    P->>F: direct_forecast(vegetable, target date)
    F->>F: h = target date - latest actual date
    alt h > 30
        F-->>P: error "reaches at most 30 days ahead"
    else 1 <= h <= 30
        F->>M: forecast(origin, horizons 1..h)
        M->>M: change = Ridge_h.predict(features)
        M-->>F: price = last_price x exp(change)
        F-->>P: forecast path
    end
    P-->>U: price card, chart, weather, sources (_render_direct_result)
```

## 5. User interaction workflow

```mermaid
flowchart TD
    START([Double-click start_app.bat]) --> CHK{.venv312 exists and imports OK?}
    CHK -- no --> ERR[Window shows the problem and waits]
    CHK -- yes --> RUN[streamlit run app/Home.py -> localhost:8501]
    RUN --> HOME[Home: accuracy cards, data in use]
    HOME --> NAV{Sidebar}
    NAV --> PRED[Predict]
    NAV --> PERF[Model Performance]
    NAV --> HIST[Price History]
    NAV --> PLAN[Cultivation Planner]
    PRED --> SRC{Data source}
    SRC -- Historical real --> DIRECT[Ridge direct forecaster]
    SRC -- Latest HARTI --> LIVE[Download bulletin, fill gap] --> DIRECT
    SRC -- Sample / Upload --> EARLY[Earlier 51-feature model<br/>optional sandbox training]
    DIRECT --> TYPE{Prediction type}
    TYPE -- Next available --> ONE[Predict -> one price]
    TYPE -- Future date --> MULTI[Generate Forecast -> path up to 30 days]
    PERF --> READ[Reads reports/metrics JSON]
    HIST --> HSRC{HARTI / CBSL / DCS} --> CHART[Charts + tables, kept separate]
    PLAN --> CROP[Crop + variety + date] --> HARV[Harvest window add_duration]
    HARV --> REACH{Harvest within 30 days of latest price?}
    REACH -- yes --> FPRICE[Forecast + likely range + selling suggestion]
    REACH -- no --> HISTP[Not available + same-season history]
    ONE & MULTI & READ & CHART & FPRICE & HISTP --> STOP([Ctrl+C in console to stop])
```

## 6. Key design decisions (be ready to justify)

| Decision | Reason |
|---|---|
| Files, not a database | One user, read-mostly, ~600k rows fit in memory; CSV is transparent and easy to back up |
| Direct (one model per horizon) instead of recursive | Recursive errors compound: MAPE 36.5 % vs 25.8 % at 15–30 days |
| Log-change target | Same model for cheap and expensive vegetables; prediction can never be negative |
| Fixed test start date | Results stay comparable as new data arrives; no peeking |
| Separate HARTI / CBSL / DCS | Different price types (wholesale/retail) and collectors — mixing would be wrong |
| Caches keyed on file modification times | App stays fast but never shows stale data after an update |
| Sandbox models saved to `models/demo` | A demo retrain can never overwrite the real model (`config.yaml` `demo_models_dir`) |
