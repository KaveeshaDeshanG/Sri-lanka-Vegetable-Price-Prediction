# 05 — Technology Stack Explanation

The versions are exactly as pinned in `requirements.txt` and installed in `.venv312` (Python 3.12.10).

> සිංහල: "Technology stack" කියන්නේ පද්ධතිය සෑදීමට භාවිතා කළ භාෂාව, පුස්තකාල (libraries) සහ මෙවලම් ලැයිස්තුවයි.

| Technology | Version | What it does in THIS project | Where (evidence) | Why chosen | සිංහල |
|---|---|---|---|---|---|
| **Python** | 3.12.10 | The only programming language | `pyproject.toml` `requires-python = ">=3.12,<3.15"` | Best data-science ecosystem; 3.15 cannot install PyMuPDF's onnxruntime | ප්‍රධාන ක්‍රමලේඛන භාෂාව |
| **Streamlit** | 1.66.1 | The web dashboard: pages, buttons, charts | `app/Home.py`, `app/pages/*.py` | A web UI in pure Python; no HTML/JavaScript needed | Python පමණක් භාවිතා කර වෙබ් අතුරුමුහුණත සාදයි |
| **pandas** | 3.1.0 | Reading CSVs, cleaning, merging, pivoting, rolling windows | everywhere, e.g. `forecast_features.py` 67–76 | The standard table library | වගු (tables) ලෙස දත්ත හසුරුවයි |
| **NumPy** | 2.5.4 | Maths: log, exp, bootstrap random sampling | `direct_forecaster.py` 84, 121, 132 | Fast arrays; used by pandas/sklearn | ගණිත ක්‍රියා |
| **scikit-learn** | 1.9.1 | Ridge, LinearRegression, RandomForest, SimpleImputer, StandardScaler, Pipeline, OneHotEncoder | `direct_forecaster.py` 19–22, 118; `src/models/trainer.py` | Reliable, well-tested ML library | Machine Learning ආකෘති |
| **XGBoost** | 3.4.1 | Gradient-boosted trees (one of the earlier compared models) | `src/models/trainer.py`; `config.yaml` `model.xgboost` | Strong tabular model to compare against | සංසන්දනය සඳහා භාවිතා කළ ආකෘතියක් |
| **joblib** | 1.6.0 | Saving/loading trained models (`.joblib` files) | `src/models/persistence.py` 21/44 | Standard for sklearn objects | පුහුණු කළ ආකෘතිය ගොනුවක ගබඩා කරයි |
| **PyMuPDF** (`pymupdf`) | 2.1 | Reads text and tables from HARTI / CBSL / DCS **PDF** bulletins | `src/data_collection/harti_live_client.py`, `cbsl_price_report_parser.py`, `dcs_retail_parser.py` | Fast, accurate PDF text with positions | PDF ගොනු කියවයි |
| **requests** | 2.34.2 | HTTP downloads (HARTI pages, PDFs, Open-Meteo, CEYPETCO) | `harti_live_client.py`, `live_weather_client.py`, `live_fuel_price_client.py` | Simple HTTP | අන්තර්ජාලයෙන් දත්ත බාගත කරයි |
| **BeautifulSoup4** | 4.15.0 | Parses the HTML archive page `daily-price.php` to find bulletin links | `harti_live_client.py` | Robust HTML parsing | HTML පිටු කියවයි |
| **PyYAML** | 6.0.3 | Reads `config/config.yaml` and `config/crop_production_regions.yaml` | `config/config_loader.py` `ConfigLoader` | Settings outside the code | සැකසුම් ගොනුව කියවයි |
| **matplotlib** | 3.11.2 | Static figures in `reports/figures` (training scripts) | `scripts/train.py` etc. | Standard plotting | රූපසටහන් (report figures) |
| **Altair** | (installed with Streamlit) | Interactive charts in the app (Predict, Price History, Model Performance) | `app/pages/*.py` `import altair` | Streamlit's native chart library | App එකේ chart |
| **Pillow** | 13.0.0 | Image crops in `scripts/validate_real_datasets.py` | that script | Checking parsed numbers against PDF images | රූප කැපීම |
| **pytest** | dev extra | 1,320 automated test cases in `tests/` | `pyproject.toml` `[project.optional-dependencies] dev` | Standard test runner | ස්වයංක්‍රීය පරීක්ෂණ |
| **Open-Meteo API** | — | Weather history and recent weather (rain, temperature, humidity) — no API key | `live_weather_client.py` 78/180/215; `config.yaml` `weather_api` | Free, no key, historical archive | කාලගුණ දත්ත |
| **CSV / JSON files** | — | All storage (no database) | `data/raw/*.csv`, `data/raw/data_manifest.json`, `reports/metrics/*.json` | Simple, transparent, versionable | දත්ත ගබඩාව — database නැත |
| **Windows batch** | — | `start_app.bat`, `scripts/run_daily_update.bat` | project root / scripts | One-click start for the user | Double-click කර app එක ආරම්භ කිරීම |

## Things NOT used (say this clearly if asked)

- **No database** (no MySQL, SQLite, MongoDB). Data lives in CSV files. *Why:* one user, read-mostly data, ~0.5 M rows fits easily in memory.
- **No Flask/Django/React backend.** Streamlit runs both the UI and the Python logic in one process.
- **No TensorFlow / PyTorch / LSTM / ARIMA / Prophet.** The models are scikit-learn and XGBoost.
- **No cloud deployment, no Docker, no login.**
- **No paid API keys.** `config.yaml` comment: "Open-Meteo requires no API key for non-commercial use".

> සිංහල: Database, login, cloud, deep learning (LSTM) **භාවිතා කර නැත**. Examiner අසුවොත් "ඔව්, එය future work" ලෙස
> අවංකව පවසන්න. හේතුව: එක් පරිශීලකයෙකු, CSV ගොනු මතකයට පහසුවෙන් ගැළපේ, සරල Ridge ආකෘතිය baseline එකට වඩා හොඳයි.

## Why Streamlit instead of Flask + React? (likely question)
- Whole app written in Python, same language as the ML code → no separate API layer.
- Built-in widgets (date picker, select box, file uploader) and caching (`@st.cache_data`, `@st.cache_resource`).
- Trade-off: less control over layout; every click re-runs the page script (handled by caching keyed on
  `data_version()`, `app/catalogue.py` line 162).
