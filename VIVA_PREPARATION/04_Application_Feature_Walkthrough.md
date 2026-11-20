# 04 — Application Feature Walkthrough

All pages were opened and used on 13 November 2026 (see `11_Project_Testing_Report.md`). Line numbers refer to the
current source files.

> සිංහල: වම් පැත්තේ sidebar එකේ පිටු 5ක් ඇත — Home, Predict, Model Performance, Price History, Cultivation Planner.
> Sidebar එකේ "Data sources" කොටස, ඇත්තටම ගොනු තිබෙන මූලාශ්‍ර පමණක් පෙන්වයි (`app/ui.py` `available_sources`, line 228).

---

## Page 1 — Home (`app/Home.py`)

**Purpose:** first impression — what the system is, how accurate it is, which data it uses.

| What you see | Where it comes from |
|---|---|
| Header "Sri Lankan Vegetable Price Prediction" | `ui.page_header` |
| Blue box: "Real data: the Ridge direct forecaster is trained … through 2026-11-06" | `DirectForecaster.load` via `_load_forecaster` (line 31); `trained_through` |
| 4 cards: Next-day error 7.5% · 4.3% lower than holding the price · 1–30 days · 31 vegetables | `_saved_accuracy` (line 36) reads `reports/metrics/direct_forecaster_metrics.json` |
| "Data in use" table: dataset, price type, first, latest, rows, source link | `load_manifest` reads `data/raw/data_manifest.json` |
| "Last daily update …" line | `latest_update` reads `data/raw/update_log.jsonl` |
| "How it works" 4 step cards + page descriptions | static text |
| 4 "Go to …" links | `st.page_link` lines 190–196 |

**Backend:** nothing is computed; it only reads saved files. If a dataset or model is missing/damaged, a red error
lists the problem (`historical_dataset_problems`, `forecaster_problems` in `app/services.py`).
**Demo:** point at the accuracy cards and say "every number is compared with simply holding the last price".

---

## Page 2 — Predict (`app/pages/1_Predict.py`)

*Redesigned on 15 November 2026: two data sources instead of four.*

**Data source** radio, exactly two options: **Upload CSV** and **Latest Real Data Prediction** (the default).
The former "Bundled sample data (synthetic)" and live "Latest HARTI Data (real)" options were removed from the
page; their functions remain in `app/services.py` and are still tested.

> සිංහල: දැන් විකල්ප දෙකක් පමණි — ඔබේම CSV ගොනුව, හෝ නිල මූලාශ්‍රවලින් එකතු කළ නවතම සැබෑ දත්ත.

### Latest Real Data Prediction: five sections
| Section | What you see | Code |
|---|---|---|
| **1. Official source status** | A card each for HARTI, CBSL, DCS: newest publication found, last checked, new data collected, failure or import problem, stale or not. **Check official sources now** button | `_render_source_status` (`1_Predict.py`); `update_status` (`app/catalogue.py`); `run_data_update` (`app/data_refresh.py`) |
| **2. Latest collected data** | Expander: newest HARTI Dambulla price of every vegetable. Then the records of the vegetable chosen in section 3: a box naming the **forecast starting point** (bulletin, min/max/midpoint, link, collection time). Four tabs: HARTI Dambulla (used by the model), HARTI other markets (model inputs), CBSL (reference only, wholesale and retail in separate tables), DCS (reference only). Each table puts date, market, min, max, midpoint, unit, document link and collection time first, with the rest reached by scrolling sideways (date and market stay pinned) | `_render_vegetable_records`; `app/source_records.py`; `data/reference/source_item_map.csv` |
| **3. Vegetable and market** | **Vegetable** dropdown (forecastable series only; it also drives section 2's tables), the list of series that can't be forecast and why, and Market (Dambulla); cards for data source, market, price type, latest data; latest actual price and last successful update | `ui.data_source_panel` |
| **4. Prediction** | Model and its inputs named ("CBSL and DCS records are not inputs"). **Forecast period**: 1 day / 7 days / 14 days / 30 days / another date. **Predict** → price, chart, weather, daily path, and a table of all four horizons with the accuracy measured for each range | `_render_real_prediction` → `direct_forecast` (`app/forecasting.py`) |
| **5. Data freshness and warnings** | Up to date, or a yellow warning (check failed, HARTI not checked, no check for 24 h, price > 3 days old) | `_render_freshness`, `update_notice` |

**What happens when you click Predict:**
1. `ForecastFeatureBuilder.build_origin_features` builds features for every vegetable and date (cached).
2. `direct_forecast` → `DirectForecaster.forecast` uses the Ridge model for that number of days: **price = last price × exp(predicted log change)**.
3. A second call gives the 30-day path, from which the 1/7/14/30 table is taken (each from its own model).

**Expected (15 Nov 2026 data, latest Beans 12 Nov Rs 430):** 1 day → 13 Nov **Rs 413.04**; 7 → 19 Nov **Rs 431.62**; 14 → 26 Nov **Rs 459.86**; 30 → 12 Dec **Rs 447.14**.

### Upload CSV
A file uploader (columns date, vegetable, market, price). `prepare_uploaded_history` (`services.py`) validates it;
errors are shown in red. Uses the earlier 51-feature model (or a sandbox model saved only to `models/demo`), never
official data. The old "Prediction type" and "Generate Forecast" controls belong to this mode now.

---

## Page 3 — Model Performance (`app/pages/2_Model_Performance.py`)

| Section | Function | Shows |
|---|---|---|
| Ridge direct forecaster | `_show_direct` (81) | 4 cards (next-day error 7.51%, gain 4.3% with 95% interval 2.6–6.0, 22–30 day error 25.1% vs 29.4%, 21 of 30 vegetables improved); bar chart of MAE by forecast range (model vs "Hold last price"); table by band; next-day comparison on the same 3,836 rows; results by vegetable |
| Earlier models | `_show_earlier_models` (172) | Linear Regression, Random Forest, XGBoost vs "Yesterday's price" — none beats it |
| Backtest | `_show_backtest` (238) | Recursive multi-day backtest (if its JSON exists) |

**Backend:** only reads `reports/metrics/*.json` (`_load_report`, line 60). No model runs here.
**Demo line:** "The green bars are always lower than the grey baseline bars — but only by 4–8%."

---

## Page 4 — Price History (`app/pages/3_Price_History.py`)

**Data source** radio (line 97):
| Source | Controls | Shows |
|---|---|---|
| HARTI daily bulletins (wholesale, by market) | Market (181), Vegetable (187), Date range (192) | Source/freshness cards, Latest/Average/Lowest/Highest, line chart (breaks at gaps > 7 days), Weather tabs Rainfall/Temperature/Humidity (Dambulla only, line 231), "View as a table" |
| CBSL daily report (wholesale and retail) | Item (102), Price type wholesale/retail (105), Markets multiselect (108) | Chart per market (Dambulla green, Pettah blue, Narahenpita amber), table with source links |
| DCS weekly bulletins (retail, Colombo District) | Vegetable (140), Years slider (144) | Weekly retail chart (points placed by week number), table |

**Backend:** `_load_history` (40), `_load_markets` (45), `_load_cbsl` (67), `_load_dcs` (56) read the CSVs; cached by `data_version`.
**Demo line:** "Wholesale and retail, and HARTI vs CBSL, are never mixed — they are different measurements."

---

## Page 5 — Cultivation Planner (`app/pages/4_Cultivation_Planner.py`)

| Control | Line | Meaning |
|---|---|---|
| Vegetable | 133 | Crops with DOA timings |
| Variety or guidance | 136 | e.g. "Bush types (crop guidance) — 45 days" |
| Market price series | 146 | Which HARTI series to price it with (or "No HARTI price series") |
| Box "Published timing …" | — | Quote + DOA source URL + how dates are calculated |
| Box "Forecast reach" | 191 | Before Generate: the forecast's last date and the latest start date that still gets a forecast |
| Start date (label changes: Planting / Transplanting / Nursery sowing date …) | 200 | Range: 2 years back to 1 year ahead |
| Weather to display | 209 | Only changes the "Recent weather" box. Defaults to the region the model uses (Beans: Badulla, Carrot: Nuwara Eliya). Never changes the forecast |
| **Generate Cultivation Plan** | 220 | Builds the plan |

**Backend:** `build_plan` (`app/cultivation.py` 305) → `harvest_window` (193) uses `add_duration` (110): days/weeks exact,
months = calendar months (never 30 days). If the harvest is within 30 days of the latest price → `forecast_with_range`
(256) gives price + likely range (10th–90th percentile of past errors, all vegetables together, `reports/metrics/forecast_error_bands.json`);
`suggest_selling_day` (270), only when the forecast covers the whole window. The plan names the starting HARTI record and the model's weather region. Otherwise → `select_price` leads with a labelled reference (section 3, "Price at harvest"): recorded prices for a passed window, else the Historical Seasonal Price Reference (`seasonal_reference`: same weeks ± 7 days, 5 most recent years, ≥ 10 prices in 3 years), else the latest actual price; a crop with no HARTI series gets "No Verified Price Available" and the missing source. `latest_start_in_reach` (198) gives the date in the "Forecast reach" box.

**Expected (tested):** Beans bush, planted 13 Nov 2026 → first harvest 28 Dec 2026 → Historical Seasonal Price Reference, Rs 425/kg (64 prices, 2021–2026; beyond
reach) + history table. Planted 20 Oct 2026 → first harvest 4 Dec 2026 → forecast **Rs 463/kg** (from Rs 430 on 12 Nov; data as of 15 Nov 2026).

---

## Numbered click-by-click demonstration (all features)

1. Double-click `start_app.bat` → browser opens **Home**. Show the accuracy cards and "Data in use" table.
2. Scroll down → "How it works" cards → click **Go to Prediction →**.
3. Predict opens on **Latest Real Data Prediction**. Section 1: read the HARTI, CBSL and DCS cards (newest publication, last checked).
4. Optional (internet): **Check official sources now** → the result message.
5. Section 2: open **Newest HARTI Dambulla price of every vegetable**; open **"5 recorded price series can't be forecast, and why"**.
6. Vegetable → **Beans**. Read the **forecast starting point** box; click **Bulletin** to show the original HARTI PDF.
7. Open the tabs: HARTI other markets, **CBSL** (wholesale and retail tables apart), **DCS** (retail, weekly).
8. Section 3: market Dambulla, wholesale, latest data card.
9. Section 4: Forecast period **7 days** → **Predict** → Rs 431.62, chart, weather card, then the **1, 7, 14 and 30 days** table with measured accuracy.
10. Choose **Another date** → pick a date → **Predict** (dates beyond 30 days can't be chosen).
11. Section 5: read the freshness message.
12. Data source → **Upload CSV** → upload `reports\final_submission\test_evidence\inputs\missing_price_column.csv` → the red error. Then `valid_upload.csv` → loads (no official data shown).
13. Sidebar → **Model Performance** → explain the bar chart (model vs hold last price), the 4.3% gain and the earlier models table.
14. Sidebar → **Price History** → HARTI: change Market to **Kandy**, change the date range, open the weather tabs (with Dambulla).
15. Switch to **CBSL daily report** → Price type **retail** → show the three markets.
16. Switch to **DCS weekly bulletins** → move the Years slider.
17. Sidebar → **Cultivation Planner** → Beans, Bush types, keep today → **Generate Cultivation Plan** → orange "Historical Seasonal Price Reference – Not a Forecast" + history table.
18. Change the Planting date to ~7 weeks ago → **Generate** → forecast price at harvest with range and selling suggestion.
19. Open **All published durations** expander.
20. Close the browser tab; press **Ctrl+C** in the console to stop.

**Possible problems during the demo:** no internet → skip step 10; first page load takes ~10 s (feature building) —
say "it is building features for 31 vegetables; it is cached afterwards".
