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

### Section 1 "Historical data" — Data source radio (line 354)
| Option | What happens | Backend |
|---|---|---|
| **Historical dataset (real)** (default) | Loads 76,719 real Dambulla prices + weather + fuel | `load_real_market_history` (`services.py` 264), cached by `_load_real_market_history_cached` (160) |
| **Bundled sample data (synthetic)** | Loads 360 made-up rows (Carrot, Tomato) labelled "Not real market data" | `load_sample_history` (`services.py` 259) |
| **Upload my own CSV** | A file uploader appears (line 365). Needs columns date, vegetable, market, price | `prepare_uploaded_history` (`services.py` 148) validates; errors shown in red |
| **Latest HARTI Data (real)** | Downloads HARTI's newest bulletin from the internet; shows its date and a **Refresh** button (407) | `fetch_harti_latest_snapshot` (`services.py` 552) → `HartiLiveClient.fetch_latest_prices` (`harti_live_client.py` 379); cached 60 min |

"Preview loaded data" expander shows the first 20 rows.

### Section 2 "Model"
- Real data → green box "Ridge direct forecaster loaded …".
- Sample/upload → the earlier 51-feature model is used, with an info box explaining why; an expander lets you **train a
  sandbox model** (`_train_sandbox_model` line 484 → `train_demo_model` `services.py` 1042) saved only to `models/demo`.

### Section 3 "Request a prediction"
| Control | Line | What it does |
|---|---|---|
| **Vegetable** dropdown | 578 | Only series the forecaster can forecast (`series_status`, `catalogue.py` 122) |
| Expander "5 recorded price series can't be forecast, and why" | ~580 | Lists Bitter Gourd (Other/Village), Brinjals (Other/Village) — not trained; Cabbage (Nuwara Eliya) — last price 2025-07-12 |
| **Market** dropdown | 596 | Dambulla |
| Cards: Data source · Market · Price type · Latest data + freshness message | `ui.data_source_panel` (`ui.py` 265) | "Latest data 2026-11-09 is 4 days old; forecasts start from that date" |
| **Prediction type** radio | 669 | "Latest / Next Available Prediction" or "Future Date Forecast (multi-step)" |
| **Date to predict** / **Forecast date** | 717 / 832 | Minimum = the day after the latest price. A past date shows a yellow warning (`past_date_notice`, `catalogue.py` 108) |
| **Predict** button | 733 | Next-day (or any single date) prediction |
| **Generate Forecast** button | 858 | Forecast path up to the chosen date |

**What happens when you click Predict / Generate Forecast (real data):**
1. `_forecast_origin_rows` (250) → `ForecastFeatureBuilder.build_origin_features` builds features for every vegetable and date (cached).
2. `_show_direct_forecast` (703) → `direct_forecast` (`app/forecasting.py` 116).
3. Horizon h = target date − latest actual date. If h > 30 → error "Cannot forecast that date: … The forecaster reaches at most 30 days ahead."
4. `DirectForecaster.forecast` (`direct_forecaster.py` 134) uses the Ridge model for horizon h; `predict_rows` (126): **price = last price × exp(predicted log change)**.
5. `_render_direct_result` (255) shows the price, a chart of recent prices + forecast, the weather used, data sources and the "Daily forecast path" expander.

**Expected (tested 13 Nov 2026):** Beans → 10 Nov 2026 → **Rs 453.74/kg**; 25 Nov 2026 (16 days ahead) → **Rs 507.96/kg**; 25 Dec 2026 → refused (46 days).

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
| Vegetable | 112 | Crops with DOA timings |
| Variety or guidance | 115 | e.g. "Bush types (crop guidance) — 45 days" |
| Market price series | 125 | Which HARTI series to price it with (or "No HARTI price series") |
| Box "Published timing …" | — | Quote + DOA source URL + how dates are calculated |
| Start date (label changes: Planting / Transplanting / Nursery sowing date …) | 153 | Range: 2 years back to 1 year ahead |
| Growing location | 161 | Only changes the "recent weather" box (alphabetical default) |
| **Generate Cultivation Plan** | 165 | Builds the plan |

**Backend:** `build_plan` (`app/cultivation.py` 276) → `harvest_window` (193) uses `add_duration` (110): days/weeks exact,
months = calendar months (never 30 days). If the harvest is within 30 days of the latest price → `forecast_with_range`
(227) gives price + likely range (10th–90th percentile of past errors, `reports/metrics/forecast_error_bands.json`);
`suggest_selling_day` (241). Otherwise → "Not available" + `same_season_history` (255) from earlier years.

**Expected (tested):** Beans bush, planted 13 Nov 2026 → first harvest 28 Dec 2026 → price "Not available" (beyond
reach) + history table. Planted 20 Oct 2026 → first harvest 4 Dec 2026 → forecast **Rs 450/kg**.

---

## Numbered click-by-click demonstration (all features)

1. Double-click `start_app.bat` → browser opens **Home**. Show the accuracy cards and "Data in use" table.
2. Scroll down → "How it works" cards → click **Go to Prediction →**.
3. Predict: keep **Historical dataset (real)**. Open **Preview loaded data**, close it.
4. Open the expander **"5 recorded price series can't be forecast, and why"** → explain stale/untrained series.
5. Vegetable → **Beans**. Show the four source cards and the freshness message.
6. Keep **Latest / Next Available Prediction**; note the yellow "already passed" warning (date = day after latest data).
7. Click **Predict** → show the price (Rs 453.74), chart, weather card, model used.
8. Choose **Future Date Forecast (multi-step)** → set Forecast date to a date ~2 weeks ahead → **Generate Forecast** → show price, path chart, open **Daily forecast path**.
9. Set a date > 30 days ahead → **Generate Forecast** → show the refusal message.
10. Data source → **Latest HARTI Data (real)** → show the bulletin date and "not a real-time feed" caption (needs internet). Click **Refresh** once.
11. Data source → **Bundled sample data (synthetic)** → show "Not real market data" and that the earlier model is used.
12. Data source → **Upload my own CSV** → upload `Final_Project_Report\testing_evidence\inputs\missing_price_column.csv` → show the red error. Then upload `valid_upload.csv` → loads.
13. Sidebar → **Model Performance** → explain the bar chart (model vs hold last price), the 4.3% gain and the earlier models table.
14. Sidebar → **Price History** → HARTI: change Market to **Kandy**, change the date range, open the weather tabs (with Dambulla).
15. Switch to **CBSL daily report** → Price type **retail** → show the three markets.
16. Switch to **DCS weekly bulletins** → move the Years slider.
17. Sidebar → **Cultivation Planner** → Beans, Bush types, keep today → **Generate Cultivation Plan** → "Not available" + history.
18. Change the Planting date to ~7 weeks ago → **Generate** → forecast price at harvest with range and selling suggestion.
19. Open **All published durations** expander.
20. Close the browser tab; press **Ctrl+C** in the console to stop.

**Possible problems during the demo:** no internet → skip step 10; first page load takes ~10 s (feature building) —
say "it is building features for 31 vegetables; it is cached afterwards".
