# 07 — Data Sources and API Integration

All numbers below come from `data/raw/data_manifest.json` (generated 2026-11-11 03:43) — read from the files themselves.

> සිංහල: මෙම ව්‍යාපෘතියේ සියලු දත්ත **සැබෑ, රජයේ/පොදු මූලාශ්‍රවලින්** ලබාගත් ඒවාය. කිසිදු දත්තයක් නිර්මාණය කර
> (fabricate) නැත. එකම ව්‍යාජ දත්ත ගොනුව "Bundled sample data (synthetic)" ලෙස පැහැදිලිව ලේබල් කර ඇත.

## 1. Datasets

| File | Source | Price type | Rows | Period | Used for |
|---|---|---|---|---|---|
| `harti_prices.csv` | HARTI daily food commodity bulletins — https://www.harti.gov.lk/daily-price.php | Wholesale Rs/kg, **midpoint** of the published range, Dambulla | 76,719 | 2015-06-22 → 2026-11-09 | **Training target** + history |
| `harti_all_markets.csv` | Same bulletins, every market (11) | Wholesale | 526,905 | 2015-06-22 → 2026-11-09 | `other_markets` features; Price History page |
| `cbsl_daily_prices.csv` | Central Bank of Sri Lanka Daily Price Report — cbsl.gov.lk price-report | Wholesale **and** retail (column `price_type`), Pettah / Dambulla / Narahenpita | 9,277 | 2026-09-01 → 2026-11-10 | Price History only (cross-check, **never** training) |
| `dcs_weekly_retail_prices.csv` | Department of Census and Statistics weekly retail prices | Retail, weekly, Colombo District | 11,439 | 2019-12 wk 4 → 2026-11 wk 1 | Price History only |
| `fuel_prices.csv` | CEYPETCO price revisions | Diesel/petrol Rs/litre | 4,161 | 2015-06-22 → 2026-11-11 | `fuel` features |
| `weather_data_by_region.csv` | Open-Meteo historical archive | Rain, temperature, humidity; 8 locations | 33,264 | 2015-06-22 → 2026-11-08 | `rain_windows`, `weather_lag_14` features |
| `app/sample_data/sample_history.csv` | **Synthetic** (made up for demo) | — | 360 | 2026-05-15 → 2026-08-12 | "Bundled sample data" option only |

**Why "midpoint"?** HARTI prints a range like "400-450". `parse_price_range` (`harti_live_client.py` line 138) turns
it into 425.

**Data gaps:** 224 bulletins could not be read (151 of them from 2022–23); HARTI does not publish on some days (holidays).
A missing day is carried forward at most 7 days in features; the Price History chart breaks the line at gaps > 7 days.

## 2. How HARTI PDFs are read (the hardest part)

1. `HartiLiveClient` (`harti_live_client.py` line 323) downloads `daily-price.php` (HTML) with **requests** and finds bulletin links with **BeautifulSoup**.
2. Downloads the PDF; **PyMuPDF** extracts words with their x/y positions.
3. `_extract_vegetable_table` (line 535) rebuilds the table rows; `split_merged_label_rows` (235) fixes rows where two vegetable names were printed together.
4. `parse_price_range` (138) converts "400-450" → 425.
5. Historical import: `scripts/build_harti_prices.py` (Dambulla) and `scripts/build_market_prices.py` (all markets).
6. `scripts/validate_real_datasets.py` crops PDF images (Pillow) to manually check parsed numbers.

> සිංහල: HARTI ට API එකක් නැහැ — PDF පමණි. PDF එකේ වචන සහ ඒවායේ ස්ථාන (x, y) කියවා, වගුව නැවත ගොඩනඟා, "400-450"
> වැනි පරාසයක මැද අගය (425) මිල ලෙස ගනී.

## 3. "Live" / real-time data — what is true

| Data | Live in the app? | How | Evidence |
|---|---|---|---|
| HARTI latest bulletin | **Yes, on demand** ("Latest HARTI Data" option) | `fetch_harti_latest_snapshot` (`app/services.py` 552) → `fetch_latest_prices` (`harti_live_client.py` 379); cached 60 min (`config.yaml` `harti.cache_minutes`); missing days between stored data and the bulletin filled from archived bulletins (`backfill_harti_price_gap`, `services.py` 705). **Not saved to disk.** | Tested 13 Nov 2026: bulletin dated 2026-11-12, 30 prices |
| Weather | Yes for prediction | `OpenMeteoWeatherClient` (`live_weather_client.py` 78): `fetch_recent` (180) → api.open-meteo.com/v1/forecast; `fetch_historical` (215) → archive-api.open-meteo.com. **No API key.** | `config.yaml` `weather_api` |
| Fuel | Yes, with fallback | `CeypetcoLiveFuelPriceClient` (`live_fuel_price_client.py` 95/131) → api.ceypetco.gov.lk; if it fails, last known price | `config.yaml` `fuel_api` |
| CBSL / DCS | No — updated by the daily script | `scripts/daily_update.py` | — |

**Important honest statement:** there is **no real-time price API** for Sri Lankan vegetables. HARTI publishes one
PDF per day, usually 1–2 days late. "Real-time" in this project means "as current as the newest published bulletin".

> සිංහල: "Real-time" යනු තත්පරයෙන් තත්පරයට යාවත්කාලීන වීම නොවේ. HARTI දිනකට එක් PDF එකක් පමණක් නිකුත් කරයි
> (සාමාන්‍යයෙන් දින 1–2ක් පසුව). App එක ලබාගත හැකි නවතම bulletin එක භාවිතා කරයි.

## 4. Keeping data up to date

`scripts/daily_update.py` (run manually or via `scripts/run_daily_update.bat`):
1. Takes a lock file (exit code 3 if another run is active).
2. Backs up changed files to `D:\ECU\Sri-lanka-Vegetable-Price-Backups\daily\`.
3. Imports new HARTI, CBSL and DCS bulletins; tops up weather and fuel.
4. Rewrites the manifest; logs to `data/raw/update_log.jsonl`. Exit code 2 if a bulletin was unreadable.
5. The running app notices changed files automatically (`data_version`, `app/catalogue.py` 162).

**Not scheduled automatically** — registering a Windows Task Scheduler job must be done by the user.
`scripts/update_data.py` is the full rebuild (slower; `--workers` default 1; weather fetched in pieces to avoid
Open-Meteo HTTP 429 "too many requests").

**Freshness labels** (`freshness`, `app/catalogue.py` 96): ≤ 3 days old = current, ≤ 30 = ageing, > 30 = stale.

## 5. Error handling for external sources
| Failure | What the app does |
|---|---|
| No internet / HARTI down | Red message "Couldn't fetch the latest HARTI bulletin"; other data sources still work |
| Open-Meteo fails | Falls back to weather already in the stored data |
| CEYPETCO fails | Uses last known fuel price |
| Bulletin unreadable in update | Logged; script exits 2; existing data untouched |
| Uploaded CSV invalid | Clear message naming the missing column / empty file (`prepare_uploaded_history`, `services.py` 148) |

## 6. Data the project does NOT have
No farm-gate prices, no supply/arrival volumes, no satellite/crop-yield data, no consumer-demand data, no
forecasts for markets other than Dambulla.
