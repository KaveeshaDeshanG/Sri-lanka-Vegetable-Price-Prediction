# 07 — Data Sources and API Integration

All numbers below come from `data/raw/data_manifest.json` (generated 2026-11-11 03:43) — read from the files themselves.

> සිංහල: මෙම ව්‍යාපෘතියේ සියලු දත්ත **සැබෑ, රජයේ/පොදු මූලාශ්‍රවලින්** ලබාගත් ඒවාය. කිසිදු දත්තයක් නිර්මාණය කර
> (fabricate) නැත. එකම ව්‍යාජ දත්ත ගොනුව (`app/sample_data/sample_history.csv`) Predict පිටුවේ තවදුරටත් පෙන්වන්නේ නැත.

## 1. Datasets

| File | Source | Price type | Rows | Period | Used for |
|---|---|---|---|---|---|
| `harti_prices.csv` | HARTI daily food commodity bulletins — https://www.harti.gov.lk/daily-price.php | Wholesale Rs/kg, **midpoint** of the published range, Dambulla | 77,011 | 2015-06-22 → 2026-11-19 | **Training target** + history |
| `harti_all_markets.csv` | Same bulletins, every market (11) | Wholesale | 529,020 | 2015-06-22 → 2026-11-19 | `other_markets` features; Price History page |
| `cbsl_daily_prices.csv` | Central Bank of Sri Lanka Daily Price Report — cbsl.gov.lk price-report | Wholesale **and** retail (column `price_type`), Pettah / Dambulla / Narahenpita | 10,551 | 2026-09-01 → 2026-11-19 | Price History only (cross-check, **never** training) |
| `dcs_weekly_retail_prices.csv` | Department of Census and Statistics weekly retail prices | Retail, weekly, Colombo District | 11,439 | 2019-12 wk 4 → 2026-11 wk 1 | Price History only |
| `fuel_prices.csv` | CEYPETCO price revisions | Diesel/petrol Rs/litre | 4,169 | 2015-06-22 → 2026-11-19 | `fuel` features |
| `weather_data_by_region.csv` | Open-Meteo historical archive | Rain, temperature, humidity; 8 locations | 33,328 | 2015-06-22 → 2026-11-16 | `rain_windows`, `weather_lag_14` features |
| `app/sample_data/sample_history.csv` | **Synthetic** (made up for demo) | — | 360 | 2026-05-15 → 2026-08-12 | Tests only; no longer offered on the Predict page (since 15 Nov 2026) |

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
| HARTI, CBSL, DCS | **Checked automatically every 3 hours** (Windows task `VegetablePriceUpdate` → `scripts/daily_update.py`), or with **Check official sources now** on the Predict page. New reports are saved, de-duplicated and logged per source | `data/raw/update_log.jsonl` (`"sources"`); Predict page section 1 | Scheduled task run verified 15 Nov 2026 |
| Weather, fuel | Topped up by the same update; the forecaster reads the stored files | `topup_weather`, `topup_fuel` | — |
| Live single-bulletin download | Code kept (`fetch_harti_latest_snapshot`, `backfill_harti_price_gap`), **no longer offered on the page** since 15 Nov 2026 | `app/services.py` | Tested 13 Nov 2026 |

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

**Scheduled since 15 Nov 2026:** Windows task `VegetablePriceUpdate`, every 3 hours from 07:00 (first timed run 13:00, exit code 0). It runs only while the PC is on.
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
