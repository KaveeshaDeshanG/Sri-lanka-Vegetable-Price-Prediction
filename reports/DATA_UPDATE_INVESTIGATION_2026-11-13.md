# Data update investigation — 13 November 2026

## Question
Are 2026 vegetable prices missing from the application, and is the latest official data being used?

## Findings (evidence, before any change)

| Check | Result | Evidence |
|---|---|---|
| 2026 rows in the Dambulla dataset | **Present**: 8,171 prices, 294 bulletin dates, every month Jan–Nov 2026 | `data/raw/harti_prices.csv` |
| 2026 rows in other datasets | Present: all markets 63,037; CBSL 9,277 (Sep–Nov 2026); DCS to 2026-11 week 1; fuel to 2026-11-11; weather to 2026-11-08 | `data/raw/*.csv` |
| Date parsing | 0 unparseable dates in any file; no dates before 2015 or after today | pandas `to_datetime` over every file |
| App loader drops rows? | No: `load_real_market_history` (`app/services.py`) returns all 76,719 prices, newest 2026-11-09 | direct call |
| Date filters | No fixed years. Price History defaults to each series' full range (`app/pages/3_Price_History.py` line 192); DCS slider to the newest year (line 144) | source |
| Caches | Keyed on file modification times (`app/catalogue.py` `data_version`); new files are picked up without a restart | verified in the running app |
| Series that really end before 2026 | Bitter Gourd (Village) 2019-04-03, Brinjals (Village) 2019-06-20, Bitter Gourd (Other) and Brinjals (Other) 2023-01-04, Cabbage (Nuwara Eliya) 2025-07-12; Pettah market 2022-01-03 | HARTI stopped publishing these; the Predict page lists them with reasons |

**Root cause of "old" data:** not missing 2026 records. The datasets were **4 days behind** (newest HARTI price
2026-11-09) because `scripts/daily_update.py` had last run at 2026-11-11 03:45 and is not scheduled automatically.
HARTI had since published bulletins for 10, 11 and 12 November. A fresh clone of the GitHub repository made before
13 Nov 2026 also had **no datasets at all** (the old `.gitignore` excluded `data/raw/*`).

## Update performed
- Full snapshot first: `D:\ECU\Sri-lanka-Vegetable-Price-Backups\manual_before_update_20261113\` (data/raw, models, reports/metrics); the script's own backup: `...\daily\20261113_200627\`.
- `scripts/daily_update.py` (existing pipeline; appends only unseen bulletins, validates rows, writes atomically).

| Dataset | Before | After | Added | Newest |
|---|---|---|---|---|
| HARTI Dambulla wholesale | 76,719 | 76,808 | 89 | 2026-11-12 |
| HARTI all markets | 526,905 | 527,575 | 670 | 2026-11-12 |
| CBSL daily (wholesale + retail) | 9,277 | 9,823 | 546 | 2026-11-13 |
| DCS weekly retail | 11,439 | 11,439 | 0 (no newer bulletin listed) | 2026-11 week 1 |
| Weather (8 locations) | 33,264 | 33,280 | 16 | 2026-11-10 (Open-Meteo archive lag) |
| Fuel | to 2026-11-11 | to 2026-11-13 | 2 days | 2026-11-13 |

Every earlier row is still present with an identical price (compared against the snapshot); 0 duplicates on
(date, vegetable, market) / (published, reported, market, vegetable) / (date, item, market, price_type).

## Model
The saved Ridge direct forecaster (`models/direct_forecaster.joblib`, trained through 2026-11-06 on history from
2024-04-25) needs no retraining to use new prices: forecasts start from the newest price (verified: Predict and
Cultivation Planner now start from 2026-11-12).

Retraining on the full 2015–2026 history was tested with `scripts/retrain_if_better.py --dry-run
--compare-history-start 2024-04-25` (held-out window 2026-09-14 to 2026-11-12, 49,641 forecasts):

| Days ahead | Full-history MAE | 2024-history MAE | Hold-last-price MAE |
|---|---|---|---|
| 1 | 18.36 | 18.36 | 19.19 |
| 2–3 | 24.79 | 25.16 | 26.56 |
| 4–7 | 32.72 | 33.72 | 34.69 |
| 8–14 | 45.60 | 47.67 | 46.33 |
| 15–21 | 61.24 | 63.51 | 58.31 |
| 22–30 | 79.84 | 78.86 | 69.57 |
| Overall | 55.44 | 56.32 | gain −6.1 % (95 % CI −9.1 to −3.3) |

**Decision: rejected, saved model kept.** The longer history helps slightly (55.44 vs 56.32), but in this most recent
window neither configuration beats holding the last price at 15–30 days. Up to 7 days ahead both still beat it
(e.g. next day 4.3 % lower). The published test-period results (2026-05-14 to 2026-11-06) are unchanged, but this
window shows the long-horizon advantage is not stable.

## Fixes made
- `scripts/retrain_if_better.py`: recording the decision crashed on NumPy booleans (`TypeError: Object of type bool is
  not JSON serializable`) — after saving a model in a real run. Added `_json_value` as the `json.dumps` fallback.
- `app/data_refresh.py` (new) + `app/Home.py`: **Check for new official data** button runs `daily_update.py` on click
  only, reporting success / unreadable bulletin / already running / failure honestly; the old data is kept on failure.
- `app/Home.py`: stale-data warning now points to the daily update (not the slow full rebuild); the retraining caption
  says "higher" when the candidate did worse instead of "-6.1 % lower".
- `.gitignore` (earlier today): datasets, models and metrics are now committed.

## Limitations
- No real-time feed exists: HARTI publishes once per working day, typically 1–2 days late.
- The daily update is not scheduled; run the button, `scripts/daily_update.py`, or register
  `scripts/run_daily_update.bat` in Windows Task Scheduler yourself.
- Open-Meteo's archive lags ~3 days, so the newest weather is 2026-11-10.
- Forecasts beyond about 14 days did worse than holding the price in the latest 60 days.
