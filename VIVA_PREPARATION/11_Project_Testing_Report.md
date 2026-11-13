# 11 — Project Testing Report

**Re-executed for this guide on 13 November 2026**, against the running app (started with `start_app.bat`, port 8501,
health `/_stcore/health` = ok). Nothing in the project was changed for testing. Status is **PASS** only if the test
actually ran and met its expected result.

| Evidence | Location |
|---|---|
| UI test runner (headless Microsoft Edge via Chrome DevTools Protocol) | `VIVA_PREPARATION\test_evidence_2026-11-13\run_ui_tests.py` |
| This run's results (13 Nov 2026 13:19) + screenshots + pytest log | `VIVA_PREPARATION\test_evidence_2026-11-13\` (`ui_results.json`, `screenshots\`, `pytest_output.txt`) |
| Permanent copy of earlier identical run (for the report) | `Final_Project_Report\testing_evidence\` |

> සිංහල: මෙහි ඇති සෑම PASS එකක්ම 2026-11-13 දින සැබවින්ම ධාවනය කළ පරීක්ෂණයකින්. ධාවනය නොකළ ඒවා "NOT TESTED" ලෙස ලකුණු කර ඇත.

## 1. Summary

| Test type | Cases | Result |
|---|---|---|
| Automated unit/integration tests (pytest) | 1,060 | **1,060 PASS, 0 FAIL** (13 Nov 2026) |
| Browser UI tests TC01–TC18 | 18 | **18 PASS, 0 FAIL** |
| Independent forecast recomputation (TC19) | 1 | **PASS** (13 Nov 2026, earlier run; values match this run's TC03/TC05) |
| Environment / startup checks | 5 | **5 PASS** |

## 2. Startup and environment

| ID | Test | Result | Evidence |
|---|---|---|---|
| E1 | Entry point is `app/Home.py`; `app.py` does not exist | PASS | `start_app.bat` line 31; file listing |
| E2 | `.venv312` is Python 3.12.10 with all imports (start_app.bat import check) | PASS | start_app.bat lines 18–25 ran without error |
| E3 | Port 8501 free before start, app listening after | PASS | `Get-NetTCPConnection` |
| E4 | Health endpoint returns `ok` | PASS | `http://localhost:8501/_stcore/health` |
| E5 | `.venv` (Python 3.15) is broken — numpy cp312 build | Known issue (documented, not used) | import error reproduced earlier |

## 3. UI test cases (executed 13 Nov 2026 13:19)

| ID | Category | Test | Expected | Actual | Status |
|---|---|---|---|---|---|
| TC01 | Functional/UI | Home page loads | Accuracy cards + "Data in use" | Rendered; "NEXT-DAY ERROR 7.5%"; "Last daily update 2026-11-11 03:45"; 0 exceptions | **PASS** |
| TC02 | Navigation | 4 pages load from sidebar | No exceptions | Predict, Model Performance, Price History, Cultivation Planner: 0 exceptions each | **PASS** |
| TC03 | Prediction | Next-day forecast, Beans | Price + model + weather | Rs 453.74/kg for 2026-11-10; model stated; weather panel shown | **PASS** |
| TC04 | Validation | Past target date flagged | Warning | "2026-11-10 has already passed (3 day(s) ago) …" | **PASS** |
| TC05 | Prediction | 16-day forecast | Price within reach | Rs 507.96/kg for 2026-11-25, 16 days ahead | **PASS** |
| TC06 | Error handling | 46 days ahead | Refused with reason | "Cannot forecast that date: 2026-12-25 is 46 days after the latest known price (2026-11-09) …"; no price shown | **PASS** |
| TC07 | Data | Unforecastable series explained | 5 listed with reasons | Bitter Gourd/Brinjals (Other, Village): too few prices; Cabbage (Nuwara Eliya): latest price 2025-07-12 (checked in screenshot — canvas table) | **PASS** |
| TC08 | Integration | Live HARTI bulletin | Newest bulletin loaded | Bulletin 2026-11-12, 30 prices, fetched 13:17; "not a real-time feed" label | **PASS** |
| TC09 | Data source | Synthetic sample labelled | "Not real market data" | 360 synthetic rows, label + earlier-model notice | **PASS** |
| TC10 | Validation | Valid CSV upload | Accepted | "Preview loaded data" shown | **PASS** |
| TC11 | Validation | CSV missing `price` | Rejected | "The CSV is missing required column(s): ['price'] …" | **PASS** |
| TC12 | Validation | Empty CSV | Rejected | "That file is empty. It needs a header row and at least one row of data." | **PASS** |
| TC13 | Functional | Price history, Kandy | Chart + stats | "Ash Plantain wholesale price at Kandy"; latest/average/lowest/highest | **PASS** |
| TC14 | Data separation | CBSL retail | Separate from wholesale | "Beans: retail price, CBSL"; Narahenpita included | **PASS** |
| TC15 | Planner | Harvest beyond reach | History, no forecast | First harvest 28 Dec 2026; price not available; history table shown | **PASS** |
| TC16 | Planner | Harvest within reach | Forecast price | First harvest 4 Dec 2026; Rs 450/kg | **PASS** |
| TC17 | Evaluation display | Accuracy vs baseline | Baseline shown | "NEXT-DAY ERROR 7.51%"; "Hold last price" series present | **PASS** |
| TC18 | Responsive | Phone width 390 px | No horizontal scroll | clientWidth 390 = scrollWidth 390 | **PASS** |
| TC19 | Correctness | Recompute forecast outside the app from `models/direct_forecaster.joblib` | Same as app | Beans h=1 Rs 453.74, h=16 Rs 507.96, last price 425 — exact match | **PASS** |

**Earlier failures (honest record):** in the first UI run (13 Nov 2026), TC05/TC06 failed because the test script's
typed date was not committed by the date widget, and TC15 used a hard-coded date. These were **test-script** problems;
the script was fixed (key events + Enter; date computed as today + 45 days) and every re-run passed. No application
code was changed.

## 4. Automated test suite (pytest)

Command: `.\.venv312\Scripts\python.exe -m pytest -q -p no:cacheprovider` (no cache files written to the project).
Result (13 Nov 2026, finished 13:28): **`1060 passed, 345 warnings in 516.80s (0:08:36)`** — 0 failed, 0 errors.
The 345 warnings are pandas/library deprecation notices (e.g. `dt.dayofweek`), not test failures. It took 8.6 minutes
because the app and UI tests were running at the same time (an earlier run alone took 162 s).

Test folders: `tests/app`, `tests/data_collection`, `tests/feature_engineering`, `tests/models`, `tests/prediction`,
`tests/preprocessing`, `tests/scripts` (+ `tests/fixtures`).

## 5. Model evaluation (offline test, not re-run today)
From `reports/metrics/direct_forecaster_metrics.json` (produced by `scripts/evaluate_forecaster.py`): next-day MAE
17.47 vs hold 18.26 (gain 4.3 %, CI 2.6–6.0); 22–30 days 65.66 vs 71.20 (gain 7.8 %, CI 4.4–11.2). The **displayed**
values were confirmed today by TC01/TC17.

## 6. NOT TESTED
| Area | Why |
|---|---|
| Real users / usability study | No participants |
| Browsers other than Microsoft Edge (Chromium) | Not available in test setup |
| Load / many simultaneous users | Single-user local app |
| Windows Task Scheduler registration of the daily update | Must be done by the user; not registered |
| `scripts/retrain_if_better.py` | Never run (would replace the model) |
| `scripts/daily_update.py` today | Not run for this guide to avoid changing data before the viva (it was run successfully on 11 Nov 2026, see `data/raw/update_log.jsonl`) |
| Database | Not applicable — no database |
| Security / penetration | Not in scope; no login |

## 7. Known issues
- `.venv` (Python 3.15) is broken — use `.venv312`.
- Streamlit listens on all network interfaces (others on the same network could open it).
- The unforecastable-series table is drawn on a canvas (can't be read by text tools; verified by screenshot).
