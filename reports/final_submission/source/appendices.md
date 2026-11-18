# Appendix A: Detailed Test Cases {#appendix}

## A.1 Test environment

Table: Test environment {#tab:testenv}

| Item | Value |
|---|---|
| Computer | AMD Ryzen 5 5500, 16 GB RAM, Windows 11 Pro |
| Python | 3.12.10 in .venv312; packages as pinned in requirements.txt |
| Automated tests | pytest, run on 17 November 2026: python -m pytest (JUnit XML kept in test_evidence/) |
| Browser tests | Script driving headless Microsoft Edge against http://localhost:8501, run on {{BT_DATE}} (results in test_evidence/browser_tests.json) |
| Data | Datasets updated to 17 November 2026; saved forecaster trained through 6 November 2026 |

Status is Pass, Fail or Blocked. An automated test case passes only if its pytest test passed in the full run; a test with several parameter cases passes only if all of them passed. A browser test passes only if the page showed the expected result.

## A.2 Automated test cases

{{AUTOMATED_TABLE}}

## A.3 Browser test cases

{{BROWSER_TABLE}}

## A.4 Test totals

{{TOTALS_TABLE}}

# Appendix B: Additional Screenshots

![Home dashboard in Sinhala](assets/screenshots/05_home_si.png){#fig:homesi width=6.0}

![Home dashboard in Tamil](assets/screenshots/06_home_ta.png){#fig:hometa width=6.0}

![Price History: DCS weekly retail prices for Colombo District](assets/screenshots/14_price_history_dcs.png){#fig:dcs width=6.0}

![Cultivation Planner: a crop without an official price series (cauliflower) receives harvest dates but no price](assets/screenshots/19_planner_no_price.png){#fig:noprice width=6.0}

![Model Performance in Tamil](assets/screenshots/22_performance_ta.png){#fig:perfta width=6.0}

# Appendix C: Configuration and Setup Guide

The commands below are run from the project folder in PowerShell.

```
py -3.12 -m venv .venv312
.venv312\Scripts\python.exe -m pip install -r requirements.txt
.venv312\Scripts\python.exe -m pip install -e ".[dev]"
.venv312\Scripts\python.exe -m streamlit run app/Home.py      # or double-click start_app.bat
.venv312\Scripts\python.exe scripts\daily_update.py             # add newly published data now
.venv312\Scripts\python.exe scripts\update_data.py --workers 1  # full rebuild since 2015 (with backup)
.venv312\Scripts\python.exe scripts\train_forecaster.py         # evaluate, then train on all data
.venv312\Scripts\python.exe scripts\retrain_if_better.py        # retrain only if a hold-out check passes
.venv312\Scripts\python.exe -m pytest                           # run all tests
```

The automatic update is registered once in Windows Task Scheduler (from a Command Prompt in the project folder):

```
schtasks /Create /TN "VegetablePriceUpdate" /TR "conhost.exe --headless \"%CD%\scripts\run_daily_update.bat\"" /SC HOURLY /MO 3 /ST 07:00 /F
schtasks /Query /TN "VegetablePriceUpdate" /V /FO LIST
```

The main settings are in config/config.yaml. The excerpt below shows the HARTI source, the fixed test start date and the folders used by the updates.

```
harti:
  archive_url: "https://www.harti.gov.lk/daily-price.php"
  request_timeout_seconds: 15
  target_market: "Dambulla"
forecasting:
  test_start: "2026-05-14"
data_update:
  harti_pdf_cache_dir: "../../Sri-lanka-Vegetable-Price-Backups/harti_pdf_archive_full"
  backup_dir: "../../Sri-lanka-Vegetable-Price-Backups"
```

# Appendix D: Selected Technical Details

**Training one model per horizon** (excerpt from src/models/direct_forecaster.py):

```
for horizon in horizons:
    rows = usable_rows(add_vegetable_columns(add_horizon_features(origin_rows, horizon), vegetables))
    if target_before is not None:
        rows = rows[rows["target_date"] < target_before]
    band = band_for(horizon)
    columns = _feature_columns(band, vegetables)
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=band.ridge_alpha))
    model.fit(rows[columns], np.log(rows["target_price"] / rows["last_price"]))
    forecaster.models[horizon] = (model, columns)
```

**Translating a text** (app/i18n.py): the English text is the key, and values are filled in after translation, so numbers are never changed.

```
st.caption(t("Forecast date **{target}**: {days} day(s) after the latest actual price ({latest}).",
             target=target.date(), days=horizon, latest=latest_actual_date.date()))
```

**Update exit codes** (scripts/daily_update.py): 0 all sources checked; 1 the run failed; 2 a newly published HARTI bulletin could not be read; 3 another update was already running; 4 at least one source could not be checked; 130 interrupted.

**Sample stored records** (harti_prices_detail.csv) keep the published range, the raw cell text and the source bulletin, so every price can be traced to its PDF.

# Appendix E: Use of AI Assistance

[Edit this statement to match your own use and your university's AI-use policy.] An AI assistant (Claude, developed by Anthropic) was used during this project to help write and review parts of the source code, audit the data and model evaluation, run and document tests, and draft and format this report. All results, figures and statements in the report were produced from or checked against the project's own source code, data, saved metrics and test runs.
