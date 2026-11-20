# 10 — Viva Questions and Answers (70 questions)

Format: **Q** → English answer (say this) → සිංහල (to understand it) → ⚠ difficult follow-up and how to answer.
Every number is from the project files (see docs 06, 07, 11).

---

## A. Project overview (1–10)

**1. Explain your project in one minute.**
It forecasts Dambulla wholesale vegetable prices 1–30 days ahead. I parse every HARTI daily PDF bulletin since
June 2015 into a clean dataset (about 77,000 Dambulla prices), add weather, fuel and other markets' prices, train a Ridge
"direct" forecaster (one model per day ahead), and show it in a Streamlit dashboard with a cultivation planner.
සිංහල: HARTI PDF වලින් දත්ත ගෙන, දින 1–30 ඉදිරියට දඹුල්ල තොග මිල අනාවැකි කියන Streamlit පද්ධතියක්.
⚠ *"So what is new?"* → The full PDF-to-dataset pipeline for 11 years and 11 markets, and an honest evaluation against the naive baseline with confidence intervals.

**2. What problem does it solve?**
Farmers and traders see only past prices in PDFs; prices swing heavily. The system gives a data-based estimate of
future prices and links planting dates to expected harvest prices.
සිංහල: අනාගත මිල ගැන දත්ත මත පදනම් වූ ඇස්තමේන්තුවක් ලබා දීම.

**3. Who are the users?** Farmers, wholesale traders, agricultural officers, students/researchers. (No real-user study was done — say so.)
සිංහල: ගොවීන්, වෙළඳුන්. සැබෑ පරිශීලක පරීක්ෂණයක් කර නැත.

**4. What are your objectives, and did you meet them?**
Collect real data ✔, build and compare ML models ✔, beat a naive baseline ✔ (4–8 % lower MAE), build a usable dashboard ✔,
real-time data — partly (latest bulletin on demand, no true real-time feed exists).
⚠ *"Partly?"* → HARTI publishes once a day, 1–2 days late; no public real-time API exists.

**5. Why Dambulla only?**
It is the largest dedicated economic centre with the longest consistent series in HARTI bulletins; the other 10 markets
are used as input features (`COMPARISON_MARKETS`, `forecast_features.py` line 27).
සිංහල: දඹුල්ල — විශාලතම වෙළඳපොළ, දිගම දත්ත මාලාව.

**6. Which vegetables?** 31 series that have enough data (≥ 60 prices, `MIN_ROWS_PER_VEGETABLE`); 5 recorded series are excluded with a reason (TC07).

**7. What is the scope limitation?** Wholesale prices only, Dambulla only, up to 30 days, no supply/arrival quantities.

**8. What is the main finding?** Short-term vegetable prices are close to a random walk: complex models (RF, XGBoost) did not beat "yesterday's price"; a carefully designed Ridge direct model beats it by 4.3 % next day and 7.8 % at 22–30 days.
සිංහල: මිල ඉතා අනපේක්ෂිතයි; සරල baseline එක පරාජය කිරීම අපහසුයි — නමුත් Ridge ආකෘතිය එය කළා.

**9. What was the most difficult part?** Reading the PDFs: layouts changed over the years, merged labels, ranges like "400-450", 224 unreadable bulletins.

**10. What would you do with more time?** Supply/arrival data, retrain on full history, prediction intervals, deployment online, user study, scheduled updates.

## B. Data (11–22)

**11. Where does the data come from?** HARTI daily bulletins (harti.gov.lk/daily-price.php), CBSL Daily Price Report, DCS weekly retail, Open-Meteo weather, CEYPETCO fuel. All public (`data_manifest.json`).

**12. How many records?** On 19 Nov 2026: 77,011 Dambulla prices (2015-06-22 → 2026-11-19); 529,020 rows across 11 markets; 33,328 weather rows; 4,169 fuel rows; CBSL 10,551; DCS 11,439. They grow with every update.

**13. Is any data fake?** Only `app/sample_data/sample_history.csv` (360 rows), labelled "Not real market data" in the app (TC09). Nothing used for training is synthetic.
සිංහල: පුහුණුවට භාවිතා කළ සියල්ල සැබෑ දත්ත.

**14. How did you get data from PDFs?** requests downloads, BeautifulSoup finds links, PyMuPDF extracts words with x/y positions, `_extract_vegetable_table` (`harti_live_client.py` 535) rebuilds rows, `parse_price_range` (138) takes the midpoint.

**15. Why the midpoint of the range?** HARTI publishes min–max; the midpoint is a single representative wholesale price. ⚠ *"Doesn't that lose information?"* → Yes, the spread is lost; using min and max separately is future work.

**16. How did you handle missing values?** Price carried forward at most 7 days (`MAX_CARRY_DAYS`); other markets at most 3; remaining gaps filled with the training median inside the model pipeline (`SimpleImputer`); weather gaps interpolated (`cleaner.py` 311).
සිංහල: දින 7කට වඩා පැරණි මිලක් "අද මිල" ලෙස නොගනී.

**17. Outliers?** Invalid prices (≤ 0, non-numeric) and impossible weather are removed (`cleaner.py` 207, 217). Real price spikes are **kept** — they are real events.
⚠ *"Why keep spikes?"* → Removing them would make the model look better than it is in real life.

**18. How did you check the parsed data is correct?** `scripts/validate_real_datasets.py` crops PDF images to compare with parsed values; plus parser unit tests in `tests/`.

**19. Why CBSL and DCS if you don't train on them?** Cross-checks shown in Price History; they are different measurements (CBSL collects its own; DCS is retail) — mixing them would be wrong.

**20. Why weather?** Rain and temperature in growing regions affect supply weeks later. Weather is taken per production region (`config/crop_production_regions.yaml`).

**21. Why fuel prices?** Transport cost; `diesel_change_30` is used only for next-day.

**22. How is the data updated?** `scripts/daily_update.py` — backups, lock file, imports new bulletins, rewrites manifest. Not scheduled automatically.

## C. Machine learning (23–42)

**23. Which algorithm does the app use?** Ridge Regression, 30 separate models (one per day ahead) — "direct multi-horizon forecasting" (`direct_forecaster.py`).

**24. What is Ridge Regression?** Linear regression plus an L2 penalty α × Σ(coefficient²) that shrinks coefficients, reducing overfitting when features are correlated.
සිංහල: Linear regression + penalty — අධික ගැළපීම (overfitting) අඩු කරයි.
⚠ *"Why α = 1000?"* → Chosen on validation windows before the test period (`scripts/model_experiments.py`); features are standardised, so a large α is reasonable with many correlated features.

**25. What does "direct" mean vs "recursive"?** Direct: a separate model for each horizon trained on the actual price h days later. Recursive: predict 1 day, feed it back. Recursive MAPE at 15–30 days was 36.5 % vs direct 25.8 %.

**26. What does the model predict exactly?** log(target price ÷ today's price). Price = last price × exp(prediction) (`predict_rows`, line 132).
⚠ *"Why log?"* → Scale-free (works for Rs 100 and Rs 1,000 vegetables), symmetric changes, never negative prices.

**27. What features?** Groups: level, momentum, calendar, rain_windows, weather_lag_14, market, fuel, other_markets, plus vegetable one-hot (`FEATURE_GROUPS`, lines 29–45).

**28. Why different features for different horizons?** Validation showed what helps: other markets matter most at 1 day; calendar/season at 22–30 days (`FORECAST_BANDS`, lines 47–52).

**29. What is StandardScaler and why?** Transforms each feature to mean 0, std 1; needed so the Ridge penalty treats all features fairly.

**30. What is SimpleImputer?** Fills missing feature values with the training median.

**31. How did you split the data?** By time, never randomly. Test = 14 May – 6 Nov 2026 (`config.yaml` `forecasting.test_start`). Settings chosen on 4 validation windows before that.
සිංහල: කාලය අනුව බෙදුවා — අනාගත දත්ත පුහුණුවට නොයයි.
⚠ *"Why not k-fold cross-validation?"* → Random folds leak future information into training for time series.

**32. What is data leakage and how did you avoid it?** Using information not available at prediction time. Other markets keyed by bulletin *published* date; weather lags never past today; target shifted only as the label; test period scored once.

**33. What metrics?** MAE, RMSE, MAPE, R², and MAE gain vs holding the last price with a 95 % bootstrap interval.

**34. Your results?** Next-day MAE Rs 17.47 (7.51 %), R² 0.969, 4.3 % better than hold (2.6–6.0); 22–30 days MAE 65.66 (25.1 %), 7.8 % better (4.4–11.2).

**35. R² is 0.969 — isn't that excellent?** It looks high only because today's price already explains most of tomorrow's. The meaningful number is the gain over the baseline: 4.3 %.
සිංහල: R² ඉහළ වුවත් එය නොමඟ යවන සුළුයි; baseline එක සමඟ සංසන්දනය වැදගත්.

**36. What is the naive baseline?** "The price h days later equals today's price." Every model is compared with it.

**37. Is 4.3 % significant?** The 95 % bootstrap interval (2,000 resamples of whole dates) is 2.6–6.0 %, above zero, so the gain is unlikely to be chance.
⚠ *"Why resample dates, not rows?"* → Vegetables on the same date move together; resampling rows would make the interval too narrow (docstring lines 71–74).

**38. Why didn't Random Forest/XGBoost win?** Next-day MAE: RF 18.75, XGB 19.84, LR 18.56 vs yesterday's price 18.17. Trees can't extrapolate beyond seen prices, and the signal is weak and noisy.

**39. Where is the model worse?** RMSE at 8–14 days (73.34 vs 72.82) and 22–30 days (115.01 vs 114.33) is slightly worse than holding — occasional larger misses. Next-day it wins for 21 of 30 vegetables, not all.

**40. Overfitting?** Controlled by Ridge penalty, time split, validation-only tuning, single test scoring. The earlier tree models did worse than the baseline on the test period, which is why they were not used.

**41. Feature importance?** An ablation (removing groups) was done: other markets most important at 1 day, price level at long horizons. No SHAP.

**42. How is the planner's "likely range" computed?** 10th–90th percentile of past forecast errors per horizon (`reports/metrics/forecast_error_bands.json`); not a formal statistical prediction interval.

## D. Implementation and technology (43–55)

**43. Why Python?** Mature data/ML libraries (pandas, scikit-learn, PyMuPDF) and Streamlit for the UI.

**44. Why Streamlit?** Web UI in pure Python, widgets and caching built in, no separate backend.
⚠ *"Limitations?"* → Every interaction re-runs the script (handled by caching); not ideal for many concurrent users.

**45. What is the entry point?** `app/Home.py`, started by `start_app.bat` line 31: `.venv312\Scripts\python.exe -m streamlit run app/Home.py`. There is no `app.py`.

**46. Why Python 3.12 and not the newest?** PyMuPDF 2.1 needs onnxruntime, which has no Python 3.15 build (`pyproject.toml`).

**47. Why no database?** One user, read-mostly data in CSV (~600k rows) fits in memory; transparent and easy to back up. A database is future work for multi-user deployment.
සිංහල: Database අවශ්‍ය නොවූයේ එක් පරිශීලකයෙකු සහ CSV ගොනු පහසුවෙන් මතකයට ගැළපෙන නිසා.

**48. How is the model saved?** joblib → `models/direct_forecaster.joblib` (`persistence.py`).

**49. How does caching work?** `@st.cache_resource` / `@st.cache_data`, keyed on `data_version()` (file modification times, `catalogue.py` 162) — fast but refreshes when data changes.

**50. How is code organised?** `src/` (collection, preprocessing, features, models, prediction), `app/` (UI), `scripts/` (build/update/train), `config/` (YAML), `tests/`.

**51. Configuration?** `config/config.yaml` holds paths, thresholds, hyperparameters, API URLs — no hard-coded values.

**52. Security?** No login, no personal data, no API keys. Runs locally. Uploaded CSVs are validated and not stored permanently. ⚠ Streamlit listens on all interfaces — on a shared network others could open it; for deployment, add authentication.

**53. Error handling?** Clear messages for missing columns, empty files, beyond-30-day dates, missing model files, network failures (TC06, TC11, TC12).

**54. Can a demo retrain damage the real model?** No — sandbox models save to `models/demo` (`config.yaml` `demo_models_dir`).

**55. How do you run it on a new PC?** Python 3.12 → `py -3.12 -m venv .venv312` → `pip install -r requirements.txt` → `start_app.bat` (doc 03).

## E. Testing (56–62)

**56. How did you test?** 1,320 automated pytest cases + 30 browser tests of the running app + an independent recomputation of a forecast + this report (doc 11).

**57. Give examples of unit tests.** PDF parsing of ranges and merged labels, cleaner rules, feature calculations, forecaster horizons, upload validation.

**58. How were UI tests run?** A script drives headless Microsoft Edge (Chrome DevTools Protocol) against the running app, clicks, types, reads results and saves screenshots.

**59. How do you know the displayed price is correct?** TC19: recomputed Beans forecasts outside the app directly from the saved model — exactly the same (Rs 453.74, Rs 507.96).

**60. What was not tested?** Real users, other browsers, load/performance, and a real source outage (simulated in tests only). The retraining script was run once as a dry run (rejected).

**61. Did any test fail?** In the first UI run, two date-input tests failed because of how the test typed into the date box (a test-script issue, not the app); fixed and re-run — all pass.

**62. Validation vs testing?** Validation windows chose the settings; the test period was used once for the final scores.

## F. Real-time, limitations, ethics, future (63–72)

**63. Is it real-time?** No true real-time exists: HARTI publishes one PDF a day, 1–3 days late. A scheduled task checks HARTI, CBSL and DCS every 3 hours and imports anything new; the Predict page shows each source's newest publication, when it was checked, and whether the check failed. ⚠ *"How do I know a prediction used the latest data?"* → Section 2 names the exact HARTI record the forecast starts from, with a link to the bulletin.
සිංහල: සම්පූර්ණ real-time නොවේ; පැය 3කට වරක් නිල මූලාශ්‍ර ස්වයංක්‍රීයව පරීක්ෂා කර නවතම bulletin එක එකතු කරයි.

**64. Why can't you forecast beyond 30 days?** Error grows to ~25 % at 22–30 days; beyond that it would be guessing. The app refuses (TC06).

**65. What if HARTI changes its PDF layout?** Parsing could fail; the daily update logs it and exits with code 2 without damaging data; tests would detect it; the parser must be updated.

**66. Biggest limitations?** No supply data; Dambulla only; modest gain; model not retrained on full history; no real-time feed; no user study; not deployed.

**67. Ethical issues?** Forecasts could influence farmers' decisions — the app shows uncertainty, the baseline, data freshness and refuses beyond 30 days. Public data only; no personal data.

**68. Could traders misuse it?** Possible in principle; the data is already public and the gain is modest. Clear uncertainty display reduces over-reliance.

**69. How would you deploy it?** Streamlit Community Cloud or a VM + Docker, scheduled daily update job, authentication, a database for multi-user.

**70. How would you improve accuracy?** Arrival/supply volumes, festival calendar, longer training history, quantile models for intervals, ensemble of Ridge and gradient boosting.

**71. What did you learn?** Real data is messy; baselines are essential; honest evaluation matters more than a high R².

**72. If you started again, what would you change?** Start from the naive baseline and time-based validation on day one, and design the PDF parser with layout tests from the beginning.

**73. Does the app work in Sinhala and Tamil?** Yes. A switcher at the top right changes every page, button, message, table and chart label; the choice is kept in the address (`?lang=si`, `?lang=ta`). Prices, dates and model inputs are the same in every language. Source names (HARTI, CBSL, DCS) and stored data are not translated.

**74. How were the translations checked?** `tests/app/test_i18n.py` checks that all 864 texts exist in all three files with the same placeholders and in the right script, and that the planner gives identical prices in every language. The wording has not yet been reviewed by a native speaker; that is listed as future work.

---

### Hard "trap" questions — short safe answers
| Trap | Answer |
|---|---|
| "Your R² is 0.97, so the model is 97 % accurate?" | No — R² is variance explained; the error is 7.5 % MAPE, and the gain over the baseline is 4.3 %. |
| "Did you use deep learning?" | No. I compared LR, RF, XGBoost and Ridge; tree models already failed the baseline, so deeper models weren't justified with this data. |
| "Is this live data?" | No. It is the newest published official data, collected every 3 hours (or on request) — not a real-time feed. HARTI usually uploads its bulletins 1 to 3 days late. |
| "Did you invent any data?" | No. Only the labelled sample file is synthetic, and it is never used for the real model. |
| "Is the planner range a confidence interval?" | No — it is the 10th–90th percentile of past forecast errors. |
