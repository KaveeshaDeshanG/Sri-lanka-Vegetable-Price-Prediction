# Data Sources

Sources and known issues for the datasets in `data/raw/`.
- **Full rebuild:** 2026-11-11, from every bulletin since 2015-06-22.
- **Daily updates:** since 2026-11-11 (see "Daily updates" below).
- **Check before the rebuild:** the rebuild was first written to a scratch folder and compared with
  the validated 2024-04-25 to 2026-11-06 datasets. Every one of the 23,809 Dambulla and 184,996
  all-market readings was identical, with no reading lost. The only additions in that period are 112
  Dambulla rows from four recovered bulletins: 2025-06-29 and 2025-07-02 to 2025-07-04. A copy of
  every file from before the rebuild is in
  `D:\ECU\Sri-lanka-Vegetable-Price-Backups\before_update_20261111_025535\`.
- **Measured results** for the 2024-2026 data are in [`DATA_QUALITY_REPORT.md`](DATA_QUALITY_REPORT.md).

**2015-2023 notes:**
- Older bulletins print prices that a second extraction method (PyMuPDF's table reader) reads
  without decimal points, for example "19000" for "190.00". That accounts for the 24,282
  "cross-check mismatches" in `harti_prices_build_report.json`. The stored values come from the word
  positions, which read the decimals correctly; checked against the 2015-06-22 bulletin in
  `tests/data_collection/test_official_bulletin_parsing.py`.
- 146 bulletins have no readable Dambulla reporting date, so those days have no price. Of the 224
  skipped bulletins, 151 are from 2022-2023, when HARTI used a one-page "Min/Max" layout this
  parser doesn't read. The per-bulletin reasons are in `harti_prices_build_report.json`.
- HARTI prints "Dambala (Wing Beans)" and "Thithbatu" rows in 2015-2023 bulletins, but no market ever
  has a price in them.

`.gitignore` excludes `data/raw/*`, so **these datasets are not committed to Git**. Back them
up separately. A downloaded copy of the repository won't include them.

## Datasets

| File | Source | Built by | How values are derived |
|---|---|---|---|
| `harti_prices.csv` | [HARTI daily food commodity bulletins](https://www.harti.gov.lk/daily-price.php), Dambulla column | `scripts/build_harti_prices.py` | Midpoint of each published "min - max" wholesale range (Rs/kg). No reading means no row. |
| `harti_prices_detail.csv` | Same | Same | Each row's low, high, raw cell text, bulletin date and source PDF URL. |
| `harti_prices_build_report.json` | Build output | Same | Bulletins yielding no prices, unknown labels, cross-check counts. |
| `harti_all_markets.csv` | Same bulletins, all 10 markets | `scripts/build_market_prices.py` | Midpoint per market. `published` is the bulletin's date and `reported` the market's own date; Meegoda and Veyangoda are reported 1 to 3 days late. The direct forecaster uses `published`. |
| `fuel_prices.csv` | [CEYPETCO historical prices](https://ceypetco.gov.lk/marketing-sales/historical-prices) | `scripts/build_fuel_prices.py` | Each published revision of Lanka Auto Diesel and Lanka Petrol 92, repeated daily until the next revision. |
| `fuel_prices_source_snapshot.json` | Same | Same | The exact published response the CSV is derived from. |
| `weather_data.csv` | [Open-Meteo Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api), Dambulla (7.8567, 80.6517) | `scripts/build_weather_data.py` | Daily rainfall, mean temperature and mean humidity, unchanged. |
| `weather_data_by_region.csv` | Same, plus the 7 regions in `config/crop_production_regions.yaml` | `scripts/build_historical_weather_by_region.py` | Same variables per region, unchanged. |

| `cbsl_daily_prices.csv` | [CBSL Daily Price Report](https://www.cbsl.gov.lk/en/statistics/economic-indicators/price-report) | `scripts/daily_update.py` | Each report's "Today" column: wholesale (Pettah, Dambulla) and retail (Pettah, Dambulla, Narahenpita), one row per date, item, market and `price_type`. "n.a." stays missing. A cross-check only: never merged into a HARTI series or used for training. |
| `harti_import_log.csv` | Daily update output | `scripts/daily_update.py` | One row per bulletin the daily update tried: URL, the PDF's SHA-256, retrieval time, status and rows added. |
| `cbsl_import_state.json` | Daily update output | Same | CBSL days with no report (weekends, holidays) and reports that couldn't be read. |
| `data_manifest.json` | Build output | `scripts/update_data.py`, `scripts/daily_update.py` | Each dataset's source, price type, rows, SHA-256 and first and latest observation, read from the file. |
| `update_log.jsonl` | Daily update output | `scripts/daily_update.py` | One line per run: what was added, what failed, and where the backup is. |

Full rebuild, from every bulletin since 2015: `scripts/update_data.py`. It backs up every dataset and
model first. Older single-purpose scripts remain: `build_harti_prices.py`, `build_market_prices.py`,
`build_fuel_prices.py`, `build_weather_data.py`, then `build_historical_weather_by_region.py`. Their
default start date is now 2015-06-22, the first bulletin in HARTI's archive. It used to be 2024-04-25,
which is why the datasets once began in 2024.

## Daily updates

There is no live price API in Sri Lanka. HARTI and CBSL publish PDF reports, checked on 2026-11-09.
The project therefore updates **from the official reports as they are published**, checking every
3 hours. They are never described as real-time. HARTI's upload delay varies: on 2026-11-11 03:45 its
newest bulletin was 2026-11-09, on 2026-11-13 20:06 it was 2026-11-12, and on 2026-11-15 11:47 it was
still 2026-11-12. So the newest price is usually 1 to 3 days old.

| Source | Published | What it gives | How it's read |
|---|---|---|---|
| HARTI daily bulletin ([daily-price.php](https://www.harti.gov.lk/daily-price.php)) | Every day, including weekends (297 Dambulla dates in 2026, never more than 4 days apart); uploaded 1 to 3 days late, at irregular times | Wholesale min-max ranges for about 10 markets, each with its own reporting date | The same parser as the full build (`extract_market_rows`, `parse_all_markets`) |
| CBSL Daily Price Report | Working days, same day | Wholesale and retail for Pettah, Dambulla and Narahenpita | `src/data_collection/cbsl_price_report_parser.py`: columns found from the printed headings. Checked on 2021, 2024 and 2026 layouts; the 2018 layout is rejected, not guessed. |

`scripts/daily_update.py` works as follows:

- It reads only bulletins not yet stored (the last 30 days are re-checked) and validates every row:
  - prices are positive and low ≤ high;
  - a reading is dated on or before its bulletin, within HARTI's reporting lag.
- It adds rows without ever changing stored ones, and records each bulletin's URL, SHA-256 and
  retrieval time.
- It extends the weather and fuel files to the newest day their sources have.
- Before any change it copies the files to `<backup_dir>/daily/<time>/`, keeping the newest 14 runs.
- A run is safe to repeat; a second run the same day adds nothing.
- Exit code 2 means a newly published HARTI bulletin couldn't be read, which usually means its
  layout changed. Nothing from it is stored until the parser is checked.

- Each source (HARTI, CBSL, DCS, weather, fuel) is checked separately. The log line's `"sources"`
  entry records, per source: `status` (`ok` or `failed`), `checked_at`, `newest_published` (HARTI's
  newest listed bulletin, CBSL's newest stored report, DCS's newest listed week), `new_records`,
  `problems` (unreadable or undownloadable reports) and `error`. A CBSL request that fails is recorded
  as an error and retried, never mistaken for "not published".
- Every run adds a line to `data/raw/update_log.jsonl`:
  - `"status": "ok"` when every source was checked;
  - `"partial"` when a source couldn't be checked: its stored data is kept and the run exits with code 4;
  - `"failed"`, with the error and exit code 1, when the run itself stopped.

  The datasets stay as they were for anything not checked.
- Once `scripts/build_market_price_ranges.py` has built `harti_all_markets_ranges.csv`, it also adds
  the other markets' published min-max ranges for new bulletins (display only; see below).

**Schedule:** the Windows task `VegetablePriceUpdate` runs `scripts/run_daily_update.bat` every 3 hours
from 07:00. It also runs as soon as possible after a start missed while the PC was off. It was
registered on 2026-11-15, and the batch file has the command to register it again. Output goes to
`logs/daily_update.log`. The Home page's **Check for new official data** button runs the same update
on request.

**What the Predict page shows** (Latest Real Data Prediction; `app/catalogue.py: update_status,
update_notice`, `app/source_records.py`):
- **Official source status:** a card per source (HARTI, CBSL, DCS) from the last completed check, with
  a **Check official sources now** button that runs the same update. A card says **Stale** when the
  newest publication is older than 3 days (HARTI), 4 days (CBSL) or 21 days (DCS; see "DCS freshness"
  below).
- **Latest collected data:** each source's records, kept apart, with every field the source publishes:
  - observation and publication dates, market or location, price type and unit;
  - minimum, maximum and average or midpoint;
  - document link and collection time.

  Fields a source doesn't publish are labelled "not published", never filled. Collection times come
  from `cbsl_daily_prices.csv` (`retrieved_at`), `harti_import_log.csv`, or the saved PDF's download
  time. CBSL and DCS items are matched to HARTI series only through the hand-reviewed
  `data/reference/source_item_map.csv`; ambiguous names (the two brinjal rows) are left unmatched.
- **Prediction:** the starting record (HARTI Dambulla, with its bulletin), the forecasts at 1, 7, 14
  and 30 days, and the accuracy measured for each range.
- **Data freshness and warnings:** a yellow warning when the last check failed, when HARTI couldn't be
  checked, when no check has completed for 24 hours, or when the newest price is more than 3 days old.
  In that last case it says whether HARTI simply hasn't published anything newer.

Only HARTI data (Dambulla and the other markets), Open-Meteo weather and CEYPETCO fuel prices are
model inputs. CBSL and DCS are reference only: CBSL collects its own prices (on 13 Nov 2026 its
Dambulla wholesale Beans was Rs 375, against HARTI's Rs 430 on 12 Nov), and DCS publishes weekly
retail averages for Colombo District.

**DCS freshness (why 21 days).** DCS labels each bulletin "week n of a month", and the app measures
its age from that week's first day (the 1st, 8th, 15th or 22nd). The bulletins' own PDF creation dates
show when each was produced. Only 20 of the 184 saved 2023–2026 PDFs carry one, giving 16 pairs of
consecutive bulletins (2024-06 to 2024-07 and 2026-07 to 2026-11). They show:
- a bulletin appears 3 to 20 days after its week starts (median 8);
- "week 4" starts on the 22nd but runs to the end of the month, so the next week can start up to
  10 days later;
- just before the next bulletin appears, the newest week is 10 to 22 days old (median 15; 20 days
  occurred twice, both times after a "week 4").

The earlier 14-day limit would have marked 10 of the 16 cycles as stale, 9 of them on a normal
schedule, a warning users would learn to ignore. The limit is now **21 days, three weekly cycles**.
Only one cycle crosses it: in October 2026, week 2 appeared 15 days after its week started, when
week 1 was 22 days old. That bulletin was genuinely late, so the warning would have been correct.

The evidence is small (16 cycles), so the limit should be reviewed as more bulletins are collected.
`daily_update.py` saves each new DCS PDF, so its creation date is kept.

**DCS "0.00" values.** DCS prints 0.00 (with a -100% change) for an item it couldn't price that week.
This happened for Butter beans in 2023-03 week 3, 2024-11 week 4 and 2025-12 week 1.
`dcs_weekly_retail_prices.csv` keeps those values exactly as published. The app reads the file through
`app/source_records.py: load_dcs_weekly`, which marks a zero or negative price or range as unavailable;
every positive price is unchanged. As a result:
- Price History doesn't plot or count those weeks, breaks the line there, and lists them in a note;
- the Predict page shows them as "not published".

**Other markets' min-max ranges.** `harti_all_markets.csv` stores each range's midpoint only.
`scripts/build_market_price_ranges.py` reads the saved bulletin PDFs again and writes
`data/raw/harti_all_markets_ranges.csv` (low and high), keeping a range only when its midpoint
equals the stored price. Its counts, including rows left out, are in
`harti_all_markets_ranges_report.json`. Display only: the forecaster never reads it.

Retraining is separate: `scripts/retrain_if_better.py` fits a candidate on everything except the most
recent 60 days of prices. It keeps the candidate only if, on those 60 unseen days, its error is clearly
lower than holding the last price, overall and in every band of days ahead. The current model and
metrics are backed up before it is replaced. Every decision is written to
`reports/metrics/retrain_decision.json`.

A running app picks up new data and a retrained model on the next page interaction: its caches are
keyed on the files' modification times (`app/catalogue.py: data_version`).

**Terms of use:** none of these sites publishes a data licence. HARTI, DOA and CBSL all state "All
rights reserved", and CBSL's `robots.txt` asks for 10 seconds between requests, which the updater
follows. Academic use with attribution and polite request rates is the working assumption. Ask HARTI
and CBSL in writing before any commercial use.

**Sources checked and not used:**
- HARTI's weekly and monthly bulletins: averages with corrupted "######" cells, and the daily data
  already covers them.
- HARTI's Power BI dashboards: no export or documented API.
- HARTI's "6666" Agri Price service: a telephone voice menu.
- DOA (no price data published).
- Dambulla Dedicated Economic Centre: no price website; its prices reach the project through HARTI
  and CBSL.
- WFP food prices on HDX: monthly, ending 2025-09-15.

`scripts/validate_real_datasets.py` validates these datasets, including `harti_all_markets.csv` (section 8
of `DATA_QUALITY_REPORT.md`), which it compares with the original HARTI PDFs cell by cell.

The 870 source PDFs are kept outside the project, because they total 635 MB, in
`D:\ECU\Sri-lanka-Vegetable-Price-Backups\harti_pdf_archive\` with a `SHA256_MANIFEST.txt`. Pass that folder
to the validator with `--cache-dir`. The four later-located bulletins are in the separate
`harti_pdf_recovered_candidates\` folder next to it, so the archive matches the datasets exactly.

## What is real, and what is derived

No value is random, estimated or interpolated. Two values are calculated from published data:
- **Vegetable price:** the midpoint of the published range. HARTI publishes a range, not one price.
- **Daily fuel price:** the latest published revision. Fuel prices stay fixed between revisions.

## Known issues

These values are kept exactly as published.

**Unusual published vegetable values** (each confirmed to match its PDF):

| Date | Vegetable | Published | Why it stands out |
|---|---|---|---|
| 2025-11-30 | Carrot | 2200 - 2500 | Other markets that day: 500 to 1400. Follows Cyclone Ditwah, when prices spiked generally, but this exact value has no independent confirmation. |
| 2025-11-02 | Beetroot | 180 - 220 | Other markets that day: 60 to 130. Dambulla next day: 80 - 120. |
| 2025-01-01 | Ladies Fingers | 30 - 40 | Other markets that day: 70 to 180. Dambulla the day before: 100 - 120. |
| 2024-07-07 | Manioc | 10 - 80 | Dambulla neighbouring days: 85 - 100 and 80 - 90. Possibly a typing error in the bulletin. |
| 2024-04-25 to 2024-05-01 | Big Onion (Imported) | about 190 - 680 | Very wide range on 7 consecutive days, and other markets show similar ranges. The midpoint may represent the market poorly. |

**Days with no Dambulla price (63 of 926):**
- 50 days have no bulletin listed.
- 7 bulletins exist, but every Dambulla cell is "-" while other markets have prices.
- 6 listings have a link that points to a folder instead of a PDF: 2025-06-29, 2025-07-02,
  2025-07-03, 2025-07-04, 2026-01-14 and 2026-05-02. None of them is in the datasets or the model.
- The first four were downloaded on 2026-11-07 from `2025/December/daily_DD-MM-YYYY.pdf`. Each is a valid
  PDF whose own date matches, and each gives 28 Dambulla prices. They're held outside the project, and
  deliberately not added: that would change the validated datasets and require retraining and
  re-evaluating the model. No alternative location has been searched for the last two.

**Sparse vegetables:**
- Cabbage (Nuwara Eliya) has 103 rows, and its last reading is 2025-07-12.
- Drumstick (181), Big Onion (Local) (292), Potato (Welimada) (419) and Potato (Nuwara Eliya)
  (423) are also sparse.

**Other notes:**
- "Eggplant" and "Brinjals" are separate rows in HARTI's bulletins. `DataTransformer` renames
  Eggplant to "Brinjal" during preprocessing, so the model sees "Brinjal" and "Brinjals".
- A CEYPETCO revision date can be up to a day later than the midnight it took effect.
- 190 PDF URLs sit in the wrong month folder on HARTI's server (for example, mid-2025
  bulletins under `2025/December/`). Every filename date matches its bulletin date.
- The Open-Meteo archive runs about 2 days behind, so `weather_data.csv` ends 2026-11-04.
  The model uses 14-day-old weather, so this affects no feature.
- The PDF download cache is not kept in the project. Each row's URL allows it to be fetched again.
