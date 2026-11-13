# Cultivation Planner: Data Findings and Acquisition Plan

Investigated 2026-11-09.

## What exists, and what the planner uses

| Need | Available? | What the planner uses |
|---|---|---|
| Planting and harvest records (dates, variety, location, conditions) | **No.** None in the project, and no public Sri Lankan dataset was found. | Nothing. No harvest-duration model was trained. |
| Published growing durations | **Yes, for 18 crop entries.** From the Department of Agriculture's HORDI crop and variety pages. | `data/reference/doa_harvest_durations.csv`, shown as a published reference range, with its source |
| Weather by location | Yes, for 8 locations, 2024-04-25 to 2026-11-04 (Open-Meteo archive) | Shown for context only |
| Market prices | Yes, Dambulla, 2024-04-25 to 2026-11-06 (HARTI bulletins) | The existing Ridge direct forecaster, and same-season history |

The existing price model can't estimate harvest duration: it was trained only on prices, weather
and fuel, with no cultivation data. It was not changed.

## How harvest-date prices are handled

The Ridge direct forecaster predicts up to **30 days** after the latest recorded price. Most
crops take 45–137 days from planting, and manioc takes 9–12 months. So, depending on the
planting date:

- **Harvest within 30 days of the latest price:** the planner shows the forecast for each harvest
  day. It also shows a likely range: the middle 80% of the forecaster's own errors on the held-out
  test period (`scripts/forecast_error_bands.py` writes `reports/metrics/forecast_error_bands.json`).
  It suggests a selling day only when that day's low estimate beats the first harvest day's forecast.
- **Harvest further ahead:** no forecast is shown. The planner shows recorded prices from the same
  weeks of earlier years, and labels them as history. With about 2½ years of data, that is at most
  two or three earlier seasons.

## Data needed for a trained harvest-duration model

1. **Farm-level crop records:** planting or transplanting date, first and last harvest date, crop,
   variety, district and agro-ecological zone, season (Yala or Maha), and irrigation. Possible
   sources are Agrarian Development Centres, Department of Agriculture extension officers and
   provincial departments, and the farm surveys HARTI runs for its cost-of-cultivation studies.
2. **Enough of them:** several hundred records per crop, spread over several seasons and zones. A
   few dozen per crop is not enough to beat the published ranges reliably.
3. **Conditions over each crop's life:** daily weather for the field's location (Open-Meteo already
   covers this), and inputs where they're recorded.
4. **Evaluation:** hold out whole seasons and whole districts, never random rows, because records
   from one farm and season are related. Compare each model with the published range, and with
   simple per-crop averages, using MAE and RMSE in days.

## Data needed for harvest-date price forecasts beyond 30 days

1. **Longer price history:** HARTI's archive lists bulletins back to 2015
   (`src/data_collection/harti_live_client.py` reads its year tables). Rebuilding about 11 years
   with `scripts/build_harti_prices.py` would give enough seasons to learn a seasonal pattern.
2. **A model trained for those horizons:** extend the direct forecaster with horizons of 60, 90 and
   120 days, and evaluate it on whole held-out years. Weather is less useful that far ahead, so
   seasonal and price-level features would carry most of the signal.
3. **Honest error ranges:** measure errors at each horizon, as `forecast_error_bands.py` does now,
   and show them with every forecast.

Until then, the planner states plainly when a harvest falls beyond the forecast's reach.
