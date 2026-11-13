# Reference data

## `doa_harvest_durations.csv`

Official harvest timing published by the Sri Lanka Department of Agriculture's Horticultural Crops
Research and Development Institute (HORDI), copied from doa.gov.lk on **2026-11-09**. Each row
keeps the source page and the sentence the numbers come from, typing errors included.

**This is agronomic guidance, not measured data and not a model output.** The Cultivation
Planner shows it as a reference range and says so. No farm-level planting and harvest records
were available, so no harvest-duration model was trained (see `reports/CULTIVATION_DATA_PLAN.md`).

| Column | Meaning |
|---|---|
| `crop` | The crop as the source names it, with a type where the source distinguishes one |
| `price_labels` | The matching vegetables in `harti_prices.csv`, separated by `;` |
| `min_days`, `max_days` | The published range in days |
| `counted_from` | What the days are counted from: `planting`, `transplanting`, `nursery sowing`, or `not stated` |
| `stage` | What the days lead to, such as `harvest` or `first harvest` |
| `scope` | `crop guidance` (a crop page), or a variety page covering one or several released varieties |
| `source_url`, `source_text` | Where the figure was published, and the sentence it comes from |

Unit conversions: weeks × 7, and months × 30.44, rounded to whole days (10–12 weeks → 70–84
days; about 4½ months → 137; 3–4 months → 91–122; 9–12 months → 274–365).

**No published figure was found** for Tomato, Cucumber, Bitter Gourd, Green Chillies and Big Onion.
The HORDI crop pages for tomato, cucumber and bitter gourd give a "Harvesting" heading without a
day count. Ash Plantain, Drumstick and Lime are perennial crops, so a single planting-to-harvest
duration doesn't apply. These crops are therefore not offered in the planner.
