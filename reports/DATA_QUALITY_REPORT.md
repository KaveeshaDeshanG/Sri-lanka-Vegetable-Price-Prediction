# Data Quality Report: Rebuilt Real Datasets

Generated 2026-11-08 20:44 by `scripts/validate_real_datasets.py`. See `reports/DATA_SOURCES.md` for sources.

## 1. Coverage

| File | Rows | First | Last | Distinct dates |
|---|---|---|---|---|
| harti_prices.csv | 23809 | 2024-04-25 | 2026-11-06 | 863 |
| fuel_prices.csv | 926 | 2024-04-25 | 2026-11-06 | 926 |
| weather_data.csv | 924 | 2024-04-25 | 2026-11-04 | 924 |
| weather_data_by_region.csv | 7392 | 2024-04-25 | 2026-11-04 | 924 |

HARTI bulletins listed for the window: **876**. Yielding no prices: **13** (see the reason for each below).
Calendar days in the price span: 926. Days with no Dambulla price at all: **63** (6.8%).
Bulletins whose Dambulla date trailed the bulletin date (allowed up to 7 days): 0.

Skipped bulletins:

| bulletin | url | reason |
|---|---|---|
| 2024-05-02 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/May/daily_02-05-2024.pdf | table found, but no usable price in it |
| 2024-09-21 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/September/daily_21-09-2024.pdf | table found, but no usable price in it |
| 2024-11-14 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/November/daily_14-11-2024.pdf | table found, but no usable price in it |
| 2025-02-19 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/February/daily_19-02-2025.pdf | table found, but no usable price in it |
| 2025-05-06 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/May/daily_06-05-2025.pdf | table found, but no usable price in it |
| 2025-06-11 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/June/daily_11-06-2025.pdf | table found, but no usable price in it |
| 2025-06-29 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2025-07-02 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2025-07-03 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2025-07-04 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2026-01-14 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2026-05-02 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ | download failed: 403 Client Error: Forbidden for url: https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/ |
| 2026-06-30 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2026/June/Vegetables Wholesale Prices (2026.06.30).pdf | table found, but no usable price in it |

Days with no Dambulla price: 2024-05-02, 2024-05-23, 2024-05-24, 2024-06-21, 2024-07-20, 2024-08-19, 2024-09-17, 2024-09-21, 2024-09-22, 2024-10-17, 2024-10-30, 2024-10-31, 2024-11-14, 2024-11-15, 2024-12-14, 2025-01-13, 2025-02-12, 2025-02-19, 2025-03-13, 2025-04-13, 2025-04-14, 2025-04-15, 2025-04-30, 2025-05-06, 2025-05-12, 2025-05-13, 2025-06-07, 2025-06-10, 2025-06-11, 2025-06-29, 2025-07-02, 2025-07-03, 2025-07-04, 2025-07-10, 2025-07-13, 2025-07-26, 2025-07-27, 2025-08-08, 2025-09-07, 2025-10-06, 2025-11-05, 2025-11-28, 2025-11-29, 2025-12-04, 2026-01-03, 2026-01-14, 2026-02-01, 2026-03-02, 2026-04-01, 2026-04-13, 2026-04-14, 2026-04-15, 2026-05-01, 2026-05-02, 2026-05-30, 2026-05-31, 2026-06-29, 2026-06-30, 2026-07-29, 2026-08-27, 2026-09-26, 2026-10-11, 2026-10-25.

## 2. Vegetable names and coverage (Rs/kg midpoint)

31 distinct labels.

| vegetable | rows | minimum | median | maximum | coverage % | longest gap (days) | first | last |
|---|---|---|---|---|---|---|---|---|
| Cabbage (Nuwara Eliya) | 103 | 75.0 | 195.0 | 490.0 | 11.1 | 183 | 2024-06-22 | 2025-07-12 |
| Drumstick | 181 | 47.5 | 130.0 | 515.0 | 19.5 | 316 | 2024-08-05 | 2026-11-06 |
| Big Onion (Local) | 292 | 95.0 | 230.0 | 400.0 | 31.5 | 266 | 2024-08-10 | 2026-11-06 |
| Potato (Welimada) | 419 | 160.0 | 210.0 | 345.0 | 45.2 | 70 | 2024-04-25 | 2026-11-06 |
| Potato (Nuwara Eliya) | 423 | 210.0 | 275.0 | 410.0 | 45.7 | 97 | 2024-05-05 | 2026-11-06 |
| Eggplant | 846 | 70.0 | 235.0 | 640.0 | 91.4 | 7 | 2024-04-25 | 2026-11-06 |
| Beetroot (Nuwara Eliya) | 848 | 65.0 | 225.0 | 725.0 | 91.6 | 9 | 2024-04-25 | 2026-11-06 |
| Manioc | 859 | 45.0 | 85.0 | 150.0 | 92.8 | 6 | 2024-04-25 | 2026-11-06 |
| Knol Khol | 860 | 37.5 | 150.0 | 425.0 | 92.9 | 4 | 2024-04-25 | 2026-11-06 |
| Beetroot | 860 | 42.5 | 190.0 | 640.0 | 92.9 | 5 | 2024-04-25 | 2026-11-06 |
| Radish | 862 | 25.0 | 65.0 | 225.0 | 93.1 | 4 | 2024-04-25 | 2026-11-06 |
| Potato (Imported) | 862 | 107.5 | 160.0 | 355.0 | 93.1 | 3 | 2024-04-25 | 2026-11-06 |
| Capsicum | 862 | 100.0 | 450.0 | 1650.0 | 93.1 | 3 | 2024-04-25 | 2026-11-06 |
| Lime | 862 | 40.0 | 300.0 | 2050.0 | 93.1 | 3 | 2024-04-25 | 2026-11-06 |
| Bitter Gourd | 862 | 95.0 | 315.0 | 740.0 | 93.1 | 3 | 2024-04-25 | 2026-11-06 |
| Brinjals | 863 | 55.0 | 240.0 | 650.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Green Chillies | 863 | 55.0 | 300.0 | 1600.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Cucumber | 863 | 17.5 | 72.5 | 210.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Cabbage (Kandy) | 863 | 47.5 | 150.0 | 540.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Carrot | 863 | 65.0 | 230.0 | 2350.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Ash Plantain | 863 | 40.0 | 115.0 | 225.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Beans | 863 | 130.0 | 375.0 | 1100.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Big Onion (Imported) | 863 | 72.5 | 162.5 | 435.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Leeks | 863 | 75.0 | 190.0 | 950.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Long Beans | 863 | 50.0 | 175.0 | 550.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Luffa | 863 | 50.0 | 195.0 | 540.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Ladies Fingers | 863 | 35.0 | 120.0 | 270.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Pumpkin | 863 | 22.5 | 97.5 | 240.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Snake Gourd | 863 | 50.0 | 155.0 | 425.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Sweet Potato | 863 | 45.0 | 110.0 | 195.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |
| Tomato | 863 | 30.0 | 190.0 | 750.0 | 93.2 | 3 | 2024-04-25 | 2026-11-06 |

Row labels in the bulletins not mapped to a tracked vegetable (fruit, section headings, or a label to review):

| label | bulletins |
|---|---|
| Low country Vegetable | 870 |
| Banana | 870 |
| Ambul(Rs/Kg) | 870 |
| Kolikuttu | 870 |
| Seeni | 870 |
| Anamalu (Rs/Fruits) | 870 |
| Papaya (Rs/Kg) | 870 |
| Other Fruits (Rs/Fruit) | 870 |
| Pineapple - Large | 870 |
| - Medium | 870 |
| - Small | 870 |
| Mango - Betti | 870 |
| Woodapple | 870 |
| Avocado | 870 |
| Orange | 870 |
| Up Country Vegetable | 790 |
| - Karathakolo | 718 |
| Passion Fruits(Rs/Fruit | 551 |
| Passion Fruits(Rs/Fruits | 167 |
| Passion Fruits(Rs/Fruits) - | 128 |
| - Karathakolomban- | 127 |
| Passion Fruits(Rs/Fru | 23 |
| - Karathako | 23 |
| - Karathakolomban - | 1 |
| Passion Fruits(Rs/Fruits) | 1 |
| - Karathakolomban | 1 |

## 3. Record integrity

| Check | Result |
|---|---|
| Duplicate (date, vegetable, market) in harti_prices.csv | 0 |
| Exact duplicate rows | 0 |
| Missing values in harti_prices.csv | 0 |
| Non-positive prices | 0 |
| Low above high | 0 |
| Price differs from (low+high)/2 | 0 |
| Market values | Dambulla |
| Build cross-check mismatches (word vs table digits) | 0 |
| High/low range at least 3x (needs review) | 8 |
| Day-over-day price change at least 3x (needs review) | 5 |
| Consecutive days with every vegetable's cell identical | 0 |

Identical consecutive days: none.

### Day-over-day changes of at least 3x

| date | vegetable | raw_cell | previous_price | price | source_pdf |
|---|---|---|---|---|---|
| 2025-11-02 | Beetroot | 180 - 220 | 60.0 | 200.0 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/December/daily_02-11-2025.pdf |
| 2025-12-01 | Cabbage (Kandy) | 80 - 120 | 415.0 | 100.0 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/December/daily_01-12-2025.pdf |
| 2025-12-01 | Carrot | 450 - 600 | 2350.0 | 525.0 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/December/daily_01-12-2025.pdf |
| 2025-12-01 | Green Chillies | 200 - 300 | 850.0 | 250.0 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/December/daily_01-12-2025.pdf |
| 2025-01-01 | Ladies Fingers | 30 - 40 | 110.0 | 35.0 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2025/January/daily_01-01-2025.pdf |

### High/low ranges of at least 3x

| date | vegetable | raw_cell | source_pdf |
|---|---|---|---|
| 2024-04-25 | Big Onion (Imported) | 190 - 680 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_25-04-2024.pdf |
| 2024-04-26 | Big Onion (Imported) | 190 - 660 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_26-04-2024.pdf |
| 2024-04-27 | Big Onion (Imported) | 185 - 660 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_27-04-2024.pdf |
| 2024-04-28 | Big Onion (Imported) | 190 - 650 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_28-04-2024.pdf |
| 2024-04-29 | Big Onion (Imported) | 180 - 660 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_29-04-2024.pdf |
| 2024-04-30 | Big Onion (Imported) | 180 - 650 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/April/daily_30-04-2024.pdf |
| 2024-05-01 | Big Onion (Imported) | 190 - 650 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/May/daily_01-05-2024.pdf |
| 2024-07-07 | Manioc | 10 - 80 | https://www.harti.gov.lk/assets/pdf/food_price/daily/eng/2024/July/daily_07-07-2024.pdf |

Rows with a low-above-high cell:

_None._

## 4. Fuel and weather

| Check | Result |
|---|---|
| Fuel: missing calendar days | 0 |
| Fuel: duplicate dates | 0 |
| Fuel: missing values | 0 |
| Fuel: non-positive prices | 0 |
| Fuel: diesel range | 274 to 407 |
| Fuel: petrol range | 292 to 434 |
| Weather: missing calendar days | 0 |
| Weather: duplicate dates | 0 |
| Weather: missing values | 0 |
| Weather: rainfall range (mm) | 0 to 200.2 |
| Weather: temperature range (C) | 21.9 to 30.2 |
| Weather: humidity range (%) | 63 to 98 |

Regional weather series:

| region | rows | first | last | dupes |
|---|---|---|---|---|
| Anuradhapura | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Badulla | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Dambulla | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Jaffna | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Kandy | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Monaragala | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Nuwara Eliya | 924 | 2024-04-25 | 2026-11-04 | 0 |
| Welimada | 924 | 2024-04-25 | 2026-11-04 | 0 |

## 5. Date alignment with the price dataset

| Check | Count | Dates |
|---|---|---|
| Price dates with no same-day fuel row | 0 |  |
| Price dates with no same-day weather row | 2 | 2026-11-05, 2026-11-06 |
| Price dates whose day-before fuel row is missing | 0 |  |
| Price dates whose 14-day-earlier weather row is missing | 0 |  |

## 6. Dates with more than one published PDF

The build keeps the first PDF for such a date.

_None._

## 7. Cross-check against the original PDFs

Random sample of 42 records (seed 20261106, 14 per calendar year). Position-based re-read matched: **42 of 42**. Image crops: `reports/figures/pdf_crosscheck_sheet_*.png`.

| id | date | vegetable | csv_price | csv_raw_cell | position_reread | position_match |
|---|---|---|---|---|---|---|
| 1 | 2024-05-16 | Leeks | 185.0 | 170 - 200 | 170 - 200 | True |
| 2 | 2024-05-30 | Ash Plantain | 135.0 | 120 - 150 | 120 - 150 | True |
| 3 | 2024-06-03 | Big Onion (Imported) | 190.0 | 170 - 210 | 170 - 210 | True |
| 4 | 2024-06-08 | Manioc | 100.0 | 90 - 110 | 90 - 110 | True |
| 5 | 2024-06-11 | Sweet Potato | 140.0 | 130 - 150 | 130 - 150 | True |
| 6 | 2024-09-01 | Manioc | 75.0 | 70 - 80 | 70 - 80 | True |
| 7 | 2024-09-09 | Snake Gourd | 140.0 | 130 - 150 | 130 - 150 | True |
| 8 | 2024-09-23 | Capsicum | 425.0 | 400 - 450 | 400 - 450 | True |
| 9 | 2024-09-27 | Potato (Welimada) | 215.0 | 210 - 220 | 210 - 220 | True |
| 10 | 2024-10-22 | Potato (Imported) | 155.0 | 150 - 160 | 150 - 160 | True |
| 11 | 2024-10-23 | Radish | 42.5 | 35 - 50 | 35 - 50 | True |
| 12 | 2024-11-01 | Leeks | 135.0 | 120 - 150 | 120 - 150 | True |
| 13 | 2024-12-08 | Radish | 85.0 | 80 - 90 | 80 - 90 | True |
| 14 | 2024-12-19 | Long Beans | 275.0 | 250 - 300 | 250 - 300 | True |
| 15 | 2025-01-02 | Capsicum | 500.0 | 450 - 550 | 450 - 550 | True |
| 16 | 2025-01-30 | Potato (Nuwara Eliya) | 355.0 | 350 - 360 | 350 - 360 | True |
| 17 | 2025-03-04 | Cabbage (Kandy) | 275.0 | 250 - 300 | 250 - 300 | True |
| 18 | 2025-04-23 | Potato (Nuwara Eliya) | 210.0 | 200 - 220 | 200 - 220 | True |
| 19 | 2025-04-27 | Cabbage (Kandy) | 255.0 | 230 - 280 | 230 - 280 | True |
| 20 | 2025-04-28 | Leeks | 165.0 | 150 - 180 | 150 - 180 | True |
| 21 | 2025-05-11 | Leeks | 190.0 | 180 - 200 | 180 - 200 | True |
| 22 | 2025-05-24 | Manioc | 70.0 | 60 - 80 | 60 - 80 | True |
| 23 | 2025-06-16 | Long Beans | 100.0 | 80 - 120 | 80 - 120 | True |
| 24 | 2025-08-03 | Potato (Nuwara Eliya) | 305.0 | 300 - 310 | 300 - 310 | True |
| 25 | 2025-08-06 | Potato (Imported) | 152.5 | 130 - 175 | 130 - 175 | True |
| 26 | 2025-08-14 | Beetroot | 60.0 | 50 - 70 | 50 - 70 | True |
| 27 | 2025-10-18 | Snake Gourd | 190.0 | 180 - 200 | 180 - 200 | True |
| 28 | 2025-11-15 | Green Chillies | 225.0 | 200 - 250 | 200 - 250 | True |
| 29 | 2026-01-11 | Radish | 90.0 | 80 - 100 | 80 - 100 | True |
| 30 | 2026-01-12 | Cabbage (Kandy) | 175.0 | 150 - 200 | 150 - 200 | True |
| 31 | 2026-01-26 | Tomato | 425.0 | 400 - 450 | 400 - 450 | True |
| 32 | 2026-03-18 | Luffa | 80.0 | 70 - 90 | 70 - 90 | True |
| 33 | 2026-04-28 | Tomato | 115.0 | 100 - 130 | 100 - 130 | True |
| 34 | 2026-04-29 | Tomato | 115.0 | 100 - 130 | 100 - 130 | True |
| 35 | 2026-06-11 | Knol Khol | 265.0 | 250 - 280 | 250 - 280 | True |
| 36 | 2026-07-28 | Leeks | 180.0 | 160 - 200 | 160 - 200 | True |
| 37 | 2026-08-24 | Beetroot (Nuwara Eliya) | 210.0 | 200 - 220 | 200 - 220 | True |
| 38 | 2026-08-24 | Snake Gourd | 140.0 | 120 - 160 | 120 - 160 | True |
| 39 | 2026-09-05 | Radish | 85.0 | 70 - 100 | 70 - 100 | True |
| 40 | 2026-09-14 | Knol Khol | 140.0 | 130 - 150 | 130 - 150 | True |
| 41 | 2026-09-24 | Green Chillies | 290.0 | 250 - 330 | 250 - 330 | True |
| 42 | 2026-10-02 | Big Onion (Imported) | 272.5 | 265 - 280 | 265 - 280 | True |

## 8. All-markets dataset (harti_all_markets.csv)

| Check | Result |
|---|---|
| Rows | 184996 |
| Duplicate (published, market, vegetable) | 0 |
| Repeated readings, same (reported, market, vegetable) with a later bulletin | 53 |
| Missing values | 0 |
| Non-positive prices | 0 |
| Market names outside the expected 10 | 0 |
| Vegetable names outside the project's vocabulary | 0 |
| Reported date after its bulletin | 0 |
| Reported date more than 7 days before its bulletin | 0 |
| Published dates with no cached source PDF | 0 |
| Cached PDFs contributing no reading | 0 |
| Published dates outside fuel_prices.csv | 0 |
| Published dates outside weather_data.csv | 2 |
| Dambulla rows missing from harti_prices.csv | 0 |
| harti_prices.csv rows missing from Dambulla rows | 0 |
| Dambulla prices that differ from harti_prices.csv | 0 |
| Prices at least 5x from their market and vegetable median (needs review) | 275 |
| Day-over-day change of at least 3x (needs review) | 26 |

### Coverage by market

`lag` is the days between a market's reported date and the bulletin that published it.

| market | rows | bulletins | first | last | vegetables | lag_min | lag_max | median_price | bulletin coverage % |
|---|---|---|---|---|---|---|---|---|---|
| Bandarawela | 17800 | 711 | 2024-05-07 | 2026-11-06 | 31 | 0 | 0 | 190.0 | 81.7 |
| Dambulla | 23809 | 863 | 2024-04-25 | 2026-11-06 | 31 | 0 | 0 | 170.0 | 99.2 |
| Kandy | 25148 | 864 | 2024-04-25 | 2026-11-06 | 31 | 0 | 0 | 190.0 | 99.3 |
| Keppetipola | 13074 | 752 | 2024-04-25 | 2026-11-06 | 21 | 0 | 0 | 172.5 | 86.4 |
| Meegoda | 21927 | 807 | 2024-04-25 | 2026-11-06 | 31 | 1 | 3 | 205.0 | 92.8 |
| Norochchole | 10662 | 752 | 2024-04-25 | 2026-10-07 | 23 | 0 | 0 | 135.0 | 86.4 |
| Nuwaraeliya | 8219 | 822 | 2024-04-25 | 2026-11-06 | 11 | 0 | 0 | 225.0 | 94.5 |
| Peliyagoda | 23806 | 866 | 2024-04-25 | 2026-11-06 | 31 | 0 | 0 | 190.0 | 99.5 |
| Thambuththegama | 21162 | 849 | 2024-04-25 | 2026-11-06 | 31 | 0 | 0 | 170.0 | 97.6 |
| Veyangoda | 19389 | 719 | 2024-04-25 | 2026-11-06 | 31 | 1 | 3 | 200.0 | 82.6 |

### Readings per vegetable and market

| vegetable | Bandarawela | Dambulla | Kandy | Keppetipola | Meegoda | Norochchole | Nuwaraeliya | Peliyagoda | Thambuththegama | Veyangoda |
|---|---|---|---|---|---|---|---|---|---|---|
| Ash Plantain | 634 | 863 | 860 | 0 | 807 | 362 | 0 | 866 | 848 | 717 |
| Beans | 711 | 863 | 864 | 752 | 806 | 0 | 822 | 866 | 849 | 719 |
| Beetroot | 23 | 860 | 838 | 4 | 788 | 750 | 0 | 839 | 841 | 690 |
| Beetroot (Nuwara Eliya) | 623 | 848 | 847 | 748 | 7 | 0 | 822 | 629 | 10 | 38 |
| Big Onion (Imported) | 709 | 863 | 860 | 742 | 801 | 0 | 0 | 606 | 848 | 714 |
| Big Onion (Local) | 173 | 292 | 240 | 123 | 246 | 0 | 0 | 150 | 228 | 215 |
| Bitter Gourd | 699 | 862 | 863 | 0 | 807 | 560 | 0 | 866 | 843 | 717 |
| Brinjal | 643 | 846 | 851 | 0 | 791 | 623 | 0 | 865 | 767 | 700 |
| Brinjals | 711 | 863 | 864 | 749 | 805 | 552 | 0 | 866 | 847 | 719 |
| Cabbage (Kandy) | 43 | 863 | 864 | 42 | 434 | 379 | 0 | 854 | 797 | 645 |
| Cabbage (Nuwara Eliya) | 671 | 103 | 834 | 709 | 454 | 0 | 822 | 100 | 128 | 289 |
| Capsicum | 684 | 862 | 863 | 752 | 807 | 568 | 821 | 866 | 840 | 717 |
| Carrot | 711 | 863 | 864 | 752 | 806 | 13 | 822 | 866 | 849 | 719 |
| Cucumber | 705 | 863 | 863 | 752 | 807 | 742 | 0 | 866 | 849 | 719 |
| Drumstick | 280 | 181 | 385 | 192 | 287 | 452 | 0 | 525 | 219 | 261 |
| Green Chillies | 710 | 863 | 863 | 751 | 807 | 749 | 0 | 866 | 849 | 719 |
| Knol Khol | 708 | 860 | 861 | 752 | 764 | 94 | 822 | 865 | 194 | 691 |
| Ladies Fingers | 700 | 863 | 860 | 0 | 805 | 745 | 0 | 866 | 849 | 719 |
| Leeks | 711 | 863 | 864 | 752 | 806 | 13 | 822 | 866 | 849 | 719 |
| Lime | 672 | 862 | 864 | 0 | 806 | 18 | 0 | 866 | 848 | 719 |
| Long Beans | 664 | 863 | 863 | 0 | 806 | 734 | 0 | 866 | 849 | 717 |
| Luffa | 672 | 863 | 863 | 0 | 801 | 635 | 0 | 866 | 848 | 714 |
| Manioc | 3 | 859 | 857 | 749 | 796 | 613 | 0 | 865 | 846 | 709 |
| Potato (Imported) | 709 | 862 | 864 | 0 | 800 | 0 | 0 | 605 | 848 | 715 |
| Potato (Nuwara Eliya) | 75 | 423 | 860 | 0 | 802 | 0 | 810 | 854 | 43 | 464 |
| Potato (Welimada) | 700 | 419 | 464 | 748 | 467 | 0 | 12 | 465 | 157 | 348 |
| Pumpkin | 710 | 863 | 864 | 752 | 807 | 302 | 0 | 866 | 848 | 718 |
| Radish | 711 | 862 | 864 | 752 | 802 | 744 | 822 | 865 | 845 | 715 |
| Snake Gourd | 620 | 863 | 860 | 0 | 798 | 645 | 0 | 866 | 848 | 711 |
| Sweet Potato | 704 | 863 | 853 | 750 | 800 | 7 | 0 | 863 | 829 | 713 |
| Tomato | 711 | 863 | 864 | 751 | 807 | 362 | 822 | 866 | 849 | 719 |

### Comparison with the source PDFs

Every reading compared with the digits in `Table.extract()`'s cells, a separate route from the word-position reading used to build the dataset. PDFs read: 870. Cells found: 184996. Matching: **184996**. Price differs: **0**. Only in the dataset: **0**. Only in the PDFs: **0**.

Price differs (up to 40):

_None._

Only in the dataset (up to 40):

_None._

Only in the PDFs (up to 40):

_None._

### Random sample (4 per market, seed 20261106)

Position-based re-read matched: **40 of 40**. Image crops: `reports/figures/market_crosscheck_sheet_*.png`.

| id | market | published | reported | vegetable | csv_price | position_reread | position_match |
|---|---|---|---|---|---|---|---|
| 1 | Bandarawela | 2024-05-31 | 2024-05-31 | Brinjals | 175.0 | 150 - 200 | True |
| 2 | Bandarawela | 2026-02-11 | 2026-02-11 | Big Onion (Imported) | 162.5 | 160 - 165 | True |
| 3 | Bandarawela | 2026-04-05 | 2026-04-05 | Ladies Fingers | 120.0 | 110 - 130 | True |
| 4 | Bandarawela | 2026-05-21 | 2026-05-21 | Carrot | 300.0 | 290 - 310 | True |
| 5 | Dambulla | 2024-10-01 | 2024-10-01 | Snake Gourd | 170.0 | 160 - 180 | True |
| 6 | Dambulla | 2024-11-11 | 2024-11-11 | Big Onion (Local) | 295.0 | 270 - 320 | True |
| 7 | Dambulla | 2025-05-21 | 2025-05-21 | Cabbage (Kandy) | 255.0 | 250 - 260 | True |
| 8 | Dambulla | 2025-12-24 | 2025-12-24 | Cabbage (Kandy) | 195.0 | 160 - 230 | True |
| 9 | Kandy | 2024-05-27 | 2024-05-27 | Beetroot (Nuwara Eliya) | 245.0 | 240 - 250 | True |
| 10 | Kandy | 2024-06-05 | 2024-06-05 | Luffa | 250.0 | 240 - 260 | True |
| 11 | Kandy | 2024-08-12 | 2024-08-12 | Manioc | 75.0 | 70 - 80 | True |
| 12 | Kandy | 2025-03-17 | 2025-03-17 | Brinjals | 140.0 | 120 - 160 | True |
| 13 | Keppetipola | 2024-11-29 | 2024-11-29 | Cabbage (Nuwara Eliya) | 70.0 | 60 - 80 | True |
| 14 | Keppetipola | 2025-01-21 | 2025-01-21 | Manioc | 95.0 | 90 - 100 | True |
| 15 | Keppetipola | 2025-09-16 | 2025-09-16 | Cucumber | 105.0 | 100 - 110 | True |
| 16 | Keppetipola | 2025-09-24 | 2025-09-24 | Carrot | 105.0 | 90 - 120 | True |
| 17 | Meegoda | 2024-11-22 | 2024-11-21 | Tomato | 210.0 | 190 - 230 | True |
| 18 | Meegoda | 2024-11-25 | 2024-11-24 | Pumpkin | 110.0 | 100 - 120 | True |
| 19 | Meegoda | 2025-01-20 | 2025-01-19 | Bitter Gourd | 305.0 | 280 - 330 | True |
| 20 | Meegoda | 2026-08-11 | 2026-08-10 | Ash Plantain | 140.0 | 130 - 150 | True |
| 21 | Norochchole | 2024-07-16 | 2024-07-16 | Radish | 35.0 | 30 - 40 | True |
| 22 | Norochchole | 2024-10-16 | 2024-10-16 | Brinjal | 135.0 | 130 - 140 | True |
| 23 | Norochchole | 2025-06-15 | 2025-06-15 | Bitter Gourd | 325.0 | 300 - 350 | True |
| 24 | Norochchole | 2026-01-08 | 2026-01-08 | Tomato | 275.0 | 200 - 350 | True |
| 25 | Nuwaraeliya | 2024-09-10 | 2024-09-10 | Tomato | 150.0 | 140 - 160 | True |
| 26 | Nuwaraeliya | 2025-01-26 | 2025-01-26 | Radish | 70.0 | 60 - 80 | True |
| 27 | Nuwaraeliya | 2025-10-18 | 2025-10-18 | Beetroot (Nuwara Eliya) | 85.0 | 80 - 90 | True |
| 28 | Nuwaraeliya | 2026-06-25 | 2026-06-25 | Tomato | 480.0 | 460 - 500 | True |
| 29 | Peliyagoda | 2024-08-27 | 2024-08-27 | Pumpkin | 80.0 | 70 - 90 | True |
| 30 | Peliyagoda | 2025-05-03 | 2025-05-03 | Cabbage (Kandy) | 360.0 | 320 - 400 | True |
| 31 | Peliyagoda | 2025-07-21 | 2025-07-21 | Drumstick | 325.0 | 200 - 450 | True |
| 32 | Peliyagoda | 2026-03-09 | 2026-03-09 | Beetroot | 65.0 | 60 - 70 | True |
| 33 | Thambuththegama | 2024-06-27 | 2024-06-27 | Luffa | 175.0 | 150 - 200 | True |
| 34 | Thambuththegama | 2024-10-28 | 2024-10-28 | Big Onion (Local) | 212.5 | 150 - 275 | True |
| 35 | Thambuththegama | 2025-04-08 | 2025-04-08 | Beans | 505.0 | 480 - 530 | True |
| 36 | Thambuththegama | 2026-04-12 | 2026-04-12 | Sweet Potato | 125.0 | 110 - 140 | True |
| 37 | Veyangoda | 2024-09-24 | 2024-09-23 | Knol Khol | 175.0 | 150 - 200 | True |
| 38 | Veyangoda | 2025-01-17 | 2025-01-16 | Ladies Fingers | 180.0 | 160 - 200 | True |
| 39 | Veyangoda | 2025-09-09 | 2025-09-08 | Capsicum | 260.0 | 240 - 280 | True |
| 40 | Veyangoda | 2026-07-28 | 2026-07-27 | Luffa | 195.0 | 170 - 220 | True |

### Prices at least 5x from their median (up to 40, highest first)

| published | market | vegetable | price |
|---|---|---|---|
| 2025-10-07 00:00:00 | Kandy | Lime | 2450.0 |
| 2025-10-09 00:00:00 | Kandy | Lime | 2450.0 |
| 2025-10-08 00:00:00 | Kandy | Lime | 2450.0 |
| 2025-11-30 00:00:00 | Dambulla | Carrot | 2350.0 |
| 2025-10-10 00:00:00 | Kandy | Lime | 2250.0 |
| 2025-10-09 00:00:00 | Meegoda | Lime | 2250.0 |
| 2025-10-08 00:00:00 | Veyangoda | Lime | 2200.0 |
| 2025-10-11 00:00:00 | Bandarawela | Lime | 2150.0 |
| 2025-10-11 00:00:00 | Peliyagoda | Lime | 2150.0 |
| 2025-10-13 00:00:00 | Meegoda | Lime | 2150.0 |
| 2025-10-14 00:00:00 | Bandarawela | Lime | 2150.0 |
| 2025-10-05 00:00:00 | Peliyagoda | Lime | 2150.0 |
| 2025-10-10 00:00:00 | Bandarawela | Lime | 2150.0 |
| 2025-10-12 00:00:00 | Peliyagoda | Lime | 2150.0 |
| 2025-10-12 00:00:00 | Bandarawela | Lime | 2150.0 |
| 2025-10-10 00:00:00 | Peliyagoda | Lime | 2150.0 |
| 2025-10-11 00:00:00 | Meegoda | Lime | 2125.0 |
| 2025-10-04 00:00:00 | Peliyagoda | Lime | 2100.0 |
| 2025-10-11 00:00:00 | Veyangoda | Lime | 2100.0 |
| 2025-10-21 00:00:00 | Peliyagoda | Lime | 2100.0 |
| 2025-10-15 00:00:00 | Bandarawela | Lime | 2100.0 |
| 2025-10-05 00:00:00 | Kandy | Lime | 2100.0 |
| 2025-10-09 00:00:00 | Veyangoda | Lime | 2100.0 |
| 2025-10-10 00:00:00 | Meegoda | Lime | 2100.0 |
| 2025-10-10 00:00:00 | Veyangoda | Lime | 2100.0 |
| 2025-10-07 00:00:00 | Peliyagoda | Lime | 2100.0 |
| 2025-10-12 00:00:00 | Meegoda | Lime | 2075.0 |
| 2025-10-08 00:00:00 | Meegoda | Lime | 2075.0 |
| 2025-10-14 00:00:00 | Meegoda | Lime | 2050.0 |
| 2025-10-09 00:00:00 | Bandarawela | Lime | 2050.0 |
| 2025-10-13 00:00:00 | Peliyagoda | Lime | 2050.0 |
| 2025-10-14 00:00:00 | Peliyagoda | Lime | 2050.0 |
| 2025-10-15 00:00:00 | Peliyagoda | Lime | 2050.0 |
| 2025-10-14 00:00:00 | Veyangoda | Lime | 2050.0 |
| 2025-10-11 00:00:00 | Dambulla | Lime | 2050.0 |
| 2025-10-12 00:00:00 | Veyangoda | Lime | 2050.0 |
| 2025-10-10 00:00:00 | Dambulla | Lime | 2050.0 |
| 2025-10-13 00:00:00 | Veyangoda | Lime | 2050.0 |
| 2025-10-12 00:00:00 | Dambulla | Lime | 2000.0 |
| 2025-10-05 00:00:00 | Dambulla | Lime | 2000.0 |
