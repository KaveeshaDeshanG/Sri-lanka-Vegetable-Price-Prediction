# 09 — Live Viva Demonstration Script (15–20 minutes)

Every step below was performed successfully on 13 November 2026 (automated UI tests TC01–TC18). Expected values
will change if you run `daily_update.py` before the viva (newer data → slightly different prices). That is normal —
say "the data was updated this morning".

> සිංහල: Viva එකට පෙර දින මෙම පියවර 2–3 වරක් පුහුණු වන්න. අන්තර්ජාලය නැත්නම් "Latest HARTI" පියවර මඟ හරින්න.

## Before the examiner arrives (5 minutes earlier)
1. Close other heavy programs. Connect to the internet (needed only for step 7).
2. Check port 8501 is free:
   ```powershell
   Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
   ```
3. Open File Explorer at the project folder. Have `Final_Project_Report\testing_evidence\inputs\` ready (CSV files for upload).
4. Optional: start the app once and open each page so caches are warm, then leave it running.

---

## 0:00–2:00 — Start the app
| Action | Say |
|---|---|
| Open PowerShell, type `cd "D:\ECU\Sri-lanka-Vegetable-Price-Prediction-main\Sri-lanka-Vegetable-Price-Prediction-main"` | "This is the project root." |
| Type `.\start_app.bat` (or double-click it) | "The launcher checks the Python 3.12 environment and required libraries, then runs Streamlit on `app/Home.py`." |
| Wait for `Local URL: http://localhost:8501`; browser opens | "It runs locally; no database or server is needed." |

## 2:00–4:00 — Home page
1. Point at the blue box: "Ridge direct forecaster, trained through 6 November 2026."
2. Point at the four cards: **Next-day error 7.5 %**, **4.3 % lower than holding the price**, 1–30 days, 31 vegetables.
   Say: "Every result is compared with the naive baseline — tomorrow's price equals today's."
3. Scroll to **Data in use**: "76,719 HARTI Dambulla prices from 2015, plus all markets, CBSL, DCS, fuel and weather — all real, with source links."
4. Point at "Last daily update …" line.

## 4:00–9:00 — Predict page (main feature)
*(Page redesigned on 15 Nov 2026: two data sources, five sections. Expected values below are from the 15 Nov data.)*

5. Sidebar → **Predict**. It opens on **Latest Real Data Prediction**. Section **1. Official source status**:
   "Every 3 hours a scheduled task checks HARTI, CBSL and DCS. Each card shows the newest publication found, when it
   was checked, and whether anything new arrived or the check failed. A failed source is never shown as up to date."
   Optional (internet): click **Check official sources now**.
6. Section **3. Vegetable and market**: open **"5 recorded price series can't be forecast, and why"**: "I don't hide stale
   series: Cabbage from Nuwara Eliya stopped in July 2025."
7. In section 3, Vegetable → **Beans**; section 2 above now shows Beans' records. Read the **forecast starting point**: HARTI Dambulla, 12 Nov 2026, Rs 430 (midpoint of 380–480).
   Click **Bulletin** to open HARTI's own PDF. Open the **CBSL** tab: "CBSL collects its own prices: Rs 375 wholesale
   on 13 Nov. Shown for comparison, never mixed with HARTI or used by the model." Open **DCS**: "Retail, weekly, Colombo."
8. Section **4. Prediction** → Forecast period **7 days** → **Predict**.
   **Expected:** Rs 431.62/kg for 19 Nov 2026. Scroll to **Forecasts at 1, 7, 14 and 30 days**: 413.04 / 431.62 /
   459.86 / 447.14, each with the accuracy measured for its range and the hold-last-price MAE beside it (all seven columns fit). "Each day uses its own Ridge model; no error feedback."
9. Choose **Another date**: "The date picker stops at 30 days after the latest price; the system can't be asked for
   more than it was tested for."
10. Section **5. Data freshness and warnings**: "It says whether the data is up to date with the official source, and
    warns if a check failed or HARTI hasn't published for more than 3 days."
11. (Skip) The live-download option no longer exists; the scheduled update replaces it.
12. Data source → **Upload CSV** → upload `missing_price_column.csv`.
    **Expected:** "The CSV is missing required column(s): ['price'] …" → "Input validation. Uploaded data is never mixed with official data."

## 9:00–11:00 — Model Performance
13. Sidebar → **Model Performance**. Point at the bar chart: model (green) vs **Hold last price** (grey) per range.
14. Say: "4.3 % lower error next day, 95 % interval 2.6–6.0 %; 7.8 % at 22–30 days. It is a modest but real gain."
15. Scroll to **Earlier models**: "Linear Regression, Random Forest and XGBoost did not beat yesterday's price — that is why I built the direct forecaster."

## 11:00–13:30 — Price History
16. Sidebar → **Price History**. HARTI: Market **Kandy**, any vegetable → chart + latest/average/lowest/highest.
17. Switch Market to **Dambulla** → open weather tabs **Rainfall / Temperature / Humidity**.
18. Data source → **CBSL daily report** → Price type **retail**: "CBSL collects its own prices; I keep it separate from HARTI."
19. Data source → **DCS weekly bulletins**: "Retail, weekly, Colombo — a different measurement."

## 13:30–16:30 — Cultivation Planner
20. Sidebar → **Cultivation Planner**. Vegetable **Beans**, Variety **Bush types** (45 days). Show the DOA quote and link, and the **Forecast reach** box: forecasts reach 12 Dec 2026, so a planting date after 28 Oct 2026 gets history, not a forecast.
21. Planting date **today** → **Generate Cultivation Plan** → harvest ~28 Dec 2026 → no forecast (beyond 30 days), so section 3 shows the orange **Historical Seasonal Price Reference – Not a Forecast**: about Rs 425/kg, the median of ~62 Dambulla prices in the same weeks of 2021–2026, with the range and years. Say: "History for this time of year, not a prediction."
22. Planting date **20 Oct 2026** → **Generate** → first harvest **4 Dec 2026**, forecast **Rs 463/kg** with a likely range; the box above names the starting record (Rs 430, 12 Nov) and the model's weather region (Badulla). Change **Weather to display** to Jaffna → the forecast doesn't change.
23. Say: "Months are counted as calendar months, not 30 days; the range comes from past forecast errors (10th–90th percentile)."

## 16:30–18:00 — Code and tests (only if asked)
24. Open `src/models/direct_forecaster.py` → lines 47–52 (bands) and 118–121 (pipeline and log target).
25. In a second PowerShell: `.\.venv312\Scripts\python.exe -m pytest -q` → (takes 3–9 minutes, so better to show `VIVA_PREPARATION\test_evidence_2026-11-13\pytest_output.txt`) "1,060 tests pass."

## 18:00–20:00 — Close
26. Close the browser tab. In the console window press **Ctrl + C** → Streamlit stops. Close the window.
27. Closing line: "In summary, the system turns HARTI's PDF bulletins into a clean 11-year dataset, forecasts Dambulla
    prices up to 30 days with a model that is measurably better than the naive baseline, and presents it honestly."

---

## If something goes wrong
| Problem | What to do / say |
|---|---|
| Page spins ~10 s first time | "It is building features for 31 vegetables; then it's cached." |
| No internet | Skip step 11: "The live option needs internet; the stored data works offline." |
| App opened on 8502 | An old copy is running — use the 8502 URL or close the old console. |
| Different price than in this script | Data was updated; explain the method instead of the number. |
| `No module named streamlit` | You used the wrong Python — use `start_app.bat`. |
| A red error on a page | Read it aloud — the app explains missing files; restart with `start_app.bat`. |
