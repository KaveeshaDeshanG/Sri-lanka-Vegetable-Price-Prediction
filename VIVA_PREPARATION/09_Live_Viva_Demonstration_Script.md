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
5. Sidebar → **Predict**. Data source stays **Historical dataset (real)**. Open/close **Preview loaded data**.
6. Open **"5 recorded price series can't be forecast, and why"**: "I don't hide stale series — Cabbage from Nuwara Eliya stopped in July 2025."
7. Vegetable → **Beans**. Point at cards: Data source, Market Dambulla, Price type wholesale, Latest data + freshness.
8. Prediction type **Latest / Next Available Prediction** → click **Predict**.
   **Expected:** Rs 453.74/kg for 10 Nov 2026 (last price Rs 425). Note the yellow notice that the date already passed:
   "The forecast starts from the latest published price, not from today."
9. Choose **Future Date Forecast (multi-step)** → Forecast date **25 Nov 2026** → **Generate Forecast**.
   **Expected:** Rs 507.96/kg, 16 days ahead. Open **Daily forecast path**: "Each day uses its own Ridge model — no error feedback."
10. Set Forecast date **25 Dec 2026** → **Generate Forecast**.
    **Expected:** "Cannot forecast that date: 2026-12-25 is 46 days after the latest known price … at most 30 days ahead."
    Say: "The system refuses instead of inventing a number."
11. Data source → **Latest HARTI Data (real)** (internet): shows the newest bulletin date (was 12 Nov 2026, 30 prices).
    Say: "HARTI has no API; this downloads and parses today's PDF. It is cached for an hour and not saved."
12. Data source → **Upload my own CSV** → upload `missing_price_column.csv`.
    **Expected:** "The CSV is missing required column(s): ['price'] …" → "Input validation."

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
20. Sidebar → **Cultivation Planner**. Vegetable **Beans**, Variety **Bush types** (45 days). Show the DOA quote and link.
21. Planting date **today** → **Generate Cultivation Plan** → harvest ~28 Dec 2026 → price **Not available** (beyond 30 days) + same-season history from earlier years.
22. Planting date **20 Oct 2026** → **Generate** → first harvest **4 Dec 2026**, forecast **Rs 450/kg** with a likely range and suggested selling day.
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
