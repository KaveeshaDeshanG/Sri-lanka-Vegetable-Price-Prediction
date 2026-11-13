# 12 — Personal Viva Speaking Script

Speak slowly. Each English part is what you say; the Sinhala part helps you remember the meaning — don't read it aloud
unless the examiner allows Sinhala.

---

## 1. Two-minute introduction

> Good morning. My name is Kaveesha Deshan, and my project is an **AI-powered Sri Lankan vegetable price prediction system**.
>
> Vegetable prices in Sri Lanka change sharply from day to day. The official source, HARTI, publishes daily prices
> only as PDF bulletins, and they show only past prices. Farmers and traders have no easy way to see where prices
> may go next.
>
> My system does three things. First, it **collects real data**: it reads every HARTI daily bulletin since June 2015 —
> about 77,000 Dambulla wholesale prices and over half a million prices across eleven markets — and adds weather from
> Open-Meteo and fuel prices from CEYPETCO.
>
> Second, it **forecasts** the Dambulla wholesale price of 31 vegetable series from one to thirty days ahead, using a
> Ridge regression model for each forecast day.
>
> Third, it **presents** everything in a Streamlit dashboard: predictions, model performance, price history and a
> cultivation planner that links a planting date to the expected harvest price.
>
> My key result is that the model is **4.3 % more accurate next day, and 7.8 % more accurate at 22 to 30 days,
> than simply assuming the price stays the same** — a modest but statistically supported improvement.

සිංහල සාරාංශය: ගැටලුව (PDF පමණි, අතීත මිල පමණි) → විසඳුම (දත්ත එකතු කිරීම, අනාවැකිය, dashboard) → ප්‍රතිඵලය (baseline එකට වඩා 4.3 %–7.8 % හොඳයි).

---

## 2. Five-minute technical explanation

> **Architecture.** It is a layered Python application on one machine. Collection scripts in `src/data_collection` and
> `scripts` download and parse the sources into CSV files in `data/raw`. The feature builder and models are in `src`.
> The Streamlit app in `app/` reads the files and saved models. There is no database — the data fits in memory and
> CSV files are transparent and easy to back up.
>
> **Data pipeline.** HARTI has no API, so I download the archive page, find each bulletin, and use PyMuPDF to read
> each word with its position, rebuild the table, and take the midpoint of each price range — for example "400 to 450"
> becomes 425. Data is cleaned: invalid prices and impossible weather values are removed, names are standardised.
>
> **Features.** For each vegetable and day I compute: today's price on a log scale, recent price changes and
> volatility, rainfall and temperature in the vegetable's growing region, the whole market's trend, fuel price
> changes, and how other markets such as Kandy and Peliyagoda compare with Dambulla. Other markets are used only from
> the date their bulletin was published, to avoid data leakage.
>
> **Model.** I first trained Linear Regression, Random Forest and XGBoost for next-day prices. None beat the naive
> "yesterday's price" baseline. So I built a **direct forecaster**: thirty separate Ridge models, one per day ahead.
> Each learns the log change from today's price to the price h days later, after median imputation and
> standardisation. Feature groups and penalty strength were chosen on four validation windows before the test period.
>
> **Evaluation.** The test period is 14 May to 6 November 2026, never used for tuning. Next-day MAE is Rs 17.47 per kg,
> or 7.5 %, against 18.26 for holding the price — 4.3 % lower, with a 95 % bootstrap interval of 2.6 to 6 %.
>
> **Testing.** 1,060 automated tests pass, plus 18 browser-driven UI tests and an independent recomputation of a
> forecast that matched the app exactly.

සිංහල: Architecture → Data pipeline (PDF කියවීම) → Features → Model (Ridge, දිනකට එක ආකෘතියක්) → Evaluation (baseline සමඟ) → Testing.

---

## 3. Demo narration (short lines while clicking — full steps in doc 09)
- "I start the app with `start_app.bat`; it checks the Python environment and runs `app/Home.py`."
- "The home page shows the accuracy against the baseline and every dataset with its source."
- "I select Beans and click Predict — Rs 453.74 per kilogram, from the latest price of Rs 425."
- "Now a 16-day forecast — each day uses its own model."
- "If I ask for 46 days ahead, the system refuses, because it reaches only 30 days."
- "This option downloads today's HARTI bulletin live."
- "Model Performance compares the model with the hold-last-price baseline for each range."
- "Price History keeps HARTI wholesale, CBSL and DCS retail separate."
- "The planner turns a planting date into a harvest window and a price where the forecast can reach."

---

## 4. Machine learning explanation (if asked "explain your model simply")
> Imagine predicting a price 10 days from now. Instead of predicting tomorrow, then using that guess to predict the
> next day — which adds error each time — I train a model that looks at today's information and directly answers
> "how much will the price change in 10 days?". I have 30 such models. Each is a Ridge regression: a weighted sum of
> features, with a penalty that keeps the weights small so it doesn't memorise noise. The answer is a percentage
> change, which I apply to today's price.

සිංහල: දින 10කට පසු මිල සඳහා, දිනෙන් දින අනාවැකි නොකියා, "දින 10කින් මිල කොපමණ % කින් වෙනස් වේද?" යන්න සෘජුවම
අනාවැකි කියන ආකෘතියක්. එවැනි ආකෘති 30ක් ඇත.

---

## 5. Real-time data explanation
> There is no public real-time API for Sri Lankan vegetable prices. HARTI publishes one PDF bulletin per day, usually
> one or two days late. My "Latest HARTI Data" option downloads and parses the newest bulletin on demand — during
> testing it fetched the 12 November bulletin with 30 prices — fills any gap from archived bulletins, and caches it
> for an hour. Weather and fuel prices are also fetched live. So the system is **as current as the newest published
> data**, and it always tells the user how old the data is.

සිංහල: "Real-time" යනු නවතම ප්‍රකාශිත bulletin එක. දත්ත කොපමණ පැරණිද යන්න app එක සැමවිටම පෙන්වයි.

---

## 6. Honest limitations
> I want to be clear about the limitations. The improvement over the baseline is real but modest — 4 to 8 %.
> At longer horizons, the model's RMSE is slightly worse than the baseline, meaning occasional bigger misses. I
> forecast only Dambulla, and I don't have supply or arrival quantities, which drive prices. The model hasn't been
> retrained on the full 2015 history. Daily updates are not scheduled automatically, the app isn't deployed online,
> and I didn't do a study with real farmers.

සිංහල: සීමාවන් අවංකව පවසන්න — එය දුර්වලතාවයක් නොව, පරිණත බවේ ලකුණකි.

---

## 7. Closing statement
> To conclude, this project turns eleven years of PDF bulletins into a clean dataset, shows that careful, honest
> modelling can beat the naive baseline for vegetable prices, and presents forecasts with clear sources, freshness and
> uncertainty. With supply data and deployment, it could become a practical tool for farmers. Thank you — I'm happy
> to answer questions.

සිංහල: අවසානය — දත්ත, අවංක ආකෘතිය, පැහැදිලි ඉදිරිපත් කිරීම, අනාගත වැඩ. "ස්තූතියි."

---

## Phrases for difficult moments
| Situation | Say |
|---|---|
| You don't know | "I haven't tested that, but my approach would be …" |
| You need time | "Let me check that in the code" → open the file named in doc 02 |
| Examiner challenges a number | "That figure is in `reports/metrics/direct_forecaster_metrics.json`; I can show it." |
| Something breaks in the demo | "Let me restart it with `start_app.bat` — the data and models are saved, so nothing is lost." |
