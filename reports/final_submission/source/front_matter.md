## Abstract

Vegetable prices in Sri Lanka change quickly, and the official price information published by the Hector Kobbekaduwa Agrarian Research and Training Institute (HARTI), the Central Bank of Sri Lanka (CBSL) and the Department of Census and Statistics (DCS) describes past prices only, in PDF bulletins that are hard to analyse. This project developed a web application that turns these bulletins into validated datasets, forecasts wholesale vegetable prices at the Dambulla Dedicated Economic Centre for 1 to 30 days ahead, and links the forecasts to cultivation planning.

A parser was built for HARTI's daily bulletin archive from June 2015, producing 76,954 Dambulla and 528,587 all-market price readings by 17 November 2026; independent checks of the 2024-2026 data against the original PDFs found no extraction errors. The datasets are kept current by an automatic update every three hours, with validation, duplicate prevention, backups, retries and recovery from interrupted runs, and every page shows how fresh each source is. The forecaster is a direct multi-horizon model with one Ridge Regression per day ahead, using only information published by the forecast date, including prices reported for other markets, regional weather and fuel prices. On a held-out test period from May to November 2026 its next-day mean absolute error was Rs 17.47 per kilogram (7.5%), 4.3% lower than holding the last price, and 7.8% lower at 22 to 30 days. Earlier Linear Regression, Random Forest and XGBoost models did not beat this baseline despite R² values of about 0.96, and a retraining candidate on the full history was rejected because it did not beat the baseline at long horizons.

The Streamlit application has five pages: a home dashboard, prediction, model performance, price history and a cultivation planner that converts published Department of Agriculture harvest durations into harvest windows and compares the current price with the expected harvest-time price. The whole interface is available in English, Sinhala and Tamil. All 1,320 automated tests and {{BT_TOTAL}} browser tests passed.

**Keywords:** vegetable price forecasting, Sri Lanka, Ridge regression, direct multi-step forecasting, cultivation planning, Streamlit, multilingual interface

## Acknowledgement

[Write your own acknowledgement here, for example thanking your supervisor, [First Supervisor's Name], and second marker, [Second Supervisor's Name], for their guidance, and your family and friends for their support.]

I also acknowledge the Hector Kobbekaduwa Agrarian Research and Training Institute, the Central Bank of Sri Lanka, the Department of Census and Statistics and the Department of Agriculture for publishing the price bulletins and crop information on which this project is built, and the Open-Meteo project for free historical weather data.
