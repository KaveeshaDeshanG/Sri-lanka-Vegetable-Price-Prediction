"""Streamlit entry point: project overview and navigation to the Prediction page."""

import sys
from pathlib import Path

# Streamlit's runner puts this script's own directory on sys.path, not the
# project root -- add the root explicitly so `import app.services` / `import
# src...` / `import config...` resolve the same way they do under pytest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import streamlit as st

st.set_page_config(page_title="Sri Lankan Vegetable Price Prediction", page_icon="🥕")

st.title("🥕 Sri Lankan Vegetable Price Prediction")
st.caption("AI-powered Sri Lankan vegetable price prediction system using historical market and weather data.")

st.info(
    "**Real data:** the deployed model (`models/best_model.joblib`) is trained on real HARTI "
    "market-price data for the Dambulla market (`data/raw/harti_prices.csv`), combined with real "
    "weather (`data/raw/weather_data.csv`) and fuel-price (`data/raw/fuel_prices.csv`) data. The "
    "**Predict** page also offers a small bundled **synthetic** sample dataset "
    "(`app/sample_data/sample_history.csv`, `source = \"SAMPLE\"`) as a convenience for trying the "
    "app without your own data -- predictions made against that sample data describe synthetic "
    "history, not a real market forecast."
)

st.header("How it works")
st.markdown(
    """
    ```
    raw market + weather data
        -> preprocessing        (src/preprocessing)     clean, standardize, merge
        -> feature engineering  (src/feature_engineering) lag / rolling / weather-lag features
        -> chronological split + categorical encoding    (fit on train only, never on test)
        -> model training       (src/models)             Linear Regression, Random Forest, XGBoost
        -> best model selected  by config's evaluation.best_model_metric
        -> prediction           (src/prediction)          reuses the trained model + fitted encoder
    ```

    The **Predict** page (left sidebar) trains a demo model on the bundled sample data the first
    time you visit it (if none exists yet under `models/`), then lets you request a prediction for
    a vegetable/market/date using that model.
    """
)

st.page_link("pages/1_Predict.py", label="Go to Prediction →", icon="📈")
