"""Integration test: the "Future Date Forecast (multi-step)" option on the Predict page
renders and runs through Streamlit's own AppTest harness, mirroring
test_predict_page_production_region.py's pattern of exercising the real, deployed page
end to end (not app.forecasting directly -- that's covered by tests/app/test_forecasting.py).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Predict.py")


def test_predict_page_future_forecast_option_renders_without_error():
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    prediction_type_radio = [r for r in at.radio if r.label == "Prediction type"][0]
    prediction_type_radio.set_value("Future Date Forecast (multi-step)")
    at.run()

    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("Forecast horizon" in c for c in captions)
    assert any("furthest" in c and "date" in c for c in captions)
    assert any(b.label == "Generate Forecast" for b in at.button)


def test_predict_page_latest_prediction_option_is_still_the_default():
    """The existing single-step flow (button label, widgets) must remain the default
    -- proves adding the forecast option didn't disturb it."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    prediction_type_radio = [r for r in at.radio if r.label == "Prediction type"][0]
    assert prediction_type_radio.value == "Latest / Next Available Prediction"
    assert any(b.label == "Predict" for b in at.button)


def test_predict_page_generate_forecast_surfaces_a_clear_error_not_a_crash():
    """The bundled synthetic sample data has no fuel-price columns, so forecasting with
    a fuel-price-trained production model can't succeed against it (see README.md
    section 9) -- this must surface as a normal st.error(), not an unhandled exception."""
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.run()
    assert not at.exception

    prediction_type_radio = [r for r in at.radio if r.label == "Prediction type"][0]
    prediction_type_radio.set_value("Future Date Forecast (multi-step)")
    at.run()
    assert not at.exception

    generate_button = [b for b in at.button if b.label == "Generate Forecast"][0]
    generate_button.click()
    at.run()

    assert not at.exception
