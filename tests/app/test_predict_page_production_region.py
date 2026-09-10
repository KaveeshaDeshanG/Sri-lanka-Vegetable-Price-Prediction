"""Integration test: the Predict page displays the resolved production region (not the
market) for a mapped vegetable, and the market-fallback message for an unmapped one --
against the REAL config/crop_production_regions.yaml, since app/pages/1_Predict.py's
CONFIG_PATH is a hardcoded "config/config.yaml" (this is the one page-level test in the
project exercising the real, deployed mapping end to end through the actual UI code,
the same pattern test_live_weather_client.py's test_real_config_still_resolves_...
already uses for testing against the real config elsewhere in this project).
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

PAGE_PATH = str(Path(__file__).resolve().parents[2] / "app" / "pages" / "1_Predict.py")


def test_predict_page_shows_production_region_for_mapped_vegetable():
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.run()

    assert not at.exception

    vegetable_box = [s for s in at.selectbox if s.label == "Vegetable"][0]
    vegetable_box.set_value("Carrot")
    at.run()

    captions = [c.value for c in at.caption]
    matching = [c for c in captions if "Weather region" in c]
    assert matching, f"no weather-region caption found among: {captions}"
    assert "Nuwara Eliya" in matching[0]
    assert "Open-Meteo" in matching[0]


def test_predict_page_shows_market_fallback_for_unmapped_vegetable():
    at = AppTest.from_file(PAGE_PATH, default_timeout=60)
    at.run()

    assert not at.exception

    vegetable_box = [s for s in at.selectbox if s.label == "Vegetable"][0]
    vegetable_box.set_value("Tomato")  # not in crop_production_regions.yaml
    at.run()

    captions = [c.value for c in at.caption]
    matching = [c for c in captions if "Weather region" in c]
    assert matching, f"no weather-region caption found among: {captions}"
    assert "no sufficiently evidenced production region" in matching[0]
