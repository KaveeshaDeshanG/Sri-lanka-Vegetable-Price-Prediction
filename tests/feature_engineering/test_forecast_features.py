"""The direct forecaster's features must use only what was public on the origin date."""

import numpy as np
import pandas as pd
import pytest

from src.feature_engineering.forecast_features import (
    FEATURE_GROUPS,
    ForecastFeatureBuilder,
    add_horizon_features,
    add_vegetable_columns,
)

ALL_FEATURES = [column for group in FEATURE_GROUPS.values() for column in group]
CUTOFF = pd.Timestamp("2025-03-01")


def _history(vegetables=("Tomato", "Beans"), days=120):
    dates = pd.date_range("2025-01-01", periods=days)
    rng = np.random.default_rng(1)
    frames = []
    for vegetable in vegetables:
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "vegetable": vegetable,
                    "market": "Dambulla",
                    "price": 100 + rng.normal(0, 5, days).cumsum().clip(-50, 300),
                    "diesel_price": 300.0,
                    "petrol_price": 350.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _panel(days=120):
    dates = pd.date_range("2025-01-01", periods=days)
    rng = np.random.default_rng(2)
    rows = []
    for market in ("Kandy", "Meegoda", "Dambulla"):
        for vegetable in ("Tomato", "Beans"):
            rows.append(pd.DataFrame({"published": dates, "market": market, "vegetable": vegetable, "price": 120 + rng.normal(0, 5, days)}))
    return pd.concat(rows, ignore_index=True)


def _weather(days=120):
    dates = pd.date_range("2025-01-01", periods=days)
    return pd.DataFrame({"date": dates, "region": "Dambulla", "rainfall_mm": 2.0 + np.arange(days) % 5, "temperature_c": 26.0, "humidity_pct": 80.0})


def _builder(tmp_path, weather=None, panel=None):
    weather_file, panel_file = tmp_path / "weather.csv", tmp_path / "panel.csv"
    (_weather() if weather is None else weather).to_csv(weather_file, index=False)
    (_panel() if panel is None else panel).to_csv(panel_file, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(
        "weather_api:\n  latitude: 7.8\n  longitude: 80.6\n"
        "paths:\n  raw_files:\n"
        f'    weather_data_by_region: "{weather_file.as_posix()}"\n'
        f'    harti_all_markets: "{panel_file.as_posix()}"\n',
        encoding="utf-8",
    )
    return ForecastFeatureBuilder(str(config))


def _with_horizon(builder, history, horizon=7):
    return add_horizon_features(builder.build_origin_features(history), horizon)


def _frozen_values(rows, horizon=7):
    known = rows[rows["origin"] <= CUTOFF - pd.Timedelta(days=horizon)].sort_values(["vegetable", "origin"])
    return known[ALL_FEATURES].to_numpy(dtype=float)


@pytest.mark.parametrize("horizon", [1, 7, 14, 30])
def test_changing_any_later_data_moves_no_earlier_feature(tmp_path, horizon):
    history = _history()
    clean = _frozen_values(_with_horizon(_builder(tmp_path), history, horizon), horizon)

    later_history = history.copy()
    later_history.loc[later_history["date"] > CUTOFF, ["price", "diesel_price", "petrol_price"]] *= 7
    later_panel = _panel()
    later_panel.loc[later_panel["published"] > CUTOFF, "price"] *= 5
    later_weather = _weather()
    later_weather.loc[later_weather["date"] > CUTOFF, ["rainfall_mm", "temperature_c", "humidity_pct"]] *= 3
    corrupted = _frozen_values(_with_horizon(_builder(tmp_path, later_weather, later_panel), later_history, horizon), horizon)

    assert np.isclose(clean, corrupted, equal_nan=True).all()


def test_an_other_market_reading_is_unknown_before_its_bulletin_is_published(tmp_path):
    """A reading published on day X can't affect features on day X-1, even if it describes X-1."""
    panel = _panel()
    published = pd.Timestamp("2025-02-10")
    changed = panel.copy()
    changed.loc[changed["published"] == published, "price"] *= 4
    history = _history()

    before = _builder(tmp_path, panel=panel).build_origin_features(history).set_index(["vegetable", "origin"])
    after = _builder(tmp_path, panel=changed).build_origin_features(history).set_index(["vegetable", "origin"])

    key = ("Tomato", published - pd.Timedelta(days=1))
    assert before.loc[key, "other_premium"] == pytest.approx(after.loc[key, "other_premium"])
    assert before.loc[("Tomato", published), "other_premium"] != pytest.approx(after.loc[("Tomato", published), "other_premium"])


def test_dambulla_is_not_counted_among_other_markets(tmp_path):
    panel = _panel()
    changed = panel.copy()
    changed.loc[changed["market"] == "Dambulla", "price"] *= 9
    history = _history()
    a = _builder(tmp_path, panel=panel).build_origin_features(history).sort_values(["vegetable", "origin"])
    b = _builder(tmp_path, panel=changed).build_origin_features(history).sort_values(["vegetable", "origin"])
    assert np.isclose(a["other_premium"], b["other_premium"], equal_nan=True).all()


def test_target_price_is_the_real_price_horizon_days_later(tmp_path):
    history = _history()
    rows = _with_horizon(_builder(tmp_path), history, 7)
    origin = pd.Timestamp("2025-02-01")
    row = rows[(rows["vegetable"] == "Tomato") & (rows["origin"] == origin)].iloc[0]
    actual = history[(history["vegetable"] == "Tomato") & (history["date"] == origin + pd.Timedelta(days=7))]["price"].iloc[0]
    assert row["target_price"] == pytest.approx(actual)
    assert row["target_date"] == origin + pd.Timedelta(days=7)


def test_a_gap_carries_the_last_price_forward_only_for_a_week(tmp_path):
    history = _history()
    gap = pd.date_range("2025-02-01", "2025-02-20")
    history = history[~((history["vegetable"] == "Tomato") & history["date"].isin(gap))]
    rows = _builder(tmp_path).build_origin_features(history)
    tomato = rows[rows["vegetable"] == "Tomato"].set_index("origin")
    assert tomato.loc["2025-02-07", "days_since_reading"] == 7
    assert tomato.loc["2025-02-07", "last_price"] == pytest.approx(tomato.loc["2025-01-31", "last_price"])
    assert np.isnan(tomato.loc["2025-02-15", "last_price"])


def test_a_live_bulletins_eggplant_rows_reach_brinjal(tmp_path):
    """A live bulletin uses HARTI's "Eggplant"; history and training call that row "Brinjal"."""
    history = _history(vegetables=("Brinjal", "Tomato"))
    panel = _panel()
    weather_file, panel_file = tmp_path / "weather.csv", tmp_path / "panel.csv"
    _weather().to_csv(weather_file, index=False)
    panel[panel["vegetable"] == "Tomato"].to_csv(panel_file, index=False)  # nothing saved for Brinjal
    config = tmp_path / "config.yaml"
    config.write_text(
        "weather_api:\n  latitude: 7.8\n  longitude: 80.6\npaths:\n  raw_files:\n"
        f'    weather_data_by_region: "{weather_file.as_posix()}"\n    harti_all_markets: "{panel_file.as_posix()}"\n',
        encoding="utf-8",
    )
    published = pd.Timestamp("2025-02-10")
    live = pd.DataFrame({"published": [published], "market": ["Kandy"], "vegetable": ["Eggplant"], "price": [150.0]})

    rows = ForecastFeatureBuilder(str(config), extra_market_rows=live).build_origin_features(history)

    brinjal = rows[(rows["vegetable"] == "Brinjal") & (rows["origin"] == published)].iloc[0]
    assert brinjal["other_level"] == pytest.approx(np.log(150.0))


def test_missing_market_data_leaves_features_empty_rather_than_failing(tmp_path):
    config = tmp_path / "config.yaml"
    weather_file = tmp_path / "weather.csv"
    _weather().to_csv(weather_file, index=False)
    config.write_text(
        f'weather_api:\n  latitude: 7.8\n  longitude: 80.6\npaths:\n  raw_files:\n    weather_data_by_region: "{weather_file.as_posix()}"\n',
        encoding="utf-8",
    )
    rows = ForecastFeatureBuilder(str(config)).build_origin_features(_history())
    assert rows["other_premium"].isna().all()
    assert add_vegetable_columns(rows, ["Tomato", "Beans"])["veg_Tomato"].sum() == 120
