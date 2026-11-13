"""Unit tests for src.models.direct_forecaster and the bulletin's all-markets parser."""

import numpy as np
import pandas as pd
import pytest

from src.data_collection.harti_live_client import HartiLiveClient, canonical_market_name, parse_all_markets
from src.feature_engineering.forecast_features import FEATURE_GROUPS
from src.models.direct_forecaster import (
    FORECAST_BANDS,
    MAX_HORIZON_DAYS,
    DirectForecaster,
    band_for,
    mae_gain_vs_holding_price,
    score_predictions,
)

SMALL_HORIZONS = range(1, 4)


def _origin_rows(days=200):
    """Features as columns of noise: enough to fit and predict with, no signal assumed."""
    rng = np.random.default_rng(3)
    calendar = pd.date_range("2025-01-01", periods=days)
    frames = []
    for vegetable, base in (("Tomato", 100.0), ("Beans", 300.0)):
        price = base * np.exp(rng.normal(0, 0.02, days).cumsum())
        frame = pd.DataFrame({"vegetable": vegetable, "origin": calendar, "actual_price": price, "last_price": price, "log_last": np.log(price)})
        frame["raw_rain"], frame["raw_temp"], frame["raw_humidity"] = 1.0, 26.0, 80.0
        for group in FEATURE_GROUPS.values():
            for column in group:
                if column not in frame:
                    frame[column] = rng.normal(0, 1, days)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_bands_cover_every_horizon_exactly_once():
    assert [day for band in FORECAST_BANDS for day in range(band.first_day, band.last_day + 1)] == list(range(1, MAX_HORIZON_DAYS + 1))
    assert band_for(1).first_day == 1 and band_for(14).first_day == 11


def test_every_configured_feature_group_exists():
    assert {group for band in FORECAST_BANDS for group in band.feature_groups} <= set(FEATURE_GROUPS)


def test_target_before_excludes_later_targets_from_training():
    rows = _origin_rows()
    cutoff = pd.Timestamp("2025-05-01")
    forecaster = DirectForecaster.fit(rows, horizons=SMALL_HORIZONS, target_before=cutoff)
    assert forecaster.trained_through < cutoff

    later = rows.copy()
    later.loc[later["origin"] >= cutoff, ["actual_price", "last_price"]] *= 50
    other = DirectForecaster.fit(later, horizons=SMALL_HORIZONS, target_before=cutoff)
    a, b = forecaster.models[1][0], other.models[1][0]
    assert np.allclose(a[-1].coef_, b[-1].coef_)


def test_forecast_returns_positive_prices_for_each_requested_horizon():
    rows = _origin_rows()
    forecaster = DirectForecaster.fit(rows, horizons=SMALL_HORIZONS)
    origin = rows["origin"].max()
    result = forecaster.forecast(rows, "Tomato", origin, [1, 2, 3])
    assert list(result["horizon"]) == [1, 2, 3]
    assert list(result["target_date"]) == [origin + pd.Timedelta(days=d) for d in (1, 2, 3)]
    assert (result["predicted_price"] > 0).all() and result["predicted_price"].notna().all()


def test_forecast_refuses_an_unknown_vegetable_and_a_missing_price():
    rows = _origin_rows()
    forecaster = DirectForecaster.fit(rows, horizons=SMALL_HORIZONS)
    with pytest.raises(ValueError, match="wasn't trained"):
        forecaster.forecast(rows, "Turnip", rows["origin"].max(), [1])
    gap = rows.copy()
    gap.loc[gap["origin"] == rows["origin"].max(), "last_price"] = np.nan
    with pytest.raises(ValueError, match="no usable price"):
        forecaster.forecast(gap, "Tomato", rows["origin"].max(), [1])


def test_forecaster_survives_saving_and_loading(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(f'paths:\n  models_dir: "{(tmp_path / "models").as_posix()}"\n', encoding="utf-8")
    rows = _origin_rows()
    forecaster = DirectForecaster.fit(rows, horizons=SMALL_HORIZONS)
    forecaster.save(str(config))
    loaded = DirectForecaster.load(str(config))
    origin = rows["origin"].max()
    pd.testing.assert_frame_equal(forecaster.forecast(rows, "Beans", origin, [1, 3]), loaded.forecast(rows, "Beans", origin, [1, 3]))


def test_score_predictions_matches_a_hand_calculation():
    result = score_predictions([100, 200], [110, 180])
    assert result["mae"] == 15
    assert result["rmse"] == pytest.approx(np.sqrt(250))
    assert result["mape"] == pytest.approx(10)


def test_gain_is_positive_when_the_model_is_closer_than_holding():
    dates = pd.date_range("2026-01-01", periods=30)
    frame = pd.DataFrame({"target_date": dates, "target_price": 100.0, "last_price": 110.0, "prediction": 105.0})
    result = mae_gain_vs_holding_price(frame)
    assert result["gain_pct"] == pytest.approx(50)
    assert result["gain_low"] == pytest.approx(50) and result["gain_high"] == pytest.approx(50)


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Peliyagoda Market", "Peliyagoda"),
        ("Peliyagod", "Peliyagoda"),
        ("Kandy 2024.11.13", "Kandy"),
        ("hambuththegam Market", "Thambuththegama"),
        ("Variety", None),
        (None, None),
    ],
)
def test_market_headers_resolve_despite_extraction_damage(header, expected):
    assert canonical_market_name(header) == expected


def test_all_markets_are_keyed_by_the_bulletin_that_published_them():
    table = [
        ["Variety", "2026.11.06", "2026.11.05"],
        [None, "Dambulla Market", "Meegoda Market"],
        ["Carrot", "100 - 200", "300 - 500"],
        ["Banana", "1 - 2", "3 - 4"],
        ["Beans", "-", "200 - 240"],
    ]
    panel = parse_all_markets(table, pd.Timestamp("2026-11-06"))
    assert set(panel["published"]) == {pd.Timestamp("2026-11-06")}
    meegoda = panel[(panel["market"] == "Meegoda") & (panel["vegetable"] == "Carrot")].iloc[0]
    assert meegoda["reported"] == pd.Timestamp("2026-11-05")  # described the day before publication
    assert meegoda["price"] == 400
    assert len(panel) == 3  # no fruit row, and no row for Dambulla's "-"


def test_real_bulletin_yields_every_market():
    with open("tests/fixtures/harti_sample_bulletin.pdf", "rb") as file:
        table = HartiLiveClient()._extract_vegetable_table(file.read())
    panel = parse_all_markets(table, pd.Timestamp("2026-08-18"))
    assert {"Dambulla", "Peliyagoda", "Kandy", "Meegoda"} <= set(panel["market"])
    assert (panel["price"] > 0).all()
