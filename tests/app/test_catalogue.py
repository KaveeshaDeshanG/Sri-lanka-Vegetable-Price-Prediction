"""The vegetable catalogue: names, support status and data freshness."""

from pathlib import Path

import pandas as pd
import pytest

from app.catalogue import HARTI_SERIES, NO_DURATION_REASONS, crop_for, data_version, display_name, freshness, past_date_notice, series_status
from app.cultivation import load_crop_timings

ROOT = Path(__file__).resolve().parents[2]


def test_display_names_are_unique_and_distinguish_the_two_brinjal_rows():
    names = [entry[0] for entry in HARTI_SERIES.values()]
    assert len(names) == len(set(names))
    assert display_name("Brinjal") == "Eggplant"  # HARTI's "Eggplant" row, called "Brinjal" by the model
    assert display_name("Brinjals") == "Brinjals"
    assert crop_for("Brinjal") == crop_for("Brinjals") == "Brinjal"
    assert display_name("Not a series") == "Not a series"


def test_every_crop_named_by_a_series_has_a_reference_or_a_stated_reason():
    timings = load_crop_timings(ROOT / "data/reference/doa_harvest_durations_v2.csv")
    with_durations = {t.crop for t in timings}
    for series, (_, crop, _) in HARTI_SERIES.items():
        if crop in with_durations:
            continue
        assert series in NO_DURATION_REASONS or crop in NO_DURATION_REASONS, series


def test_every_reference_price_label_is_a_known_series():
    timings = load_crop_timings(ROOT / "data/reference/doa_harvest_durations_v2.csv")
    for timing in timings:
        for label in timing.price_labels:
            assert label in HARTI_SERIES and crop_for(label) == timing.crop, (timing.crop, label)


@pytest.mark.parametrize(("age", "level"), [(0, "current"), (3, "current"), (4, "ageing"), (30, "ageing"), (31, "stale"), (500, "stale")])
def test_freshness_levels(age, level):
    today = pd.Timestamp("2026-11-09")
    assert freshness(today - pd.Timedelta(days=age), today).level == level


def _history(rows):
    return pd.DataFrame(rows, columns=["date", "vegetable", "price"]).assign(date=lambda frame: pd.to_datetime(frame["date"]))


def test_stale_and_untrained_series_are_not_offered_for_prediction():
    history = _history(
        [
            ("2026-11-08", "Carrot", 100.0),
            ("2025-07-12", "Cabbage (Nuwara Eliya)", 90.0),
            ("2026-11-08", "Brand New", 50.0),
            ("2026-11-08", "Nothing", None),
        ]
    )
    status = series_status(history, ["Carrot", "Cabbage (Nuwara Eliya)"], today="2026-11-09").set_index("series")
    assert status.loc["Carrot", "prediction_supported"]
    assert not status.loc["Cabbage (Nuwara Eliya)", "prediction_supported"]
    assert "2025-07-12" in status.loc["Cabbage (Nuwara Eliya)", "reason"]
    assert not status.loc["Brand New", "prediction_supported"]
    assert "Nothing" not in status.index  # no price at all: not listed as a series


def test_without_a_forecaster_nothing_is_offered():
    status = series_status(_history([("2026-11-08", "Carrot", 100.0)]), None, today="2026-11-09")
    assert not status["prediction_supported"].any()


def test_a_past_target_date_is_flagged_and_today_or_later_is_not():
    notice = past_date_notice("2026-11-07", "2026-11-06", today="2026-11-11")
    assert "2026-11-07 has already passed" in notice and "4 day(s) ago" in notice and "2026-11-06" in notice
    assert past_date_notice("2026-11-11", "2026-11-06", today="2026-11-11") is None
    assert past_date_notice("2026-12-01", "2026-11-06", today="2026-11-11") is None


def test_data_version_changes_when_a_file_changes(tmp_path):
    (tmp_path / "a.csv").write_text("x\n1\n")
    before = data_version((str(tmp_path),))
    assert data_version((str(tmp_path),)) == before
    (tmp_path / "a.csv").write_text("x\n1\n2\n")
    assert data_version((str(tmp_path),)) != before


def test_only_sources_with_data_are_listed(monkeypatch):
    from app import ui
    from config.config_loader import ConfigLoader

    monkeypatch.chdir(ROOT)
    listed = ui.available_sources()
    assert "HARTI daily bulletins (wholesale)" in listed
    config = ConfigLoader("config/config.yaml")
    for name, key in ui._SOURCE_FILES:
        path = config.get(key) if key.startswith("paths.") else key
        assert (name in listed) == bool(path and (ROOT / path).exists()), name
