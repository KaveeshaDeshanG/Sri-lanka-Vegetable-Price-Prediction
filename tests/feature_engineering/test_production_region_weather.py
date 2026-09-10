"""Unit tests for src.feature_engineering.production_region_weather.apply_production_region_weather."""

from pathlib import Path

import pandas as pd

from src.feature_engineering.production_region_weather import apply_production_region_weather


def _mapping_yaml(tmp_path: Path) -> Path:
    mapping_path = tmp_path / "crop_production_regions.yaml"
    mapping_path.write_text(
        "regions:\n"
        "  nuwara_eliya:\n"
        '    name: "Nuwara Eliya"\n'
        "    latitude: 6.97078\n"
        "    longitude: 80.78286\n"
        "vegetable_production_regions:\n"
        "  Carrot:\n"
        "    region: nuwara_eliya\n"
        '    evidence: "test"\n'
        "    confidence: high\n",
        encoding="utf-8",
    )
    return mapping_path


def _market_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "weather_api:\n  latitude: 7.8567\n  longitude: 80.6517\n", encoding="utf-8"
    )
    return config_path


def _region_weather_csv(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "weather_data_by_region.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _merged_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


# 1. Mapped vegetable's weather is overridden by its production region's reading -----


def test_mapped_vegetable_weather_is_overridden(tmp_path):
    region_csv = _region_weather_csv(
        tmp_path,
        [
            {"date": "2026-01-01", "region": "Nuwara Eliya", "rainfall_mm": 4.2, "temperature_c": 17.1, "humidity_pct": 84},
        ],
    )
    df = _merged_df(
        [
            {"date": "2026-01-01", "vegetable": "Carrot", "market": "Dambulla",
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
        ]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(region_csv),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    row = result.iloc[0]
    assert row["rainfall_mm"] == 4.2
    assert row["temperature_c"] == 17.1
    assert row["humidity_pct"] == 84


# 2. Unmapped vegetable's weather is left completely unchanged -----------------------


def test_unmapped_vegetable_weather_is_unchanged(tmp_path):
    region_csv = _region_weather_csv(
        tmp_path,
        [{"date": "2026-01-01", "region": "Nuwara Eliya", "rainfall_mm": 4.2, "temperature_c": 17.1, "humidity_pct": 84}],
    )
    df = _merged_df(
        [
            {"date": "2026-01-01", "vegetable": "Tomato", "market": "Dambulla",
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
        ]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(region_csv),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    row = result.iloc[0]
    assert row["rainfall_mm"] == 0.0
    assert row["temperature_c"] == 29.8
    assert row["humidity_pct"] == 65


# 3. Only the EXACT matching date is used -- no future/other-date leakage ------------


def test_only_exact_date_match_is_used_not_a_nearby_date(tmp_path):
    """A region reading for a DIFFERENT date must never override this row's weather --
    proves the join is exact-date, so no future (or past) weather value leaks in."""
    region_csv = _region_weather_csv(
        tmp_path,
        [
            {"date": "2026-01-01", "region": "Nuwara Eliya", "rainfall_mm": 4.2, "temperature_c": 17.1, "humidity_pct": 84},
            {"date": "2026-01-02", "region": "Nuwara Eliya", "rainfall_mm": 99.9, "temperature_c": 99.9, "humidity_pct": 99},
        ],
    )
    df = _merged_df(
        [
            {"date": "2026-01-01", "vegetable": "Carrot", "market": "Dambulla",
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
        ]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(region_csv),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    row = result.iloc[0]
    assert row["temperature_c"] == 17.1  # 2026-01-01's own reading
    assert row["temperature_c"] != 99.9  # never the next day's reading


# 4. Date not covered by the region series: falls back to existing (market) reading --


def test_date_not_covered_by_region_series_leaves_existing_weather_unchanged(tmp_path):
    region_csv = _region_weather_csv(
        tmp_path,
        [{"date": "2026-01-01", "region": "Nuwara Eliya", "rainfall_mm": 4.2, "temperature_c": 17.1, "humidity_pct": 84}],
    )
    df = _merged_df(
        [
            {"date": "2026-06-01", "vegetable": "Carrot", "market": "Dambulla",  # not in region_csv
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
        ]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(region_csv),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    row = result.iloc[0]
    assert row["temperature_c"] == 29.8  # unchanged, since no NE reading exists for this date


# 5. Missing region-weather file: everything unchanged, no error ---------------------


def test_missing_region_weather_file_leaves_everything_unchanged(tmp_path):
    df = _merged_df(
        [{"date": "2026-01-01", "vegetable": "Carrot", "market": "Dambulla",
          "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65}]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(tmp_path / "does_not_exist.csv"),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    pd.testing.assert_frame_equal(result, df)


# 6. Market is never consulted for region resolution ----------------------------------


def test_market_column_does_not_affect_which_region_weather_is_applied(tmp_path):
    """Same vegetable, two different markets -- both must resolve to the SAME
    production region (Nuwara Eliya), proving market location and production/weather
    location are kept independent."""
    region_csv = _region_weather_csv(
        tmp_path,
        [{"date": "2026-01-01", "region": "Nuwara Eliya", "rainfall_mm": 4.2, "temperature_c": 17.1, "humidity_pct": 84}],
    )
    df = _merged_df(
        [
            {"date": "2026-01-01", "vegetable": "Carrot", "market": "Dambulla",
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
            {"date": "2026-01-01", "vegetable": "Carrot", "market": "SomeOtherMarket",
             "rainfall_mm": 0.0, "temperature_c": 29.8, "humidity_pct": 65},
        ]
    )

    result = apply_production_region_weather(
        df,
        region_weather_path=str(region_csv),
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )

    assert (result["temperature_c"] == 17.1).all()


# 7. Missing vegetable/date column: leaves df unchanged, doesn't raise ---------------


def test_missing_vegetable_column_leaves_df_unchanged(tmp_path):
    df = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "rainfall_mm": [0.0]})
    result = apply_production_region_weather(
        df,
        mapping_path=str(_mapping_yaml(tmp_path)),
        config_path=str(_market_config(tmp_path)),
    )
    pd.testing.assert_frame_equal(result, df)
