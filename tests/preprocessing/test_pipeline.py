"""Integration test for the full preprocessing pipeline, using Phase 2's test fixtures."""

import pandas as pd

from src.preprocessing.pipeline import PreprocessingPipeline


def test_full_pipeline_runs_end_to_end_and_saves_csv(tmp_path):
    output_path = tmp_path / "processed_prices.csv"
    pipeline = PreprocessingPipeline(config_path="tests/fixtures/test_config.yaml")

    merged = pipeline.run(output_path=str(output_path))

    assert output_path.exists()
    assert len(merged) > 0
    assert "price" in merged.columns
    assert "rainfall_mm" in merged.columns
    assert pd.api.types.is_datetime64_any_dtype(merged["date"])

    saved = pd.read_csv(output_path)
    assert len(saved) == len(merged)


def test_full_pipeline_runs_without_fuel_configured(tmp_path):
    """test_config.yaml has no paths.raw_files.fuel_prices -- pipeline must work exactly as before."""
    output_path = tmp_path / "processed_prices.csv"
    pipeline = PreprocessingPipeline(config_path="tests/fixtures/test_config.yaml")

    merged = pipeline.run(output_path=str(output_path))

    assert len(merged) > 0
    assert "diesel_price" not in merged.columns
    assert "petrol_price" not in merged.columns


def test_full_pipeline_attaches_fuel_columns_when_configured(tmp_path):
    output_path = tmp_path / "processed_prices.csv"
    pipeline = PreprocessingPipeline(config_path="tests/fixtures/test_config_with_fuel.yaml")

    merged = pipeline.run(output_path=str(output_path))

    assert len(merged) > 0
    assert "diesel_price" in merged.columns
    assert "petrol_price" in merged.columns
    assert not merged["diesel_price"].isna().all()
