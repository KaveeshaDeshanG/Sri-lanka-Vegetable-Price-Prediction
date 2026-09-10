"""Unit and integration tests for src.feature_engineering.feature_engineer.FeatureEngineer."""

import pandas as pd
import pytest

from src.feature_engineering.feature_engineer import FeatureEngineer, FeatureEngineeringError
from src.preprocessing.pipeline import PreprocessingPipeline


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer(config_path="config/config.yaml")


def _sample_merged_df() -> pd.DataFrame:
    """A small DataFrame shaped like Phase 6's merge output (market + weather + source)."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-08", "2024-01-15", "2024-01-01", "2024-01-08", "2024-01-15"]
            ),
            "vegetable": ["Carrot"] * 3 + ["Tomato"] * 3,
            "market": ["Dambulla"] * 6,
            "source": ["HARTI"] * 6,
            "price": [100.0, 105.0, 102.0, 80.0, 82.0, 78.0],
            "rainfall_mm": [5.0, 10.0, 0.0, 5.0, 10.0, 0.0],
            "temperature_c": [28.0, 29.0, 27.0, 28.0, 29.0, 27.0],
            "humidity_pct": [70.0, 75.0, 72.0, 70.0, 75.0, 72.0],
        }
    )


def _sample_merged_df_with_fuel() -> pd.DataFrame:
    """Same shape as _sample_merged_df(), plus the fuel-price columns merge_fuel_prices() adds."""
    df = _sample_merged_df()
    df["diesel_price"] = [300.0, 310.0, 320.0, 300.0, 310.0, 320.0]
    df["petrol_price"] = [350.0, 360.0, 370.0, 350.0, 360.0, 370.0]
    return df


def test_build_features_raises_on_missing_required_columns(engineer):
    df = _sample_merged_df().drop(columns=["price"])
    with pytest.raises(FeatureEngineeringError, match="price"):
        engineer.build_features(df)


def test_build_features_raises_on_completely_empty_dataframe(engineer):
    with pytest.raises(FeatureEngineeringError):
        engineer.build_features(pd.DataFrame())


def test_build_features_handles_zero_rows_with_correct_schema(engineer):
    df = _sample_merged_df().iloc[0:0]
    result = engineer.build_features(df)
    assert len(result) == 0
    assert "price_lag_1d" in result.columns
    assert "price_rolling_mean_7d" in result.columns
    assert "rainfall_mm_lag14d" in result.columns


def test_build_features_adds_expected_columns(engineer):
    result = engineer.build_features(_sample_merged_df())
    for expected in (
        "day_of_week",
        "month",
        "price_lag_1d",
        "price_lag_7d",
        "price_lag_14d",
        "price_lag_30d",
        "price_rolling_mean_7d",
        "price_rolling_std_7d",
        "rainfall_mm_lag14d",
        "temperature_c_lag14d",
        "humidity_pct_lag14d",
    ):
        assert expected in result.columns


def test_build_features_adds_fuel_price_lag_columns_when_fuel_present(engineer):
    result = engineer.build_features(_sample_merged_df_with_fuel())
    assert "diesel_price_lag_1d" in result.columns
    assert "petrol_price_lag_1d" in result.columns


def test_build_features_omits_fuel_price_lag_columns_when_fuel_absent(engineer):
    """No fuel_price columns in the input -- must degrade gracefully, exactly like weather."""
    result = engineer.build_features(_sample_merged_df())
    assert "diesel_price_lag_1d" not in result.columns
    assert "petrol_price_lag_1d" not in result.columns


def test_split_features_target_excludes_raw_fuel_columns_but_keeps_lagged(engineer):
    built = engineer.build_features(_sample_merged_df_with_fuel())
    X, y = engineer.split_features_target(built, drop_incomplete_rows=False)

    # raw same-day fuel columns must never reach X -- same leakage-prevention
    # reasoning as same-day weather columns.
    assert "diesel_price" not in X.columns
    assert "petrol_price" not in X.columns
    # the lagged versions must remain available as features.
    assert "diesel_price_lag_1d" in X.columns
    assert "petrol_price_lag_1d" in X.columns


def test_split_features_target_excludes_target_and_identifier_columns(engineer):
    built = engineer.build_features(_sample_merged_df())
    X, y = engineer.split_features_target(built, drop_incomplete_rows=False)

    assert "price" not in X.columns  # the target must never leak into X
    assert "date" not in X.columns
    assert "source" not in X.columns
    # raw same-day weather is superseded by the lagged version, see weather_features.py
    assert "rainfall_mm" not in X.columns
    assert "temperature_c" not in X.columns
    assert "humidity_pct" not in X.columns
    assert "rainfall_mm_lag14d" in X.columns

    assert y.name == "price"
    assert len(X) == len(y)


def test_split_features_target_drops_rows_with_incomplete_engineered_features(engineer):
    built = engineer.build_features(_sample_merged_df())
    X_complete, y_complete = engineer.split_features_target(built, drop_incomplete_rows=True)
    X_full, y_full = engineer.split_features_target(built, drop_incomplete_rows=False)

    assert len(X_complete) < len(X_full)  # price_lag_30d can never be populated by this fixture
    assert len(X_complete) == len(y_complete)
    # every remaining row has a fully populated feature set
    assert not X_complete.isna().any().any()


def test_split_features_target_never_fills_nan_with_zero(engineer):
    built = engineer.build_features(_sample_merged_df())
    _, y_full = engineer.split_features_target(built, drop_incomplete_rows=False)
    X_full, _ = engineer.split_features_target(built, drop_incomplete_rows=False)
    # price_lag_30d is NaN for every row in this short fixture -- must stay NaN, not become 0
    assert X_full["price_lag_30d"].isna().all()
    assert not (X_full["price_lag_30d"] == 0).any()


def test_save_feature_dataset_writes_csv(engineer, tmp_path):
    built = engineer.build_features(_sample_merged_df())
    output_path = tmp_path / "feature_dataset.csv"
    saved_path = engineer.save_feature_dataset(built, output_path=str(output_path))

    assert output_path.exists()
    assert saved_path == str(output_path)
    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(built)


def test_split_features_target_require_target_false_keeps_nan_target_rows(engineer):
    """Inference support: a row with an unknown (NaN) target must survive when require_target=False."""
    built = engineer.build_features(_sample_merged_df())
    built.loc[0, "price"] = float("nan")

    X_with_target_required, y_with_target_required = engineer.split_features_target(
        built, drop_incomplete_rows=False, require_target=True
    )
    X_without_requiring_target, y_without_requiring_target = engineer.split_features_target(
        built, drop_incomplete_rows=False, require_target=False
    )

    assert len(X_with_target_required) == len(built) - 1  # the NaN-target row was dropped
    assert len(X_without_requiring_target) == len(built)  # kept, target value simply left NaN
    assert y_without_requiring_target.isna().sum() == 1


def test_split_features_target_require_target_default_matches_prior_behavior(engineer):
    """require_target defaults to True, so existing callers see no behavior change."""
    built = engineer.build_features(_sample_merged_df())
    X_default, y_default = engineer.split_features_target(built, drop_incomplete_rows=False)
    X_explicit, y_explicit = engineer.split_features_target(
        built, drop_incomplete_rows=False, require_target=True
    )
    pd.testing.assert_frame_equal(X_default, X_explicit)
    pd.testing.assert_series_equal(y_default, y_explicit)


def test_feature_engineering_integrates_with_preprocessing_pipeline_fixtures(tmp_path):
    """End-to-end: raw fixtures -> PreprocessingPipeline -> FeatureEngineer, no real data needed."""
    preprocessing = PreprocessingPipeline(config_path="tests/fixtures/test_config.yaml")
    merged = preprocessing.run(output_path=str(tmp_path / "processed_prices.csv"))

    engineer = FeatureEngineer(config_path="tests/fixtures/test_config.yaml")
    features_df = engineer.build_features(merged)
    X, y = engineer.split_features_target(features_df, drop_incomplete_rows=False)

    assert len(features_df) == len(merged)
    assert "price_lag_7d" in features_df.columns
    assert len(X) == len(y)
    assert "price" not in X.columns

    saved_path = engineer.save_feature_dataset(
        features_df, output_path=str(tmp_path / "feature_dataset.csv")
    )
    assert pd.read_csv(saved_path).shape[0] == len(features_df)
