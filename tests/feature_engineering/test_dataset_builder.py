"""Unit and integration tests for src.feature_engineering.dataset_builder.TrainTestDatasetBuilder."""

import pandas as pd
import pytest

from src.feature_engineering.dataset_builder import TrainTestDatasetBuilder
from src.feature_engineering.feature_engineer import FeatureEngineeringError


def _daily_group_df(
    start: str,
    n_days: int,
    vegetable: str,
    market: str,
    price_start: float,
    price_step: float = 1.0,
    source: str = "HARTI",
) -> pd.DataFrame:
    """n_days of consecutive daily rows for one (vegetable, market) group, price rising by price_step/day."""
    dates = pd.date_range(start, periods=n_days, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "vegetable": [vegetable] * n_days,
            "market": [market] * n_days,
            "source": [source] * n_days,
            "price": [price_start + price_step * i for i in range(n_days)],
            "rainfall_mm": [5.0] * n_days,
            "temperature_c": [28.0] * n_days,
            "humidity_pct": [70.0] * n_days,
        }
    )


def _daily_group_df_with_fuel(
    start: str,
    n_days: int,
    vegetable: str,
    market: str,
    price_start: float,
    price_step: float = 1.0,
    source: str = "HARTI",
) -> pd.DataFrame:
    """Same shape as _daily_group_df(), plus the fuel-price columns merge_fuel_prices() adds."""
    df = _daily_group_df(start, n_days, vegetable, market, price_start, price_step, source)
    df["diesel_price"] = [300.0] * n_days
    df["petrol_price"] = [350.0] * n_days
    return df


@pytest.fixture
def builder() -> TrainTestDatasetBuilder:
    return TrainTestDatasetBuilder(config_path="config/config.yaml")


# 1. Chronological order is respected -------------------------------------------


def test_chronological_order_is_respected(builder):
    merged = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    assert len(X_train) == len(y_train) == 15
    assert len(X_test) == len(y_test) == 5
    # price rises monotonically with date in this fixture, so train (older) < test (newer)
    assert y_train.max() < y_test.min()


# 2. test_size defaults to config's model.test_size when omitted ---------------


def test_build_uses_config_test_size_when_not_specified(builder):
    merged = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    X_train, X_test, y_train, y_test = builder.build(merged, drop_incomplete_rows=False)

    # config/config.yaml has model.test_size: 0.2 -- same result as passing 0.2 explicitly
    assert len(X_train) == len(y_train) == 16
    assert len(X_test) == len(y_test) == 4


# 3. Encoder is fitted using only training categories ---------------------------


def test_encoder_is_fitted_only_using_training_categories(builder):
    # 20 days of one group, followed immediately by 5 days of a second group that
    # only exists in what test_size=0.2 will place in the test split.
    train_period = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    test_only_period = _daily_group_df(
        "2024-01-21", n_days=5, vegetable="RadishNew", market="Jaffna", price_start=200.0
    )
    merged = pd.concat([train_period, test_only_period], ignore_index=True)

    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.2, drop_incomplete_rows=False)
    assert len(X_train) == 20
    assert len(X_test) == 5

    # a category never seen in training gets no learned one-hot column at all
    assert "vegetable_RadishNew" not in X_train.columns
    assert "market_Jaffna" not in X_train.columns
    assert "vegetable_RadishNew" not in X_test.columns
    assert "market_Jaffna" not in X_test.columns

    # the training-only category is present and correctly one-hot encoded in train
    assert (X_train["vegetable_Carrot"] == 1).all()
    assert (X_train["market_Dambulla"] == 1).all()


# 4. Unseen test categories do not break transformation --------------------------


def test_unseen_test_categories_do_not_break_transformation_and_encode_as_zero(builder):
    train_period = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    test_only_period = _daily_group_df(
        "2024-01-21", n_days=5, vegetable="RadishNew", market="Jaffna", price_start=200.0
    )
    merged = pd.concat([train_period, test_only_period], ignore_index=True)

    # must not raise
    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.2, drop_incomplete_rows=False)

    vegetable_columns = [c for c in X_test.columns if c.startswith("vegetable_")]
    market_columns = [c for c in X_test.columns if c.startswith("market_")]
    assert (X_test[vegetable_columns] == 0).all(axis=None)
    assert (X_test[market_columns] == 0).all(axis=None)


# 5. Train and test feature columns are identical --------------------------------


def test_train_and_test_feature_columns_are_identical(builder):
    train_period = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    test_only_period = _daily_group_df(
        "2024-01-21", n_days=5, vegetable="RadishNew", market="Jaffna", price_start=200.0
    )
    merged = pd.concat([train_period, test_only_period], ignore_index=True)

    X_train, X_test, _, _ = builder.build(merged, test_size=0.2, drop_incomplete_rows=False)
    assert list(X_train.columns) == list(X_test.columns)


# 6. Target column is never accidentally included in X ---------------------------


def test_target_column_is_not_included_in_features(builder):
    merged = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    X_train, X_test, _, _ = builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    assert "price" not in X_train.columns
    assert "price" not in X_test.columns
    assert "date" not in X_train.columns
    assert "date" not in X_test.columns


# 7. No future information leaks into training features --------------------------


def test_no_future_information_leaks_into_training_features(builder):
    """Changing only the test-period prices must not change X_train/y_train at all."""
    baseline = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)

    perturbed = baseline.copy()
    is_future = perturbed["date"] >= "2024-01-16"  # the 5 rows that will land in the test split
    perturbed.loc[is_future, "price"] = perturbed.loc[is_future, "price"] + 100_000.0

    X_train_a, X_test_a, y_train_a, y_test_a = builder.build(baseline, test_size=0.25, drop_incomplete_rows=False)
    X_train_b, X_test_b, y_train_b, y_test_b = builder.build(perturbed, test_size=0.25, drop_incomplete_rows=False)

    pd.testing.assert_frame_equal(X_train_a, X_train_b)
    pd.testing.assert_series_equal(y_train_a, y_train_b)

    # sanity check the perturbation actually reached the test split, so the
    # assertions above are verifying leak-safety, not a no-op
    assert not y_test_a.equals(y_test_b)


# 8. No unexpected NaN in the output ------------------------------------------------


def test_no_unexpected_nan_with_default_drop_incomplete_rows(builder):
    group_a = _daily_group_df("2024-01-01", n_days=50, vegetable="Carrot", market="Dambulla", price_start=100.0)
    group_b = _daily_group_df("2024-01-01", n_days=50, vegetable="Tomato", market="Pettah", price_start=80.0)
    merged = pd.concat([group_a, group_b], ignore_index=True)

    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.2)  # drop_incomplete_rows defaults True

    assert len(X_train) > 0
    assert len(X_test) > 0
    assert not X_train.isna().any().any()
    assert not X_test.isna().any().any()
    assert not y_train.isna().any()
    assert not y_test.isna().any()


# 9. Existing errors from the composed pieces still propagate, not swallowed ------


def test_fitted_encoder_is_available_after_build_and_is_actually_fitted(builder):
    merged = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    encoder = builder.fitted_encoder
    assert hasattr(encoder, "encoder_")  # fitted state exists (see categorical_features.py)

    # the fitted encoder must be usable standalone, on new data, without refitting
    unseen = pd.DataFrame({"vegetable": ["Cabbage"], "market": ["Jaffna"], "price_lag_1d": [1.0]})
    encoded = encoder.transform(unseen)
    assert "vegetable_Cabbage" not in encoded.columns  # never learned -- would mean it got refit


def test_fitted_encoder_raises_before_build_is_called():
    builder = TrainTestDatasetBuilder(config_path="config/config.yaml")
    with pytest.raises(RuntimeError, match="build"):
        _ = builder.fitted_encoder


# 10. Fuel-price integration: retained lag, excluded raw, split/encoding unaffected -----------


def test_build_retains_fuel_lag_features_and_excludes_raw_fuel_columns(builder):
    merged = _daily_group_df_with_fuel(
        "2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0
    )
    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    assert "diesel_price_lag_1d" in X_train.columns
    assert "petrol_price_lag_1d" in X_train.columns
    assert "diesel_price_lag_1d" in X_test.columns
    assert "petrol_price_lag_1d" in X_test.columns

    assert "diesel_price" not in X_train.columns
    assert "petrol_price" not in X_train.columns
    assert "diesel_price" not in X_test.columns
    assert "petrol_price" not in X_test.columns


def test_build_chronological_order_respected_with_fuel_present(builder):
    merged = _daily_group_df_with_fuel(
        "2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0
    )
    X_train, X_test, y_train, y_test = builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    assert len(X_train) == len(y_train) == 15
    assert len(X_test) == len(y_test) == 5
    # price rises monotonically with date in this fixture, so train (older) < test (newer) --
    # i.e. train dates are strictly earlier than test dates, with no overlap.
    assert y_train.max() < y_test.min()


def test_build_uses_config_test_size_0_2_with_fuel_present(builder):
    merged = _daily_group_df_with_fuel(
        "2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0
    )
    X_train, X_test, y_train, y_test = builder.build(merged, drop_incomplete_rows=False)

    # config/config.yaml has model.test_size: 0.2 -- unaffected by fuel columns being present
    assert len(X_train) == len(y_train) == 16
    assert len(X_test) == len(y_test) == 4


def test_encoder_is_fitted_only_on_train_when_fuel_columns_are_present(builder):
    """Preprocessing (the categorical encoder) must still be fitted on train only, and fuel's
    numeric lag columns must pass through untouched rather than being treated as categorical."""
    merged = _daily_group_df_with_fuel(
        "2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0
    )
    builder.build(merged, test_size=0.25, drop_incomplete_rows=False)

    encoder = builder.fitted_encoder
    assert hasattr(encoder, "encoder_")
    assert "diesel_price_lag_1d" not in encoder.categorical_columns
    assert "petrol_price_lag_1d" not in encoder.categorical_columns


def test_build_raises_when_merged_df_is_missing_a_required_column(builder):
    merged = _daily_group_df("2024-01-01", n_days=20, vegetable="Carrot", market="Dambulla", price_start=100.0)
    merged = merged.drop(columns=["price"])

    with pytest.raises(FeatureEngineeringError, match="price"):
        builder.build(merged)
