"""Unit and integration tests for src.feature_engineering.categorical_features.CategoricalEncoder."""

import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from src.feature_engineering.categorical_features import CategoricalEncoder, CategoricalFeatureEncodingError
from src.feature_engineering.feature_engineer import FeatureEngineer


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "vegetable": ["Carrot", "Tomato", "Carrot", "Brinjal"],
            "market": ["Dambulla", "Pettah", "Pettah", "Dambulla"],
            "price_lag_7d": [100.0, 90.0, 105.0, 60.0],
            "price": [102.0, 91.0, 104.0, 61.0],
        }
    )


@pytest.fixture
def encoder() -> CategoricalEncoder:
    return CategoricalEncoder()


# 1 & 2. Known vegetable / known market encoding -----------------------------


def test_known_vegetable_is_encoded_as_a_one_hot_column(encoder):
    encoded = encoder.fit_transform(_train_df())
    assert encoded.loc[0, "vegetable_Carrot"] == 1.0
    assert encoded.loc[0, "vegetable_Tomato"] == 0.0
    assert encoded.loc[0, "vegetable_Brinjal"] == 0.0


def test_known_market_is_encoded_as_a_one_hot_column(encoder):
    encoded = encoder.fit_transform(_train_df())
    assert encoded.loc[0, "market_Dambulla"] == 1.0
    assert encoded.loc[0, "market_Pettah"] == 0.0


# 3. Multiple categories ------------------------------------------------------


def test_multiple_distinct_categories_each_get_their_own_column(encoder):
    encoded = encoder.fit_transform(_train_df())
    vegetable_columns = [c for c in encoded.columns if c.startswith("vegetable_")]
    market_columns = [c for c in encoded.columns if c.startswith("market_")]
    assert sorted(vegetable_columns) == ["vegetable_Brinjal", "vegetable_Carrot", "vegetable_Tomato"]
    assert sorted(market_columns) == ["market_Dambulla", "market_Pettah"]
    # every row is a valid one-hot vector: exactly one 1 per category group
    assert (encoded[vegetable_columns].sum(axis=1) == 1).all()
    assert (encoded[market_columns].sum(axis=1) == 1).all()


# 4 & 5. Unknown vegetable / unknown market -----------------------------------


def test_unknown_vegetable_at_transform_time_does_not_crash_and_is_all_zero(encoder):
    encoder.fit(_train_df())
    unseen = pd.DataFrame({"vegetable": ["Cabbage"], "market": ["Dambulla"], "price_lag_7d": [50.0]})
    encoded = encoder.transform(unseen)
    vegetable_columns = [c for c in encoded.columns if c.startswith("vegetable_")]
    assert (encoded[vegetable_columns] == 0).all(axis=None)


def test_unknown_market_at_transform_time_does_not_crash_and_is_all_zero(encoder):
    encoder.fit(_train_df())
    unseen = pd.DataFrame({"vegetable": ["Carrot"], "market": ["Jaffna"], "price_lag_7d": [50.0]})
    encoded = encoder.transform(unseen)
    market_columns = [c for c in encoded.columns if c.startswith("market_")]
    assert (encoded[market_columns] == 0).all(axis=None)


# 6. Consistent output columns -------------------------------------------------


def test_output_columns_are_identical_between_training_and_prediction_transforms(encoder):
    train_encoded = encoder.fit_transform(_train_df())
    predict_df = pd.DataFrame(
        {"vegetable": ["Cabbage"], "market": ["Jaffna"], "price_lag_7d": [70.0], "price": [999.0]}
    )
    predict_encoded = encoder.transform(predict_df)
    assert list(train_encoded.columns) == list(predict_encoded.columns)


# 7. Fit/transform separation ---------------------------------------------------


def test_transform_before_fit_raises_not_fitted_error(encoder):
    with pytest.raises(NotFittedError):
        encoder.transform(_train_df())


def test_fit_returns_self_and_does_not_transform(encoder):
    result = encoder.fit(_train_df())
    assert result is encoder
    assert hasattr(encoder, "encoder_")  # fitted state now exists


# 8. Transforming prediction data using the already-fitted encoder -------------


def test_transform_reuses_the_fitted_vocabulary_not_a_new_one(encoder):
    encoder.fit(_train_df())
    # a category present in training but NOT in this new batch must still get its own column
    single_row = pd.DataFrame({"vegetable": ["Carrot"], "market": ["Pettah"], "price_lag_7d": [1.0]})
    encoded = encoder.transform(single_row)
    assert "vegetable_Tomato" in encoded.columns  # learned at fit time, not present in this batch
    assert "vegetable_Brinjal" in encoded.columns
    assert encoded.loc[0, "vegetable_Tomato"] == 0.0
    assert encoded.loc[0, "vegetable_Carrot"] == 1.0


# 9. Missing required categorical columns ---------------------------------------


def test_fit_raises_when_a_categorical_column_is_missing(encoder):
    with pytest.raises(CategoricalFeatureEncodingError, match="market"):
        encoder.fit(_train_df().drop(columns=["market"]))


def test_transform_raises_when_a_categorical_column_is_missing(encoder):
    encoder.fit(_train_df())
    with pytest.raises(CategoricalFeatureEncodingError, match="vegetable"):
        encoder.transform(_train_df().drop(columns=["vegetable"]))


# 10. Empty input ------------------------------------------------------------------


def test_transform_on_zero_rows_after_fit_returns_empty_with_correct_columns(encoder):
    encoder.fit(_train_df())
    empty = _train_df().iloc[0:0]
    encoded = encoder.transform(empty)
    assert len(encoded) == 0
    assert "vegetable_Carrot" in encoded.columns
    assert "market_Dambulla" in encoded.columns


def test_fit_on_zero_rows_raises_rather_than_learning_no_categories(encoder):
    empty = _train_df().iloc[0:0]
    with pytest.raises(ValueError):
        encoder.fit(empty)


# 11. No accidental target leakage -------------------------------------------------


def test_target_column_passes_through_completely_unchanged(encoder):
    df = _train_df()
    encoded = encoder.fit_transform(df)
    assert list(encoded["price"]) == list(df["price"])  # untouched, not derived from encoding
    new_columns = set(encoded.columns) - set(df.columns)
    assert all(c.startswith("vegetable_") or c.startswith("market_") for c in new_columns)


def test_encoder_never_touches_non_categorical_columns(encoder):
    df = _train_df()
    encoded = encoder.fit_transform(df)
    assert list(encoded["price_lag_7d"]) == list(df["price_lag_7d"])


# 12. Compatibility with the current FeatureEngineer output ------------------------


def test_encoder_is_compatible_with_feature_engineer_output():
    merged = pd.DataFrame(
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
    fe = FeatureEngineer(config_path="config/config.yaml")
    featured = fe.build_features(merged)
    X, y = fe.split_features_target(featured, drop_incomplete_rows=False)

    encoder = CategoricalEncoder()
    X_encoded = encoder.fit_transform(X)

    assert "vegetable" not in X_encoded.columns
    assert "market" not in X_encoded.columns
    assert "vegetable_Carrot" in X_encoded.columns
    assert "market_Dambulla" in X_encoded.columns
    assert "price_lag_7d" in X_encoded.columns  # numeric engineered features pass through
    assert len(X_encoded) == len(X) == len(y)
