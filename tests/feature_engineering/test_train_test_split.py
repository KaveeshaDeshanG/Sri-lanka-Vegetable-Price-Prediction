"""Unit tests for src.feature_engineering.train_test_split.chronological_train_test_split."""

import pandas as pd
import pytest

from src.feature_engineering.train_test_split import ChronologicalSplitError, chronological_train_test_split


def _sequential_df(n: int) -> pd.DataFrame:
    """n rows dated 2024-01-01, 2024-01-02, ... in that same (already sorted) order."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "value": list(range(n)),
        }
    )


# 1. Normal 80/20 chronological split -----------------------------------------


def test_default_80_20_split_sizes():
    df = _sequential_df(10)
    train, test = chronological_train_test_split(df, test_size=0.2)
    assert len(train) == 8
    assert len(test) == 2


# 2. Oldest rows in train, newest rows in test --------------------------------


def test_oldest_rows_go_to_train_and_newest_to_test():
    df = _sequential_df(10)
    train, test = chronological_train_test_split(df, test_size=0.2)
    assert train["date"].max() < test["date"].min()
    assert list(train["value"]) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert list(test["value"]) == [8, 9]


# 3. Original row order is preserved (using an input that is NOT date-sorted) --


def test_original_row_order_is_preserved_even_when_input_is_unsorted():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-05", "2024-01-01", "2024-01-03", "2024-01-02", "2024-01-04", "2024-01-09", "2024-01-10"]
            ),
            "value": [5, 1, 3, 2, 4, 9, 10],
        }
    )
    train, test = chronological_train_test_split(df, test_size=0.2)
    # membership is correct by date (the two newest, 9 and 10, are held out)...
    assert sorted(train["value"]) == [1, 2, 3, 4, 5]
    assert sorted(test["value"]) == [9, 10]
    # ...but each half keeps its ORIGINAL relative order, not date-sorted order
    assert list(train["value"]) == [5, 1, 3, 2, 4]
    assert list(test["value"]) == [9, 10]


# 4. Input DataFrame is not mutated --------------------------------------------


def test_input_dataframe_is_not_mutated():
    df = _sequential_df(10)
    original = df.copy()
    chronological_train_test_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(df, original)


# 5. test_size = 0.2 (explicit and via config default) -------------------------


def test_test_size_0_2_explicit():
    df = _sequential_df(20)
    train, test = chronological_train_test_split(df, test_size=0.2)
    assert len(test) == 4
    assert len(train) == 16


def test_test_size_defaults_to_config_value_when_omitted():
    df = _sequential_df(20)
    # config/config.yaml has model.test_size: 0.2 -- same result as passing 0.2 explicitly
    train, test = chronological_train_test_split(df, config_path="config/config.yaml")
    assert len(test) == 4
    assert len(train) == 16


# 6. Different valid test sizes -------------------------------------------------


@pytest.mark.parametrize("test_size,expected_train,expected_test", [(0.1, 9, 1), (0.3, 7, 3), (0.5, 5, 5)])
def test_different_valid_test_sizes(test_size, expected_train, expected_test):
    df = _sequential_df(10)
    train, test = chronological_train_test_split(df, test_size=test_size)
    assert len(train) == expected_train
    assert len(test) == expected_test


# 7 & 8. Invalid test_size ------------------------------------------------------


@pytest.mark.parametrize("bad_test_size", [0, -0.1, 1, 1.5])
def test_invalid_test_size_is_rejected(bad_test_size):
    df = _sequential_df(10)
    with pytest.raises(ChronologicalSplitError, match="test_size"):
        chronological_train_test_split(df, test_size=bad_test_size)


# 9. Small dataset behavior -----------------------------------------------------


def test_small_dataset_that_still_yields_both_splits():
    df = _sequential_df(2)
    train, test = chronological_train_test_split(df, test_size=0.2)
    assert len(train) == 1
    assert len(test) == 1


def test_single_row_dataset_raises_rather_than_returning_an_empty_train_set():
    df = _sequential_df(1)
    with pytest.raises(ChronologicalSplitError, match="empty"):
        chronological_train_test_split(df, test_size=0.2)


def test_empty_dataframe_returns_two_empty_frames_without_raising():
    df = _sequential_df(0)
    train, test = chronological_train_test_split(df, test_size=0.2)
    assert len(train) == 0
    assert len(test) == 0
    assert list(train.columns) == list(df.columns)


# 10. Missing/invalid date column behavior --------------------------------------


def test_missing_date_column_raises():
    df = _sequential_df(10).drop(columns=["date"])
    with pytest.raises(ChronologicalSplitError, match="date"):
        chronological_train_test_split(df, test_size=0.2)


def test_unparseable_date_values_raise():
    df = pd.DataFrame({"date": ["2024-01-01", "not-a-date", "2024-01-03"], "value": [1, 2, 3]})
    with pytest.raises(ChronologicalSplitError, match="unparseable"):
        chronological_train_test_split(df, test_size=0.3)


def test_missing_date_values_raise():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", None, "2024-01-03"]), "value": [1, 2, 3]})
    with pytest.raises(ChronologicalSplitError, match="missing"):
        chronological_train_test_split(df, test_size=0.3)


def test_custom_date_column_name_is_respected():
    df = pd.DataFrame(
        {"report_date": pd.date_range("2024-01-01", periods=10, freq="D"), "value": list(range(10))}
    )
    train, test = chronological_train_test_split(df, test_size=0.2, date_column="report_date")
    assert len(train) == 8
    assert len(test) == 2


# 11. Duplicate dates handled deterministically, no random reordering ----------


def test_duplicate_dates_are_split_deterministically_and_repeatably():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03"]
            ),
            "value": [0, 1, 2, 3, 4, 5],
        }
    )
    train_a, test_a = chronological_train_test_split(df, test_size=0.2)
    train_b, test_b = chronological_train_test_split(df, test_size=0.2)
    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a, test_b)
    # ties on 2024-01-01 keep their original relative order (0 before 1 before 2)
    assert list(train_a.loc[train_a["date"] == "2024-01-01", "value"]) == [0, 1, 2]


# 12. Compatible with FeatureEngineer output ------------------------------------


def test_split_composes_with_feature_engineer_output():
    from src.feature_engineering.feature_engineer import FeatureEngineer

    merged = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "vegetable": ["Carrot"] * 12,
            "market": ["Dambulla"] * 12,
            "price": [float(100 + i) for i in range(12)],
        }
    )
    fe = FeatureEngineer(config_path="config/config.yaml")
    featured = fe.build_features(merged)

    train_df, test_df = chronological_train_test_split(featured, test_size=0.25, config_path="config/config.yaml")
    assert len(train_df) == 9
    assert len(test_df) == 3
    assert train_df["date"].max() < test_df["date"].min()

    X_train, y_train = fe.split_features_target(train_df, drop_incomplete_rows=False)
    X_test, y_test = fe.split_features_target(test_df, drop_incomplete_rows=False)
    assert len(X_train) == len(y_train) == 9
    assert len(X_test) == len(y_test) == 3


def test_split_composes_with_feature_engineer_output_including_fuel():
    """Same composition as above, but with fuel-price columns present: the split itself must
    stay chronological/non-overlapping, and fuel lag features must survive while raw fuel
    columns are excluded -- exactly like weather."""
    from src.feature_engineering.feature_engineer import FeatureEngineer

    merged = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "vegetable": ["Carrot"] * 12,
            "market": ["Dambulla"] * 12,
            "price": [float(100 + i) for i in range(12)],
            "diesel_price": [300.0 + i for i in range(12)],
            "petrol_price": [350.0 + i for i in range(12)],
        }
    )
    fe = FeatureEngineer(config_path="config/config.yaml")
    featured = fe.build_features(merged)

    train_df, test_df = chronological_train_test_split(
        featured, test_size=0.25, config_path="config/config.yaml"
    )
    assert len(train_df) == 9
    assert len(test_df) == 3
    assert train_df["date"].max() < test_df["date"].min()  # no overlap, train strictly earlier

    X_train, y_train = fe.split_features_target(train_df, drop_incomplete_rows=False)
    X_test, y_test = fe.split_features_target(test_df, drop_incomplete_rows=False)
    assert len(X_train) == len(y_train) == 9
    assert len(X_test) == len(y_test) == 3

    assert "diesel_price_lag_1d" in X_train.columns
    assert "diesel_price_lag_1d" in X_test.columns
    assert "petrol_price_lag_1d" in X_train.columns
    assert "petrol_price_lag_1d" in X_test.columns
    assert "diesel_price" not in X_train.columns
    assert "diesel_price" not in X_test.columns
    assert "petrol_price" not in X_train.columns
    assert "petrol_price" not in X_test.columns
