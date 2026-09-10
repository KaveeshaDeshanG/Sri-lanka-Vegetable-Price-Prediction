"""Chronological (non-shuffled) train/test split for the time-series training dataset."""

import logging

import numpy as np
import pandas as pd

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class ChronologicalSplitError(Exception):
    """Raised when a chronological split cannot be performed safely or as requested."""


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float | None = None,
    date_column: str | None = None,
    config_path: str = "config/config.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, test) by date, oldest rows in train, newest in test.

    Not sklearn.model_selection.train_test_split: that function shuffles by
    default, and even with shuffle=False it splits on row *position*, not on
    the date column's actual values. This project's rows aren't guaranteed
    to already be in date order by the time they reach this function (Phase
    6/8 preserve whatever order their input had -- see lag_features.py's and
    rolling_features.py's own docstrings), so the split must be defined by
    date, not position: a position-based split on unsorted input would leak
    "future" rows into train and "past" rows into test.

    Design:
      1. Sort order is computed (via np.argsort, not df.sort_values()) with
         kind="stable", so two rows sharing the same date keep their
         original relative order rather than being reordered unpredictably.
         This resolves ties deterministically without introducing
         randomness -- there's no meaningful sub-day ordering to prefer.
      2. train/test *membership* is decided from that date order (oldest
         test_size-complement fraction -> train, newest -> test), but the
         rows returned in each half keep their ORIGINAL relative order from
         df, via boolean masking -- not the date-sorted order. If df is
         already date-sorted (the normal case, since
         transformer.sort_chronologically() runs upstream) these coincide;
         if it isn't, this still avoids silently reordering rows that
         weren't already in date order for some other reason.

    Args:
        test_size: fraction of rows (by date rank, not by calendar time
            span) held out for the test set. Must be in (0, 1) -- exclusive
            on both ends: 0 or 1 would produce a training or test set with
            no rows at all. Defaults to config's model.test_size when not
            given, so this one value isn't duplicated as a second hardcoded
            default here.
        date_column: defaults to config's data_schema.date_column -- the
            same column every other stage of this project already uses,
            not a new field invented for this function.
        config_path: only read when test_size or date_column is omitted.

    Raises:
        ChronologicalSplitError: if test_size is outside (0, 1), the date
            column is missing, any date is missing/unparseable, or the
            requested split would leave train or test with zero rows for a
            non-empty df.
    """
    resolved_test_size, resolved_date_column = _resolve_config_defaults(
        test_size, date_column, config_path
    )
    _validate_test_size(resolved_test_size)

    if resolved_date_column not in df.columns:
        message = f"chronological_train_test_split: missing required date column '{resolved_date_column}'"
        logger.error(message)
        raise ChronologicalSplitError(message)

    if df.empty:
        logger.info("chronological_train_test_split: input is empty; returning two empty frames")
        return df.copy().reset_index(drop=True), df.copy().reset_index(drop=True)

    dates = _coerce_and_validate_dates(df[resolved_date_column], resolved_date_column)

    row_count = len(df)
    train_count = int(np.floor(row_count * (1 - resolved_test_size)))
    test_count = row_count - train_count
    if train_count == 0 or test_count == 0:
        message = (
            f"chronological_train_test_split: {row_count} row(s) with test_size="
            f"{resolved_test_size} would leave an empty train or test set "
            f"(train={train_count}, test={test_count}); provide more rows or a "
            f"different test_size"
        )
        logger.error(message)
        raise ChronologicalSplitError(message)

    # Stable argsort: ties (duplicate dates) keep their original relative
    # order instead of being shuffled by the sort algorithm.
    date_rank_order = np.argsort(dates.to_numpy(), kind="stable")
    train_positions = date_rank_order[:train_count]

    train_mask = np.zeros(row_count, dtype=bool)
    train_mask[train_positions] = True

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    test_df = df.loc[~train_mask].copy().reset_index(drop=True)

    logger.info(
        "chronological_train_test_split: %d row(s) -> train=%d (%s to %s), test=%d (%s to %s)",
        row_count,
        len(train_df),
        train_df[resolved_date_column].min(),
        train_df[resolved_date_column].max(),
        len(test_df),
        test_df[resolved_date_column].min(),
        test_df[resolved_date_column].max(),
    )
    return train_df, test_df


def _resolve_config_defaults(
    test_size: float | None, date_column: str | None, config_path: str
) -> tuple[float, str]:
    if test_size is not None and date_column is not None:
        return test_size, date_column

    config = ConfigLoader(config_path)
    resolved_test_size = test_size if test_size is not None else config.get("model.test_size", 0.2)
    resolved_date_column = (
        date_column if date_column is not None else config.get("data_schema.date_column", "date")
    )
    return resolved_test_size, resolved_date_column


def _validate_test_size(test_size: float) -> None:
    if not isinstance(test_size, (int, float)) or isinstance(test_size, bool) or not (0 < test_size < 1):
        message = f"chronological_train_test_split: test_size must be in (0, 1), got {test_size!r}"
        logger.error(message)
        raise ChronologicalSplitError(message)


def _coerce_and_validate_dates(date_series: pd.Series, date_column: str) -> pd.Series:
    """Coerce to datetime if needed (defensive, matching date_features.py) and reject any NaT."""
    dates = date_series
    if not pd.api.types.is_datetime64_any_dtype(dates):
        dates = pd.to_datetime(dates, errors="coerce")

    missing_count = int(dates.isna().sum())
    if missing_count > 0:
        message = (
            f"chronological_train_test_split: column '{date_column}' has {missing_count} "
            "missing or unparseable date value(s); cannot determine a chronological order for them"
        )
        logger.error(message)
        raise ChronologicalSplitError(message)
    return dates
