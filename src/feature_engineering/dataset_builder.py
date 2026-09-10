"""Production integration: FeatureEngineer -> chronological split -> leakage-safe categorical encoding."""

import logging

import pandas as pd

from .categorical_features import CategoricalEncoder
from .feature_engineer import FeatureEngineer
from .train_test_split import chronological_train_test_split

logger = logging.getLogger(__name__)


class TrainTestDatasetBuilder:
    """Turns a merged market+weather DataFrame into a leakage-safe, model-ready train/test dataset.

    Composes three already-independent, already-tested pieces in the exact
    sequence categorical_features.py's and feature_engineer.py's docstrings
    describe as the intended call order:

      1. FeatureEngineer.build_features() -- adds date/lag/rolling/weather-lag
         features to the *full* historical DataFrame, before any split. This
         must happen first: lag/rolling features for the earliest test-period
         rows legitimately depend on train-period history (see
         lag_features.py/rolling_features.py), the same way a deployed model
         would have access to all prior history at prediction time. Every one
         of those generators only ever looks backward in time (exact-day lag
         join, closed="left" rolling, backward weather lag), so building
         features before the split cannot leak a test-period value into any
         train-period row's features.
      2. chronological_train_test_split() -- splits the *already-featured*
         DataFrame by date, oldest rows to train, newest to test.
      3. FeatureEngineer.split_features_target() -- applied independently to
         each split, producing (X, y) with the target/date/source/raw-weather
         columns excluded, exactly as it already does for a single DataFrame.
         Applying it per-split (rather than splitting X/y first) means a row
         dropped from one split for incomplete features never affects the
         other split.
      4. CategoricalEncoder -- fit ONLY on X_train (never on X_test, never on
         the two splits combined), then used to transform both X_train and
         X_test. This is the one step this class exists to add: fitting a
         one-hot encoder on the full dataset before splitting would leak the
         test set's category vocabulary into training, exactly what
         categorical_features.py's own docstring warns against. A category
         present only in the test split is therefore never learned and is
         encoded as all-zero there via the encoder's existing
         handle_unknown="ignore" behavior, not specially handled here.

    None of the four steps' own logic is reimplemented here -- this class
    only sequences existing, already-tested components.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = config_path
        self._feature_engineer = FeatureEngineer(config_path=config_path)
        self._fitted_encoder: CategoricalEncoder | None = None

    @property
    def fitted_encoder(self) -> CategoricalEncoder:
        """The CategoricalEncoder fit during the most recent build() call.

        Exists so a caller that also needs to persist this state for later
        inference (see src/prediction) doesn't have to fit its own encoder
        on prediction data -- reusing this exact fitted instance is the only
        correct way to encode new prediction input the same way training
        data was encoded (see categorical_features.py's leakage warning).

        Raises:
            RuntimeError: if build() has not been called yet.
        """
        if self._fitted_encoder is None:
            raise RuntimeError(
                "TrainTestDatasetBuilder: build() must be called before fitted_encoder is available."
            )
        return self._fitted_encoder

    def build(
        self,
        merged_df: pd.DataFrame,
        test_size: float | None = None,
        drop_incomplete_rows: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Build a leakage-safe (X_train, X_test, y_train, y_test) from a merged DataFrame.

        Args:
            merged_df: preprocessing's merged market+weather output, not yet
                feature-engineered or split.
            test_size: forwarded to chronological_train_test_split(); defaults
                to config's model.test_size when omitted.
            drop_incomplete_rows: forwarded to
                FeatureEngineer.split_features_target(); see that method's
                docstring. Applied independently to the train and test
                splits, after splitting, so a row dropped from one split
                never affects the other.

        Returns:
            (X_train, X_test, y_train, y_test), matching the same argument
            order sklearn.model_selection.train_test_split returns. X_train
            and X_test always have identical columns, in identical order.

        Raises:
            FeatureEngineeringError: if merged_df is missing a column
                build_features() requires.
            ChronologicalSplitError: if the requested split can't be
                performed (e.g. too few rows, bad test_size, bad dates).
            CategoricalFeatureEncodingError: if the vegetable/market columns
                are missing after split_features_target().
            ValueError: if the training split ends up with zero rows --
                CategoricalEncoder.fit() cannot learn a vocabulary from no
                data (see categorical_features.py).
        """
        featured_df = self._feature_engineer.build_features(merged_df)

        train_df, test_df = chronological_train_test_split(
            featured_df, test_size=test_size, config_path=self._config_path
        )

        X_train, y_train = self._feature_engineer.split_features_target(
            train_df, drop_incomplete_rows=drop_incomplete_rows
        )
        X_test, y_test = self._feature_engineer.split_features_target(
            test_df, drop_incomplete_rows=drop_incomplete_rows
        )

        encoder = CategoricalEncoder()
        X_train_encoded = encoder.fit_transform(X_train)
        X_test_encoded = encoder.transform(X_test)
        self._fitted_encoder = encoder

        logger.info(
            "TrainTestDatasetBuilder: built train=%d row(s)/%d column(s), "
            "test=%d row(s)/%d column(s)",
            len(X_train_encoded),
            X_train_encoded.shape[1],
            len(X_test_encoded),
            X_test_encoded.shape[1],
        )
        return X_train_encoded, X_test_encoded, y_train, y_test
