"""One-hot encoding for nominal categorical features (vegetable, market)."""

import logging

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


class CategoricalFeatureEncodingError(Exception):
    """Raised when a required categorical column is missing from the input DataFrame."""


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """One-hot encodes nominal categorical columns (vegetable, market by default).

    Why one-hot, not label/integer encoding: vegetable and market are
    nominal categories with no natural order. Assigning "Carrot"=0,
    "Tomato"=1, "Brinjal"=2 would hand a linear model a fabricated ordinal
    relationship, and even a tree model a fabricated notion of "distance"
    between categories that isn't real. Config has no evidence of a huge
    category vocabulary (no vegetable/market list runs into the hundreds),
    so the extra columns one-hot encoding produces are an acceptable
    trade-off for not inventing an ordering that doesn't exist.

    handle_unknown="ignore": the real vegetable/market vocabulary isn't
    known yet -- no real HARTI/Dambulla data has been loaded, and
    transformer.py's alias tables are explicitly marked illustrative, not
    validated against real exports (see src/preprocessing/transformer.py).
    A vegetable or market seen at prediction time but not during training
    must not crash the pipeline; handle_unknown="ignore" encodes it as all
    zeros across that column's one-hot columns instead of raising.

    sklearn-pipeline compatible on purpose: inherits BaseEstimator +
    TransformerMixin (get_params/set_params/clone, and a free
    fit_transform() that just calls fit().transform()), so this can be
    dropped into a Pipeline/ColumnTransformer, or joblib.dump()'d directly --
    see src/prediction/predictor.py's InferenceArtifact, which does exactly
    the latter.

    Deliberately NOT wired into FeatureEngineer.build_features() or
    split_features_target(): those run on the full historical dataset
    before any train/test split exists. Fitting a OneHotEncoder there would
    fit it on the union of what will later become train and test data --
    exactly the kind of leakage this class exists to avoid. It is instead
    fit only on the training split, by TrainTestDatasetBuilder.build() (see
    src/feature_engineering/dataset_builder.py), the composing layer that
    owns the split and is therefore the only correct place to fit this.
    """

    def __init__(self, categorical_columns: tuple[str, ...] = ("vegetable", "market")) -> None:
        # sklearn convention: __init__ only stores constructor args verbatim,
        # with matching attribute names -- get_params()/set_params()/clone()
        # (used by e.g. GridSearchCV in a later phase) depend on this.
        self.categorical_columns = categorical_columns

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        """Learn the category vocabulary from X. Call this on a TRAINING split only.

        Raises:
            CategoricalFeatureEncodingError: if a configured categorical
                column is missing from X.
        """
        self._validate_required_columns(X)
        self.encoder_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.encoder_.fit(X[list(self.categorical_columns)])
        logger.info(
            "CategoricalEncoder: fitted on %d row(s); categories learned: %s",
            len(X),
            dict(zip(self.categorical_columns, (list(c) for c in self.encoder_.categories_))),
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Replace X's categorical columns with one-hot columns; pass everything else through.

        Output column order is always: passthrough columns (in X's own
        order), then one-hot columns (in the order learned at fit time).
        This is identical on every call once fitted -- including a single
        unseen row at prediction time -- so it stays aligned with whatever
        a model was trained on.

        Raises:
            NotFittedError: if called before fit().
            CategoricalFeatureEncodingError: if a configured categorical
                column is missing from X.
        """
        check_is_fitted(self, "encoder_")
        self._validate_required_columns(X)

        categorical_columns = list(self.categorical_columns)
        encoded_column_names = self.encoder_.get_feature_names_out(categorical_columns)

        # sklearn's OneHotEncoder.transform() itself rejects a 0-row input
        # ("Found array with 0 sample(s)"), unlike every other generator in
        # this package, which returns a correctly-shaped empty result
        # instead of raising. Matching that established, more lenient
        # behavior here -- an empty prediction batch should not crash.
        if len(X) == 0:
            encoded_df = pd.DataFrame(columns=encoded_column_names, dtype="float64", index=X.index)
        else:
            encoded = self.encoder_.transform(X[categorical_columns])
            encoded_df = pd.DataFrame(encoded, columns=encoded_column_names, index=X.index)

        passthrough_columns = [c for c in X.columns if c not in categorical_columns]
        return pd.concat([X[passthrough_columns], encoded_df], axis=1)

    def _validate_required_columns(self, X: pd.DataFrame) -> None:
        missing = [c for c in self.categorical_columns if c not in X.columns]
        if missing:
            message = f"CategoricalEncoder: missing required categorical column(s): {missing}"
            logger.error(message)
            raise CategoricalFeatureEncodingError(message)
