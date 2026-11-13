"""Loads the persisted model + fitted feature-pipeline state, and predicts from prepared features."""

import logging
from dataclasses import dataclass

import pandas as pd
from sklearn.base import BaseEstimator

from src.feature_engineering.categorical_features import CategoricalEncoder
from src.models.artifact_metadata import (
    ArtifactConsistencyError,
    ArtifactMetadataNotFoundError,
    load_artifact_metadata,
    validate_artifact_consistency,
)
from src.models.persistence import load_model, save_model

logger = logging.getLogger(__name__)

DEFAULT_MODEL_FILENAME = "best_model.joblib"
DEFAULT_TRANSFORMATION_FILENAME = "feature_pipeline_state.joblib"

__all__ = [
    "ArtifactConsistencyError",
    "ModelArtifactNotFoundError",
    "TransformationArtifactNotFoundError",
    "FeatureAlignmentError",
    "InferenceArtifact",
    "save_inference_artifact",
    "load_inference_artifact",
    "Predictor",
]


class ModelArtifactNotFoundError(Exception):
    """Raised when the persisted trained model cannot be found."""


class TransformationArtifactNotFoundError(Exception):
    """Raised when the persisted fitted feature-pipeline state (encoder + feature columns) cannot be found."""


class FeatureAlignmentError(Exception):
    """Raised when a row's engineered features don't match what the model was trained on."""


@dataclass(frozen=True)
class InferenceArtifact:
    """The fitted, non-model state training produces that inference must reuse, never refit.

    categorical_encoder: the exact CategoricalEncoder instance
        TrainTestDatasetBuilder.build() fit on the training split (exposed
        via its `fitted_encoder` property). Refitting a new encoder on
        prediction input would learn a different -- and likely incomplete
        -- category vocabulary and column order than the model was trained
        against; that is exactly the leakage/inconsistency categorical_features.py's
        own docstring warns about, just at inference time instead of training time.
    feature_columns: the exact, ordered list of X_train's columns after
        encoding. Used to defensively re-align/validate prediction-time
        features before calling model.predict() -- most estimators don't
        verify column *order* by name, so a silent mismatch would feed the
        model's weights the wrong columns without raising anything.
    training_run_id: optional identifier shared with the ArtifactMetadata
        (src/models/artifact_metadata.py) a training run also persists.
        Defaults to None so every existing caller constructing
        InferenceArtifact(encoder, feature_columns) positionally is
        unaffected -- only src/models/pipeline.py's real training entry
        points set this, to let Predictor.load() detect a model file and a
        feature-pipeline-state file that were saved by two different
        training runs (e.g. a demo retrain overwriting one but not the
        other) rather than silently combining them.
    """

    categorical_encoder: CategoricalEncoder
    feature_columns: list[str]
    training_run_id: str | None = None


def save_inference_artifact(
    artifact: InferenceArtifact,
    filename: str = DEFAULT_TRANSFORMATION_FILENAME,
    output_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> str:
    """Persist the fitted feature-pipeline state, for later reuse by Predictor.load().

    Whatever trains and persists the model (see src/models) should also
    call this, passing InferenceArtifact(dataset_builder.fitted_encoder,
    list(X_train.columns)) -- there is no other correct source for either
    value. output_dir overrides config's paths.models_dir, exactly like
    save_model()'s own parameter of the same name -- used by demo/sandbox
    training runs that must never write into the production models directory.
    """
    return save_model(artifact, filename=filename, output_dir=output_dir, config_path=config_path)


def load_inference_artifact(
    filename: str = DEFAULT_TRANSFORMATION_FILENAME,
    input_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> InferenceArtifact:
    """Load the fitted feature-pipeline state saved by save_inference_artifact().

    Raises:
        TransformationArtifactNotFoundError: if the artifact file doesn't exist.
    """
    try:
        return load_model(filename=filename, input_dir=input_dir, config_path=config_path)
    except (FileNotFoundError, OSError) as error:
        raise TransformationArtifactNotFoundError(
            f"Predictor: no fitted feature-pipeline state found at '{filename}' under "
            "config's paths.models_dir -- train and persist one (TrainTestDatasetBuilder's "
            "fitted_encoder + save_inference_artifact()) before predicting"
        ) from error


class Predictor:
    """Wraps a trained model and the fitted feature-pipeline state it was trained with.

    Given an already feature-engineered (but not yet encoded) DataFrame --
    i.e. FeatureEngineer.build_features() + split_features_target() output,
    not raw input -- encodes it with the SAME fitted CategoricalEncoder used
    at training time and predicts. This class never fits anything and never
    sees a train/test split; it only consumes already-fitted state.
    """

    def __init__(self, model: BaseEstimator, inference_artifact: InferenceArtifact) -> None:
        self._model = model
        self._artifact = inference_artifact

    @classmethod
    def load(
        cls,
        config_path: str = "config/config.yaml",
        model_filename: str = DEFAULT_MODEL_FILENAME,
        transformation_filename: str = DEFAULT_TRANSFORMATION_FILENAME,
        models_dir: str | None = None,
    ) -> "Predictor":
        """Load the persisted best model and its fitted feature-pipeline state.

        models_dir overrides config's paths.models_dir for both files (and for the
        artifact metadata check below) -- used to load a demo/sandbox predictor from
        its own separate directory (see app/services.py::train_demo_model()) without
        ever touching the production models directory.

        If an ArtifactMetadata file (src/models/artifact_metadata.py) is present
        alongside the two artifacts, the loaded model and InferenceArtifact are
        validated against it before returning -- catching a model file and a
        feature-pipeline-state file that were saved by two different training runs.
        Older artifact sets with no metadata file are loaded as before, unvalidated.

        Raises:
            ModelArtifactNotFoundError: if the model file doesn't exist.
            TransformationArtifactNotFoundError: if the feature-pipeline
                state file doesn't exist.
            ArtifactConsistencyError: if a metadata file exists but disagrees
                with the loaded model and/or feature-pipeline state.
        """
        try:
            model = load_model(filename=model_filename, input_dir=models_dir, config_path=config_path)
        except (FileNotFoundError, OSError) as error:
            raise ModelArtifactNotFoundError(
                f"Predictor: no trained model found at '{model_filename}' under config's "
                "paths.models_dir -- train and persist one with ModelTrainingPipeline first"
            ) from error

        artifact = load_inference_artifact(
            filename=transformation_filename, input_dir=models_dir, config_path=config_path
        )

        try:
            metadata = load_artifact_metadata(input_dir=models_dir, config_path=config_path)
        except ArtifactMetadataNotFoundError:
            logger.info("Predictor.load: no artifact metadata found -- skipping consistency validation")
        else:
            validate_artifact_consistency(model, artifact, metadata)

        return cls(model, artifact)

    @property
    def feature_columns(self) -> list[str]:
        return list(self._artifact.feature_columns)

    def known_vegetables(self) -> list[str]:
        """The vegetables training saw."""
        encoder = self._artifact.categorical_encoder
        return list(dict(zip(encoder.categorical_columns, encoder.encoder_.categories_))["vegetable"])

    def unseen_categories(self, **categories: str) -> list[str]:
        """Which of the given categories (e.g. vegetable="Drumstick") training never saw.

        The encoder turns such a value into all zeros rather than failing, so the model
        still returns a number, with nothing vegetable-specific behind it. Callers should
        warn about any value returned here.
        """
        encoder = self._artifact.categorical_encoder
        known = dict(zip(encoder.categorical_columns, encoder.encoder_.categories_))
        return [value for column, value in categories.items() if column in known and value not in known[column]]

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Encode X with the training-fitted encoder, align to the training feature schema, and predict.

        Never fits the encoder -- a category unseen during training is
        encoded as all-zero by the encoder's existing
        handle_unknown="ignore" behavior, not learned as a new column here.

        Raises:
            FeatureAlignmentError: if a required feature column is missing
                or NaN after encoding+alignment. This means the input's
                historical context was insufficient to compute a required
                lag/rolling/weather-lag feature -- it is never papered over
                with a fabricated value (some models, notably XGBoost, would
                otherwise silently "predict" from NaN input via their own
                missing-value handling).
        """
        encoded = self._artifact.categorical_encoder.transform(X)
        aligned = encoded.reindex(columns=self._artifact.feature_columns)

        incomplete_rows = aligned.isna().any(axis=1)
        if incomplete_rows.any():
            incomplete_columns = sorted(aligned.columns[aligned.loc[incomplete_rows].isna().any()])
            raise FeatureAlignmentError(
                f"Predictor: cannot predict for row(s) with index {list(aligned.index[incomplete_rows])} "
                f"-- required feature(s) {incomplete_columns} are missing or NaN after encoding. This "
                "usually means the input's historical context doesn't reach far enough back to compute "
                "a lag/rolling/weather-lag feature (see config.feature_engineering for the required "
                "windows); Predictor never substitutes a fabricated value for it."
            )

        predictions = self._model.predict(aligned)
        logger.info("Predictor: predicted %d row(s)", len(aligned))
        return pd.Series(predictions, index=X.index, name="prediction")
