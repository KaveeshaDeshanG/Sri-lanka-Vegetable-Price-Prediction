"""Unit tests for src.models.artifact_metadata: the model/encoder/feature-list consistency check."""

from dataclasses import dataclass

import pytest

from src.models.artifact_metadata import (
    ArtifactConsistencyError,
    ArtifactMetadata,
    ArtifactMetadataNotFoundError,
    load_artifact_metadata,
    new_training_run_id,
    save_artifact_metadata,
    validate_artifact_consistency,
)


class _FakeModel:
    """Stands in for a real fitted estimator -- only its class name matters here."""


@dataclass(frozen=True)
class _FakeInferenceArtifact:
    feature_columns: list
    training_run_id: str | None = None


def _metadata(**overrides) -> ArtifactMetadata:
    defaults = dict(
        artifact_version="1",
        training_run_id="run-123",
        trained_at="2026-08-20T00:00:00",
        model_type="_FakeModel",
        model_name="fake_model",
        training_data={"source": "test", "row_count": 10},
        feature_count=2,
        feature_columns=["a", "b"],
        metrics={"mae": 1.0, "rmse": 1.5, "r2": 0.9},
    )
    defaults.update(overrides)
    return ArtifactMetadata(**defaults)


# 1. new_training_run_id ------------------------------------------------------------


def test_new_training_run_id_returns_distinct_strings():
    first = new_training_run_id()
    second = new_training_run_id()
    assert isinstance(first, str) and first
    assert first != second


# 2. save/load round trip -----------------------------------------------------------


def test_save_and_load_artifact_metadata_round_trips(tmp_path):
    metadata = _metadata()
    save_artifact_metadata(metadata, output_dir=str(tmp_path))

    loaded = load_artifact_metadata(input_dir=str(tmp_path))
    assert loaded == metadata


def test_load_artifact_metadata_raises_when_missing(tmp_path):
    with pytest.raises(ArtifactMetadataNotFoundError):
        load_artifact_metadata(input_dir=str(tmp_path))


# 3. validate_artifact_consistency ---------------------------------------------------


def test_validate_artifact_consistency_passes_for_a_genuinely_matching_set():
    metadata = _metadata()
    model = _FakeModel()
    artifact = _FakeInferenceArtifact(feature_columns=["a", "b"], training_run_id="run-123")

    validate_artifact_consistency(model, artifact, metadata)  # must not raise


def test_validate_artifact_consistency_raises_on_training_run_id_mismatch():
    metadata = _metadata(training_run_id="run-123")
    model = _FakeModel()
    artifact = _FakeInferenceArtifact(feature_columns=["a", "b"], training_run_id="run-999")

    with pytest.raises(ArtifactConsistencyError, match="training_run_id"):
        validate_artifact_consistency(model, artifact, metadata)


def test_validate_artifact_consistency_raises_on_model_type_mismatch():
    class _OtherModel:
        pass

    metadata = _metadata()
    artifact = _FakeInferenceArtifact(feature_columns=["a", "b"], training_run_id="run-123")

    with pytest.raises(ArtifactConsistencyError, match="model_type"):
        validate_artifact_consistency(_OtherModel(), artifact, metadata)


def test_validate_artifact_consistency_raises_on_feature_columns_mismatch():
    metadata = _metadata(feature_columns=["a", "b"], feature_count=2)
    model = _FakeModel()
    artifact = _FakeInferenceArtifact(feature_columns=["a", "c"], training_run_id="run-123")

    with pytest.raises(ArtifactConsistencyError, match="feature_columns"):
        validate_artifact_consistency(model, artifact, metadata)


def test_validate_artifact_consistency_raises_when_feature_column_order_differs():
    """Order matters -- Predictor reindexes onto this exact list, so a permuted
    match must be rejected even though every column name is present."""
    metadata = _metadata(feature_columns=["a", "b"], feature_count=2)
    model = _FakeModel()
    artifact = _FakeInferenceArtifact(feature_columns=["b", "a"], training_run_id="run-123")

    with pytest.raises(ArtifactConsistencyError, match="feature_columns"):
        validate_artifact_consistency(model, artifact, metadata)


def test_validate_artifact_consistency_skips_run_id_check_for_older_artifacts_without_one():
    """An InferenceArtifact saved before training_run_id existed has training_run_id=None --
    that alone must not be treated as a mismatch; model_type/feature_columns are still checked."""
    metadata = _metadata()
    model = _FakeModel()
    artifact = _FakeInferenceArtifact(feature_columns=["a", "b"], training_run_id=None)

    validate_artifact_consistency(model, artifact, metadata)  # must not raise
