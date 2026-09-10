"""Persists a small JSON side-car tying model + encoder + feature list to one training run.

Two independent joblib files (best_model.joblib, feature_pipeline_state.joblib) can
individually be missing, stale, or -- as happened once in this project's history --
overwritten by an unrelated training run (a demo/sandbox run reusing the production
filename), leaving a *structurally valid but internally inconsistent* pair: a model
fit on one feature schema loaded next to an encoder/feature list from another. Neither
half raises on its own; only a joint consistency check catches it.

This module is that check. Every production training run writes one ArtifactMetadata
JSON alongside the model + inference artifact it describes; Predictor.load() validates
the loaded model and InferenceArtifact against it before returning. Deliberately a
plain JSON side-car (same pattern as src/models/metrics_report.py) rather than a new
field bolted onto the model or encoder objects themselves, so it stays optional and
backward compatible: artifacts saved before this module existed simply have no
metadata file, and validation is skipped rather than failing closed against artifacts
that were never wrong, just older than this check.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

DEFAULT_METADATA_FILENAME = "artifact_metadata.json"
ARTIFACT_VERSION = "1"


class ArtifactMetadataNotFoundError(Exception):
    """Raised when no persisted artifact metadata file exists."""


class ArtifactConsistencyError(Exception):
    """Raised when the loaded model/encoder/feature-list do not belong to the same training run."""


@dataclass(frozen=True)
class ArtifactMetadata:
    """Everything needed to confirm a loaded model + InferenceArtifact are mutually consistent.

    training_run_id: generated once per training run and also stamped onto the
        InferenceArtifact that same run persists (see predictor.py) -- the strongest
        available check, since it fails a mismatch even between two artifact pairs
        that happen to share a model type and feature count.
    model_type: type(best_model).__name__ (e.g. "RandomForestRegressor"), checked
        against the loaded model object directly -- catches a swapped model file
        even if training_run_id is absent (older artifact).
    """

    artifact_version: str
    training_run_id: str
    trained_at: str
    model_type: str
    model_name: str
    training_data: dict[str, Any]
    feature_count: int
    feature_columns: list[str]
    metrics: dict[str, float]


def new_training_run_id() -> str:
    """A fresh identifier for one training run, shared by the model, InferenceArtifact, and metadata it produces."""
    return uuid.uuid4().hex


def save_artifact_metadata(
    metadata: ArtifactMetadata,
    filename: str = DEFAULT_METADATA_FILENAME,
    output_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> str:
    """Persist an ArtifactMetadata as JSON under <output_dir or config's paths.models_dir>/<filename>."""
    config = ConfigLoader(config_path)
    resolved_dir = Path(output_dir or config.get("paths.models_dir", "models"))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / filename
    resolved_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    logger.info("Artifact metadata: saved to '%s' (training_run_id=%s)", resolved_path, metadata.training_run_id)
    return str(resolved_path)


def load_artifact_metadata(
    filename: str = DEFAULT_METADATA_FILENAME,
    input_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> ArtifactMetadata:
    """Load a previously-saved ArtifactMetadata.

    Raises:
        ArtifactMetadataNotFoundError: if the metadata file doesn't exist -- callers
            that must tolerate older artifact sets predating this module should
            catch this and skip validation, not treat it as fatal.
    """
    config = ConfigLoader(config_path)
    resolved_dir = Path(input_dir or config.get("paths.models_dir", "models"))
    resolved_path = resolved_dir / filename
    if not resolved_path.exists():
        raise ArtifactMetadataNotFoundError(f"Artifact metadata: no file at '{resolved_path}'")
    data = json.loads(resolved_path.read_text(encoding="utf-8"))
    return ArtifactMetadata(**data)


def validate_artifact_consistency(model: Any, inference_artifact: Any, metadata: ArtifactMetadata) -> None:
    """Confirm a loaded model + InferenceArtifact genuinely belong to the persisted metadata's training run.

    Checked, in order of strength:
      1. training_run_id -- if the InferenceArtifact carries one (newer artifacts
         always do; see predictor.InferenceArtifact), it must match metadata's exactly.
      2. model_type -- the loaded model's class name must match what metadata recorded.
      3. feature_columns -- the InferenceArtifact's feature list must match metadata's,
         both in length and in exact order (order matters: Predictor reindexes onto
         this list, so a silently-permuted match here would still fail correctness
         downstream even though every column name is present).

    Raises:
        ArtifactConsistencyError: on any mismatch above, naming exactly what
            disagreed -- callers must not proceed to predict from a mismatched set.
    """
    artifact_run_id = getattr(inference_artifact, "training_run_id", None)
    if artifact_run_id is not None and artifact_run_id != metadata.training_run_id:
        raise ArtifactConsistencyError(
            f"Artifact consistency: loaded model's inference artifact has training_run_id "
            f"'{artifact_run_id}', but the persisted metadata records training_run_id "
            f"'{metadata.training_run_id}' -- the model, encoder, and feature list under this "
            "models directory do not all come from the same training run."
        )

    actual_model_type = type(model).__name__
    if actual_model_type != metadata.model_type:
        raise ArtifactConsistencyError(
            f"Artifact consistency: loaded model is a '{actual_model_type}', but the persisted "
            f"metadata records model_type '{metadata.model_type}' from training_run_id "
            f"'{metadata.training_run_id}' -- best_model.joblib does not match the recorded "
            "training run."
        )

    actual_feature_columns = list(inference_artifact.feature_columns)
    if actual_feature_columns != list(metadata.feature_columns):
        raise ArtifactConsistencyError(
            f"Artifact consistency: the loaded feature-pipeline state's feature_columns "
            f"({len(actual_feature_columns)} column(s)) do not match the persisted metadata's "
            f"feature_columns ({len(metadata.feature_columns)} column(s)) from training_run_id "
            f"'{metadata.training_run_id}' -- feature_pipeline_state.joblib does not match the "
            "recorded training run."
        )
