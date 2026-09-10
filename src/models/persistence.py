"""Generic joblib persistence: save/load a fitted object under config's paths.models_dir.

Named around "model" because that was its first use (see src/models), but
the implementation is deliberately generic -- src/prediction reuses these
same two functions to persist the fitted CategoricalEncoder + feature
column list inference needs, rather than duplicating this path-resolution
and joblib I/O for a second artifact type.
"""

import logging
from pathlib import Path
from typing import Any

import joblib

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def save_model(
    model: Any,
    filename: str,
    output_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> str:
    """Save a fitted object to <output_dir or config's paths.models_dir>/<filename> via joblib.

    Args:
        filename: e.g. "best_model.joblib". The *directory* is config-driven
            (paths.models_dir); which specific file gets written is the
            caller's choice, not something config prescribes.
        output_dir: overrides config's paths.models_dir.
    """
    config = ConfigLoader(config_path)
    resolved_dir = Path(output_dir or config.get("paths.models_dir", "models"))
    resolved_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = resolved_dir / filename
    joblib.dump(model, resolved_path)
    logger.info("Model persistence: saved model to '%s'", resolved_path)
    return str(resolved_path)


def load_model(
    filename: str,
    input_dir: str | None = None,
    config_path: str = "config/config.yaml",
) -> Any:
    """Load a previously-saved object from <input_dir or config's paths.models_dir>/<filename>."""
    config = ConfigLoader(config_path)
    resolved_dir = Path(input_dir or config.get("paths.models_dir", "models"))
    resolved_path = resolved_dir / filename
    model = joblib.load(resolved_path)
    logger.info("Model persistence: loaded model from '%s'", resolved_path)
    return model
