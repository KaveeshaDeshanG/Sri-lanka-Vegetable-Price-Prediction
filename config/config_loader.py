"""Loads and provides typed access to config/config.yaml."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Distinct from None so a key explicitly set to `null` in YAML is treated the
# same as a key that's absent entirely -- both fail get_required().
_MISSING = object()


class ConfigKeyError(Exception):
    """Raised when a required configuration key is missing or explicitly null."""


class ConfigLoader:
    """Loads config.yaml once and exposes dotted-path lookups.

    Example:
        config = ConfigLoader()
        lag_days = config.get("feature_engineering.lag_periods_days")
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self._config_path = Path(config_path)
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {self._config_path.resolve()}"
            )

        try:
            with self._config_path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"Failed to parse YAML in {self._config_path}: {error}"
            ) from error

        if not isinstance(data, dict):
            raise ValueError(
                f"Configuration file {self._config_path} did not parse to a dictionary."
            )
        return data

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Look up a value using a dotted path, e.g. 'paths.raw_files.harti_prices'."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_required(self, dotted_key: str) -> Any:
        """Look up a value that must be present and non-null.

        Use this instead of get() for values with no safe fallback -- most
        notably raw file paths, where silently defaulting to some made-up
        path would fail later with a far more confusing error.

        Raises:
            ConfigKeyError: if the key is absent or its value is null.
        """
        value = self.get(dotted_key, default=_MISSING)
        if value is _MISSING or value is None:
            message = (
                f"Missing required configuration key '{dotted_key}' in {self._config_path}"
            )
            logger.error(message)
            raise ConfigKeyError(message)
        return value

    @property
    def data(self) -> dict[str, Any]:
        """The full parsed configuration dictionary."""
        return self._data
