"""Typed, immutable configuration objects loaded from ``configs/config.yaml``.

Design note (SE4ML): configuration is *data*, not code. Every threshold the
business may want to renegotiate (decision cut-off, risk tiers, release gates)
is externalised here so that changing a policy never requires a code review of
the inference path -- one of Sculley's "configuration debt" mitigations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class ConfigurationError(RuntimeError):
    """Raised when the configuration file is missing, unreadable or invalid."""


@dataclass(frozen=True)
class ModelConfig:
    """Hyper-parameters and the frozen feature contract of the estimator."""

    algorithm: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    random_state: int
    test_size: float
    default_threshold: float
    target: str
    features: List[str]

    @property
    def n_features(self) -> int:
        """Size of the frozen feature contract."""
        return len(self.features)


@dataclass(frozen=True)
class QualityGates:
    """Release gates. A build that violates any of these must not ship."""

    min_accuracy: float
    min_f1: float
    min_roc_auc: float
    max_brier_score: float
    max_latency_ms: float
    max_missing_fraction: float
    max_psi: float


@dataclass(frozen=True)
class BusinessRules:
    """Underwriting policy constants applied by the validation filter."""

    min_annual_income: int
    max_loan_to_income_multiple: float
    risk_tier_low_threshold: float
    risk_tier_medium_threshold: float


@dataclass(frozen=True)
class Settings:
    """Root configuration aggregate handed to every component."""

    name: str
    version: str
    env: str
    paths: Dict[str, str]
    model: ModelConfig
    gates: QualityGates
    rules: BusinessRules
    log_level: str = "INFO"
    _root: Path = field(default=PROJECT_ROOT, repr=False)

    def path(self, key: str) -> Path:
        """Resolve a configured *relative* path against the project root."""
        if key not in self.paths:
            raise ConfigurationError(f"Unknown path key '{key}' in configuration.")
        return self._root / self.paths[key]


def _read_yaml(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"Malformed YAML in {config_path}: {exc}") from exc


def load_settings(config_path: Path | str | None = None) -> Settings:
    """Load and validate settings.

    Raises:
        ConfigurationError: if the file is absent, malformed, or a mandatory
            section/key is missing. Failing loudly at start-up is deliberate --
            a service that silently boots with default thresholds is worse than
            one that refuses to boot.
    """
    path = Path(config_path or os.environ.get("LOAN_RISK_CONFIG", DEFAULT_CONFIG_PATH))
    raw = _read_yaml(path)

    try:
        model = ModelConfig(**raw["model"])
        gates = QualityGates(**raw["quality_gates"])
        rules = BusinessRules(**raw["business_rules"])
        loaded = Settings(
            name=raw["app"]["name"],
            version=raw["app"]["version"],
            env=raw["app"]["env"],
            paths=raw["paths"],
            model=model,
            gates=gates,
            rules=rules,
            log_level=raw.get("logging", {}).get("level", "INFO"),
            _root=path.resolve().parents[1],
        )
    except KeyError as exc:
        raise ConfigurationError(f"Missing mandatory config key: {exc}") from exc
    except TypeError as exc:
        raise ConfigurationError(f"Invalid config schema: {exc}") from exc

    if not loaded.model.features:
        raise ConfigurationError("model.features must not be empty.")
    return loaded


settings = load_settings()
