"""Inference component: artifact loading + the pipe-and-filter scoring path.

The Assignment I pipe-and-filter design is preserved, but each filter is now a
method on a cohesive ``RiskPredictor`` object with an explicit contract, and
the model artifact is held by a small ``ModelRegistry`` so the API layer never
touches ``joblib`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from loan_risk.config import Settings, settings as default_settings
from loan_risk.exceptions import BusinessRuleViolation, ModelNotLoadedError
from loan_risk.logging_utils import Timer, get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class RiskAssessment:
    """Value object returned by the scoring path."""

    is_approved: bool
    probability: float
    risk_tier: str
    net_worth: int
    debt_to_income: float
    latency_ms: float
    model_version: str

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view used by the API layer and the audit log."""
        return {
            "is_approved": self.is_approved,
            "probability": self.probability,
            "risk_tier": self.risk_tier,
            "net_worth": self.net_worth,
            "debt_to_income": self.debt_to_income,
            "latency_ms": self.latency_ms,
            "model_version": self.model_version,
        }


class ModelRegistry:
    """Loads and holds the serialised pipeline (single point of truth)."""

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.pipeline = None
        self.version: str = "unloaded"
        self.training_metrics: Optional[Dict[str, float]] = None

    @property
    def is_loaded(self) -> bool:
        """True once an artifact has been successfully loaded."""
        return self.pipeline is not None

    def load(self, path: Path | str | None = None) -> bool:
        """Load the artifact. Returns ``True`` on success, ``False`` otherwise.

        Deliberately non-raising: a serving process should start in a
        *degraded* state and report it on ``/health`` rather than crash-loop,
        so that an orchestrator can keep the previous replica serving traffic.
        """
        # pylint: disable=import-outside-toplevel
        import joblib  # deferred: keeps joblib off the fast test-import path

        target = Path(path) if path else self.config.path("model_artifact")
        if not target.exists():
            logger.warning("model_artifact_missing", extra={"path": str(target)})
            return False
        try:
            bundle = joblib.load(target)
            self.pipeline = bundle["pipeline"]
            self.version = bundle.get("version", "unknown")
            self.training_metrics = bundle.get("metrics")
        except (KeyError, EOFError, OSError, ValueError) as exc:
            logger.error(
                "model_artifact_load_failed",
                extra={"path": str(target), "error": str(exc)},
            )
            return False

        logger.info(
            "model_artifact_loaded",
            extra={"path": str(target), "model_version": self.version},
        )
        return True


class RiskPredictor:
    """Pipe-and-filter scoring path.

    Filter 1 :meth:`validate_business_rules` -> underwriting policy
    Filter 2 :meth:`to_frame`                -> payload to model input frame
    Filter 3 :meth:`score`                   -> probability from the estimator
    Filter 4 :meth:`assign_risk_tier`        -> probability to business tier
    """

    def __init__(self, registry: ModelRegistry, config: Settings | None = None) -> None:
        self.registry = registry
        self.config = config or default_settings

    # -- Filter 1 --------------------------------------------------------
    def validate_business_rules(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Apply underwriting policy. Pure: never mutates ``payload``."""
        rules = self.config.rules
        income = payload["AnnualIncome"]
        amount = payload["LoanAmount"]

        if income < rules.min_annual_income:
            logger.warning(
                "business_rule_rejected",
                extra={"rule": "min_annual_income", "value": income},
            )
            raise BusinessRuleViolation(
                f"Annual income {income} is below the minimum assessable "
                f"threshold of {rules.min_annual_income}."
            )
        if amount > income * rules.max_loan_to_income_multiple:
            logger.warning(
                "business_rule_rejected",
                extra={
                    "rule": "max_loan_to_income_multiple",
                    "loan_amount": amount,
                    "annual_income": income,
                },
            )
            raise BusinessRuleViolation(
                f"Requested amount exceeds "
                f"{rules.max_loan_to_income_multiple:.0f}x annual income."
            )
        return payload

    # -- Filter 2 --------------------------------------------------------
    @staticmethod
    def to_frame(payload: Dict[str, Any]) -> pd.DataFrame:
        """Wrap a single application in a one-row dataframe."""
        return pd.DataFrame([payload])

    # -- Filter 3 --------------------------------------------------------
    def score(self, frame: pd.DataFrame) -> float:
        """Return P(approved). Raises if no artifact is loaded."""
        if not self.registry.is_loaded:
            logger.error("inference_without_model")
            raise ModelNotLoadedError("No model artifact is loaded.")
        proba = self.registry.pipeline.predict_proba(frame)[0][1]
        return float(proba)

    # -- Filter 4 --------------------------------------------------------
    def assign_risk_tier(self, probability: float, previous_defaults: int) -> str:
        """Map a probability (plus a hard knock-out) to LOW / MEDIUM / HIGH."""
        rules = self.config.rules
        if previous_defaults:
            return "HIGH"
        if probability >= rules.risk_tier_low_threshold:
            return "LOW"
        if probability >= rules.risk_tier_medium_threshold:
            return "MEDIUM"
        return "HIGH"

    # -- Pipe ------------------------------------------------------------
    def predict(self, payload: Dict[str, Any]) -> RiskAssessment:
        """Run the four filters end to end and emit one audit log record."""
        with Timer() as timer:
            validated = self.validate_business_rules(payload)
            frame = self.to_frame(validated)
            probability = self.score(frame)
            approved = probability >= self.config.model.default_threshold
            tier = self.assign_risk_tier(probability, validated["PreviousLoanDefaults"])

        assessment = RiskAssessment(
            is_approved=bool(approved),
            probability=round(probability, 4),
            risk_tier=tier,
            net_worth=int(validated["NetWorth"]),
            debt_to_income=round(float(validated["DebtToIncomeRatio"]), 4),
            latency_ms=round(timer.elapsed_ms, 2),
            model_version=self.registry.version,
        )
        logger.info("loan_application_scored", extra=assessment.as_dict())
        return assessment
