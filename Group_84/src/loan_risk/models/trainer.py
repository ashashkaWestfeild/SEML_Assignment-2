"""Model training component (OOP) with built-in quality gates.

``ModelTrainer`` owns the whole train -> evaluate -> gate -> persist lifecycle.
It trains a scikit-learn ``Pipeline`` whose first stage is the *same*
``FeatureEngineer`` used at serving time, which is how training/serving skew is
structurally prevented rather than merely documented.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from loan_risk.config import Settings, settings as default_settings
from loan_risk.exceptions import ModelTrainingError
from loan_risk.features.engineering import FeatureEngineer
from loan_risk.logging_utils import Timer, get_logger

logger = get_logger(__name__)


@dataclass
class ModelMetrics:
    """Model-quality metrics (Objective 2.8a).

    ``accuracy`` / ``f1`` are the headline discrimination metrics; ``roc_auc``
    is threshold-independent; ``brier_score`` measures *calibration* -- vital
    for underwriting, where the probability itself drives the risk tier and
    pricing, not just the 0/1 decision.
    """

    accuracy: float
    f1: float
    precision: float
    recall: float
    roc_auc: float
    brier_score: float
    log_loss: float
    n_train: int
    n_test: int

    def as_dict(self) -> Dict[str, float]:
        """Rounded, JSON-serialisable view written to reports/metrics."""
        return {
            k: (round(v, 6) if isinstance(v, float) else v)
            for k, v in asdict(self).items()
        }


class ModelTrainer:
    """Train, evaluate, gate and persist the loan-approval classifier.

    CRITICAL FUNCTION #3 for the error-handling requirement: training failures,
    degenerate label distributions and quality-gate breaches are each logged at
    a distinct level and surfaced as :class:`ModelTrainingError`.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings
        self.pipeline: Optional[Pipeline] = None
        self.metrics: Optional[ModelMetrics] = None

    # ------------------------------------------------------------------ build
    def build_pipeline(self, algorithm: Optional[str] = None) -> Pipeline:
        """Assemble the FeatureEngineer -> estimator pipeline."""
        algorithm = algorithm or self.config.model.algorithm
        model_cfg = self.config.model

        if algorithm == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=model_cfg.n_estimators,
                max_depth=model_cfg.max_depth,
                min_samples_leaf=model_cfg.min_samples_leaf,
                random_state=model_cfg.random_state,
                # n_jobs=1 is a *correctness* choice, not a performance
                # oversight. With n_jobs=-1 the per-tree vote accumulation
                # happens in non-deterministic thread order, so two identical
                # requests can differ at ~1e-16 -- enough to flip an applicant
                # sitting exactly on the 0.50 cut-off. Regulated lending
                # decisions must be bit-reproducible for audit, and inference
                # is already ~2 ms per request, so there is nothing to buy.
                # Caught by tests/test_model_inference.py::
                # test_prediction_is_invariant_to_repeated_calls.
                n_jobs=1,
            )
            steps = [("features", FeatureEngineer()), ("model", estimator)]
        elif algorithm == "logistic_regression":
            steps = [
                ("features", FeatureEngineer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000, random_state=model_cfg.random_state
                    ),
                ),
            ]
        else:
            logger.error("unknown_algorithm_requested", extra={"algorithm": algorithm})
            raise ModelTrainingError(f"Unsupported algorithm '{algorithm}'")

        logger.info(
            "pipeline_built", extra={"algorithm": algorithm, "n_steps": len(steps)}
        )
        return Pipeline(steps)

    # ------------------------------------------------------------------ train
    def train(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        algorithm: Optional[str] = None,
        evaluate: bool = True,
    ) -> Pipeline:
        """Fit the pipeline and (optionally) compute held-out metrics."""
        if len(features) != len(labels):
            logger.error(
                "xy_length_mismatch",
                extra={"n_features": len(features), "n_labels": len(labels)},
            )
            raise ModelTrainingError("Feature/label row counts differ.")

        n_classes = labels.nunique()
        if n_classes < 2:
            logger.error(
                "degenerate_label_distribution", extra={"n_classes": int(n_classes)}
            )
            raise ModelTrainingError(
                "Training data contains a single class; cannot fit a classifier."
            )

        stratify = labels if (evaluate and labels.value_counts().min() >= 2) else None
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=self.config.model.test_size,
                random_state=self.config.model.random_state,
                stratify=stratify,
            )
        else:
            x_train, y_train = features, labels
            x_test, y_test = features, labels

        self.pipeline = self.build_pipeline(algorithm)

        try:
            with Timer() as timer:
                self.pipeline.fit(x_train, y_train)
        except Exception as exc:  # noqa: BLE001 - re-raised as a domain error
            logger.error("model_training_failed", extra={"error": str(exc)})
            raise ModelTrainingError(f"Training failed: {exc}") from exc

        logger.info(
            "model_training_completed",
            extra={
                "algorithm": algorithm or self.config.model.algorithm,
                "n_train": int(len(x_train)),
                "train_seconds": round(timer.elapsed_ms / 1000.0, 3),
            },
        )

        if evaluate:
            self.metrics = self.evaluate(x_test, y_test)
        return self.pipeline

    # --------------------------------------------------------------- evaluate
    def evaluate(self, features: pd.DataFrame, labels: pd.Series) -> ModelMetrics:
        """Compute the model-quality metric suite on a held-out split."""
        if self.pipeline is None:
            raise ModelTrainingError("evaluate() called before train().")

        probabilities = self.pipeline.predict_proba(features)[:, 1]
        predictions = (probabilities >= self.config.model.default_threshold).astype(int)

        metrics = ModelMetrics(
            accuracy=float(accuracy_score(labels, predictions)),
            f1=float(f1_score(labels, predictions, zero_division=0)),
            precision=float(precision_score(labels, predictions, zero_division=0)),
            recall=float(recall_score(labels, predictions, zero_division=0)),
            roc_auc=float(roc_auc_score(labels, probabilities)),
            brier_score=float(brier_score_loss(labels, probabilities)),
            log_loss=float(log_loss(labels, np.clip(probabilities, 1e-9, 1 - 1e-9))),
            n_train=int(len(features)),
            n_test=int(len(labels)),
        )
        logger.info("model_evaluated", extra=metrics.as_dict())
        return metrics

    # ------------------------------------------------------------------ gates
    def enforce_quality_gates(self, metrics: Optional[ModelMetrics] = None) -> None:
        """Fail the build when a release gate is breached."""
        metrics = metrics or self.metrics
        if metrics is None:
            raise ModelTrainingError("No metrics available to gate on.")

        gates = self.config.gates
        breaches = []
        if metrics.accuracy < gates.min_accuracy:
            breaches.append(f"accuracy {metrics.accuracy:.4f} < {gates.min_accuracy}")
        if metrics.f1 < gates.min_f1:
            breaches.append(f"f1 {metrics.f1:.4f} < {gates.min_f1}")
        if metrics.roc_auc < gates.min_roc_auc:
            breaches.append(f"roc_auc {metrics.roc_auc:.4f} < {gates.min_roc_auc}")
        if metrics.brier_score > gates.max_brier_score:
            breaches.append(f"brier {metrics.brier_score:.4f} > {gates.max_brier_score}")

        if breaches:
            logger.error("quality_gate_breached", extra={"breaches": breaches})
            raise ModelTrainingError("Quality gates breached: " + "; ".join(breaches))
        logger.info("quality_gates_passed", extra=metrics.as_dict())

    # ----------------------------------------------------------------- persist
    def save(self, path: Path | str | None = None) -> Path:
        """Serialise the fitted pipeline to disk."""
        if self.pipeline is None:
            raise ModelTrainingError("save() called before train().")

        target = Path(path) if path else self.config.path("model_artifact")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump(
                {
                    "pipeline": self.pipeline,
                    "features": self.config.model.features,
                    "metrics": self.metrics.as_dict() if self.metrics else None,
                    "version": self.config.version,
                },
                target,
            )
        except OSError as exc:
            logger.error(
                "model_persist_failed", extra={"path": str(target), "error": str(exc)}
            )
            raise ModelTrainingError(f"Could not write artifact to {target}") from exc

        logger.info("model_artifact_saved", extra={"path": str(target)})
        return target

    # ------------------------------------------------------------- comparison
    def compare_algorithms(
        self, features: pd.DataFrame, labels: pd.Series
    ) -> Tuple[Dict[str, Dict[str, float]], str]:
        """Train every candidate and return their metrics plus the winner.

        Documents the algorithm-vs-softgoal trade-off carried over from the
        GR4ML Analytics Design View in Assignment I.
        """
        results: Dict[str, Dict[str, float]] = {}
        for algorithm in ("logistic_regression", "random_forest"):
            trainer = ModelTrainer(self.config)
            trainer.train(features, labels, algorithm=algorithm)
            results[algorithm] = trainer.metrics.as_dict()  # type: ignore[union-attr]

        winner = max(results, key=lambda k: results[k]["roc_auc"])
        logger.info("algorithm_comparison_completed", extra={"winner": winner})
        return results, winner
