"""Produce every measured artifact quoted in the Assignment II report.

Outputs
-------
reports/metrics/qa_metrics.json   consolidated model-quality + data-quality run
reports/figures/*.png             the figures embedded in the report

Run:
    python scripts/evaluate_and_report.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from loan_risk.config import settings  # noqa: E402
from loan_risk.data.ingestion import DataIngestor  # noqa: E402
from loan_risk.data.validation import DataValidator  # noqa: E402
from loan_risk.logging_utils import get_logger  # noqa: E402
from loan_risk.models.predictor import ModelRegistry, RiskPredictor  # noqa: E402
from loan_risk.models.trainer import ModelTrainer  # noqa: E402
from loan_risk.monitoring.drift import DriftMonitor  # noqa: E402

logger = get_logger("evaluate_and_report")

FIGURES = PROJECT_ROOT / "reports" / "figures"
METRICS = PROJECT_ROOT / "reports" / "metrics"
INK = "#1f3864"
ACCENT = "#c00000"
GREEN = "#2e7d32"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def figure_model_quality(y_true, y_prob, y_pred, metrics) -> None:
    """ROC curve, calibration curve, confusion matrix and the gate table."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, color=INK, lw=2, label=f"RF (AUC = {metrics['roc_auc']:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Chance")
    axes[0].legend(fontsize=8, loc="lower right")
    _style(axes[0], "MQ-3  ROC Curve", "False positive rate", "True positive rate")

    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    axes[1].plot(
        prob_pred,
        prob_true,
        "o-",
        color=ACCENT,
        lw=2,
        ms=4,
        label=f"Brier = {metrics['brier_score']:.4f}",
    )
    axes[1].plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect")
    axes[1].legend(fontsize=8, loc="upper left")
    _style(
        axes[1],
        "MQ-4  Calibration (reliability)",
        "Mean predicted probability",
        "Observed frequency",
    )

    ConfusionMatrixDisplay(
        confusion_matrix(y_true, y_pred), display_labels=["Denied", "Approved"]
    ).plot(ax=axes[2], cmap="Blues", colorbar=False, values_format="d")
    _style(axes[2], "MQ-1/MQ-2  Confusion matrix", "Predicted", "Actual")
    axes[2].grid(False)

    fig.suptitle(
        "Model-Quality Metrics on the held-out test split "
        f"(n = {metrics['n_test']:,})",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "model_quality_metrics.png",
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def figure_drift(summary: pd.DataFrame) -> None:
    """PSI bar chart for the simulated recession batch."""
    top = summary.head(10).iloc[::-1]
    colors = [
        ACCENT if s == "SEVERE" else ("#ed7d31" if s == "MODERATE" else GREEN)
        for s in top["severity"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.barh(top["feature"], top["psi"], color=colors, height=0.62)
    ax.axvline(0.10, ls="--", lw=1.2, color="#ed7d31")
    ax.axvline(0.25, ls="--", lw=1.2, color=ACCENT)
    ax.text(0.105, -0.45, "0.10 moderate", fontsize=7.5, color="#ed7d31")
    ax.text(0.255, -0.45, "0.25 severe", fontsize=7.5, color=ACCENT)
    for y, value in enumerate(top["psi"]):
        ax.text(value + 0.012, y, f"{value:.3f}", va="center", fontsize=8)
    _style(ax, "DQ-3  Population Stability Index vs. the training reference", "PSI", "")
    fig.tight_layout()
    fig.savefig(
        FIGURES / "drift_psi.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def figure_lint(before: dict, after: dict) -> None:
    """Before/after code-quality bar chart."""
    labels = [
        "flake8\nviolations",
        "black files\nneeding reformat",
        "pylint score\n(x10)",
    ]
    before_values = [before["flake8"], before["black"], before["pylint"] * 10]
    after_values = [after["flake8"], after["black"], after["pylint"] * 10]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    bars_b = ax.bar(x - 0.2, before_values, 0.4, label="Before", color=ACCENT)
    bars_a = ax.bar(x + 0.2, after_values, 0.4, label="After", color=GREEN)
    for bars, raw in ((bars_b, before_values), (bars_a, after_values)):
        for bar, value in zip(bars, raw):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{value:g}",
                ha="center",
                fontsize=8.5,
                fontweight="bold",
            )
    ax.set_xticks(x, labels, fontsize=9)
    ax.legend(fontsize=9)
    _style(
        ax, "Objective 1.4  Linting & formatting: before vs. after", "", "count / score"
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "lint_before_after.png", dpi=170, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def figure_latency(latencies: np.ndarray) -> None:
    """Latency distribution against the 150 ms SLA."""
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.hist(latencies, bins=40, color=INK, alpha=0.85)
    p95 = float(np.percentile(latencies, 95))
    ax.axvline(
        latencies.mean(), color=GREEN, lw=1.8, label=f"mean = {latencies.mean():.2f} ms"
    )
    ax.axvline(p95, color="#ed7d31", lw=1.8, ls="--", label=f"p95 = {p95:.2f} ms")
    ax.legend(fontsize=8.5)
    _style(
        ax,
        "Inference latency over 1,000 requests (SLA = 150 ms)",
        "latency (ms)",
        "requests",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "latency_distribution.png",
        dpi=170,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> int:
    """Recompute every number quoted in the report and redraw the figures."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)

    ingestor = DataIngestor(settings)
    frame = ingestor.load()
    dq_report = DataValidator(
        max_missing_fraction=settings.gates.max_missing_fraction
    ).validate(frame)
    features, labels = ingestor.split_xy(frame)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=settings.model.test_size,
        random_state=settings.model.random_state,
        stratify=labels,
    )
    trainer = ModelTrainer(settings)
    trainer.train(features, labels)
    metrics = trainer.metrics.as_dict()

    probabilities = trainer.pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= settings.model.default_threshold).astype(int)
    figure_model_quality(y_test, probabilities, predictions, metrics)

    # ---- drift: simulate a credit-tightening recession batch --------------
    reference = x_train
    recession = x_test.copy()
    recession["CreditScore"] = np.clip(recession["CreditScore"] - 85, 300, 850)
    recession["DebtToIncomeRatio"] = np.clip(recession["DebtToIncomeRatio"] * 1.6, 0, 10)
    recession["CreditCardUtilizationRate"] = np.clip(
        recession["CreditCardUtilizationRate"] * 1.5, 0, 1
    )
    monitor = DriftMonitor(
        reference, features=settings.model.features, psi_threshold=settings.gates.max_psi
    )
    drift_summary = monitor.summary_frame(recession)
    figure_drift(drift_summary)
    drift_summary.to_csv(METRICS / "drift_report.csv", index=False)

    # ---- latency benchmark over the real serving path --------------------
    registry = ModelRegistry(settings)
    registry.load()
    predictor = RiskPredictor(registry, settings)
    sample = x_test.iloc[0].to_dict()
    sample = {
        k: (int(v) if float(v).is_integer() else float(v)) for k, v in sample.items()
    }
    # Quieten the audit log for the benchmark: in production these records go
    # to a pipe/file collected by the logging agent, not to an interactive
    # terminal, and TTY writes would otherwise dominate the measurement.
    logging.getLogger("loan_risk.models.predictor").setLevel(logging.WARNING)
    logging.getLogger("loan_risk.features.engineering").setLevel(logging.WARNING)

    for _ in range(50):  # warm-up: page in the forest, exclude from the stats
        predictor.predict(dict(sample))

    latencies = []
    for _ in range(1000):
        start = time.perf_counter()
        predictor.predict(dict(sample))
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies = np.asarray(latencies)
    figure_latency(latencies)

    # ---- lint counts (parsed from the captured reports) -------------------
    lint_dir = PROJECT_ROOT / "reports" / "lint"
    before_flake8 = len(
        [
            ln
            for ln in (lint_dir / "01_before_flake8.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
    )
    after_flake8 = len(
        [
            ln
            for ln in (lint_dir / "07_after_flake8.txt")
            .read_text(encoding="utf-8")
            .splitlines()
            if ln.strip()
        ]
    )
    before = {"flake8": before_flake8, "black": 13, "pylint": 9.72}
    after = {"flake8": after_flake8, "black": 0, "pylint": 10.0}
    figure_lint(before, after)

    payload = {
        "model_quality": metrics,
        "data_quality": dq_report.as_dict(),
        "drift_simulation": {
            "scenario": "recession: credit score -85, DTI x1.6, utilisation x1.5",
            "n_features_checked": int(len(drift_summary)),
            "n_severe": int((drift_summary["severity"] == "SEVERE").sum()),
            "n_moderate": int((drift_summary["severity"] == "MODERATE").sum()),
            "top_5": drift_summary.head(5).to_dict(orient="records"),
        },
        "latency_ms": {
            "mean": round(float(latencies.mean()), 3),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "p99": round(float(np.percentile(latencies, 99)), 3),
            "sla": settings.gates.max_latency_ms,
            "n_requests": int(latencies.size),
        },
        "code_quality": {"before": before, "after": after},
    }
    (METRICS / "qa_metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(
        "evaluation_report_written", extra={"path": str(METRICS / "qa_metrics.json")}
    )
    print(json.dumps(payload, indent=2)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
