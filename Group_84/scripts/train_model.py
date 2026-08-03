"""Training entrypoint -- the PRODUCTION counterpart of the research notebook.

Run:
    python scripts/train_model.py

Contrast with ``notebooks/research_prototype.ipynb``: this script is
deterministic (seeded), configuration-driven, validated, gated, logged and
importable. Every step is a call into ``loan_risk``; the script itself holds no
business logic, so the exact same code paths are exercised by the test suite.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loan_risk.config import settings  # noqa: E402
from loan_risk.data.ingestion import DataIngestor  # noqa: E402
from loan_risk.data.validation import DataValidator  # noqa: E402
from loan_risk.exceptions import LoanRiskError  # noqa: E402
from loan_risk.logging_utils import get_logger  # noqa: E402
from loan_risk.models.trainer import ModelTrainer  # noqa: E402

logger = get_logger("train_model")


def main() -> int:
    """Ingest -> validate -> compare algorithms -> train -> gate -> persist."""
    try:
        ingestor = DataIngestor(settings)
        frame = ingestor.load()

        validator = DataValidator(
            max_missing_fraction=settings.gates.max_missing_fraction
        )
        dq_report = validator.validate(frame, strict=True)

        features, labels = ingestor.split_xy(frame)

        trainer = ModelTrainer(settings)
        comparison, winner = trainer.compare_algorithms(features, labels)
        logger.info("algorithm_winner", extra={"winner": winner})

        trainer.train(features, labels, algorithm=winner)
        trainer.enforce_quality_gates()
        artifact_path = trainer.save()

        # Freeze the training distribution as the drift reference profile.
        reference_path = settings.path("reference_data")
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        features.sample(
            n=min(5000, len(features)), random_state=settings.model.random_state
        ).to_csv(reference_path, index=False)

        metrics_dir = settings.path("metrics_dir")
        metrics_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "winner": winner,
            "algorithm_comparison": comparison,
            "chosen_model_metrics": trainer.metrics.as_dict(),  # type: ignore[union-attr]
            "data_quality": dq_report.as_dict(),
            "artifact": str(artifact_path),
        }
        (metrics_dir / "training_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info("training_pipeline_completed", extra={"artifact": str(artifact_path)})
        return 0

    except LoanRiskError as exc:
        logger.error(
            "training_pipeline_failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
