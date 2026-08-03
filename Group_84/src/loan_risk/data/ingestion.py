"""Data ingestion component (OOP).

``DataIngestor`` is the *only* place in the system that touches the filesystem
for training data. Isolating I/O behind one class means the rest of the
pipeline can be unit-tested with in-memory frames, and swapping CSV for a
feature store later touches exactly one file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from loan_risk.config import Settings, settings as default_settings
from loan_risk.exceptions import DataIngestionError
from loan_risk.logging_utils import get_logger

logger = get_logger(__name__)


class DataIngestor:
    """Load a dataset from disk and split it into features and target.

    CRITICAL FUNCTION #1 for the error-handling / logging requirement:
    every failure mode (missing file, empty file, unparseable CSV, missing
    columns) is caught, logged at the appropriate level and re-raised as a
    single domain exception so callers never have to catch ``OSError``.
    """

    def __init__(self, config: Settings | None = None) -> None:
        self.config = config or default_settings

    def load(self, path: Path | str | None = None) -> pd.DataFrame:
        """Read a CSV into a dataframe.

        Raises:
            DataIngestionError: file missing, empty, or unparseable.
        """
        source = Path(path) if path else self.config.path("processed_data")
        logger.info("data_ingestion_started", extra={"source": str(source)})

        if not source.exists():
            logger.error(
                "data_source_missing",
                extra={"source": str(source), "hint": "run scripts/build_dataset.py"},
            )
            raise DataIngestionError(f"Dataset not found at {source}")

        try:
            frame = pd.read_csv(source)
        except pd.errors.EmptyDataError as exc:
            logger.error("data_source_empty", extra={"source": str(source)})
            raise DataIngestionError(f"Dataset at {source} is empty") from exc
        except (pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
            logger.error(
                "data_source_unreadable",
                extra={"source": str(source), "error": str(exc)},
            )
            raise DataIngestionError(f"Could not parse {source}: {exc}") from exc

        if frame.empty:
            logger.error("data_source_zero_rows", extra={"source": str(source)})
            raise DataIngestionError(f"Dataset at {source} contains zero rows")

        duplicates = int(frame.duplicated().sum())
        if duplicates:
            # Recoverable: we keep going but the operator must know.
            logger.warning(
                "duplicate_rows_detected",
                extra={"duplicate_rows": duplicates, "total_rows": len(frame)},
            )

        logger.info(
            "data_ingestion_completed",
            extra={"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
        )
        return frame

    def split_xy(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Return ``(X, y)`` restricted to the configured feature contract.

        Raises:
            DataIngestionError: if the target or any configured feature is
                absent -- a silent column drop here would train a model on a
                different feature space than the one served, which is the
                classic training/serving skew defect.
        """
        target = self.config.model.target
        missing_target = target not in frame.columns
        missing_features = [
            c for c in self.config.model.features if c not in frame.columns
        ]

        if missing_target or missing_features:
            logger.error(
                "feature_contract_violation",
                extra={
                    "missing_target": target if missing_target else None,
                    "missing_features": missing_features,
                },
            )
            raise DataIngestionError(
                f"Feature contract violated. missing_target={missing_target}, "
                f"missing_features={missing_features}"
            )

        features = frame[self.config.model.features].copy()
        labels = frame[target].copy()
        logger.info(
            "xy_split_completed",
            extra={
                "n_rows": int(len(features)),
                "n_features": int(features.shape[1]),
                "positive_rate": round(float(labels.mean()), 4),
            },
        )
        return features, labels
