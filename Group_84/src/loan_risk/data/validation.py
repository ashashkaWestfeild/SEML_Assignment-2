"""Data-quality gate: declarative schema contract + measurable DQ metrics.

Two of the four Data-Quality metrics required by Objective 2.8(b) are produced
here:

    DQ-1  Schema conformance rate  -- % of declared columns that are present
                                      AND within their declared type/range.
    DQ-2  Missing-value fraction   -- overall and worst-column null ratio.

(The remaining two -- PSI drift and KS drift -- live in
``loan_risk.monitoring.drift``.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from loan_risk.exceptions import SchemaValidationError
from loan_risk.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ColumnSpec:
    """Declared contract for one column."""

    name: str
    dtype: str  # "int" | "float"
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    nullable: bool = False


# The data contract for the 22 modelling features. This is version-controlled
# alongside the code, so a schema change is a reviewable diff -- not a surprise
# discovered by a NaN in production.
LOAN_SCHEMA: Tuple[ColumnSpec, ...] = (
    ColumnSpec("Age", "int", 18, 100),
    ColumnSpec("AnnualIncome", "int", 0, 10_000_000),
    ColumnSpec("CreditScore", "int", 300, 850),
    ColumnSpec("EmploymentStatus", "int", 0, 2),
    ColumnSpec("EducationLevel", "int", 0, 4),
    ColumnSpec("LoanAmount", "int", 0, 10_000_000),
    ColumnSpec("LoanDuration", "int", 1, 480),
    ColumnSpec("MonthlyDebtPayments", "int", 0, 1_000_000),
    ColumnSpec("CreditCardUtilizationRate", "float", 0.0, 1.0),
    ColumnSpec("DebtToIncomeRatio", "float", 0.0, 10.0),
    ColumnSpec("BankruptcyHistory", "int", 0, 1),
    ColumnSpec("LoanPurpose", "int", 0, 4),
    ColumnSpec("PreviousLoanDefaults", "int", 0, 1),
    ColumnSpec("PaymentHistory", "int", 0, 600),
    ColumnSpec("LengthOfCreditHistory", "int", 0, 90),
    ColumnSpec("SavingsAccountBalance", "int", 0, 100_000_000),
    ColumnSpec("CheckingAccountBalance", "int", 0, 100_000_000),
    ColumnSpec("TotalLiabilities", "int", 0, 100_000_000),
    ColumnSpec("JobTenure", "int", 0, 70),
    ColumnSpec("NetWorth", "int", None, None),
    ColumnSpec("LoanToIncomeRatio", "float", 0.0, 100.0),
    ColumnSpec("SavingsToLoanRatio", "float", 0.0, 1000.0),
)


@dataclass
class DataQualityReport:
    """Result of a validation run -- serialisable for the metrics dashboard."""

    n_rows: int
    n_columns: int
    schema_conformance_rate: float
    missing_value_fraction: float
    worst_column_missing_fraction: float
    worst_column: Optional[str]
    violations: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when no contract violation was recorded."""
        return not self.violations

    def as_dict(self) -> Dict[str, object]:
        """JSON-serialisable view for the metrics dashboard."""
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "schema_conformance_rate": round(self.schema_conformance_rate, 4),
            "missing_value_fraction": round(self.missing_value_fraction, 6),
            "worst_column_missing_fraction": round(self.worst_column_missing_fraction, 6),
            "worst_column": self.worst_column,
            "n_violations": len(self.violations),
            "violations": self.violations,
            "passed": self.passed,
        }


class DataValidator:
    """Validate a dataframe against :data:`LOAN_SCHEMA`.

    CRITICAL FUNCTION #2 for the error-handling requirement: soft breaches are
    logged as WARNING and accumulated; a hard breach (``strict=True``) is logged
    as ERROR and raises :class:`SchemaValidationError`.
    """

    def __init__(
        self,
        schema: Tuple[ColumnSpec, ...] = LOAN_SCHEMA,
        max_missing_fraction: float = 0.02,
    ) -> None:
        self.schema = schema
        self.max_missing_fraction = max_missing_fraction

    @staticmethod
    def _check_range(series: pd.Series, spec: ColumnSpec) -> List[str]:
        """Range half of the column contract. Extracted to keep ``validate``
        under the cyclomatic-complexity budget enforced by flake8 (C901)."""
        clean = series.dropna()
        findings: List[str] = []
        if spec.minimum is not None and (clean < spec.minimum).any():
            findings.append(f"below_minimum:{spec.name}")
        if spec.maximum is not None and (clean > spec.maximum).any():
            findings.append(f"above_maximum:{spec.name}")
        return findings

    def _check_column(self, frame: pd.DataFrame, spec: ColumnSpec) -> List[str]:
        """Return every contract violation for a single column."""
        if spec.name not in frame.columns:
            return [f"missing_column:{spec.name}"]

        series = frame[spec.name]
        findings: List[str] = []
        if not spec.nullable and series.isna().any():
            findings.append(f"nulls_in_non_nullable:{spec.name}")
        if not pd.api.types.is_numeric_dtype(series):
            findings.append(f"non_numeric_dtype:{spec.name}")
        else:
            findings.extend(self._check_range(series, spec))
        return findings

    def validate(self, frame: pd.DataFrame, strict: bool = False) -> DataQualityReport:
        """Run every contract check and return a :class:`DataQualityReport`."""
        if frame is None or frame.empty:
            logger.error("validation_called_on_empty_frame")
            raise SchemaValidationError("Cannot validate an empty dataframe.")

        violations: List[str] = []
        conforming_columns = 0
        for spec in self.schema:
            findings = self._check_column(frame, spec)
            violations.extend(findings)
            conforming_columns += int(not findings)

        # ---- DQ metric 1: schema conformance rate --------------------------
        conformance = conforming_columns / len(self.schema)

        # ---- DQ metric 2: missing-value fractions --------------------------
        null_fractions = frame.isna().mean()
        overall_missing = float(frame.isna().to_numpy().mean())
        worst_column = str(null_fractions.idxmax()) if len(null_fractions) else None
        worst_fraction = float(null_fractions.max()) if len(null_fractions) else 0.0

        if overall_missing > self.max_missing_fraction:
            violations.append(
                f"missing_fraction_exceeded:{overall_missing:.4f}"
                f">{self.max_missing_fraction}"
            )

        report = DataQualityReport(
            n_rows=int(frame.shape[0]),
            n_columns=int(frame.shape[1]),
            schema_conformance_rate=conformance,
            missing_value_fraction=overall_missing,
            worst_column_missing_fraction=worst_fraction,
            worst_column=worst_column,
            violations=violations,
        )

        if violations:
            logger.warning(
                "data_quality_violations_detected",
                extra={
                    "n_violations": len(violations),
                    "violations": violations[:10],
                    "schema_conformance_rate": round(conformance, 4),
                },
            )
            if strict:
                logger.error(
                    "data_quality_gate_failed",
                    extra={"n_violations": len(violations)},
                )
                raise SchemaValidationError(
                    f"{len(violations)} schema violation(s) detected.", violations
                )
        else:
            logger.info(
                "data_quality_gate_passed",
                extra={
                    "rows": report.n_rows,
                    "schema_conformance_rate": 1.0,
                    "missing_value_fraction": round(overall_missing, 6),
                },
            )
        return report
