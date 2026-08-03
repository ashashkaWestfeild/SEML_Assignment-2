"""Domain exception hierarchy.

A single root (``LoanRiskError``) lets the API layer distinguish *our* failures
from unexpected ones, and map each to the correct HTTP status code instead of
leaking a bare 500 for every problem.
"""

from __future__ import annotations


class LoanRiskError(Exception):
    """Base class for every error raised by this application."""


class DataIngestionError(LoanRiskError):
    """Raised when the dataset cannot be read or is structurally unusable."""


class SchemaValidationError(LoanRiskError):
    """Raised when a dataframe violates the declared data contract."""

    def __init__(self, message: str, violations: list[str] | None = None) -> None:
        super().__init__(message)
        self.violations = violations or []


class FeatureEngineeringError(LoanRiskError):
    """Raised when derived features cannot be computed."""


class ModelTrainingError(LoanRiskError):
    """Raised when training fails or the trained model misses a quality gate."""


class ModelNotLoadedError(LoanRiskError):
    """Raised when inference is requested before an artifact has been loaded."""


class BusinessRuleViolation(LoanRiskError):
    """Raised when an application breaches underwriting policy (client error)."""
