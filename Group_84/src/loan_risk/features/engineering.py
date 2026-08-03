"""Feature engineering -- the PRODUCTION counterpart of
``notebooks/research_prototype.ipynb``.

This module is the reference example for Objective 1.2 (research code vs.
production code). The notebook computes the same two derived ratios, but this
version adds everything the notebook cannot offer:

  * a stable, declared *contract* (``FEATURE_ORDER``) so training and serving
    always see identical columns in identical order;
  * pure functions (no globals, no hidden state) -> unit-testable;
  * defensive division and explicit error semantics;
  * structured logging;
  * a scikit-learn ``TransformerMixin`` wrapper so the same object can be
    dropped into a ``Pipeline`` and serialised with the model, eliminating
    training/serving skew.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from loan_risk.config import settings
from loan_risk.exceptions import FeatureEngineeringError
from loan_risk.logging_utils import get_logger

logger = get_logger(__name__)

FEATURE_ORDER: List[str] = list(settings.model.features)

#: Columns the raw application must supply before derivation.
BASE_COLUMNS: List[str] = [
    c for c in FEATURE_ORDER if c not in ("LoanToIncomeRatio", "SavingsToLoanRatio")
]

_EPSILON = 1.0  # +1 smoothing: makes every ratio total -- never a ZeroDivisionError


def safe_ratio(
    numerator: pd.Series | float, denominator: pd.Series | float
) -> pd.Series | float:
    """Divide with +1 smoothing so the denominator can never be zero.

    Pure function -- no I/O, no globals -- which is exactly why it can be
    property-tested in isolation (see ``tests/test_unit_features.py``).
    """
    return numerator / (denominator + _EPSILON)


def compute_loan_to_income(loan_amount, annual_income):
    """Leverage ratio: how many times the annual income is being borrowed."""
    return safe_ratio(loan_amount, annual_income)


def compute_savings_to_loan(savings_balance, loan_amount):
    """Cushion ratio: liquid savings held against the requested principal."""
    return safe_ratio(savings_balance, loan_amount)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    # The `X`/`y` parameter names and the unused `fit` arguments are imposed by
    # the scikit-learn transformer contract, so the checks are waived here only.
    # pylint: disable=invalid-name,unused-argument
    """Stateless, sklearn-compatible transformer producing derived features.

    Being a real transformer (rather than a loose function called in two
    places) means the *identical* object is pickled inside the served pipeline,
    so a feature definition can never drift between the training script and the
    API process.
    """

    def __init__(self, feature_order: Iterable[str] | None = None) -> None:
        self.feature_order = list(feature_order) if feature_order else FEATURE_ORDER

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureEngineer":  # noqa: N803
        """No parameters are learned; ``fit`` exists to satisfy the sklearn API."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Return a frame containing exactly ``feature_order``, in order.

        Raises:
            FeatureEngineeringError: if a required base column is absent or the
                derived values are not finite.
        """
        if not isinstance(X, pd.DataFrame):
            raise FeatureEngineeringError(
                f"Expected a pandas DataFrame, received {type(X).__name__}"
            )

        frame = X.copy()
        missing = [c for c in BASE_COLUMNS if c not in frame.columns]
        if missing:
            logger.error("feature_base_columns_missing", extra={"missing": missing})
            raise FeatureEngineeringError(f"Missing base columns: {missing}")

        frame["LoanToIncomeRatio"] = compute_loan_to_income(
            frame["LoanAmount"], frame["AnnualIncome"]
        )
        frame["SavingsToLoanRatio"] = compute_savings_to_loan(
            frame["SavingsAccountBalance"], frame["LoanAmount"]
        )

        derived = frame[["LoanToIncomeRatio", "SavingsToLoanRatio"]]
        if not np.isfinite(derived.to_numpy(dtype=float)).all():
            logger.error("derived_features_not_finite")
            raise FeatureEngineeringError(
                "Derived ratios contain NaN/inf -- check inputs for nulls."
            )

        out_of_range = int((frame["LoanToIncomeRatio"] > 5.0).sum())
        if out_of_range:
            # Recoverable: the model still scores these, but the operator should
            # know that unusually leveraged applications are arriving.
            logger.warning(
                "high_leverage_applications_detected",
                extra={"count": out_of_range, "threshold_loan_to_income": 5.0},
            )

        logger.info(
            "feature_engineering_completed",
            extra={"rows": int(len(frame)), "n_features": len(self.feature_order)},
        )
        return frame[self.feature_order]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """sklearn introspection hook -- keeps the contract discoverable."""
        return np.asarray(self.feature_order, dtype=object)
