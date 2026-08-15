"""TEST TYPE 1 - UNIT TESTS.

Scope: one function or class at a time, no I/O, no model, no HTTP. These are
the fastest layer of the pyramid and are what makes refactoring safe.

Selection principle: a test earns its place only if it could fail while the
underwriting logic is wrong. Assertions that merely re-check Python semantics
(that ``a / b`` grows with ``a``, that a pure function is pure) were removed --
they pass regardless of whether this system works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.config import settings
from loan_risk.exceptions import FeatureEngineeringError
from loan_risk.features.engineering import FEATURE_ORDER, FeatureEngineer, safe_ratio

pytestmark = pytest.mark.unit


# ── pure helper functions ────────────────────────────────────────────────
def test_safe_ratio_never_divides_by_zero():
    """The +1 smoothing must make a zero denominator harmless.

    This is the only interesting property of ``safe_ratio``: an applicant
    declaring zero income must not crash the scoring path.
    """
    assert safe_ratio(50.0, 0.0) == pytest.approx(50.0)
    assert safe_ratio(100.0, 9.0) == pytest.approx(10.0)
    assert np.isfinite(safe_ratio(1.0, 0.0))


# ── FeatureEngineer transformer ──────────────────────────────────────────
def test_transformer_emits_the_declared_feature_contract(synthetic_frame):
    """Training and serving must always see the same columns, in order."""
    out = FeatureEngineer().fit_transform(synthetic_frame)
    assert list(out.columns) == FEATURE_ORDER
    assert out.shape == (len(synthetic_frame), settings.model.n_features)


def test_transformer_does_not_mutate_its_input(synthetic_frame):
    """A transformer with side effects breaks every downstream assumption."""
    before = synthetic_frame.copy(deep=True)
    FeatureEngineer().transform(synthetic_frame)
    pd.testing.assert_frame_equal(synthetic_frame, before)


def test_transformer_recomputes_derived_columns_from_base_columns(synthetic_frame):
    """Derived ratios must be recomputed, not trusted from the input frame.

    A stale or poisoned ``LoanToIncomeRatio`` arriving from upstream must never
    reach the estimator.
    """
    poisoned = synthetic_frame.copy()
    poisoned["LoanToIncomeRatio"] = -999.0
    out = FeatureEngineer().transform(poisoned)
    assert (out["LoanToIncomeRatio"] > 0).all()


def _frame_missing_a_base_column(frame):
    return frame.drop(columns=["CreditScore"])


def _frame_with_a_null_income(frame):
    broken = frame.copy()
    broken.loc[broken.index[0], "AnnualIncome"] = np.nan
    return broken


@pytest.mark.parametrize(
    "make_input,label",
    [
        (_frame_missing_a_base_column, "missing base column"),
        (_frame_with_a_null_income, "non-finite derived value"),
    ],
)
def test_transformer_rejects_malformed_input(synthetic_frame, make_input, label):
    """Every malformed input must fail loudly as one domain exception."""
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer().transform(make_input(synthetic_frame))


# ── configuration ────────────────────────────────────────────────────────
def test_settings_are_immutable():
    """Frozen dataclasses stop a request handler from editing global policy."""
    with pytest.raises(Exception):
        settings.model.default_threshold = 0.99  # type: ignore[misc]
