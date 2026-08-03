"""TEST TYPE 1 - UNIT TESTS.

Scope: one function or class at a time, no I/O, no model, no HTTP. These are
the fastest layer of the pyramid and are what makes refactoring safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.config import ConfigurationError, load_settings, settings
from loan_risk.exceptions import FeatureEngineeringError
from loan_risk.features.engineering import (
    FEATURE_ORDER,
    FeatureEngineer,
    compute_loan_to_income,
    compute_savings_to_loan,
    safe_ratio,
)

pytestmark = pytest.mark.unit


# ── pure helper functions ────────────────────────────────────────────────
def test_safe_ratio_matches_expected_value():
    assert safe_ratio(100.0, 9.0) == pytest.approx(10.0)


def test_safe_ratio_never_divides_by_zero():
    """The +1 smoothing must make a zero denominator harmless."""
    assert safe_ratio(50.0, 0.0) == pytest.approx(50.0)
    assert np.isfinite(safe_ratio(1.0, 0.0))


def test_compute_loan_to_income_is_monotonic_in_loan_amount():
    low = compute_loan_to_income(10_000, 100_000)
    high = compute_loan_to_income(80_000, 100_000)
    assert high > low


def test_compute_savings_to_loan_is_monotonic_in_savings():
    assert compute_savings_to_loan(50_000, 20_000) > compute_savings_to_loan(
        1_000, 20_000
    )


# ── FeatureEngineer transformer ──────────────────────────────────────────
def test_transformer_emits_the_declared_feature_contract(synthetic_frame):
    engineer = FeatureEngineer()
    out = engineer.fit_transform(synthetic_frame)
    assert list(out.columns) == FEATURE_ORDER
    assert out.shape == (len(synthetic_frame), settings.model.n_features)


def test_transformer_is_deterministic(synthetic_frame):
    engineer = FeatureEngineer()
    first = engineer.transform(synthetic_frame)
    second = engineer.transform(synthetic_frame)
    pd.testing.assert_frame_equal(first, second)


def test_transformer_does_not_mutate_its_input(synthetic_frame):
    """A transformer with side effects breaks every downstream assumption."""
    before = synthetic_frame.copy(deep=True)
    FeatureEngineer().transform(synthetic_frame)
    pd.testing.assert_frame_equal(synthetic_frame, before)


def test_transformer_recomputes_derived_columns_from_base_columns(synthetic_frame):
    """Derived ratios must be recomputed, not trusted from the input frame."""
    poisoned = synthetic_frame.copy()
    poisoned["LoanToIncomeRatio"] = -999.0
    out = FeatureEngineer().transform(poisoned)
    assert (out["LoanToIncomeRatio"] > 0).all()


def test_transformer_rejects_missing_base_columns(synthetic_frame):
    broken = synthetic_frame.drop(columns=["CreditScore"])
    with pytest.raises(FeatureEngineeringError, match="Missing base columns"):
        FeatureEngineer().transform(broken)


def test_transformer_rejects_non_dataframe_input():
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer().transform([1, 2, 3])


def test_transformer_raises_on_non_finite_derived_values(synthetic_frame):
    broken = synthetic_frame.copy()
    broken.loc[broken.index[0], "AnnualIncome"] = np.nan
    with pytest.raises(FeatureEngineeringError):
        FeatureEngineer().transform(broken)


# ── configuration ────────────────────────────────────────────────────────
def test_settings_expose_a_non_empty_feature_contract():
    assert settings.model.n_features == len(FEATURE_ORDER) > 0


def test_settings_are_immutable():
    """Frozen dataclasses stop a request handler from editing global policy."""
    with pytest.raises(Exception):
        settings.model.default_threshold = 0.99  # type: ignore[misc]


def test_missing_config_file_fails_loudly(tmp_path):
    with pytest.raises(ConfigurationError):
        load_settings(tmp_path / "does_not_exist.yaml")
