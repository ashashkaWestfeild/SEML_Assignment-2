"""TEST TYPE 3 (cont.) - ML BEHAVIOURAL TESTS: INFERENCE (Objective 2.7b).

Three families, following Zinkevich / "ML Test Score" practice:

  * SHAPE & RANGE      -- outputs are structurally valid for every input.
  * DIRECTIONAL        -- a change the domain says should push the score one
                          way actually does (monotonic expectations).
  * INVARIANCE         -- a change the domain says is irrelevant leaves the
                          score untouched.

Parametrised across multiple boundary cases to ensure robustness.
"""

from __future__ import annotations

import pytest

from loan_risk.config import settings
from loan_risk.models.predictor import ModelRegistry, RiskPredictor

pytestmark = pytest.mark.ml

APPLICATION = {
    "Age": 45,
    "AnnualIncome": 95_000,
    "CreditScore": 720,
    "EmploymentStatus": 0,
    "EducationLevel": 4,
    "LoanAmount": 25_000,
    "LoanDuration": 36,
    "MonthlyDebtPayments": 400,
    "CreditCardUtilizationRate": 0.25,
    "DebtToIncomeRatio": 0.15,
    "BankruptcyHistory": 0,
    "LoanPurpose": 3,
    "PreviousLoanDefaults": 0,
    "PaymentHistory": 28,
    "LengthOfCreditHistory": 15,
    "SavingsAccountBalance": 35_000,
    "CheckingAccountBalance": 8_000,
    "TotalLiabilities": 30_000,
    "JobTenure": 10,
    "NetWorth": 220_000,
}


@pytest.fixture
def predictor(trained_pipeline) -> RiskPredictor:
    registry = ModelRegistry(settings)
    registry.pipeline = trained_pipeline
    registry.version = "test-2.0.0"
    return RiskPredictor(registry, settings)


def _score(predictor: RiskPredictor, **overrides) -> float:
    payload = {**APPLICATION, **overrides}
    return predictor.score(predictor.to_frame(payload))


# ── shape & range ────────────────────────────────────────────────────────
def test_single_prediction_returns_a_well_formed_assessment(predictor):
    """Output must contain all required fields with valid types and ranges."""
    result = predictor.predict(dict(APPLICATION))
    assert isinstance(result.is_approved, bool)
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_tier in {"LOW", "MEDIUM", "HIGH"}
    assert result.latency_ms >= 0.0


@pytest.mark.parametrize("credit_score", [300, 500, 650, 750, 850])
def test_probability_is_bounded_across_credit_score_range(predictor, credit_score):
    """Probability must stay in [0, 1] for all valid credit scores,
    including boundary values at the schema extremes (300, 850).
    """
    prob = _score(predictor, CreditScore=credit_score)
    assert 0.0 <= prob <= 1.0


# ── directional expectations ─────────────────────────────────────────────
def test_higher_credit_score_does_not_reduce_approval_probability(predictor):
    """Core domain assertion: better credit -> higher/equal approval."""
    poor = _score(predictor, CreditScore=520)
    excellent = _score(predictor, CreditScore=800)
    assert excellent >= poor, (poor, excellent)


@pytest.mark.parametrize(
    "low,high",
    [
        (300, 500),
        (500, 700),
        (700, 850),
    ],
)
def test_directional_credit_score_across_boundary_pairs(predictor, low, high):
    """Directional expectation must hold across multiple credit score
    boundary pairs, not just a single convenient pair.
    """
    score_low = _score(predictor, CreditScore=low)
    score_high = _score(predictor, CreditScore=high)
    assert score_high >= score_low, (
        f"CreditScore {low}->{high}: " f"prob {score_low:.4f}->{score_high:.4f}"
    )


def test_lower_debt_to_income_does_not_reduce_probability(predictor):
    """Less debt relative to income should be at least as good."""
    high_dti = _score(predictor, DebtToIncomeRatio=0.8)
    low_dti = _score(predictor, DebtToIncomeRatio=0.05)
    assert low_dti >= high_dti, f"DTI 0.8->{0.05}: prob {high_dti:.4f}->{low_dti:.4f}"


# ── invariance expectations ──────────────────────────────────────────────
def test_prediction_is_invariant_to_repeated_calls(predictor):
    """Determinism: the same payload must always yield the same score.

    This test caught a real defect with n_jobs=-1 causing non-deterministic
    thread ordering. The estimator is now pinned to n_jobs=1.
    """
    scores = [_score(predictor) for _ in range(5)]
    for score in scores[1:]:
        assert score == pytest.approx(scores[0], abs=1e-12, rel=0)
    rounded = {round(score, 4) for score in scores}
    assert len(rounded) == 1


def test_prediction_invariant_to_education_level_change(predictor):
    """Education level has low feature importance; changing it should
    not drastically alter the approval probability (within 0.15).
    """
    score_high_ed = _score(predictor, EducationLevel=4)
    score_low_ed = _score(predictor, EducationLevel=0)
    assert abs(score_high_ed - score_low_ed) < 0.15, (
        f"Education 4 vs 0 caused large shift: "
        f"{score_high_ed:.4f} vs {score_low_ed:.4f}"
    )
