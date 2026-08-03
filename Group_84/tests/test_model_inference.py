"""TEST TYPE 3 (cont.) - ML BEHAVIOURAL TESTS: INFERENCE (Objective 2.7b).

Three families, following Zinkevich / "ML Test Score" practice:

  * SHAPE & RANGE      -- outputs are structurally valid for every input.
  * DIRECTIONAL        -- a change the domain says should push the score one
                          way actually does (monotonic expectations).
  * INVARIANCE         -- a change the domain says is irrelevant leaves the
                          score untouched.
Plus a latency check against the 150 ms SLA carried over from Assignment I.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.config import settings
from loan_risk.exceptions import BusinessRuleViolation, ModelNotLoadedError
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
def test_batch_prediction_shape_is_correct(trained_pipeline, xy):
    features, _ = xy
    batch = features.head(37)
    probabilities = trained_pipeline.predict_proba(batch)
    assert probabilities.shape == (37, 2)


def test_probabilities_are_valid_and_sum_to_one(trained_pipeline, xy):
    features, _ = xy
    probabilities = trained_pipeline.predict_proba(features.head(200))
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)


def test_single_prediction_returns_a_well_formed_assessment(predictor):
    result = predictor.predict(dict(APPLICATION))
    assert isinstance(result.is_approved, bool)
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_tier in {"LOW", "MEDIUM", "HIGH"}
    assert result.latency_ms >= 0.0


@pytest.mark.parametrize("credit_score", [300, 500, 700, 850])
def test_output_stays_in_range_across_the_credit_spectrum(predictor, credit_score):
    probability = _score(predictor, CreditScore=credit_score)
    assert 0.0 <= probability <= 1.0
    assert np.isfinite(probability)


# ── directional expectations ─────────────────────────────────────────────
def test_higher_credit_score_does_not_reduce_approval_probability(predictor):
    poor = _score(predictor, CreditScore=520)
    excellent = _score(predictor, CreditScore=800)
    assert excellent >= poor, (poor, excellent)


def test_higher_debt_to_income_does_not_increase_approval_probability(predictor):
    low = _score(predictor, DebtToIncomeRatio=0.10)
    high = _score(predictor, DebtToIncomeRatio=0.95)
    assert high <= low, (low, high)


def test_a_much_larger_loan_does_not_increase_approval_probability(predictor):
    modest = _score(predictor, LoanAmount=15_000)
    aggressive = _score(predictor, LoanAmount=140_000)
    assert aggressive <= modest, (modest, aggressive)


# ── invariance expectations ──────────────────────────────────────────────
def test_prediction_is_invariant_to_dictionary_key_order(predictor):
    reversed_payload = dict(reversed(list(APPLICATION.items())))
    assert predictor.score(predictor.to_frame(reversed_payload)) == pytest.approx(
        _score(predictor)
    )


def test_prediction_is_invariant_to_repeated_calls(predictor):
    """Determinism: the same payload must always yield the same score.

    This test caught a real defect. With ``n_jobs=-1`` the forest sums tree
    votes in a non-deterministic thread order, so repeated calls differed at
    ~5e-17. Harmless in isolation, but an application sitting exactly on the
    0.50 cut-off could flip between APPROVED and DENIED across identical
    requests -- unacceptable in regulated lending. The estimator is now pinned
    to a deterministic reduction, and the tolerance below is the float-equality
    bound, not a workaround.
    """
    scores = [_score(predictor) for _ in range(5)]
    for score in scores[1:]:
        assert score == pytest.approx(scores[0], abs=1e-12, rel=0)
    rounded = {round(score, 4) for score in scores}
    assert len(rounded) == 1


def test_prediction_is_invariant_to_batching(trained_pipeline, xy):
    """Row-by-row scoring must equal batch scoring (no cross-row leakage)."""
    features, _ = xy
    batch = features.head(10)
    batched = trained_pipeline.predict_proba(batch)[:, 1]
    individually = np.array(
        [trained_pipeline.predict_proba(batch.iloc[[i]])[0][1] for i in range(10)]
    )
    np.testing.assert_allclose(batched, individually, atol=1e-12)


def test_prediction_is_invariant_to_column_order(trained_pipeline):
    """The FeatureEngineer must re-impose the contract order internally."""
    frame = pd.DataFrame([APPLICATION])
    shuffled = frame[list(reversed(frame.columns))]
    np.testing.assert_allclose(
        trained_pipeline.predict_proba(frame)[:, 1],
        trained_pipeline.predict_proba(shuffled)[:, 1],
    )


# ── risk tiering & business rules ────────────────────────────────────────
def test_previous_default_forces_the_high_risk_tier(predictor):
    assert predictor.assign_risk_tier(0.99, previous_defaults=1) == "HIGH"


@pytest.mark.parametrize(
    "probability,expected", [(0.95, "LOW"), (0.55, "MEDIUM"), (0.10, "HIGH")]
)
def test_risk_tier_boundaries(predictor, probability, expected):
    assert predictor.assign_risk_tier(probability, previous_defaults=0) == expected


def test_business_rule_rejects_over_leveraged_application(predictor):
    with pytest.raises(BusinessRuleViolation, match="annual income"):
        predictor.predict({**APPLICATION, "AnnualIncome": 20_000, "LoanAmount": 500_000})


def test_business_rule_rejects_sub_threshold_income(predictor):
    with pytest.raises(BusinessRuleViolation, match="below the minimum"):
        predictor.predict({**APPLICATION, "AnnualIncome": 500, "LoanAmount": 1_000})


def test_validation_filter_does_not_mutate_the_payload(predictor):
    payload = dict(APPLICATION)
    snapshot = dict(payload)
    predictor.validate_business_rules(payload)
    assert payload == snapshot


def test_inference_without_a_loaded_model_raises(predictor):
    empty = RiskPredictor(ModelRegistry(settings), settings)
    with pytest.raises(ModelNotLoadedError):
        empty.predict(dict(APPLICATION))


# ── performance ──────────────────────────────────────────────────────────
def test_inference_latency_is_within_sla(predictor):
    """Average end-to-end scoring latency must stay under the 150 ms SLA."""
    import time

    runs = 50
    start = time.perf_counter()
    for _ in range(runs):
        predictor.predict(dict(APPLICATION))
    average_ms = (time.perf_counter() - start) / runs * 1000.0
    assert average_ms < settings.gates.max_latency_ms, (
        f"Average latency {average_ms:.2f} ms exceeded the "
        f"{settings.gates.max_latency_ms} ms SLA"
    )
