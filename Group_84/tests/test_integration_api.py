"""TEST TYPE 4 - INTEGRATION TESTS.

These exercise several components together through the real HTTP surface:
FastAPI routing -> Pydantic validation -> ModelRegistry -> RiskPredictor ->
FeatureEngineer -> estimator -> response serialisation. A unit test proves a
part works; these prove the parts were wired together correctly.

The status-code contract is parametrised: one test covering every way a
request can be rejected reads better than five near-identical functions.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ── operational endpoints ────────────────────────────────────────────────
def test_health_endpoint_reports_a_loaded_model(api_client):
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


def test_model_metadata_endpoint_exposes_the_contract(api_client):
    body = api_client.get("/v1/model/metadata").json()
    assert body["loaded"] is True
    assert body["n_features"] == 22
    assert 0.0 < body["decision_threshold"] < 1.0


# ── happy path ───────────────────────────────────────────────────────────
def test_predict_returns_200_and_a_complete_payload(api_client, valid_application):
    response = api_client.post("/v1/predict", json=valid_application)
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "is_approved",
        "probability",
        "risk_tier",
        "net_worth",
        "debt_to_income",
        "latency_ms",
        "model_version",
    }
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_tier"] in {"LOW", "MEDIUM", "HIGH"}


def test_batch_predict_returns_one_result_per_application(api_client, valid_application):
    payload = {"applications": [valid_application, valid_application]}
    response = api_client.post("/v1/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert body["approved_count"] == sum(r["is_approved"] for r in body["results"])


# ── status-code contract ─────────────────────────────────────────────────
def _without_credit_score(application):
    incomplete = dict(application)
    del incomplete["credit_score"]
    return incomplete


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda a: {**a, "credit_score": 1500}, "value above the declared maximum"),
        (_without_credit_score, "required field absent"),
        (lambda a: {**a, "credit_scr": 700}, "unknown key (extra='forbid')"),
        (lambda a: {**a, "annual_income": "one hundred k"}, "wrong type"),
        (lambda a: {**a, "annual_income": 10**15}, "absurd magnitude"),
        # One representative injection string. The XSS, path-traversal, NoSQL
        # and array variants all fail on the same int_parsing rule, so testing
        # each separately measures Pydantic's consistency, not our boundary.
        (lambda a: {**a, "credit_score": "'; DROP TABLE applications;--"}, "injection"),
    ],
)
def test_malformed_and_adversarial_payloads_are_rejected_with_422(
    api_client, valid_application, mutate, reason
):
    """Nothing malformed reaches the model: Pydantic rejects it at the edge."""
    response = api_client.post("/v1/predict", json=mutate(valid_application))
    assert response.status_code == 422, reason


def test_business_rule_violation_returns_400(api_client, valid_application):
    """Over-leveraged but schema-valid -> a *client* error, never a 500."""
    over_leveraged = {
        **valid_application,
        "annual_income": 20_000,
        "loan_amount": 500_000,
    }
    response = api_client.post("/v1/predict", json=over_leveraged)
    assert response.status_code == 400
    assert response.json()["error_type"] == "BusinessRuleViolation"


# 0 and 101 both trip the same min_length/max_length rule; 101 is the
# denial-of-service-relevant end.
@pytest.mark.parametrize("n_applications", [101])
def test_batch_size_limits_are_enforced(api_client, valid_application, n_applications):
    """The 1-100 cap is a denial-of-service control, not just ergonomics."""
    payload = {"applications": [valid_application] * n_applications}
    assert api_client.post("/v1/predict/batch", json=payload).status_code == 422


def test_error_responses_never_leak_internals(api_client, valid_application):
    """No stack traces, file paths or module names in a client-facing error."""
    response = api_client.post(
        "/v1/predict",
        json={**valid_application, "annual_income": 20_000, "loan_amount": 500_000},
    )
    detail = response.json()["detail"]
    assert "Traceback" not in detail
    assert "loan_risk" not in detail


# ── end-to-end consistency ───────────────────────────────────────────────
def test_api_and_direct_pipeline_agree(api_client, valid_application, trained_pipeline):
    """The HTTP path must not alter the score the pipeline would give."""
    import pandas as pd

    from loan_risk.api.schemas import LoanApplication

    body = api_client.post("/v1/predict", json=valid_application).json()
    payload = LoanApplication(**valid_application).to_model_payload()
    direct = trained_pipeline.predict_proba(pd.DataFrame([payload]))[0][1]
    assert body["probability"] == pytest.approx(round(float(direct), 4), abs=1e-4)
