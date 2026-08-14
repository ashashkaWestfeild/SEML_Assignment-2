"""TEST TYPE 2 - INTEGRATION TESTS.

These exercise several components together through the real HTTP surface:
FastAPI routing -> Pydantic validation -> ModelRegistry -> RiskPredictor ->
FeatureEngineer -> estimator -> response serialisation. A unit test proves a
part works; these prove the parts were wired together correctly.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


# ── operational endpoints ────────────────────────────────────────────────
def test_health_endpoint_reports_a_loaded_model(api_client):
    """GET /health must return 200 with model_loaded=True when registry is primed."""
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True


# ── happy path ───────────────────────────────────────────────────────────
def test_predict_returns_200_and_a_complete_payload(api_client, valid_application):
    """POST /v1/predict with a valid payload must return 200 and all expected fields."""
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


# ── schema validation (422) ──────────────────────────────────────────────
def test_predict_rejects_out_of_range_credit_score_with_422(
    api_client, valid_application
):
    """credit_score=1500 exceeds the declared maximum (850) and must be rejected."""
    payload = {**valid_application, "credit_score": 1500}
    response = api_client.post("/v1/predict", json=payload)
    assert response.status_code == 422


# ── business rule rejection (400) ────────────────────────────────────────
def test_predict_rejects_overleveraged_application_with_400(
    api_client, valid_application
):
    """Loan amount exceeding 5x annual income must trigger a business rule rejection."""
    payload = {**valid_application, "annual_income": 20000, "loan_amount": 500000}
    response = api_client.post("/v1/predict", json=payload)
    assert response.status_code == 400
    assert "BusinessRuleViolation" in response.json()["error_type"]


# ── unknown field rejection (422) ────────────────────────────────────────
def test_predict_rejects_unknown_fields_with_422(api_client, valid_application):
    """extra='forbid' in Pydantic schema must reject payloads with unknown keys."""
    payload = {**valid_application, "unknown_field": 999}
    response = api_client.post("/v1/predict", json=payload)
    assert response.status_code == 422
