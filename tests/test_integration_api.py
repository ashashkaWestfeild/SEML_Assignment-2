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


# ── operational endpoints ────────────────────────────────────────────────
def test_model_metadata_endpoint_exposes_the_contract(api_client):
    body = api_client.get("/v1/model/metadata").json()
    assert body["loaded"] is True
    assert body["n_features"] == 22
    assert 0.0 < body["decision_threshold"] < 1.0


def test_openapi_schema_is_generated(api_client):
    """The contract must be machine-readable for client generation."""
    schema = api_client.get("/openapi.json").json()
    assert "/v1/predict" in schema["paths"]
    assert "/v1/predict/batch" in schema["paths"]


def test_unknown_route_returns_404(api_client):
    assert api_client.get("/v1/does-not-exist").status_code == 404


# ── batch endpoint ───────────────────────────────────────────────────────
def test_batch_predict_returns_one_result_per_application(api_client, valid_application):
    payload = {"applications": [valid_application, valid_application]}
    response = api_client.post("/v1/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert body["approved_count"] == sum(r["is_approved"] for r in body["results"])


def test_oversized_batch_is_rejected(api_client, valid_application):
    """The 100-application cap is a denial-of-service control."""
    payload = {"applications": [valid_application] * 101}
    assert api_client.post("/v1/predict/batch", json=payload).status_code == 422


def test_empty_batch_is_rejected(api_client):
    response = api_client.post("/v1/predict/batch", json={"applications": []})
    assert response.status_code == 422


# ── status-code contract ─────────────────────────────────────────────────
def test_missing_required_field_returns_422(api_client, valid_application):
    incomplete = dict(valid_application)
    del incomplete["credit_score"]
    assert api_client.post("/v1/predict", json=incomplete).status_code == 422


def test_wrong_type_returns_422(api_client, valid_application):
    response = api_client.post(
        "/v1/predict", json={**valid_application, "annual_income": "one hundred k"}
    )
    assert response.status_code == 422


# ── security-relevant input handling ─────────────────────────────────────
@pytest.mark.parametrize(
    "malicious",
    [
        "'; DROP TABLE applications;--",
        "<script>alert(1)</script>",
        "../../etc/passwd",
        {"$ne": None},
        [1, 2, 3],
    ],
)
def test_adversarial_payloads_are_rejected_at_the_boundary(
    api_client, valid_application, malicious
):
    """Injection strings never reach the model: Pydantic rejects them with 422."""
    response = api_client.post(
        "/v1/predict", json={**valid_application, "credit_score": malicious}
    )
    assert response.status_code == 422


def test_extreme_numeric_values_are_rejected(api_client, valid_application):
    response = api_client.post(
        "/v1/predict", json={**valid_application, "annual_income": 10**15}
    )
    assert response.status_code == 422


def test_error_responses_never_leak_internals(api_client, valid_application):
    """No stack traces, file paths or module names in a client-facing error."""
    response = api_client.post(
        "/v1/predict",
        json={**valid_application, "annual_income": 20000, "loan_amount": 500000},
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
