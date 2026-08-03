"""FastAPI serving layer.

API-design decisions worth defending:

  * Resource-oriented, versioned paths (``/v1/predict``) so a breaking change
    can ship as ``/v2`` without stranding existing callers.
  * Correct, differentiated status codes:
        200 scored | 422 schema violation (FastAPI/Pydantic, automatic)
        400 business-rule rejection | 503 model not loaded | 500 unexpected
  * Declared ``response_model`` on every route, so the OpenAPI contract is
    generated from the code and cannot drift from it.
  * Domain exceptions are translated by dedicated handlers, so internal
    messages and stack traces never leak to the client.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from loan_risk.api.schemas import (
    BatchLoanApplications,
    BatchRiskAssessmentResponse,
    ErrorResponse,
    HealthResponse,
    LoanApplication,
    RiskAssessmentResponse,
)
from loan_risk.config import settings
from loan_risk.exceptions import BusinessRuleViolation, ModelNotLoadedError
from loan_risk.logging_utils import get_logger
from loan_risk.models.predictor import ModelRegistry, RiskPredictor

logger = get_logger(__name__)

registry = ModelRegistry()
predictor = RiskPredictor(registry)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the artifact once at start-up; stay degraded rather than crash."""
    if registry.load():
        logger.info("service_started", extra={"model_version": registry.version})
    else:
        logger.warning(
            "service_started_degraded", extra={"reason": "model artifact unavailable"}
        )
    yield
    logger.info("service_stopped")


app = FastAPI(
    title=settings.name,
    version=settings.version,
    description=(
        "Real-time consumer-credit risk assessment. Submit an application and "
        "receive an approval decision, a calibrated probability and a risk tier."
    ),
    lifespan=lifespan,
)


# --------------------------------------------------------------- error mapping
@app.exception_handler(BusinessRuleViolation)
async def business_rule_handler(_: Request, exc: BusinessRuleViolation) -> JSONResponse:
    """Policy rejection is a *client* error -> 400, not 500."""
    logger.warning("request_rejected_by_business_rule", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            detail=str(exc), error_type="BusinessRuleViolation"
        ).model_dump(),
    )


@app.exception_handler(ModelNotLoadedError)
async def model_missing_handler(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
    """No artifact -> 503 so load balancers retry elsewhere."""
    logger.error("request_failed_model_unavailable", extra={"detail": str(exc)})
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            detail="Model is not available.", error_type="ModelNotLoadedError"
        ).model_dump(),
    )


# ---------------------------------------------------------------------- routes
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe for Kubernetes."""
    return HealthResponse(
        status="healthy" if registry.is_loaded else "degraded",
        model_loaded=registry.is_loaded,
        model_version=registry.version,
        environment=settings.env,
        api_version=settings.version,
    )


@app.get("/v1/model/metadata", tags=["ops"])
def model_metadata() -> dict:
    """Expose the served model's identity and its offline training metrics."""
    return {
        "model_version": registry.version,
        "loaded": registry.is_loaded,
        "n_features": settings.model.n_features,
        "decision_threshold": settings.model.default_threshold,
        "training_metrics": registry.training_metrics,
    }


@app.post(
    "/v1/predict",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse, "description": "Business-rule rejection"},
        422: {"model": ErrorResponse, "description": "Schema validation failure"},
        503: {"model": ErrorResponse, "description": "Model unavailable"},
    },
)
def predict(application: LoanApplication) -> RiskAssessmentResponse:
    """Score a single loan application."""
    assessment = predictor.predict(application.to_model_payload())
    return RiskAssessmentResponse(**assessment.as_dict())


@app.post(
    "/v1/predict/batch",
    response_model=BatchRiskAssessmentResponse,
    status_code=status.HTTP_200_OK,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict_batch(batch: BatchLoanApplications) -> BatchRiskAssessmentResponse:
    """Score up to 100 applications in a single round trip."""
    results = [
        RiskAssessmentResponse(**predictor.predict(item.to_model_payload()).as_dict())
        for item in batch.applications
    ]
    approved = sum(r.is_approved for r in results)
    logger.info("batch_scored", extra={"count": len(results), "approved_count": approved})
    return BatchRiskAssessmentResponse(
        results=results, count=len(results), approved_count=approved
    )
