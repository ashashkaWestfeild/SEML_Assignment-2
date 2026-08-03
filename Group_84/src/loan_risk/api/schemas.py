"""Pydantic request/response schemas -- the API's data contract.

Good-API-design notes:
  * Every field is bounded (``ge``/``le``), so malformed or adversarial values
    are rejected at the edge with 422 before reaching the model.
  * ``extra="forbid"`` rejects unknown keys instead of silently ignoring them,
    which catches client-side typos (``credit_scr``) that would otherwise be
    scored with a default value.
  * ``json_schema_extra`` supplies a worked example, so the generated
    OpenAPI/Swagger page is self-service documentation.
"""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE_APPLICATION = {
    "age": 45,
    "annual_income": 95000,
    "credit_score": 720,
    "employment_status": 0,
    "education_level": 4,
    "loan_amount": 25000,
    "loan_duration": 36,
    "monthly_debt_payments": 400,
    "credit_card_utilization_rate": 0.25,
    "debt_to_income_ratio": 0.15,
    "bankruptcy_history": 0,
    "loan_purpose": 3,
    "previous_loan_defaults": 0,
    "payment_history": 28,
    "length_of_credit_history": 15,
    "savings_account_balance": 35000,
    "checking_account_balance": 8000,
    "total_liabilities": 30000,
    "job_tenure": 10,
    "net_worth": 220000,
}


class LoanApplication(BaseModel):
    """One consumer-credit application submitted for scoring."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": EXAMPLE_APPLICATION},
    )

    age: int = Field(..., ge=18, le=100, description="Applicant age in years")
    annual_income: int = Field(
        ..., gt=0, le=10_000_000, description="Gross annual income"
    )
    credit_score: int = Field(..., ge=300, le=850, description="Bureau score (300-850)")
    employment_status: int = Field(
        ..., ge=0, le=2, description="0=Employed, 1=Self-Employed, 2=Unemployed"
    )
    education_level: int = Field(
        ...,
        ge=0,
        le=4,
        description="0=Associate, 1=Bachelor, 2=Doctorate, 3=High School, 4=Master",
    )
    loan_amount: int = Field(..., gt=0, le=10_000_000, description="Principal requested")
    loan_duration: int = Field(..., ge=12, le=120, description="Term in months")
    monthly_debt_payments: int = Field(..., ge=0, le=1_000_000)
    credit_card_utilization_rate: float = Field(..., ge=0.0, le=1.0)
    debt_to_income_ratio: float = Field(..., ge=0.0, le=10.0)
    bankruptcy_history: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    loan_purpose: int = Field(
        ...,
        ge=0,
        le=4,
        description="0=Auto, 1=Debt Consolidation, 2=Education, 3=Home, 4=Other",
    )
    previous_loan_defaults: int = Field(..., ge=0, le=1, description="0=No, 1=Yes")
    payment_history: int = Field(..., ge=0, le=600, description="Months of clean history")
    length_of_credit_history: int = Field(..., ge=0, le=90)
    savings_account_balance: int = Field(..., ge=0, le=100_000_000)
    checking_account_balance: int = Field(..., ge=0, le=100_000_000)
    total_liabilities: int = Field(..., ge=0, le=100_000_000)
    job_tenure: int = Field(..., ge=0, le=70)
    net_worth: int = Field(..., description="Total assets minus total liabilities")

    def to_model_payload(self) -> dict:
        """Translate snake_case API fields to the model's PascalCase contract."""
        return {
            "Age": self.age,
            "AnnualIncome": self.annual_income,
            "CreditScore": self.credit_score,
            "EmploymentStatus": self.employment_status,
            "EducationLevel": self.education_level,
            "LoanAmount": self.loan_amount,
            "LoanDuration": self.loan_duration,
            "MonthlyDebtPayments": self.monthly_debt_payments,
            "CreditCardUtilizationRate": self.credit_card_utilization_rate,
            "DebtToIncomeRatio": self.debt_to_income_ratio,
            "BankruptcyHistory": self.bankruptcy_history,
            "LoanPurpose": self.loan_purpose,
            "PreviousLoanDefaults": self.previous_loan_defaults,
            "PaymentHistory": self.payment_history,
            "LengthOfCreditHistory": self.length_of_credit_history,
            "SavingsAccountBalance": self.savings_account_balance,
            "CheckingAccountBalance": self.checking_account_balance,
            "TotalLiabilities": self.total_liabilities,
            "JobTenure": self.job_tenure,
            "NetWorth": self.net_worth,
        }


class BatchLoanApplications(BaseModel):
    """Bounded batch request -- the cap is itself a denial-of-service control."""

    model_config = ConfigDict(extra="forbid")

    applications: List[LoanApplication] = Field(..., min_length=1, max_length=100)


class RiskAssessmentResponse(BaseModel):
    """Scoring result returned to the caller."""

    is_approved: bool = Field(..., description="Decision at the configured threshold")
    probability: float = Field(..., ge=0.0, le=1.0, description="P(approved)")
    risk_tier: Literal["LOW", "MEDIUM", "HIGH"]
    net_worth: int
    debt_to_income: float
    latency_ms: float = Field(..., description="Server-side scoring latency")
    model_version: str


class BatchRiskAssessmentResponse(BaseModel):
    """Batch scoring result plus a small roll-up for the caller's dashboard."""

    results: List[RiskAssessmentResponse]
    count: int
    approved_count: int


class HealthResponse(BaseModel):
    """Liveness/readiness payload consumed by the orchestrator."""

    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_version: str
    environment: str
    api_version: str


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    detail: str
    error_type: str
