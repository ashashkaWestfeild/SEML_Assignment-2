"""Shared pytest fixtures.

Fixtures build every artifact the tests need *in memory* (synthetic frames, a
small trained pipeline, a TestClient with a stub registry), so the suite runs
in seconds on a clean checkout with no trained model on disk -- a hard
requirement for running QA inside CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from loan_risk.config import settings  # noqa: E402
from loan_risk.models.trainer import ModelTrainer  # noqa: E402

BASE_APPLICATION = {
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


def make_synthetic_frame(n_rows: int = 4000, seed: int = 7) -> pd.DataFrame:
    """Small, schema-valid dataset with a genuine, learnable signal."""
    rng = np.random.default_rng(seed)
    annual_income = rng.integers(20_000, 250_000, n_rows)
    loan_amount = (annual_income * rng.uniform(0.05, 1.5, n_rows)).astype(int)
    savings = (annual_income * rng.uniform(0.0, 0.8, n_rows)).astype(int)
    credit_score = np.clip(rng.normal(670, 80, n_rows).astype(int), 300, 850)
    dti = np.round(np.clip(rng.beta(2, 5, n_rows) * 1.2, 0, 5), 3)
    cc_util = np.round(np.clip(rng.beta(2, 5, n_rows), 0, 1), 3)
    prev_default = (rng.uniform(0, 1, n_rows) < 0.15).astype(int)

    frame = pd.DataFrame(
        {
            "Age": rng.integers(21, 70, n_rows),
            "AnnualIncome": annual_income,
            "CreditScore": credit_score,
            "EmploymentStatus": rng.integers(0, 3, n_rows),
            "EducationLevel": rng.integers(0, 5, n_rows),
            "LoanAmount": loan_amount,
            "LoanDuration": rng.choice([12, 24, 36, 60], n_rows),
            "MonthlyDebtPayments": rng.integers(0, 4000, n_rows),
            "CreditCardUtilizationRate": cc_util,
            "DebtToIncomeRatio": dti,
            "BankruptcyHistory": (rng.uniform(0, 1, n_rows) < 0.06).astype(int),
            "LoanPurpose": rng.integers(0, 5, n_rows),
            "PreviousLoanDefaults": prev_default,
            "PaymentHistory": rng.integers(0, 48, n_rows),
            "LengthOfCreditHistory": rng.integers(0, 40, n_rows),
            "SavingsAccountBalance": savings,
            "CheckingAccountBalance": rng.integers(0, 60_000, n_rows),
            "TotalLiabilities": rng.integers(0, 300_000, n_rows),
            "JobTenure": rng.integers(0, 30, n_rows),
            "NetWorth": rng.integers(-50_000, 900_000, n_rows),
        }
    )
    frame["LoanToIncomeRatio"] = frame["LoanAmount"] / (frame["AnnualIncome"] + 1)
    frame["SavingsToLoanRatio"] = frame["SavingsAccountBalance"] / (
        frame["LoanAmount"] + 1
    )

    # Latent creditworthiness -> label (linear + interaction terms).
    z = (
        1.4 * (credit_score - 670) / 80
        - 2.0 * dti
        - 1.5 * cc_util
        - 1.6 * prev_default
        - 1.2 * (frame["LoanToIncomeRatio"] > 1.0).astype(float)
        + 0.6
        + rng.normal(0, 0.4, n_rows)
    )
    frame["LoanApproved"] = (1 / (1 + np.exp(-z)) > 0.5).astype(int)
    return frame


@pytest.fixture(scope="session")
def synthetic_frame() -> pd.DataFrame:
    return make_synthetic_frame()


@pytest.fixture(scope="session")
def xy(synthetic_frame: pd.DataFrame):
    return (
        synthetic_frame[settings.model.features].copy(),
        synthetic_frame[settings.model.target].copy(),
    )


@pytest.fixture(scope="session")
def trained_pipeline(xy):
    """A real fitted pipeline -- built once and shared across the suite."""
    features, labels = xy
    trainer = ModelTrainer(settings)
    trainer.train(features, labels)
    return trainer.pipeline


@pytest.fixture(scope="session")
def api_client(trained_pipeline):
    """TestClient whose registry is primed with the in-memory pipeline."""
    from fastapi.testclient import TestClient

    from loan_risk.api import app as api_module

    with TestClient(api_module.app) as client:
        # Override AFTER start-up: the lifespan hook loads the on-disk artifact,
        # and the tests must assert against the in-memory pipeline they built.
        api_module.registry.pipeline = trained_pipeline
        api_module.registry.version = "test-2.0.0"
        api_module.registry.training_metrics = {"accuracy": 0.9}
        yield client


@pytest.fixture
def valid_application() -> dict:
    return dict(BASE_APPLICATION)
