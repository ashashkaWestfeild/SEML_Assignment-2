# ============================================================================
# ASSIGNMENT I CODE - VERBATIM, UNMODIFIED.
#
# This is Group 84's own prior submission (app/pipeline.py), kept as the
# authentic 'before' half of the research-vs-production comparison in
# Objective 1.2. It is real code that really shipped, not an illustrative
# example written to be criticised.
#
# Note extract_features() below: it computes LoanToIncomeRatio and
# SavingsToLoanRatio inline for SERVING, while data/prepare_data.py
# computes the same two formulas independently for TRAINING. Two copies of
# one definition is the training/serving skew that Assignment II removes.
# ============================================================================

import time
import pandas as pd
from app.schemas import LoanApplicationInput, LoanApprovalResult
from app.config import settings

# =========================================================
# FILTER 1 : INPUT VALIDATION FILTER
# Business-rule validation checks on raw input
# Quality Attribute: Robustness
# =========================================================
def validate_input(app_input: LoanApplicationInput) -> LoanApplicationInput:
    if app_input.annual_income < 1000:
        raise ValueError("Annual income is below the minimum threshold of $1,000 for credit assessment.")

    if app_input.loan_amount > app_input.annual_income * 5:
        raise ValueError("Requested loan amount exceeds 500% of the applicant's annual income.")

    return app_input

# =========================================================
# FILTER 2 : FEATURE EXTRACTION FILTER
# Extract and normalize features, compute derivatives
# Quality Attribute: Maintainability
# =========================================================
def extract_features(app_input: LoanApplicationInput) -> tuple:
    # Derived features
    loan_to_income_ratio = app_input.loan_amount / (app_input.annual_income + 1)
    savings_to_loan_ratio = app_input.savings_account_balance / (app_input.loan_amount + 1)

    # Building DataFrame for model input matching training columns
    feature_vector = pd.DataFrame([[
        app_input.age,
        app_input.annual_income,
        app_input.credit_score,
        app_input.employment_status,
        app_input.education_level,
        app_input.loan_amount,
        app_input.loan_duration,
        app_input.monthly_debt_payments,
        app_input.credit_card_utilization_rate,
        app_input.debt_to_income_ratio,
        app_input.bankruptcy_history,
        app_input.loan_purpose,
        app_input.previous_loan_defaults,
        app_input.payment_history,
        app_input.length_of_credit_history,
        app_input.savings_account_balance,
        app_input.checking_account_balance,
        app_input.total_liabilities,
        app_input.job_tenure,
        app_input.net_worth,
        loan_to_income_ratio,
        savings_to_loan_ratio,
    ]], columns=settings.FEATURES)

    return feature_vector, app_input.net_worth, app_input.debt_to_income_ratio

# =========================================================
# FILTER 3 : MODEL RUNNING FILTER
# Run classification inference using the trained Random Forest model
# Quality Attribute: Performance
# =========================================================
def run_model(feature_vector: pd.DataFrame, model) -> tuple:
    probabilities = model.predict_proba(feature_vector)[0]
    prob_approved = float(probabilities[1])
    is_approved = prob_approved >= settings.DEFAULT_DECISION_THRESHOLD
    return is_approved, prob_approved

# =========================================================
# FILTER 4 : RESPONSE FORMATTER FILTER
# Determine risk tiering and construct output schema
# Quality Attribute: Reliability
# =========================================================
def format_response(
    is_approved: bool,
    prob: float,
    net_worth: float,
    debt_to_income: float,
    previous_defaults: int,
    latency_ms: float
) -> LoanApprovalResult:
    if prob >= 0.70 and previous_defaults == 0:
        risk_tier = "LOW"
    elif prob >= 0.40 and previous_defaults == 0:
        risk_tier = "MEDIUM"
    else:
        risk_tier = "HIGH"

    result = LoanApprovalResult(
        is_approved=is_approved,
        probability=round(prob, 4),
        risk_tier=risk_tier,
        net_worth=net_worth,
        debt_to_income=round(debt_to_income, 4),
        latency_ms=round(latency_ms, 2)
    )
    return result

# =========================================================
# PIPELINE EXECUTION (PIPE CONNECTING FILTERS)
# =========================================================
def execute_pipeline(app_input: LoanApplicationInput, model) -> LoanApprovalResult:
    start_time = time.perf_counter()

    # Stage 1: Validation (Filter 1)
    validated_input = validate_input(app_input)

    # Stage 2: Feature Extraction (Filter 2)
    features, net_worth, debt_to_income = extract_features(validated_input)

    # Stage 3: Inference (Filter 3)
    is_approved, prob = run_model(features, model)

    # Calculate latency before format step
    latency_ms = (time.perf_counter() - start_time) * 1000.0

    # Stage 4: Formatting (Filter 4)
    result = format_response(
        is_approved,
        prob,
        net_worth,
        debt_to_income,
        validated_input.previous_loan_defaults,
        latency_ms
    )
    return result
