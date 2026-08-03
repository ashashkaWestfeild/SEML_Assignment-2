import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Dynamically resolve paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(SCRIPT_DIR, "Loan.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "loan_data_processed.csv")


def prepare_data():
    print("[DATA PREP] Loading raw Kaggle dataset...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"  Raw shape: {df.shape}")

    # =========================================================
    # STEP 1: Drop non-predictive columns
    # =========================================================
    drop_cols = ['ApplicationDate', 'RiskScore', 'InterestRate',
                 'BaseInterestRate', 'MonthlyLoanPayment', 'TotalDebtToIncomeRatio']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"  After dropping non-predictive columns: {df.shape}")

    # =========================================================
    # STEP 2: Remove redundant features (correlation > 0.95)
    # =========================================================
    # MonthlyIncome ~ AnnualIncome (r=0.99) -> drop MonthlyIncome
    # Experience ~ Age (r=0.98) -> drop Experience
    # TotalAssets ~ NetWorth (r=0.98) -> drop TotalAssets
    redundant = ['MonthlyIncome', 'Experience', 'TotalAssets']
    df = df.drop(columns=[c for c in redundant if c in df.columns])
    print(f"  After removing redundant features: {df.shape}")

    # =========================================================
    # STEP 3: Remove low-importance features (importance < 0.005)
    # =========================================================
    low_importance = ['MaritalStatus', 'NumberOfDependents', 'HomeOwnershipStatus',
                      'NumberOfOpenCreditLines', 'NumberOfCreditInquiries',
                      'UtilityBillsPaymentHistory']
    df = df.drop(columns=[c for c in low_importance if c in df.columns])
    print(f"  After removing low-importance features: {df.shape}")

    # =========================================================
    # STEP 4: Encode categorical variables
    # =========================================================
    cat_mappings = {
        'EmploymentStatus': {'Employed': 0, 'Self-Employed': 1, 'Unemployed': 2},
        'EducationLevel': {"Associate": 0, "Bachelor": 1, "Doctorate": 2, "High School": 3, "Master": 4},
        'LoanPurpose': {'Auto': 0, 'Debt Consolidation': 1, 'Education': 2, 'Home': 3, 'Other': 4}
    }
    for col, mapping in cat_mappings.items():
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].map(mapping).fillna(0).astype(int)

    # =========================================================
    # STEP 5: Engineer derived features
    # =========================================================
    df['LoanToIncomeRatio'] = df['LoanAmount'] / (df['AnnualIncome'] + 1)
    df['SavingsToLoanRatio'] = df['SavingsAccountBalance'] / (df['LoanAmount'] + 1)
    print(f"  After feature engineering: {df.shape}")

    # =========================================================
    # STEP 6: Save processed dataset
    # =========================================================
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[DATA PREP] Processed dataset saved to: {OUTPUT_PATH}")
    print(f"  Final columns ({len(df.columns)}): {list(df.columns)}")


if __name__ == "__main__":
    prepare_data()
