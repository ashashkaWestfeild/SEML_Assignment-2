"""Generate ``notebooks/research_prototype.ipynb``.

The notebook is the deliberately *un-engineered* prototype used as the "research
code" half of the research-vs-production comparison (Objective 1.2). It is kept
under version control precisely so the contrast with
``src/loan_risk/features/engineering.py`` is concrete and reviewable.

Run:
    python scripts/build_research_notebook.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "notebooks" / "research_prototype.ipynb"

HEADER = """\
# Loan Approval - exploratory prototype  *(RESEARCH CODE)*

**Group 84 | AIMLCZG546 SEML | Assignment II**

This notebook is the *before* artefact for Objective 1.2. It is genuine
exploratory code: it was written to answer "is there signal here?" as fast as
possible, and it succeeds at that. It is also unfit to serve traffic, and the
report explains exactly why, defect by defect.

**Do not fix this notebook.** Its value is as evidence. The engineered
counterpart is `src/loan_risk/features/engineering.py`.
"""

CELLS: list[tuple[str, str]] = [
    (
        "markdown",
        "## 1. Load the data\n\nPath is whatever was on the analyst's laptop "
        "that afternoon.",
    ),
    (
        "code",
        "import pandas as pd, numpy as np\n"
        "import os, sys, json\n"
        "from sklearn.ensemble import RandomForestClassifier\n"
        "from sklearn.model_selection import train_test_split\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.metrics import accuracy_score\n"
        "\n"
        'df = pd.read_csv("C:/Users/analyst/Desktop/loan_data_processed.csv")   '
        "# hardcoded local path\n"
        "print(df.shape)\n"
        "print( df.head() )",
    ),
    ("markdown", "## 2. Poke at it"),
    (
        "code",
        "tmp = df.describe()\n"
        "X = df.drop('LoanApproved',axis=1)\n"
        "y = df['LoanApproved']",
    ),
    ("markdown", "## 3. Feature ideas\n\nTry a few, keep whatever looks good."),
    (
        "code",
        "df['LoanToIncomeRatio'] = df['LoanAmount']/df['AnnualIncome']       "
        "# NOTE: blows up when income is 0\n"
        "df['SavingsToLoanRatio'] = df['SavingsAccountBalance']/df['LoanAmount']\n"
        "df['ratio2'] = df['MonthlyDebtPayments']*12/df['AnnualIncome']\n"
        "df['x'] = df['CreditScore']/850\n"
        "FEATURES=['Age','AnnualIncome','CreditScore','LoanAmount',"
        "'LoanToIncomeRatio','SavingsToLoanRatio','ratio2','x',"
        "'DebtToIncomeRatio','CreditCardUtilizationRate','BankruptcyHistory',"
        "'PreviousLoanDefaults']",
    ),
    ("markdown", "## 4. Fit something"),
    (
        "code",
        "X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],y,"
        "test_size=0.2)      # no random_state -> not reproducible\n"
        "m = RandomForestClassifier()\n"
        "m.fit(X_train,y_train)\n"
        "p=m.predict(X_test)\n"
        'print("acc",accuracy_score(y_test,p))',
    ),
    (
        "code",
        "def get_ratio(a,b) :\n"
        "    return a/b        # no zero guard, no types, no docstring",
    ),
    ("markdown", "## 5. Importances"),
    ("code", "imp = m.feature_importances_\nplt.barh(FEATURES,imp) ; plt.show()"),
    (
        "code",
        "# TODO: clean this up before the demo\n"
        "# TODO: why does the score move every run?\n"
        "try:\n"
        "    m.predict(pd.DataFrame())\n"
        "except:\n"
        "    pass                # bare except swallows everything",
    ),
    (
        "markdown",
        "## What is wrong with this notebook\n\n"
        "| # | Defect | Consequence in production |\n"
        "|---|---|---|\n"
        "| 1 | Hard-coded absolute path | Runs on exactly one machine |\n"
        "| 2 | `train_test_split` without `random_state` | Metrics move every "
        "run; nothing is reproducible or auditable |\n"
        "| 3 | `LoanAmount/AnnualIncome` with no guard | `ZeroDivisionError` / "
        "`inf` on a zero-income application |\n"
        "| 4 | Feature list retyped by hand | Training/serving skew the moment "
        "one copy changes |\n"
        "| 5 | Dead variables (`tmp`, `X`, `y`, `ratio2`, `x`) | Reader cannot "
        "tell what is load-bearing |\n"
        "| 6 | `except:` / `pass` | Every failure is silent |\n"
        "| 7 | No logging, no tests, no types | Nothing is observable or "
        "verifiable |\n"
        "| 8 | Logic trapped in notebook cells | Cannot be imported, reused or "
        "unit-tested |\n"
        "| 9 | Two `TODO`s, one of which is the reproducibility bug | Known "
        "defects ship |\n\n"
        "Each of these is addressed in `src/loan_risk/features/engineering.py`; "
        "the mapping is tabulated in the report.",
    ),
]


def main() -> int:
    """Write the research-prototype notebook to disk."""
    notebook = nbf.v4.new_notebook()
    notebook.cells.append(nbf.v4.new_markdown_cell(HEADER))
    for kind, source in CELLS:
        if kind == "markdown":
            notebook.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            notebook.cells.append(nbf.v4.new_code_cell(source))

    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUT)
    print(f"Wrote {OUT} ({len(notebook.cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
