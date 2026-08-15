"""TEST TYPE 2 - DATA VALIDATION / DATA-QUALITY TESTS.

These assert on the *data contract* rather than on code behaviour. In an ML
system the data is as much a dependency as a library is, so it needs its own
regression suite: schema conformance, missing values and distribution drift.

Families of near-identical checks are parametrised rather than written out one
per function -- same assertions, fewer things for a reader to hold in mind.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from loan_risk.data.ingestion import DataIngestor
from loan_risk.data.validation import DataValidator
from loan_risk.exceptions import DataIngestionError, SchemaValidationError
from loan_risk.monitoring.drift import (
    PSI_SEVERE,
    DriftMonitor,
    population_stability_index,
)

pytestmark = pytest.mark.data


# ── DQ metric 1: schema conformance ──────────────────────────────────────
def test_clean_data_conforms_fully(synthetic_frame):
    report = DataValidator().validate(synthetic_frame)
    assert report.passed
    assert report.schema_conformance_rate == 1.0


@pytest.mark.parametrize(
    "column,value,expected_violation",
    [
        ("CreditScore", 9999, "above_maximum:CreditScore"),
        ("AnnualIncome", -1, "below_minimum:AnnualIncome"),
        ("Age", 5, "below_minimum:Age"),
    ],
)
def test_out_of_range_values_are_flagged(
    synthetic_frame, column, value, expected_violation
):
    """Each declared bound must actually be enforced, not just documented."""
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], column] = value
    report = DataValidator().validate(bad)
    assert expected_violation in report.violations
    assert report.schema_conformance_rate < 1.0


def test_missing_column_is_flagged(synthetic_frame):
    bad = synthetic_frame.drop(columns=["JobTenure"])
    assert "missing_column:JobTenure" in DataValidator().validate(bad).violations


def test_strict_mode_aborts_the_build_on_violation(synthetic_frame):
    """The training entrypoint runs strict=True; a breach must stop the run."""
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], "Age"] = 5
    with pytest.raises(SchemaValidationError):
        DataValidator().validate(bad, strict=True)


def test_validator_rejects_empty_frame():
    with pytest.raises(SchemaValidationError):
        DataValidator().validate(pd.DataFrame())


# ── DQ metric 2: missing values ──────────────────────────────────────────
def test_missing_value_fraction_is_measured(synthetic_frame):
    dirty = synthetic_frame.copy().astype({"PaymentHistory": "float"})
    # Null a *fraction* of rows, not a fixed count, so this holds at any
    # fixture size.
    n_null = int(len(dirty) * 0.20)
    dirty.loc[dirty.index[:n_null], "PaymentHistory"] = np.nan
    report = DataValidator(max_missing_fraction=0.001).validate(dirty)
    assert report.missing_value_fraction > 0
    assert report.worst_column == "PaymentHistory"
    assert any(v.startswith("missing_fraction_exceeded") for v in report.violations)
    assert "nulls_in_non_nullable:PaymentHistory" in report.violations


# ── DQ metrics 3 & 4: drift (PSI, KS) ────────────────────────────────────
def test_psi_separates_stable_from_shifted_distributions():
    """PSI must be ~0 on identical samples and severe on a real shift."""
    rng = np.random.default_rng(0)
    reference = rng.normal(650, 60, 5000)
    assert population_stability_index(reference, reference.copy()) < 0.01

    shifted = rng.normal(560, 60, 5000)  # a 90-point credit-score collapse
    assert population_stability_index(reference, shifted) > PSI_SEVERE


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError):
        population_stability_index([], [1.0, 2.0])


def test_drift_monitor_flags_only_the_shifted_feature(synthetic_frame):
    """A recession hits CreditScore only; the others must stay STABLE."""
    current = synthetic_frame.copy()
    current["CreditScore"] = np.clip(current["CreditScore"] - 120, 300, 850)

    monitor = DriftMonitor(
        synthetic_frame.copy(), features=["CreditScore", "Age", "LoanAmount"]
    )
    summary = monitor.summary_frame(current)

    credit_row = summary.loc[summary["feature"] == "CreditScore"].iloc[0]
    age_row = summary.loc[summary["feature"] == "Age"].iloc[0]
    assert credit_row["severity"] == "SEVERE"
    assert credit_row["ks_p_value"] < 0.05  # DQ-4: the KS test agrees
    assert age_row["severity"] == "STABLE"


# ── ingestion contract ───────────────────────────────────────────────────
def test_ingestor_reads_a_valid_csv(tmp_path, synthetic_frame):
    path = tmp_path / "clean.csv"
    synthetic_frame.to_csv(path, index=False)
    loaded = DataIngestor().load(path)
    assert loaded.shape == synthetic_frame.shape


@pytest.mark.parametrize(
    "filename,content",
    [
        ("missing.csv", None),  # file never created
        ("empty.csv", ""),  # zero bytes -> EmptyDataError
        ("headers.csv", "Age,AnnualIncome\n"),  # parses, but zero rows
        ("ragged.csv", "a,b\n1,2\n3,4,5,6\n"),  # ParserError
    ],
)
def test_ingestor_rejects_unusable_files(tmp_path, filename, content):
    """Every I/O failure mode surfaces as one domain exception, not OSError."""
    path = tmp_path / filename
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(DataIngestionError):
        DataIngestor().load(path)


def test_ingestor_warns_but_continues_on_duplicate_rows(
    tmp_path, synthetic_frame, caplog
):
    """Duplicates are recoverable: WARNING, not ERROR, and the load succeeds.

    Asserts the log-level policy the report states, not merely that the code
    runs -- a duplicate that vanished silently would be worse than one that is
    reported and kept.
    """
    path = tmp_path / "dupes.csv"
    pd.concat([synthetic_frame.head(50)] * 2, ignore_index=True).to_csv(path, index=False)

    with caplog.at_level(logging.WARNING, logger="loan_risk.data.ingestion"):
        loaded = DataIngestor().load(path)

    assert len(loaded) == 100
    assert any(r.message == "duplicate_rows_detected" for r in caplog.records)


def test_ingestor_rejects_a_frame_missing_contract_columns(synthetic_frame):
    """A silent column drop would train on a different space than we serve."""
    with pytest.raises(DataIngestionError, match="Feature contract violated"):
        DataIngestor().split_xy(synthetic_frame.drop(columns=["NetWorth"]))


def test_ingestor_split_returns_aligned_features_and_labels(synthetic_frame):
    features, labels = DataIngestor().split_xy(synthetic_frame)
    assert len(features) == len(labels)
    assert set(labels.unique()).issubset({0, 1})
