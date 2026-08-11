"""TEST TYPE 2 - DATA VALIDATION / DATA-QUALITY TESTS.

These assert on the *data contract* rather than on code behaviour. In an ML
system the data is as much a dependency as a library is, so it needs its own
regression suite: schema conformance, missing values and distribution drift.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from loan_risk.data.ingestion import DataIngestor
from loan_risk.data.validation import LOAN_SCHEMA, DataValidator
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


def test_out_of_range_credit_score_is_flagged(synthetic_frame):
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], "CreditScore"] = 9999
    report = DataValidator().validate(bad)
    assert "above_maximum:CreditScore" in report.violations
    assert report.schema_conformance_rate < 1.0


def test_negative_income_is_flagged(synthetic_frame):
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], "AnnualIncome"] = -1
    assert "below_minimum:AnnualIncome" in DataValidator().validate(bad).violations


def test_missing_column_is_flagged(synthetic_frame):
    bad = synthetic_frame.drop(columns=["JobTenure"])
    assert "missing_column:JobTenure" in DataValidator().validate(bad).violations


def test_strict_mode_raises_on_violation(synthetic_frame):
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], "Age"] = 5
    with pytest.raises(SchemaValidationError):
        DataValidator().validate(bad, strict=True)


def test_validator_rejects_empty_frame():
    with pytest.raises(SchemaValidationError):
        DataValidator().validate(pd.DataFrame())


def test_every_declared_column_has_a_spec():
    """Guard against a schema entry being added twice or left unnamed."""
    names = [spec.name for spec in LOAN_SCHEMA]
    assert len(names) == len(set(names))


# ── DQ metric 2: missing values ──────────────────────────────────────────
def test_missing_value_fraction_is_measured(synthetic_frame):
    dirty = synthetic_frame.copy().astype({"PaymentHistory": "float"})
    # Null out a *fraction* of rows, not a fixed count, so the assertion holds
    # whatever size the fixture frame is.
    n_null = int(len(dirty) * 0.20)
    dirty.loc[dirty.index[:n_null], "PaymentHistory"] = np.nan
    report = DataValidator(max_missing_fraction=0.001).validate(dirty)
    assert report.missing_value_fraction > 0
    assert report.worst_column == "PaymentHistory"
    assert any(v.startswith("missing_fraction_exceeded") for v in report.violations)
    assert "nulls_in_non_nullable:PaymentHistory" in report.violations


# ── DQ metrics 3 & 4: drift (PSI, KS) ────────────────────────────────────
def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    sample = rng.normal(0, 1, 5000)
    assert population_stability_index(sample, sample.copy()) < 0.01


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = rng.normal(650, 60, 5000)
    shifted = rng.normal(560, 60, 5000)  # a 90-point credit-score collapse
    assert population_stability_index(reference, shifted) > PSI_SEVERE


def test_psi_rejects_empty_samples():
    with pytest.raises(ValueError):
        population_stability_index([], [1.0, 2.0])


def test_drift_monitor_flags_only_the_shifted_feature(synthetic_frame):
    reference = synthetic_frame.copy()
    current = synthetic_frame.copy()
    current["CreditScore"] = np.clip(current["CreditScore"] - 120, 300, 850)

    monitor = DriftMonitor(reference, features=["CreditScore", "Age", "LoanAmount"])
    summary = monitor.summary_frame(current)

    credit_row = summary.loc[summary["feature"] == "CreditScore"].iloc[0]
    age_row = summary.loc[summary["feature"] == "Age"].iloc[0]
    assert credit_row["severity"] == "SEVERE"
    assert credit_row["ks_p_value"] < 0.05
    assert age_row["severity"] == "STABLE"


# ── ingestion contract ───────────────────────────────────────────────────
def test_ingestor_reads_a_valid_csv(tmp_path, synthetic_frame):
    """The happy path: a well-formed CSV round-trips into a dataframe."""
    path = tmp_path / "clean.csv"
    synthetic_frame.to_csv(path, index=False)
    loaded = DataIngestor().load(path)
    assert loaded.shape == synthetic_frame.shape
    assert list(loaded.columns) == list(synthetic_frame.columns)


def test_ingestor_rejects_a_missing_file(tmp_path):
    with pytest.raises(DataIngestionError):
        DataIngestor().load(tmp_path / "nope.csv")


def test_ingestor_rejects_a_completely_empty_file(tmp_path):
    """A zero-byte file raises pandas' EmptyDataError -> one domain error."""
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DataIngestionError, match="empty"):
        DataIngestor().load(path)


def test_ingestor_rejects_a_header_only_file(tmp_path):
    """Headers but no rows parses fine, so it needs its own explicit guard."""
    path = tmp_path / "headers.csv"
    path.write_text("Age,AnnualIncome,CreditScore\n", encoding="utf-8")
    with pytest.raises(DataIngestionError, match="zero rows"):
        DataIngestor().load(path)


def test_ingestor_rejects_an_unparseable_file(tmp_path):
    """Ragged rows raise pandas' ParserError -> the same domain error."""
    path = tmp_path / "ragged.csv"
    path.write_text("a,b\n1,2\n3,4,5,6,7\n", encoding="utf-8")
    with pytest.raises(DataIngestionError):
        DataIngestor().load(path)


def test_ingestor_warns_but_continues_on_duplicate_rows(
    tmp_path, synthetic_frame, caplog
):
    """Duplicates are recoverable: WARNING, not ERROR, and the load succeeds.

    This asserts the log-level policy stated in the report, not just that the
    code runs -- a duplicate that silently vanished would be worse than one
    that is loudly reported and kept.
    """
    path = tmp_path / "dupes.csv"
    doubled = pd.concat([synthetic_frame.head(50)] * 2, ignore_index=True)
    doubled.to_csv(path, index=False)

    with caplog.at_level(logging.WARNING, logger="loan_risk.data.ingestion"):
        loaded = DataIngestor().load(path)

    assert len(loaded) == 100
    assert any(r.message == "duplicate_rows_detected" for r in caplog.records)
    assert all(r.levelno == logging.WARNING for r in caplog.records)


def test_ingestor_rejects_a_frame_missing_contract_columns(synthetic_frame):
    with pytest.raises(DataIngestionError, match="Feature contract violated"):
        DataIngestor().split_xy(synthetic_frame.drop(columns=["NetWorth"]))


def test_ingestor_split_returns_contract_ordered_features(synthetic_frame):
    features, labels = DataIngestor().split_xy(synthetic_frame)
    assert len(features) == len(labels)
    assert set(labels.unique()).issubset({0, 1})
