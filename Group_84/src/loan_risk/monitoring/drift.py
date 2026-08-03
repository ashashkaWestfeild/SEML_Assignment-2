"""Data-drift detection (Objective 2.8b, metrics DQ-3 and DQ-4).

    DQ-3  Population Stability Index (PSI) -- the banking-industry standard for
          covariate shift. Conventional reading:
              PSI < 0.10  stable | 0.10-0.25 moderate shift | > 0.25 severe.
    DQ-4  Kolmogorov-Smirnov two-sample statistic + p-value -- a distribution-
          free test that flags shifts PSI can miss in the tails.

Both compare a *live* window against the frozen training reference profile.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from loan_risk.logging_utils import get_logger

logger = get_logger(__name__)

PSI_MODERATE = 0.10
PSI_SEVERE = 0.25


@dataclass
class DriftResult:
    """Per-feature drift verdict."""

    feature: str
    psi: float
    ks_statistic: float
    ks_p_value: float

    @property
    def severity(self) -> str:
        """Bucket the PSI into STABLE / MODERATE / SEVERE."""
        if self.psi >= PSI_SEVERE:
            return "SEVERE"
        if self.psi >= PSI_MODERATE:
            return "MODERATE"
        return "STABLE"

    def as_dict(self) -> Dict[str, object]:
        """JSON-serialisable view for the drift dashboard."""
        return {
            "feature": self.feature,
            "psi": round(self.psi, 4),
            "ks_statistic": round(self.ks_statistic, 4),
            "ks_p_value": round(self.ks_p_value, 6),
            "severity": self.severity,
        }


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], n_bins: int = 10
) -> float:
    """Compute PSI between a reference and a current sample.

    Quantile bins are taken from the *reference* so the baseline stays fixed;
    empty buckets are floored at 1e-6 to keep the logarithm finite.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.size == 0 or current.size == 0:
        raise ValueError("PSI requires non-empty reference and current samples.")

    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.unique(np.percentile(reference, quantiles))
    if edges.size < 2:  # constant reference column
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, bins=edges)[0] / reference.size
    cur_pct = np.histogram(current, bins=edges)[0] / current.size
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class DriftMonitor:
    """Compare a live batch against the frozen training reference profile."""

    def __init__(
        self,
        reference: pd.DataFrame,
        features: Sequence[str] | None = None,
        psi_threshold: float = PSI_SEVERE,
    ) -> None:
        if reference.empty:
            raise ValueError("Reference profile must not be empty.")
        self.reference = reference
        self.features = list(features) if features else list(reference.columns)
        self.psi_threshold = psi_threshold

    def compare(self, current: pd.DataFrame) -> List[DriftResult]:
        """Return one :class:`DriftResult` per shared numeric feature."""
        results: List[DriftResult] = []
        for feature in self.features:
            if feature not in current.columns:
                logger.warning(
                    "drift_feature_absent_in_current", extra={"feature": feature}
                )
                continue
            ref_values = self.reference[feature].dropna().to_numpy(dtype=float)
            cur_values = current[feature].dropna().to_numpy(dtype=float)
            if ref_values.size == 0 or cur_values.size == 0:
                continue

            psi = population_stability_index(ref_values, cur_values)
            ks_stat, p_value = stats.ks_2samp(ref_values, cur_values)
            results.append(DriftResult(feature, psi, float(ks_stat), float(p_value)))

        drifted = [r for r in results if r.psi >= self.psi_threshold]
        if drifted:
            logger.warning(
                "data_drift_detected",
                extra={
                    "n_drifted_features": len(drifted),
                    "features": [r.feature for r in drifted],
                },
            )
        else:
            logger.info(
                "no_significant_drift", extra={"n_features_checked": len(results)}
            )
        return results

    def summary_frame(self, current: pd.DataFrame) -> pd.DataFrame:
        """Drift results as a dataframe, worst first -- ready for the report."""
        rows = [r.as_dict() for r in self.compare(current)]
        return (
            pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
        )
