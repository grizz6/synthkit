"""Drift detection: has production moved away from the profile fixtures are built on?

Test fixtures silently rot as production data drifts, and nothing normally tells you. This
compares a fresh dataframe against a profile's stored marginals, one column at a time, using
the Kolmogorov-Smirnov statistic for numeric/datetime columns and total variation distance for
categorical/boolean ones, and reports which columns moved past a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from synthkit.marginals import (
    OTHER_CATEGORY,
    BooleanMarginal,
    CategoricalMarginal,
    DatetimeMarginal,
    NumericMarginal,
    to_epoch_seconds,
)
from synthkit.profile import ALL_NULL_KIND, Profile

REFERENCE_SAMPLE_SIZE = 2000
DEFAULT_DRIFT_THRESHOLD = 0.1


def _numeric_drift(marginal_dict: dict[str, Any], current: pd.Series) -> float:
    marginal = NumericMarginal.from_dict(marginal_dict)
    reference = marginal.sample(np.linspace(0.0, 1.0, REFERENCE_SAMPLE_SIZE))
    clean = current.dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return 0.0
    return float(ks_2samp(clean, reference).statistic)


def _datetime_drift(marginal_dict: dict[str, Any], current: pd.Series) -> float:
    marginal = DatetimeMarginal.from_dict(marginal_dict)
    reference = marginal.sample(np.linspace(0.0, 1.0, REFERENCE_SAMPLE_SIZE))
    reference_epoch = to_epoch_seconds(reference).astype(float)
    clean = current.dropna()
    if clean.empty:
        return 0.0
    current_epoch = to_epoch_seconds(clean).astype(float)
    return float(ks_2samp(current_epoch, reference_epoch).statistic)


def _categorical_drift(marginal_dict: dict[str, Any], current: pd.Series) -> float:
    marginal = CategoricalMarginal.from_dict(marginal_dict)
    reference_probs = dict(zip(marginal.categories, marginal.probabilities, strict=True))
    raw_current_probs = current.dropna().astype(str).value_counts(normalize=True).to_dict()

    if OTHER_CATEGORY in reference_probs:
        # Fold any category the profile doesn't recognize into __other__ too, so it isn't
        # counted twice: once as its own missing key, once as unmatched __other__ mass.
        current_probs: dict[str, float] = {}
        for category, prob in raw_current_probs.items():
            key = category if category in reference_probs else OTHER_CATEGORY
            current_probs[key] = current_probs.get(key, 0.0) + prob
    else:
        current_probs = raw_current_probs

    categories = set(reference_probs) | set(current_probs)
    return 0.5 * sum(
        abs(reference_probs.get(c, 0.0) - current_probs.get(c, 0.0)) for c in categories
    )


def _boolean_drift(marginal_dict: dict[str, Any], current: pd.Series) -> float:
    marginal = BooleanMarginal.from_dict(marginal_dict)
    clean = current.dropna()
    if clean.empty:
        return 0.0
    current_rate = float(clean.astype(bool).mean())
    return abs(marginal.probability_true - current_rate)


_DRIFT_FUNCTIONS = {
    "continuous": _numeric_drift,
    "discrete": _numeric_drift,
    "datetime": _datetime_drift,
    "categorical": _categorical_drift,
    "boolean": _boolean_drift,
}


@dataclass
class DriftReport:
    column_drift: dict[str, float]
    drifted_columns: list[str]
    passed: bool

    @property
    def max_drift(self) -> float:
        return max(self.column_drift.values(), default=0.0)


def compute_drift(
    profile: Profile, current: pd.DataFrame, threshold: float = DEFAULT_DRIFT_THRESHOLD
) -> DriftReport:
    column_drift: dict[str, float] = {}

    for column in profile.columns:
        if column not in current.columns:
            continue

        ctype = profile.column_types[column]
        if ctype == ALL_NULL_KIND or ctype not in _DRIFT_FUNCTIONS:
            continue

        marginal_dict = profile.marginals[column]
        column_drift[column] = _DRIFT_FUNCTIONS[ctype](marginal_dict, current[column])

    drifted = [c for c, score in column_drift.items() if score > threshold]

    return DriftReport(column_drift=column_drift, drifted_columns=drifted, passed=not drifted)
