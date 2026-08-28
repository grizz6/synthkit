"""The top-level Python API: synthkit.fit, synthkit.emit, synthkit.check.

Thin wrappers over Profile and privacy.check, so normal usage doesn't require reaching into
submodules for anything but constraint classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from scipy.stats import ks_2samp

from synthkit import privacy as _privacy
from synthkit.constraints import Constraint
from synthkit.marginals import to_epoch_seconds
from synthkit.profile import Profile
from synthkit.types import ColumnType

NUMERIC_LIKE_TYPES = {"continuous", "discrete"}


def fit(
    df: pd.DataFrame,
    column_types: dict[str, ColumnType] | None = None,
    constraints: list[Constraint] | None = None,
) -> Profile:
    return Profile.fit(df, column_types=column_types, constraints=constraints)


def emit(
    profile: Profile,
    n: int,
    seed: int,
    key_pools: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    return profile.emit(n=n, seed=seed, key_pools=key_pools)


def _fidelity_by_column(
    real: pd.DataFrame, synthetic: pd.DataFrame, column_types: dict[str, str]
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for column, ctype in column_types.items():
        if column not in real.columns or column not in synthetic.columns:
            continue

        if ctype in NUMERIC_LIKE_TYPES:
            real_values = real[column].dropna().astype(float)
            synthetic_values = synthetic[column].dropna().astype(float)
        elif ctype == "datetime":
            real_values = to_epoch_seconds(real[column].dropna())
            synthetic_values = to_epoch_seconds(synthetic[column].dropna())
        else:
            continue

        if len(real_values) == 0 or len(synthetic_values) == 0:
            continue

        scores[column] = float(ks_2samp(real_values, synthetic_values).statistic)

    return scores


@dataclass
class CheckReport:
    dcr_ratio: float
    exact_matches: int
    rare_combination_leaks: int
    passed: bool
    ks_by_column: dict[str, float]


def check(
    synthetic: pd.DataFrame,
    profile: Profile,
    real: pd.DataFrame,
    holdout_fraction: float = 0.2,
    min_dcr_ratio: float = 1.0,
    seed: int = 0,
) -> CheckReport:
    privacy_report = _privacy.check(
        synthetic,
        real,
        profile.column_types,
        holdout_fraction=holdout_fraction,
        min_dcr_ratio=min_dcr_ratio,
        seed=seed,
    )
    ks_by_column = _fidelity_by_column(real, synthetic, profile.column_types)

    return CheckReport(
        dcr_ratio=privacy_report.dcr_ratio,
        exact_matches=privacy_report.exact_matches,
        rare_combination_leaks=privacy_report.rare_combination_leaks,
        passed=privacy_report.passed,
        ks_by_column=ks_by_column,
    )
