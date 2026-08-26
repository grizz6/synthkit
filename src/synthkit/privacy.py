"""The privacy check: turning "the profile contains no real records" from an assertion into a
measurement.

The key move is the baseline. A synthetic row being "close" to some real row means nothing on
its own — the question is whether it is *closer than real data naturally is to itself*. Holding
out a slice of real rows during fitting and comparing the synthetic distance-to-closest-record
against the holdout's own distance-to-closest-record gives exactly that baseline: if synthetic
rows are systematically closer to the training data than an untouched holdout is, the model is
memorizing rather than generalizing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NUMERIC_LIKE_TYPES = {"continuous", "discrete"}
DEFAULT_RARE_COMBINATION_THRESHOLD = 5


def _column_distance(query: pd.Series, reference: pd.Series, column_type: str) -> np.ndarray:
    if column_type in NUMERIC_LIKE_TYPES:
        q = query.to_numpy(dtype=float)
        r = reference.to_numpy(dtype=float)
        combined = np.concatenate([q, r])
        value_range = np.nanmax(combined) - np.nanmin(combined) or 1.0
        dist = np.abs(q[:, None] - r[None, :]) / value_range
    elif column_type == "datetime":
        q = pd.to_datetime(query).to_numpy().astype("datetime64[s]").astype("float64")
        r = pd.to_datetime(reference).to_numpy().astype("datetime64[s]").astype("float64")
        combined = np.concatenate([q, r])
        value_range = (np.nanmax(combined) - np.nanmin(combined)) or 1.0
        dist = np.abs(q[:, None] - r[None, :]) / value_range
    else:
        q = query.astype(str).to_numpy()
        r = reference.astype(str).to_numpy()
        dist = (q[:, None] != r[None, :]).astype(float)

    return np.nan_to_num(dist, nan=1.0)


def gower_distance_matrix(
    query: pd.DataFrame, reference: pd.DataFrame, column_types: dict[str, str]
) -> np.ndarray:
    """Pairwise Gower distance between every row of `query` and every row of `reference`.

    Gower distance handles mixed numeric/categorical types by normalizing each column's
    contribution to [0, 1] before averaging, so no single high-range numeric column dominates
    the result. This is O(n * m) in the number of rows on each side; fine for the profile
    sizes this package targets, not meant for million-row comparisons.
    """
    columns = [c for c in query.columns if c in column_types]
    total = np.zeros((len(query), len(reference)))

    for column in columns:
        total += _column_distance(query[column], reference[column], column_types[column])

    return total / max(len(columns), 1)


def distance_to_closest_record(
    query: pd.DataFrame, reference: pd.DataFrame, column_types: dict[str, str]
) -> np.ndarray:
    """For each row in `query`, its Gower distance to the nearest row in `reference`."""
    distances = gower_distance_matrix(query, reference, column_types)
    return distances.min(axis=1)


def count_exact_matches(synthetic: pd.DataFrame, real: pd.DataFrame) -> int:
    real_rows = set(map(tuple, real.astype(str).to_numpy()))
    synthetic_rows = map(tuple, synthetic.astype(str).to_numpy())
    return sum(1 for row in synthetic_rows if row in real_rows)


def count_rare_combination_leaks(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: list[str],
    threshold: int = DEFAULT_RARE_COMBINATION_THRESHOLD,
) -> int:
    """How many synthetic rows reproduce a real combination that appeared fewer than
    `threshold` times — a rare combination is close to re-identifying, so any reproduction of
    one at all is worth flagging even though a single exact-match check would miss it."""
    real_counts = real[columns].astype(str).value_counts()
    rare_combinations = set(real_counts[real_counts < threshold].index)
    synthetic_combinations = map(tuple, synthetic[columns].astype(str).to_numpy())
    return sum(1 for combo in synthetic_combinations if combo in rare_combinations)


@dataclass
class PrivacyReport:
    dcr_ratio: float
    exact_matches: int
    rare_combination_leaks: int
    passed: bool


def check(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    column_types: dict[str, str],
    holdout_fraction: float = 0.2,
    min_dcr_ratio: float = 1.0,
    dcr_percentile: float = 5.0,
    rare_combination_threshold: int = DEFAULT_RARE_COMBINATION_THRESHOLD,
    seed: int = 0,
) -> PrivacyReport:
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(len(real))
    n_holdout = max(1, int(len(real) * holdout_fraction))

    holdout = real.iloc[shuffled_index[:n_holdout]]
    training = real.iloc[shuffled_index[n_holdout:]]

    synthetic_dcr = distance_to_closest_record(synthetic, training, column_types)
    holdout_dcr = distance_to_closest_record(holdout, training, column_types)

    synthetic_p = np.percentile(synthetic_dcr, dcr_percentile)
    holdout_p = np.percentile(holdout_dcr, dcr_percentile) or 1e-9
    ratio = float(synthetic_p / holdout_p)

    exact_matches = count_exact_matches(synthetic, real)

    categorical_columns = [
        c for c, t in column_types.items() if t in ("categorical", "boolean") and c in real.columns
    ]
    rare_leaks = (
        count_rare_combination_leaks(
            synthetic, real, categorical_columns, threshold=rare_combination_threshold
        )
        if len(categorical_columns) >= 2
        else 0
    )

    return PrivacyReport(
        dcr_ratio=ratio,
        exact_matches=exact_matches,
        rare_combination_leaks=rare_leaks,
        passed=ratio >= min_dcr_ratio and exact_matches == 0 and rare_leaks == 0,
    )
