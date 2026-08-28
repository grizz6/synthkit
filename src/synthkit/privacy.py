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

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from synthkit.marginals import to_epoch_seconds

NUMERIC_LIKE_TYPES = {"continuous", "discrete"}
DEFAULT_RARE_COMBINATION_THRESHOLD = 5


def _finite_range(combined: np.ndarray) -> float:
    """The max-minus-min of `combined`, ignoring NaN/+-inf, falling back to 1.0 when that's
    not computable (no finite values at all, or every finite value is identical).

    A plain `nanmax - nanmin` breaks in two ways a real column can actually hit: an all-NaN
    slice raises a RuntimeWarning and returns NaN, and a genuine +-inf value (a division
    result, a sentinel) makes the range infinite, which silently turns every per-row distance
    into 0/inf and floods the caller's terminal with "invalid value encountered" warnings.
    """
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return 1.0
    value_range = finite.max() - finite.min()
    return float(value_range) if value_range else 1.0


def _column_distance(query: pd.Series, reference: pd.Series, column_type: str) -> np.ndarray:
    if column_type in NUMERIC_LIKE_TYPES:
        q = query.to_numpy(dtype=float)
        r = reference.to_numpy(dtype=float)
        value_range = _finite_range(np.concatenate([q, r]))
        with np.errstate(invalid="ignore"):
            dist = np.abs(q[:, None] - r[None, :]) / value_range
    elif column_type == "datetime":
        q = to_epoch_seconds(query).astype("float64")
        r = to_epoch_seconds(reference).astype("float64")
        value_range = _finite_range(np.concatenate([q, r]))
        with np.errstate(invalid="ignore"):
            dist = np.abs(q[:, None] - r[None, :]) / value_range
    else:
        q = query.astype(str).to_numpy()
        r = reference.astype(str).to_numpy()
        dist = (q[:, None] != r[None, :]).astype(float)

    # Gower distance for a single column is meant to live in [0, 1]. Left alone, a genuine
    # +-inf value in the data would turn `nan_to_num`'s default posinf/neginf fill into
    # numpy's largest finite float, which would dominate every other column's contribution
    # once averaged in gower_distance_matrix. Clamping keeps one bad value from drowning out
    # everything else instead of silently making the whole check meaningless.
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(dist, 0.0, 1.0)


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


def _rare_combination_leak_mask(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: list[str],
    threshold: int,
) -> np.ndarray:
    real_counts = real[columns].astype(str).value_counts()
    rare_combinations = set(real_counts[real_counts < threshold].index)
    synthetic_combinations = list(map(tuple, synthetic[columns].astype(str).to_numpy()))
    return np.array([combo in rare_combinations for combo in synthetic_combinations])


def count_rare_combination_leaks(
    synthetic: pd.DataFrame,
    real: pd.DataFrame,
    columns: list[str],
    threshold: int = DEFAULT_RARE_COMBINATION_THRESHOLD,
) -> int:
    """How many synthetic rows reproduce a real combination of exactly `columns` that appeared
    fewer than `threshold` times — a rare combination is close to re-identifying, so any
    reproduction of one at all is worth flagging even though a single exact-match check would
    miss it.

    Passing many columns at once makes almost every combination "rare" purely from
    dimensionality (a handful of categorical columns can each have a small number of rare
    combinations, but their full cross product mostly consists of combinations seen once or
    twice, even with no real re-identification risk). `check()` accounts for this by scoring
    pairs of columns rather than every categorical column jointly; call this function directly
    with a larger column list only if you specifically want that stricter joint check.
    """
    return int(_rare_combination_leak_mask(synthetic, real, columns, threshold).sum())


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
    # Score pairs of categorical columns rather than every categorical column jointly: with
    # more than a couple of categorical columns, almost every combination in their full cross
    # product is "rare" purely from dimensionality, which would flag most rows regardless of
    # actual re-identification risk. A leak is any synthetic row whose value on *some* pair
    # reproduces a combination that was rare for that pair in the real data.
    if len(categorical_columns) >= 2:
        leak_mask = np.zeros(len(synthetic), dtype=bool)
        for a, b in itertools.combinations(categorical_columns, 2):
            leak_mask |= _rare_combination_leak_mask(
                synthetic, real, [a, b], rare_combination_threshold
            )
        rare_leaks = int(leak_mask.sum())
    else:
        rare_leaks = 0

    return PrivacyReport(
        dcr_ratio=ratio,
        exact_matches=exact_matches,
        rare_combination_leaks=rare_leaks,
        passed=ratio >= min_dcr_ratio and exact_matches == 0 and rare_leaks == 0,
    )
