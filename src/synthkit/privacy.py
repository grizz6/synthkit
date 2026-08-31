"""The privacy check: turning "the profile contains no real records" from an assertion into a
measurement.

A synthetic row being close to some real row means nothing on its own; the question is
whether it's closer than real data naturally is to itself. Holding out a slice of real rows
and comparing the synthetic distance-to-closest-record against the holdout's own gives that
baseline: if synthetic rows are systematically closer to training data than an untouched
holdout is, the model is memorizing rather than generalizing.
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
    """Max minus min of `combined`, ignoring NaN/+-inf, falling back to 1.0 when that's not
    computable."""
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return 1.0
    value_range = finite.max() - finite.min()
    return float(value_range) if value_range else 1.0


def _column_distance(
    query: pd.Series,
    reference: pd.Series,
    column_type: str,
    value_range: float | None = None,
) -> np.ndarray:
    if column_type in NUMERIC_LIKE_TYPES:
        q = query.to_numpy(dtype=float)
        r = reference.to_numpy(dtype=float)
        if value_range is None:
            value_range = _finite_range(np.concatenate([q, r]))
        with np.errstate(invalid="ignore"):
            dist = np.abs(q[:, None] - r[None, :]) / value_range
    elif column_type == "datetime":
        q = to_epoch_seconds(query).astype("float64")
        r = to_epoch_seconds(reference).astype("float64")
        if value_range is None:
            value_range = _finite_range(np.concatenate([q, r]))
        with np.errstate(invalid="ignore"):
            dist = np.abs(q[:, None] - r[None, :]) / value_range
    else:
        q = query.astype(str).to_numpy()
        r = reference.astype(str).to_numpy()
        dist = (q[:, None] != r[None, :]).astype(float)

    # Gower distance for a single column should live in [0, 1]; clamp so a stray +-inf value
    # can't dominate every other column's contribution once averaged.
    dist = np.nan_to_num(dist, nan=1.0, posinf=1.0, neginf=1.0)
    return np.clip(dist, 0.0, 1.0)


def compute_value_ranges(
    query: pd.DataFrame, reference: pd.DataFrame, column_types: dict[str, str]
) -> dict[str, float]:
    """Each numeric/datetime column's normalization range, computed once over the full
    query + reference data so that batched calls all normalize consistently."""
    ranges: dict[str, float] = {}
    for column, ctype in column_types.items():
        if column not in query.columns or column not in reference.columns:
            continue
        if ctype in NUMERIC_LIKE_TYPES:
            combined = np.concatenate(
                [query[column].to_numpy(dtype=float), reference[column].to_numpy(dtype=float)]
            )
        elif ctype == "datetime":
            combined = np.concatenate(
                [
                    to_epoch_seconds(query[column]).astype(float),
                    to_epoch_seconds(reference[column]).astype(float),
                ]
            )
        else:
            continue
        ranges[column] = _finite_range(combined)
    return ranges


def gower_distance_matrix(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    column_types: dict[str, str],
    value_ranges: dict[str, float] | None = None,
) -> np.ndarray:
    """Pairwise Gower distance between every row of `query` and every row of `reference`.

    Handles mixed numeric/categorical types by normalizing each column's contribution to
    [0, 1] before averaging. O(n * m) in row count on each side. Pass `value_ranges` (from
    `compute_value_ranges`) when calling this repeatedly over slices of the same pair, as
    `distance_to_closest_record` does.
    """
    columns = [c for c in query.columns if c in column_types and c in reference.columns]
    total = np.zeros((len(query), len(reference)))

    for column in columns:
        value_range = None if value_ranges is None else value_ranges.get(column)
        total += _column_distance(
            query[column], reference[column], column_types[column], value_range
        )

    return total / max(len(columns), 1)


DEFAULT_DCR_BATCH_SIZE = 500


def distance_to_closest_record(
    query: pd.DataFrame,
    reference: pd.DataFrame,
    column_types: dict[str, str],
    batch_size: int = DEFAULT_DCR_BATCH_SIZE,
) -> np.ndarray:
    """For each row in `query`, its Gower distance to the nearest row in `reference`.

    Both sides are batched so peak memory is O(batch_size^2) rather than
    O(len(query) * len(reference)). This is exact, not an approximation: only the running
    per-row minimum survives past each reference batch, and every batch normalizes against
    the same precomputed value ranges so results don't shift between batches.
    """
    if len(query) <= batch_size and len(reference) <= batch_size:
        return gower_distance_matrix(query, reference, column_types).min(axis=1)

    value_ranges = compute_value_ranges(query, reference, column_types)
    result = np.full(len(query), np.inf)

    for q_start in range(0, len(query), batch_size):
        q_batch = query.iloc[q_start : q_start + batch_size]
        batch_min = np.full(len(q_batch), np.inf)

        for r_start in range(0, len(reference), batch_size):
            r_batch = reference.iloc[r_start : r_start + batch_size]
            distances = gower_distance_matrix(q_batch, r_batch, column_types, value_ranges)
            batch_min = np.minimum(batch_min, distances.min(axis=1))

        result[q_start : q_start + len(q_batch)] = batch_min

    return result


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
    """How many synthetic rows reproduce a real combination of `columns` that appeared fewer
    than `threshold` times in the real data.

    Passing many columns at once makes almost every combination "rare" purely from
    dimensionality, so `check()` scores pairs of columns instead of the full cross product.
    Call this directly with a larger column list only if you want that stricter joint check.
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
    n_holdout = max(1, min(len(real) - 1, int(len(real) * holdout_fraction)))

    if len(real) < 2:
        raise ValueError(
            f"real dataset has only {len(real)} row(s); check() needs at least 2 to split "
            "into a holdout and a training set"
        )

    holdout = real.iloc[shuffled_index[:n_holdout]]
    training = real.iloc[shuffled_index[n_holdout:]]

    synthetic_dcr = distance_to_closest_record(synthetic, training, column_types)
    holdout_dcr = distance_to_closest_record(holdout, training, column_types)

    synthetic_p = np.percentile(synthetic_dcr, dcr_percentile)
    holdout_p = np.percentile(holdout_dcr, dcr_percentile)

    if holdout_p == 0 and synthetic_p == 0:
        # Both sides sit exactly on some training row at this percentile, which usually means
        # the dataset doesn't have enough entropy for even real holdout rows to come out
        # distinct, not that synthkit is memorizing. Report the neutral ratio instead of
        # dividing by a fallback epsilon and reading as a hard failure.
        ratio = 1.0
    else:
        ratio = float(synthetic_p / (holdout_p or 1e-9))

    exact_matches = count_exact_matches(synthetic, real)

    categorical_columns = [
        c for c, t in column_types.items() if t in ("categorical", "boolean") and c in real.columns
    ]
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
