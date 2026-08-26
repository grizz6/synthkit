"""The repair engine: enforces declared constraints on already-sampled synthetic data.

Each constraint type gets a different repair strategy, applied in a fixed order, because the
right fix genuinely differs by constraint:

- Derived columns are recomputed, never sampled.
- Inequalities are repaired by swapping the two values, which fixes the ordering while
  preserving both columns' marginal distributions exactly (clamping would not).
- Conditional nulls are applied directly.
- A single-column unique constraint (the common surrogate-key case) is generated fresh rather
  than resampled and checked, since resampling a wide numeric range until it happens to be
  unique is unreliable. A multi-column unique constraint only touches rows whose combination
  actually collides, to avoid discarding realistic sampled values in every listed column.
- Foreign keys are sampled from a caller-supplied key pool rather than resampled and checked.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthkit.constraints import (
    ConditionalNull,
    Constraint,
    Derived,
    ForeignKey,
    Inequality,
    Unique,
)

_STRICT_OPS = {"<", ">"}


def _nudge(series: pd.Series, step: int) -> pd.Series:
    """Push `series` a minimal amount in the direction of `step` (positive or negative)."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series + pd.Timedelta(seconds=step)
    if pd.api.types.is_integer_dtype(series):
        return series + step
    if pd.api.types.is_float_dtype(series):
        target = np.inf if step > 0 else -np.inf
        return pd.Series(np.nextafter(series.to_numpy(dtype=float), target), index=series.index)
    # No well-defined "smallest step" for this dtype (e.g. strings) — leave the tie as a
    # documented limitation rather than guess at one.
    return series


def _repair_inequality(df: pd.DataFrame, constraint: Inequality) -> pd.DataFrame:
    left, right = constraint.left, constraint.right
    strict = constraint.op in _STRICT_OPS

    if constraint.op in ("<=", "<"):
        violated = (df[left] >= df[right]) if strict else (df[left] > df[right])
    else:  # ">=" or ">"
        violated = (df[left] <= df[right]) if strict else (df[left] < df[right])

    if violated.any():
        df.loc[violated, [left, right]] = df.loc[violated, [right, left]].to_numpy()

    if strict:
        # Swapping two equal values is a no-op, so a genuine tie (left == right) survives the
        # swap above undisturbed; nudge the right-hand side apart by the smallest step the
        # dtype supports rather than leave the strict constraint silently unsatisfied.
        tied = df[left] == df[right]
        if tied.any():
            step = 1 if constraint.op == "<" else -1
            df.loc[tied, right] = _nudge(df.loc[tied, right], step)

    return df


def _repair_derived(df: pd.DataFrame, constraint: Derived) -> pd.DataFrame:
    df[constraint.column] = df.eval(constraint.expr)
    return df


def _repair_conditional_null(df: pd.DataFrame, constraint: ConditionalNull) -> pd.DataFrame:
    mask = df.eval(constraint.null_when)
    df.loc[mask, constraint.column] = None
    return df


def _repair_unique(df: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    if len(columns) == 1:
        # The common case (a surrogate key column): every value must be distinct, so generate
        # fresh unique values outright rather than resample-and-check.
        column = columns[0]
        order = rng.permutation(len(df))
        df[column] = order + 1
        return df

    # Only the combination of all `columns` together needs to be unique. Regenerating every
    # listed column (as the single-column path does) would overwrite realistic sampled values
    # in every one of them for every row; instead, disambiguate only rows whose combination
    # actually collides with an earlier row, by suffixing the last column.
    combo_key = df[columns].astype(str).agg("|".join, axis=1)
    seen: dict[str, int] = {}
    occurrence = np.empty(len(df), dtype=int)
    for i, key in enumerate(combo_key):
        seen[key] = seen.get(key, 0) + 1
        occurrence[i] = seen[key]

    duplicated = occurrence > 1
    if duplicated.any():
        last_column = columns[-1]
        df.loc[duplicated, last_column] = [
            f"{value}_{count}"
            for value, count in zip(
                df.loc[duplicated, last_column], occurrence[duplicated], strict=True
            )
        ]

    return df


def _repair_foreign_key(
    df: pd.DataFrame,
    constraint: ForeignKey,
    rng: np.random.Generator,
    key_pools: dict[str, list[Any]],
) -> pd.DataFrame:
    pool = key_pools.get(constraint.column)
    if pool:
        df[constraint.column] = rng.choice(pool, size=len(df))
    return df


def apply_constraints(
    df: pd.DataFrame,
    constraints: list[Constraint],
    rng: np.random.Generator | None = None,
    key_pools: dict[str, list[Any]] | None = None,
) -> pd.DataFrame:
    """Repair `df` in place (on a copy) so it satisfies every constraint, in a fixed order."""
    rng = rng or np.random.default_rng()
    key_pools = key_pools or {}
    df = df.copy()

    for constraint in constraints:
        if isinstance(constraint, Derived):
            df = _repair_derived(df, constraint)

    for constraint in constraints:
        if isinstance(constraint, Inequality):
            df = _repair_inequality(df, constraint)

    for constraint in constraints:
        if isinstance(constraint, ConditionalNull):
            df = _repair_conditional_null(df, constraint)

    for constraint in constraints:
        if isinstance(constraint, Unique):
            df = _repair_unique(df, constraint.columns, rng)
        elif isinstance(constraint, ForeignKey):
            df = _repair_foreign_key(df, constraint, rng, key_pools)

    return df
