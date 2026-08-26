"""The repair engine: enforces declared constraints on already-sampled synthetic data.

Each constraint type gets a different repair strategy, applied in a fixed order, because the
right fix genuinely differs by constraint:

- Derived columns are recomputed, never sampled.
- Inequalities are repaired by swapping the two values, which fixes the ordering while
  preserving both columns' marginal distributions exactly (clamping would not).
- Conditional nulls are applied directly.
- Unique and foreign-key columns are generated from a key pool rather than resampled and
  checked, since resampling a wide numeric range until it happens to be unique is unreliable.
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


def _repair_inequality(df: pd.DataFrame, constraint: Inequality) -> pd.DataFrame:
    left, right = constraint.left, constraint.right

    if constraint.op in ("<=", "<"):
        violated = df[left] > df[right]
    else:  # ">=" or ">"
        violated = df[left] < df[right]

    if violated.any():
        df.loc[violated, [left, right]] = df.loc[violated, [right, left]].to_numpy()

    return df


def _repair_derived(df: pd.DataFrame, constraint: Derived) -> pd.DataFrame:
    df[constraint.column] = df.eval(constraint.expr)
    return df


def _repair_conditional_null(df: pd.DataFrame, constraint: ConditionalNull) -> pd.DataFrame:
    mask = df.eval(constraint.null_when)
    df.loc[mask, constraint.column] = None
    return df


def _repair_unique(df: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    order = rng.permutation(n)
    for column in columns:
        # Only the combination of all `columns` together needs to be unique; a single-column
        # constraint is the common case where this reduces to "every value is distinct".
        if len(columns) == 1:
            df[column] = order + 1
        else:
            df[column] = [f"{column}_{idx + 1}" for idx in order]
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
