"""The repair engine: enforces declared constraints on already-sampled synthetic data.

Each constraint type gets a different repair strategy, applied in a fixed order:

- Derived columns are recomputed, never sampled.
- Inequalities are repaired by swapping the two values, which preserves both columns'
  marginal distributions exactly (clamping would not).
- Conditional nulls are applied directly.
- A single-column unique constraint is generated fresh; a multi-column one only touches rows
  whose combination actually collides, to avoid discarding realistic values elsewhere.
- Foreign keys are sampled from a caller-supplied key pool.
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
    # No well-defined "smallest step" for this dtype (e.g. strings); leave the tie as-is.
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
        # Swapping two equal values is a no-op, so a genuine tie survives the swap above.
        # Nudge the right-hand side apart by the smallest step the dtype supports.
        tied = df[left] == df[right]
        if tied.any():
            step = 1 if constraint.op == "<" else -1
            try:
                df.loc[tied, right] = _nudge(df.loc[tied, right], step)
            except OverflowError:
                # A datetime tie sitting at pd.Timestamp.max/.min (a common "never expires" or
                # "no start date" sentinel) overflows when nudged further in that direction.
                # Nudge the other side the opposite way instead.
                df.loc[tied, left] = _nudge(df.loc[tied, left], -step)

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
        column = columns[0]
        order = rng.permutation(len(df))
        df[column] = order + 1
        return df

    # Only the combination of all `columns` needs to be unique. Disambiguate only rows whose
    # combination actually collides with an earlier row, by suffixing the last column, rather
    # than overwriting every listed column for every row. Keys are tuples, not joined strings,
    # so ("a|b", "c") and ("a", "b|c") aren't mistaken for the same combination.
    combos = [tuple(row) for row in df[columns].astype(str).itertuples(index=False)]
    taken = set(combos)
    seen: set[tuple[str, ...]] = set()
    last_column = columns[-1]
    last_position = df.columns.get_loc(last_column)

    for i, combo in enumerate(combos):
        if combo not in seen:
            seen.add(combo)
            continue
        # A suffixed value can itself already be a real row (e.g. "k_2"), so keep counting
        # until the new combination is free.
        value = df.iat[i, last_position]
        count = 2
        while (candidate := (*combo[:-1], f"{value}_{count}")) in taken:
            count += 1
        df.iat[i, last_position] = candidate[-1]
        taken.add(candidate)
        seen.add(candidate)

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
