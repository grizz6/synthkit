"""Column type inference.

Classifying a column correctly is the first thing that happens and everything downstream
depends on it, so the heuristics here are deliberately conservative and overridable: a caller
can always pass an explicit type through `Profile.fit(..., column_types={...})` rather than
trust the guess.
"""

from __future__ import annotations

import enum

import pandas as pd

CATEGORICAL_MAX_CARDINALITY = 20
TEXT_CARDINALITY_RATIO = 0.5
MIN_ROWS_FOR_IDENTIFIER = 10


class ColumnType(str, enum.Enum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    IDENTIFIER = "identifier"
    TEXT = "text"


def infer_column_type(series: pd.Series) -> ColumnType:
    """Guess a single column's type from its values.

    Order matters: boolean and datetime dtypes are checked first because pandas already
    disambiguates them at the dtype level, then numeric and string columns are split by
    cardinality.
    """
    non_null = series.dropna()
    n = len(non_null)

    if n == 0:
        return ColumnType.CATEGORICAL

    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME

    if pd.api.types.is_numeric_dtype(series):
        nunique = non_null.nunique()

        if pd.api.types.is_integer_dtype(series) and nunique < CATEGORICAL_MAX_CARDINALITY:
            return ColumnType.CATEGORICAL

        # A float column where every observed value happens to be a whole number should be
        # treated as discrete (emit whole numbers), not continuous.
        is_whole_valued = (non_null == non_null.round()).all()
        if is_whole_valued and nunique < CATEGORICAL_MAX_CARDINALITY:
            return ColumnType.CATEGORICAL
        if is_whole_valued:
            return ColumnType.DISCRETE

        return ColumnType.CONTINUOUS

    # Everything else is treated as string-like.
    nunique = non_null.nunique()

    if n >= MIN_ROWS_FOR_IDENTIFIER and nunique == n:
        return ColumnType.IDENTIFIER

    if nunique <= CATEGORICAL_MAX_CARDINALITY:
        return ColumnType.CATEGORICAL

    if nunique / n > TEXT_CARDINALITY_RATIO:
        return ColumnType.TEXT

    return ColumnType.CATEGORICAL


def infer_all_types(
    df: pd.DataFrame, overrides: dict[str, ColumnType] | None = None
) -> dict[str, ColumnType]:
    """Infer types for every column, letting `overrides` win where given."""
    overrides = overrides or {}
    return {
        column: overrides.get(column, infer_column_type(df[column])) for column in df.columns
    }
