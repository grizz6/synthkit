"""The Profile: the single artifact this whole package exists to produce.

`Profile.fit` reads a real dataframe once and reduces it to marginals, a null-pattern model,
and a copula correlation matrix — no real rows. `Profile.emit` reverses that into as many
synthetic rows as asked for, deterministically, from nothing but the profile and a seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthkit import serialization
from synthkit.constraints import Constraint, constraints_to_dicts, parse_constraints
from synthkit.copula import GaussianCopula, category_pseudo_uniform, rank_transform_to_uniform
from synthkit.marginals import (
    BooleanMarginal,
    CategoricalMarginal,
    DatetimeMarginal,
    IdentifierMarginal,
    NumericMarginal,
    TextMarginal,
)
from synthkit.nulls import NullModel
from synthkit.repair import apply_constraints
from synthkit.types import ColumnType, infer_all_types

ALL_NULL_KIND = "all_null"

TYPE_TO_MARGINAL_CLASS: dict[ColumnType, type[Any]] = {
    ColumnType.CONTINUOUS: NumericMarginal,
    ColumnType.DISCRETE: NumericMarginal,
    ColumnType.CATEGORICAL: CategoricalMarginal,
    ColumnType.BOOLEAN: BooleanMarginal,
    ColumnType.DATETIME: DatetimeMarginal,
    ColumnType.IDENTIFIER: IdentifierMarginal,
    ColumnType.TEXT: TextMarginal,
}

MARGINAL_CLASS_BY_KIND: dict[str, type[Any]] = {
    "numeric": NumericMarginal,
    "categorical": CategoricalMarginal,
    "boolean": BooleanMarginal,
    "datetime": DatetimeMarginal,
    "identifier": IdentifierMarginal,
    "text": TextMarginal,
}

# Only these types participate in the copula: identifiers are regenerated from a pattern
# rather than modeled, and free text has no natural notion of rank correlation.
COPULA_ELIGIBLE_TYPES = {
    ColumnType.CONTINUOUS,
    ColumnType.DISCRETE,
    ColumnType.CATEGORICAL,
    ColumnType.BOOLEAN,
    ColumnType.DATETIME,
}

MIN_COMPLETE_ROWS_FOR_COPULA = 10


def _pseudo_uniform(ctype: ColumnType, values: pd.Series, marginal: Any) -> np.ndarray:
    """Map a column's real values to uniform pseudo-observations for fitting the copula."""
    if ctype in (ColumnType.CONTINUOUS, ColumnType.DISCRETE):
        return rank_transform_to_uniform(values.to_numpy(dtype=float))

    if ctype == ColumnType.DATETIME:
        epoch = pd.to_datetime(values).to_numpy().astype("datetime64[s]").astype("int64")
        return rank_transform_to_uniform(epoch.astype(float))

    if ctype == ColumnType.CATEGORICAL:
        return category_pseudo_uniform(values.to_numpy(), marginal)

    if ctype == ColumnType.BOOLEAN:
        # Consistent with BooleanMarginal.sample's own convention (u < p_true => True): True's
        # interval is [0, p_true), False's is [p_true, 1).
        p_true = marginal.probability_true
        midpoint_true = p_true / 2
        midpoint_false = p_true + (1 - p_true) / 2
        return np.array([midpoint_true if bool(v) else midpoint_false for v in values])

    raise ValueError(f"{ctype} is not copula-eligible")


def _marginal_from_dict(data: dict[str, Any]) -> Any:
    return MARGINAL_CLASS_BY_KIND[data["kind"]].from_dict(data)


def _sample_column(ctype: ColumnType, marginal_dict: dict[str, Any], u: np.ndarray) -> np.ndarray:
    marginal = _marginal_from_dict(marginal_dict)
    return marginal.sample(u)


@dataclass
class Profile:
    columns: list[str]
    column_types: dict[str, str]
    marginals: dict[str, dict[str, Any]]
    null_columns: list[str]
    null_model: dict[str, Any]
    copula_columns: list[str]
    copula: dict[str, Any] | None
    n_rows_fit: int
    constraints: list[dict[str, Any]]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        column_types: dict[str, ColumnType] | None = None,
        constraints: list[Constraint] | str | Path | None = None,
    ) -> Profile:
        # A profile is a JSON artifact, and JSON object keys are always strings -- a profile
        # fit on a dataframe with e.g. integer column names would save and reload with
        # column_types keyed by "0"/"1"/... while self.columns stayed [0, 1, ...], and emit()
        # would then KeyError looking up an int in a str-keyed dict. Coercing to strings once,
        # up front, means every internal structure (columns, column_types, marginals, ...) is
        # consistently string-keyed from the start, matching what save/load always produces
        # anyway, instead of working by coincidence until the first round trip through disk.
        df = df.rename(columns=str)
        if column_types:
            # A caller overriding a non-string-named column (column_types={0: ...}) is keying
            # by the original identifier; the rename above means lookups now happen by str(0),
            # so the override dict needs the same treatment or it would silently never match.
            column_types = {str(k): v for k, v in column_types.items()}

        duplicated = df.columns[df.columns.duplicated()].unique().tolist()
        if duplicated:
            # df[col] returns a DataFrame instead of a Series for a duplicated name, which
            # breaks every .isnull()/.dropna() call downstream with a confusing pandas
            # internals error ("truth value of a Series is ambiguous") far from the actual
            # cause. Fail clearly here instead, at the one place that knows what's wrong.
            # Checking after the str() rename above also catches the rarer case of two
            # differently-typed columns (an int 1 and a string "1") colliding once stringified.
            raise ValueError(f"duplicate column name(s) in the input dataframe: {duplicated}")

        columns = list(df.columns)
        types = infer_all_types(df, overrides=column_types)

        marginals: dict[str, dict[str, Any]] = {}
        stored_types: dict[str, str] = {}

        for col in columns:
            if df[col].isnull().all():
                marginals[col] = {"kind": ALL_NULL_KIND}
                stored_types[col] = ALL_NULL_KIND
                continue

            ctype = types[col]
            marginal_class = TYPE_TO_MARGINAL_CLASS[ctype]
            marginals[col] = marginal_class.fit(df[col]).to_dict()
            stored_types[col] = ctype.value

        null_columns = [c for c in columns if df[c].isnull().any()]
        null_model = NullModel.fit(df, null_columns).to_dict()

        copula_columns = [
            c
            for c in columns
            if types[c] in COPULA_ELIGIBLE_TYPES and marginals[c]["kind"] != ALL_NULL_KIND
        ]

        copula_dict: dict[str, Any] | None = None
        if copula_columns:
            complete = df.dropna(subset=copula_columns)
            if len(complete) >= MIN_COMPLETE_ROWS_FOR_COPULA:
                uniform_columns = {
                    col: _pseudo_uniform(
                        types[col], complete[col], _marginal_from_dict(marginals[col])
                    )
                    for col in copula_columns
                }
                # A constant column (every fitted row has the same value) has a degenerate,
                # zero-variance pseudo-uniform, which drives the correlation matrix to NaN and
                # crashes Cholesky sampling downstream. Correlation with a constant is
                # undefined anyway, so drop it from the copula; it still gets sampled
                # independently from its own marginal in emit().
                uniform_columns = {
                    col: u for col, u in uniform_columns.items() if np.unique(u).size > 1
                }
                copula_columns = [c for c in copula_columns if c in uniform_columns]

                if uniform_columns:
                    copula_dict = GaussianCopula.fit(uniform_columns).to_dict()
            else:
                copula_columns = []

        parsed_constraints = parse_constraints(constraints) if constraints else []

        return cls(
            columns=columns,
            column_types=stored_types,
            marginals=marginals,
            null_columns=null_columns,
            null_model=null_model,
            copula_columns=copula_columns,
            copula=copula_dict,
            n_rows_fit=len(df),
            constraints=constraints_to_dicts(parsed_constraints),
        )

    def emit(
        self,
        n: int,
        seed: int,
        key_pools: dict[str, list[Any]] | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        result: dict[str, np.ndarray] = {}

        if self.copula_columns and self.copula is not None:
            copula_obj = GaussianCopula.from_dict(self.copula)
            u_map = copula_obj.sample(n, rng)
            for col in self.copula_columns:
                ctype = ColumnType(self.column_types[col])
                result[col] = _sample_column(ctype, self.marginals[col], u_map[col])

        for col in self.columns:
            if col in result:
                continue

            marginal_dict = self.marginals[col]
            if marginal_dict["kind"] == ALL_NULL_KIND:
                result[col] = np.array([None] * n, dtype=object)
                continue

            ctype = ColumnType(self.column_types[col])
            if ctype == ColumnType.IDENTIFIER:
                result[col] = IdentifierMarginal.from_dict(marginal_dict).sample(n, rng)
            elif ctype == ColumnType.TEXT:
                result[col] = TextMarginal.from_dict(marginal_dict).sample(n, rng)
            else:
                # A copula-eligible column that got skipped for lack of complete rows: fall
                # back to independent sampling rather than dropping the column.
                u = rng.uniform(0, 1, n)
                result[col] = _sample_column(ctype, marginal_dict, u)

        frame = pd.DataFrame(result, columns=self.columns)

        null_model_obj = NullModel.from_dict(self.null_model)
        if null_model_obj.columns:
            mask = null_model_obj.sample(n, rng)
            for col in null_model_obj.columns:
                frame.loc[mask[col].to_numpy(), col] = None

        if self.constraints:
            constraints = parse_constraints(self.constraints)
            frame = apply_constraints(frame, constraints, rng, key_pools)

        return frame

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "column_types": self.column_types,
            "marginals": self.marginals,
            "null_columns": self.null_columns,
            "null_model": self.null_model,
            "copula_columns": self.copula_columns,
            "copula": self.copula,
            "n_rows_fit": self.n_rows_fit,
            "constraints": self.constraints,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Profile:
        return cls(**data)

    def save(self, path: str | Path) -> None:
        serialization.dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> Profile:
        return cls.from_dict(serialization.load(path))
