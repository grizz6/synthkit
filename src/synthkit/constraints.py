"""The constraints DSL.

The copula gets the statistics right and the business rules wrong: it has no idea that
`created_at` must precede `updated_at`, or that `total` is `subtotal + tax` rather than an
independently distributed number. Constraints are declared separately from the statistical
model and enforced afterward by `repair.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

VALID_INEQUALITY_OPS = {"<=", "<", ">=", ">"}


@dataclass
class Inequality:
    left: str
    op: str
    right: str

    def __post_init__(self) -> None:
        if self.op not in VALID_INEQUALITY_OPS:
            raise ValueError(f"unsupported inequality operator: {self.op!r}")


@dataclass
class Derived:
    # expr is evaluated with pandas' DataFrame.eval. A column name that isn't a valid Python
    # identifier (a hyphen, a space, a leading digit) needs backtick-quoting, e.g.
    # "`sub-total` + tax", the same as pandas' own eval/query syntax.
    column: str
    expr: str


@dataclass
class ConditionalNull:
    # null_when is evaluated the same way as Derived.expr; see the note there.
    column: str
    null_when: str


@dataclass
class Unique:
    columns: list[str]


@dataclass
class ForeignKey:
    column: str
    references: str


Constraint = Inequality | Derived | ConditionalNull | Unique | ForeignKey

_CONSTRAINT_TYPES: dict[str, type] = {
    "inequality": Inequality,
    "derived": Derived,
    "conditional_null": ConditionalNull,
    "unique": Unique,
    "foreign_key": ForeignKey,
}


def _constraint_from_dict(data: dict[str, Any]) -> Constraint:
    kind = data["type"]
    if kind not in _CONSTRAINT_TYPES:
        raise ValueError(f"unknown constraint type: {kind!r}")
    kwargs = {k: v for k, v in data.items() if k != "type"}
    return _CONSTRAINT_TYPES[kind](**kwargs)


def _constraint_to_dict(constraint: Constraint) -> dict[str, Any]:
    kind = next(name for name, cls in _CONSTRAINT_TYPES.items() if isinstance(constraint, cls))
    return {"type": kind, **vars(constraint)}


def parse_constraints(
    spec: str | Path | Sequence[dict[str, Any] | Constraint] | dict[str, Any],
) -> list[Constraint]:
    """Parse constraints from a YAML file path, a list of dicts or `Constraint` instances
    (the Python API accepts `sk.Inequality(...)` directly), or `{"constraints": [...]}`."""
    if isinstance(spec, (str, Path)):
        data = yaml.safe_load(Path(spec).read_text())
    else:
        data = spec

    if isinstance(data, dict):
        data = data.get("constraints", [])

    return [_coerce_constraint(item) for item in data]


def _coerce_constraint(item: dict[str, Any] | Constraint) -> Constraint:
    if isinstance(item, dict):
        return _constraint_from_dict(item)
    return item


def constraints_to_dicts(constraints: list[Constraint]) -> list[dict[str, Any]]:
    return [_constraint_to_dict(c) for c in constraints]


def referenced_columns(constraint: Constraint) -> list[str]:
    """The dataframe columns a constraint names directly.

    Expression strings (`Derived.expr`, `ConditionalNull.null_when`) are deliberately not
    parsed: pandas' own eval already reports an undefined name inside one, and parsing its
    expression grammar here to find column references would be a second, worse implementation
    of that. This covers only the fields that name a column outright, which is what a typo in
    a hand-written YAML constraint file usually lands in.
    """
    if isinstance(constraint, Inequality):
        return [constraint.left, constraint.right]
    if isinstance(constraint, Unique):
        return list(constraint.columns)
    return [constraint.column]
