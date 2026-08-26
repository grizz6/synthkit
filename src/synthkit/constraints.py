"""The constraints DSL.

The copula gets the statistics right and the business rules wrong: it has no idea that
`created_at` must precede `updated_at`, or that `total` is `subtotal + tax` rather than an
independently distributed number. Constraints are declared separately from the statistical
model and enforced afterward by `repair.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

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
    column: str
    expr: str


@dataclass
class ConditionalNull:
    column: str
    null_when: str


@dataclass
class Unique:
    columns: list[str]


@dataclass
class ForeignKey:
    column: str
    references: str


Constraint = Union[Inequality, Derived, ConditionalNull, Unique, ForeignKey]

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


def parse_constraints(spec: str | Path | list[dict[str, Any]] | dict[str, Any]) -> list[Constraint]:
    """Parse constraints from a YAML file path, a raw list of dicts, or `{"constraints": [...]}`."""
    if isinstance(spec, (str, Path)):
        data = yaml.safe_load(Path(spec).read_text())
    else:
        data = spec

    if isinstance(data, dict):
        data = data.get("constraints", [])

    return [_constraint_from_dict(d) for d in data]


def constraints_to_dicts(constraints: list[Constraint]) -> list[dict[str, Any]]:
    return [_constraint_to_dict(c) for c in constraints]
