"""Canonical JSON for profiles.

A profile is meant to be diffable in a pull request and byte-identical across a save/load
round trip, so serialization can't rely on whatever key order and float formatting Python
happens to produce. `canonicalize` walks a structure once, converts numpy scalars to native
Python types, rounds floats to a fixed precision, and sorts dict keys recursively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

FLOAT_PRECISION = 12


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), FLOAT_PRECISION)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    return value


def dumps(data: dict[str, Any]) -> str:
    canonical = canonicalize(data)
    return json.dumps(canonical, sort_keys=True, indent=2) + "\n"


def dump(data: dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(dumps(data))


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())
