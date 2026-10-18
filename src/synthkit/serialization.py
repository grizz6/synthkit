"""Canonical JSON for profiles.

A profile is meant to be diffable in a pull request and byte-identical across a save/load
round trip, so serialization can't rely on whatever key order and float formatting Python
happens to produce. `canonicalize` walks a structure once, converts numpy scalars to native
Python types, rounds floats to a fixed number of significant digits, and sorts dict keys
recursively.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

# Significant digits, not decimal places: a fixed count of decimal places rounds a column
# measured around 1e-13 down to zero. 15 is the most a float64 always round-trips exactly.
FLOAT_SIGNIFICANT_DIGITS = 15


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda kv: str(kv[0]))
        return {str(k): canonicalize(v) for k, v in items}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(f"{float(value):.{FLOAT_SIGNIFICANT_DIGITS}g}")
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
