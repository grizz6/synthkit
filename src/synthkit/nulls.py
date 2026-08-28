"""Joint null-pattern modeling.

A per-column null rate is wrong whenever nulls are correlated across columns (cancelled_at is
null exactly when status != 'cancelled', for example). This fits the joint distribution over
which combination of columns is null in each row, then samples whole patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class NullModel:
    """The joint distribution of null patterns across a set of columns."""

    columns: list[str]
    patterns: list[list[bool]]
    probabilities: list[float]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> NullModel:
        if not columns:
            return cls(columns=[], patterns=[], probabilities=[])

        mask = df[columns].isnull()
        pattern_tuples = list(mask.itertuples(index=False, name=None))

        first_seen: dict[tuple[bool, ...], int] = {}
        for position, pattern in enumerate(pattern_tuples):
            first_seen.setdefault(pattern, position)

        counts: dict[tuple[bool, ...], int] = {}
        for pattern in pattern_tuples:
            counts[pattern] = counts.get(pattern, 0) + 1

        total = len(pattern_tuples)
        ordered = sorted(counts, key=lambda p: (-counts[p], first_seen[p]))

        return cls(
            columns=columns,
            patterns=[list(p) for p in ordered],
            probabilities=[counts[p] / total for p in ordered],
        )

    def sample(self, n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Draw `n` rows of boolean null masks, one column per fitted column."""
        if not self.columns:
            return pd.DataFrame(index=range(n))

        indices = rng.choice(len(self.patterns), size=n, p=self.probabilities)
        chosen = np.array(self.patterns, dtype=bool)[indices]
        return pd.DataFrame(chosen, columns=self.columns)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": self.columns,
            "patterns": self.patterns,
            "probabilities": self.probabilities,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NullModel:
        return cls(
            columns=data["columns"],
            patterns=data["patterns"],
            probabilities=data["probabilities"],
        )
