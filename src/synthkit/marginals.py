"""Per-column marginal distributions.

Each marginal is fit non-parametrically: no named distribution is assumed, because real
columns are rarely normal, lognormal, or gamma in the way a textbook expects. Instead each
marginal stores enough of the empirical shape to invert a uniform draw back into a realistic
value, which is exactly the interface the Gaussian copula in `copula.py` needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_N_KNOTS = 100


@dataclass
class NumericMarginal:
    """An empirical CDF stored as quantile knots, for continuous or discrete numeric columns."""

    quantile_levels: list[float]
    quantile_values: list[float]
    is_integer: bool

    @classmethod
    def fit(cls, values: pd.Series, n_knots: int = DEFAULT_N_KNOTS) -> "NumericMarginal":
        clean = values.dropna().to_numpy(dtype=float)
        if clean.size == 0:
            raise ValueError("cannot fit a numeric marginal on an all-null column")

        levels = np.linspace(0.0, 1.0, n_knots)
        knots = np.quantile(clean, levels)
        # Quantiles of a sample with ties can produce a non-monotonic-looking but technically
        # non-decreasing sequence; enforce non-decreasing explicitly so interpolation is sane.
        knots = np.maximum.accumulate(knots)
        is_integer = bool(np.all(clean == np.round(clean)))

        return cls(
            quantile_levels=levels.tolist(),
            quantile_values=knots.tolist(),
            is_integer=is_integer,
        )

    def sample(self, u: np.ndarray) -> np.ndarray:
        """Inverse-transform uniform draws `u` into values via linear interpolation."""
        values = np.interp(u, self.quantile_levels, self.quantile_values)
        if self.is_integer:
            values = np.round(values)
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "numeric",
            "quantile_levels": self.quantile_levels,
            "quantile_values": self.quantile_values,
            "is_integer": self.is_integer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NumericMarginal":
        return cls(
            quantile_levels=data["quantile_levels"],
            quantile_values=data["quantile_values"],
            is_integer=data["is_integer"],
        )


OTHER_CATEGORY = "__other__"
DEFAULT_MAX_CATEGORIES = 50


@dataclass
class CategoricalMarginal:
    """A category -> frequency table, ordered by descending frequency.

    The order is significant, not cosmetic: `copula.py` maps a uniform draw into this same
    ordered list of intervals, so two marginals fit on the same data must always produce the
    same order for the copula's correlation structure to mean anything.
    """

    categories: list[str]
    probabilities: list[float]
    other_mass: float

    @classmethod
    def fit(
        cls, values: pd.Series, max_categories: int = DEFAULT_MAX_CATEGORIES
    ) -> "CategoricalMarginal":
        clean = values.dropna().astype(str)
        if clean.empty:
            raise ValueError("cannot fit a categorical marginal on an all-null column")

        first_seen = {}
        for position, value in enumerate(clean):
            first_seen.setdefault(value, position)

        counts = clean.value_counts()
        ordered = sorted(counts.index, key=lambda c: (-counts[c], first_seen[c]))

        total = len(clean)
        other_mass = 0.0
        kept = ordered

        if len(ordered) > max_categories:
            kept = ordered[: max_categories - 1]
            tail = ordered[max_categories - 1 :]
            other_mass = sum(counts[c] for c in tail) / total

        probabilities = [counts[c] / total for c in kept]

        if other_mass > 0:
            kept = [*kept, OTHER_CATEGORY]
            probabilities = [*probabilities, other_mass]

        return cls(categories=kept, probabilities=probabilities, other_mass=other_mass)

    def sample(self, u: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(self.probabilities)
        # Guard against floating point drift leaving the top edge just under 1.0.
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, u, side="right")
        indices = np.clip(indices, 0, len(self.categories) - 1)
        return np.array(self.categories, dtype=object)[indices]

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "categorical",
            "categories": self.categories,
            "probabilities": self.probabilities,
            "other_mass": self.other_mass,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CategoricalMarginal":
        return cls(
            categories=data["categories"],
            probabilities=data["probabilities"],
            other_mass=data["other_mass"],
        )
