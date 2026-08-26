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


@dataclass
class BooleanMarginal:
    """A single Bernoulli probability."""

    probability_true: float

    @classmethod
    def fit(cls, values: pd.Series) -> "BooleanMarginal":
        clean = values.dropna()
        if clean.empty:
            raise ValueError("cannot fit a boolean marginal on an all-null column")
        return cls(probability_true=float(clean.mean()))

    def sample(self, u: np.ndarray) -> np.ndarray:
        return u < self.probability_true

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "boolean", "probability_true": self.probability_true}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BooleanMarginal":
        return cls(probability_true=data["probability_true"])


# Candidate granularities in seconds, largest first: prefer the coarsest granularity that
# every observed timestamp is an exact multiple of, so daily data emits at midnight rather
# than at an arbitrary second.
GRANULARITY_CANDIDATES_SECONDS = [86400, 3600, 60, 1]


def _infer_granularity_seconds(epoch_seconds: np.ndarray) -> int:
    for granularity in GRANULARITY_CANDIDATES_SECONDS:
        if np.all(epoch_seconds % granularity == 0):
            return granularity
    return 1


@dataclass
class DatetimeMarginal:
    """A numeric marginal over epoch seconds, re-quantized to the observed granularity."""

    numeric: NumericMarginal
    granularity_seconds: int

    @classmethod
    def fit(cls, values: pd.Series) -> "DatetimeMarginal":
        clean = pd.to_datetime(values.dropna())
        if clean.empty:
            raise ValueError("cannot fit a datetime marginal on an all-null column")

        # Cast through datetime64[s] rather than dividing a nanosecond int64 by 1e9: pandas'
        # default datetime unit varies by version (ns historically, us/s increasingly), so
        # going straight to whole seconds sidesteps having to know which one we got.
        epoch_seconds = clean.to_numpy().astype("datetime64[s]").astype("int64")
        granularity = _infer_granularity_seconds(epoch_seconds)
        numeric = NumericMarginal.fit(pd.Series(epoch_seconds.astype(float)))

        return cls(numeric=numeric, granularity_seconds=granularity)

    def sample(self, u: np.ndarray) -> np.ndarray:
        raw_seconds = self.numeric.sample(u)
        quantized = np.round(raw_seconds / self.granularity_seconds) * self.granularity_seconds
        return pd.to_datetime(quantized.astype("int64"), unit="s")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "datetime",
            "numeric": self.numeric.to_dict(),
            "granularity_seconds": self.granularity_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatetimeMarginal":
        return cls(
            numeric=NumericMarginal.from_dict(data["numeric"]),
            granularity_seconds=data["granularity_seconds"],
        )
