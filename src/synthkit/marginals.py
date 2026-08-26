"""Per-column marginal distributions.

Each marginal is fit non-parametrically: no named distribution is assumed, because real
columns are rarely normal, lognormal, or gamma in the way a textbook expects. Instead each
marginal stores enough of the empirical shape to invert a uniform draw back into a realistic
value, which is exactly the interface the Gaussian copula in `copula.py` needs.
"""

from __future__ import annotations

import re
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


_PREFIX_DIGITS_PATTERN = re.compile(r"^(\D*)(\d+)$")
DEFAULT_TOKEN_CHARSET = "abcdefghijklmnopqrstuvwxyz0123456789"


@dataclass
class IdentifierMarginal:
    """A regenerated identifier format. Identifiers are never modeled statistically — per the
    project's own rule, doing so would just be memorizing real keys — only their *shape* is.
    """

    style: str  # "sequential" or "random_token"
    prefix: str = ""
    digit_width: int = 0
    token_length: int = 8

    @classmethod
    def fit(cls, values: pd.Series) -> "IdentifierMarginal":
        clean = values.dropna().astype(str)
        if clean.empty:
            raise ValueError("cannot fit an identifier marginal on an all-null column")

        matches = [_PREFIX_DIGITS_PATTERN.fullmatch(v) for v in clean]
        if all(matches):
            prefixes = {m.group(1) for m in matches}  # type: ignore[union-attr]
            if len(prefixes) == 1:
                width = max(len(m.group(2)) for m in matches)  # type: ignore[union-attr]
                return cls(style="sequential", prefix=next(iter(prefixes)), digit_width=width)

        avg_length = int(round(clean.str.len().mean()))
        return cls(style="random_token", token_length=max(avg_length, 4))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.style == "sequential":
            start = int(rng.integers(0, 1_000_000))
            numbers = start + np.arange(n)
            width = max(self.digit_width, len(str(numbers[-1])) if n else self.digit_width)
            return np.array([f"{self.prefix}{num:0{width}d}" for num in numbers], dtype=object)

        charset = np.array(list(DEFAULT_TOKEN_CHARSET))
        seen: set[str] = set()
        tokens: list[str] = []
        while len(tokens) < n:
            batch = rng.choice(charset, size=(n - len(tokens), self.token_length))
            for row in batch:
                token = "".join(row)
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return np.array(tokens, dtype=object)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "identifier",
            "style": self.style,
            "prefix": self.prefix,
            "digit_width": self.digit_width,
            "token_length": self.token_length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IdentifierMarginal":
        return cls(
            style=data["style"],
            prefix=data.get("prefix", ""),
            digit_width=data.get("digit_width", 0),
            token_length=data.get("token_length", 8),
        )


DEFAULT_MAX_WORD_POOL = 5000


@dataclass
class TextMarginal:
    """Free text, approximated by resampling words from the observed vocabulary.

    This is deliberately not a language model and not a verbatim bootstrap of real strings —
    reproducing an exact original sentence would defeat the entire point of the package. What
    it preserves is rough vocabulary and sentence length, nothing more. Free-text columns that
    contain PII (names, notes, complaint text) should be reviewed by a human before a profile
    built from them is shared; word-level resampling is not by itself a privacy guarantee.
    """

    word_pool: list[str]
    mean_word_count: float
    std_word_count: float

    @classmethod
    def fit(cls, values: pd.Series, max_pool: int = DEFAULT_MAX_WORD_POOL) -> "TextMarginal":
        clean = values.dropna().astype(str)
        if clean.empty:
            raise ValueError("cannot fit a text marginal on an all-null column")

        word_counts = clean.str.split().str.len()
        words = [word for value in clean for word in value.split()]

        if len(words) > max_pool:
            rng = np.random.default_rng(0)
            words = list(rng.choice(words, size=max_pool, replace=False))
        if not words:
            words = [""]

        return cls(
            word_pool=words,
            mean_word_count=float(word_counts.mean()),
            std_word_count=float(word_counts.std(ddof=0) or 0.0),
        )

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.std_word_count > 0:
            counts = rng.normal(self.mean_word_count, self.std_word_count, size=n)
        else:
            counts = np.full(n, self.mean_word_count)
        counts = np.clip(np.round(counts), 1, None).astype(int)

        pool = np.array(self.word_pool)
        return np.array(
            [" ".join(rng.choice(pool, size=c)) for c in counts],
            dtype=object,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "text",
            "word_pool": self.word_pool,
            "mean_word_count": self.mean_word_count,
            "std_word_count": self.std_word_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TextMarginal":
        return cls(
            word_pool=data["word_pool"],
            mean_word_count=data["mean_word_count"],
            std_word_count=data["std_word_count"],
        )
