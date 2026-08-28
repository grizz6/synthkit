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
    def fit(cls, values: pd.Series, n_knots: int = DEFAULT_N_KNOTS) -> NumericMarginal:
        clean = values.dropna().to_numpy(dtype=float)
        if clean.size == 0:
            raise ValueError("cannot fit a numeric marginal on an all-null column")

        levels = np.linspace(0.0, 1.0, n_knots)
        with np.errstate(invalid="ignore"):
            knots = np.quantile(clean, levels)
        # A genuine +-inf value in the data (a division result, a sentinel) makes numpy's
        # quantile interpolation hit `0 * inf = nan` for any knot that lands exactly on a
        # sample index next to it -- even though the mathematically sensible answer is just
        # that neighboring value. Left alone, every quantile knot at or above the first NaN
        # becomes NaN too (np.maximum.accumulate propagates NaN once it appears), and
        # `sample()` would silently emit NaN for a chunk of rows instead of a number. Forward-
        # then-backward-filling from the nearest valid neighbor is the same fix pandas itself
        # uses for exactly this class of gap, and every neighbor is at most 1/n_knots away in
        # quantile space, so the correction is small everywhere except the tail(s) actually
        # touching +-inf.
        if np.isnan(knots).any():
            knots = pd.Series(knots).ffill().bfill().to_numpy()
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
    def from_dict(cls, data: dict[str, Any]) -> NumericMarginal:
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
    value_dtype: str = "str"

    @classmethod
    def fit(
        cls, values: pd.Series, max_categories: int = DEFAULT_MAX_CATEGORIES
    ) -> CategoricalMarginal:
        # A low-cardinality integer or whole-valued float column is classified as categorical
        # (see types.py) but should still emit ints/floats, not the string representation used
        # internally for ordering and counting.
        if pd.api.types.is_bool_dtype(values):
            value_dtype = "bool"
        elif pd.api.types.is_integer_dtype(values):
            value_dtype = "int"
        elif pd.api.types.is_float_dtype(values):
            value_dtype = "float"
        else:
            value_dtype = "str"

        clean = values.dropna().astype(str)
        if clean.empty:
            raise ValueError("cannot fit a categorical marginal on an all-null column")

        counts = clean.value_counts()

        if value_dtype == "bool":
            ordered = sorted(counts.index, key=lambda c: c == "True")
        elif value_dtype in ("int", "float"):
            # A low-cardinality *numeric* column (a rating, a small count) still has a natural
            # order that the copula's rank correlation with other numeric columns depends on.
            # Ordering by descending frequency instead -- the right call for a genuinely
            # nominal categorical, see the `else` branch -- would scramble that order (real
            # example: a 1-5 rating whose value counts happen to peak at 3 then 4 gets ordered
            # [3, 4, 2, 5, 1], which no longer corresponds to rating order at all) and silently
            # destroy any correlation this column has with anything else.
            ordered = sorted(counts.index, key=float)
        else:
            first_seen: dict[str, int] = {}
            for position, value in enumerate(clean):
                first_seen.setdefault(value, position)
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

        return cls(
            categories=kept,
            probabilities=probabilities,
            other_mass=other_mass,
            value_dtype=value_dtype,
        )

    def sample(self, u: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(self.probabilities)
        # Guard against floating point drift leaving the top edge just under 1.0.
        cumulative[-1] = 1.0
        indices = np.searchsorted(cumulative, u, side="right")
        indices = np.clip(indices, 0, len(self.categories) - 1)
        values = np.array(self.categories, dtype=object)[indices]

        if self.value_dtype not in ("int", "float", "bool"):
            return values

        other_mask = values == OTHER_CATEGORY
        if not other_mask.any():
            if self.value_dtype == "int":
                return values.astype(np.int64)
            if self.value_dtype == "float":
                return values.astype(np.float64)
            return values.astype(str) == "True"

        # Some rows landed in the merged tail bucket, which has no single real value to cast
        # back to (only reachable via an explicit column_types override forcing a
        # high-cardinality numeric column into CATEGORICAL). Representing those as NaN keeps
        # every other row's dtype correct instead of leaving the whole column stringified.
        result = np.full(len(values), np.nan, dtype=np.float64)
        result[~other_mask] = values[~other_mask].astype(np.float64)
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "categorical",
            "categories": self.categories,
            "probabilities": self.probabilities,
            "other_mass": self.other_mass,
            "value_dtype": self.value_dtype,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CategoricalMarginal:
        return cls(
            categories=data["categories"],
            probabilities=data["probabilities"],
            other_mass=data["other_mass"],
            value_dtype=data.get("value_dtype", "str"),
        )


@dataclass
class BooleanMarginal:
    """A single Bernoulli probability."""

    probability_true: float

    @classmethod
    def fit(cls, values: pd.Series) -> BooleanMarginal:
        clean = values.dropna()
        if clean.empty:
            raise ValueError("cannot fit a boolean marginal on an all-null column")
        return cls(probability_true=float(clean.mean()))

    def sample(self, u: np.ndarray) -> np.ndarray:
        return u < self.probability_true

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "boolean", "probability_true": self.probability_true}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BooleanMarginal:
        return cls(probability_true=data["probability_true"])


def to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """Convert a datetime Series to whole epoch seconds (int64), handling every wrinkle in one
    place instead of six call sites each getting it slightly differently.

    Two things go wrong with the naive `.to_numpy().astype("datetime64[s]")`:

    1. pandas' default datetime64 unit varies (ns historically, us/s increasingly, and can
       differ between two Series parsed at different times), so a bare `.astype("int64")`
       without normalizing the unit first can compare nanoseconds against seconds elsewhere in
       the pipeline -- a factor-of-a-billion mismatch that made a KS test report two
       essentially-identical distributions as completely disjoint before this existed.
    2. A timezone-aware Series has no meaningful direct cast to `datetime64[s]` at all --
       numpy has no timezone concept, so casting straight to it silently normalizes to UTC
       with nothing but a UserWarning as the hint, and if that shift isn't a whole number of
       hours, whatever granularity daily/hourly data actually had gets thrown off, producing
       output that doesn't land on the boundaries it should. Converting explicitly first makes
       that normalization a deliberate, silent-only-because-it's-correct choice: the emitted
       profile never carries timezone information, every value comes back as a naive UTC
       timestamp.
    """
    # Normalize to a DatetimeIndex rather than working with whatever came in: a plain Series
    # exposes tz/tz_convert only through its .dt accessor, while DatetimeMarginal.sample's own
    # return type (a DatetimeIndex) has no .dt accessor at all and would raise AttributeError
    # on the very same call that works fine for a Series.
    dt = pd.DatetimeIndex(pd.to_datetime(values))
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_numpy().astype("datetime64[s]").astype("int64")


# Candidate granularities in seconds, largest first: prefer the coarsest granularity that
# every observed timestamp is an exact multiple of, so daily data emits at midnight rather
# than at an arbitrary second.
GRANULARITY_CANDIDATES_SECONDS = [86400, 3600, 60, 1]


def _infer_granularity(epoch_seconds: np.ndarray) -> tuple[int, int]:
    """The coarsest granularity every timestamp is a multiple of, and the phase (offset from
    the Unix epoch) that multiple is relative to.

    Checking `epoch_seconds % granularity == 0` directly anchors the grid to the Unix epoch,
    which only detects "daily" for data that happens to land on UTC midnight. A column
    recorded as local midnight in a fixed-UTC-offset timezone (say, every day at 05:00 UTC)
    is just as genuinely daily, but every value has a constant 5-hour phase relative to the
    epoch that the naive check would call "hourly" instead of noticing the phase and stripping
    it out. Anchoring the check to one of the observed values instead of the epoch makes the
    detection invariant to that constant offset; the phase is still returned so `sample()` can
    reproduce it, since quantizing back to the bare Unix-epoch grid would silently move every
    generated timestamp by however many hours the offset was.
    """
    anchor = int(epoch_seconds.min())
    for granularity in GRANULARITY_CANDIDATES_SECONDS:
        if np.all((epoch_seconds - anchor) % granularity == 0):
            return granularity, anchor % granularity
    return 1, 0


@dataclass
class DatetimeMarginal:
    """A numeric marginal over epoch seconds, re-quantized to the observed granularity."""

    numeric: NumericMarginal
    granularity_seconds: int
    phase_seconds: int = 0

    @classmethod
    def fit(cls, values: pd.Series) -> DatetimeMarginal:
        clean = values.dropna()
        if clean.empty:
            raise ValueError("cannot fit a datetime marginal on an all-null column")

        epoch_seconds = to_epoch_seconds(clean)
        granularity, phase = _infer_granularity(epoch_seconds)
        numeric = NumericMarginal.fit(pd.Series(epoch_seconds.astype(float)))

        return cls(numeric=numeric, granularity_seconds=granularity, phase_seconds=phase)

    def sample(self, u: np.ndarray) -> pd.DatetimeIndex:
        raw_seconds = self.numeric.sample(u)
        shifted = raw_seconds - self.phase_seconds
        quantized = (
            np.round(shifted / self.granularity_seconds) * self.granularity_seconds
            + self.phase_seconds
        )
        return pd.to_datetime(quantized.astype("int64"), unit="s")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "datetime",
            "numeric": self.numeric.to_dict(),
            "granularity_seconds": self.granularity_seconds,
            "phase_seconds": self.phase_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatetimeMarginal:
        return cls(
            numeric=NumericMarginal.from_dict(data["numeric"]),
            granularity_seconds=data["granularity_seconds"],
            phase_seconds=data.get("phase_seconds", 0),
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
    def fit(cls, values: pd.Series) -> IdentifierMarginal:
        clean = values.dropna().astype(str)
        if clean.empty:
            raise ValueError("cannot fit an identifier marginal on an all-null column")

        matches = [_PREFIX_DIGITS_PATTERN.fullmatch(v) for v in clean]
        if all(matches):
            prefixes = {m.group(1) for m in matches}  # type: ignore[union-attr]
            if len(prefixes) == 1:
                width = max(len(m.group(2)) for m in matches)  # type: ignore[union-attr]
                return cls(style="sequential", prefix=next(iter(prefixes)), digit_width=width)

        avg_length = int(clean.str.len().mean().round())
        return cls(style="random_token", token_length=max(avg_length, 4))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.style == "sequential":
            # Keep the emitted numbers within the fitted digit width whenever `n` allows, so
            # e.g. fitted IDs like CUST001-CUST050 (digit_width=3) come back out as 3-digit
            # numbers rather than a uniformly random 6-digit blowout that no longer resembles
            # the source format.
            max_value = 10**self.digit_width - 1
            highest_start = max(max_value - n + 1, 0)
            start = int(rng.integers(0, highest_start + 1))
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
    def from_dict(cls, data: dict[str, Any]) -> IdentifierMarginal:
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
    def fit(cls, values: pd.Series, max_pool: int = DEFAULT_MAX_WORD_POOL) -> TextMarginal:
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
    def from_dict(cls, data: dict[str, Any]) -> TextMarginal:
        return cls(
            word_pool=data["word_pool"],
            mean_word_count=data["mean_word_count"],
            std_word_count=data["std_word_count"],
        )
