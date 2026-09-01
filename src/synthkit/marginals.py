"""Per-column marginal distributions, fit non-parametrically (no assumed named distribution).

Each marginal stores enough of the empirical shape to invert a uniform draw back into a
realistic value, which is the interface the Gaussian copula in copula.py needs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_N_KNOTS = 100


def _encode_non_finite(value: float) -> float | str:
    if value == float("inf"):
        return "Infinity"
    if value == float("-inf"):
        return "-Infinity"
    return value


def _decode_non_finite(value: float | str) -> float:
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    return float(value)


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
        # A +-inf value can produce NaN knots near it (0 * inf during interpolation); fill
        # from the nearest valid neighbor rather than let NaN propagate into sample().
        if np.isnan(knots).any():
            knots = pd.Series(knots).ffill().bfill().to_numpy()
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
            # A real +-inf value in the source column can end up as a knot (see fit() above).
            # json.dumps would emit the bare, non-standard tokens Infinity/-Infinity for that,
            # which most JSON parsers outside Python reject, so encode them as strings instead.
            "quantile_values": [_encode_non_finite(v) for v in self.quantile_values],
            "is_integer": self.is_integer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumericMarginal:
        return cls(
            quantile_levels=data["quantile_levels"],
            quantile_values=[_decode_non_finite(v) for v in data["quantile_values"]],
            is_integer=data["is_integer"],
        )


OTHER_CATEGORY = "__other__"
DEFAULT_MAX_CATEGORIES = 50


def escape_other_category(value: str) -> str:
    """Disambiguate a real value from the reserved long-tail-bucket sentinel.

    OTHER_CATEGORY is added to `categories` as its own entry for the synthesized tail bucket
    (see CategoricalMarginal.fit). A real value equal to it verbatim would otherwise collide:
    two entries in `categories` sharing one label breaks anything that looks a category up by
    name, notably copula.py's category_pseudo_uniform. Both fitting and that lookup need to
    apply this same escape to a raw value before treating it as a category label.
    """
    return f"{OTHER_CATEGORY}_actual" if value == OTHER_CATEGORY else value


@dataclass
class CategoricalMarginal:
    """A category -> frequency table.

    Order matters here, not just for display: copula.py maps a uniform draw into this same
    ordered list of intervals, so the order has to be consistent between fitting and sampling.
    """

    categories: list[str]
    probabilities: list[float]
    other_mass: float
    value_dtype: str = "str"

    @classmethod
    def fit(
        cls, values: pd.Series, max_categories: int = DEFAULT_MAX_CATEGORIES
    ) -> CategoricalMarginal:
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

        if (clean == OTHER_CATEGORY).any():
            clean = clean.map(escape_other_category)

        counts = clean.value_counts()

        if value_dtype == "bool":
            ordered = sorted(counts.index, key=lambda c: c == "True")
        elif value_dtype in ("int", "float"):
            # Numeric categoricals (a rating, a small count) have a real order that the
            # copula's correlation with other numeric columns depends on. Ordering by
            # frequency instead would scramble it, so order by value here.
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
            if value_dtype in ("int", "float"):
                # `ordered` is sorted by value, not frequency, to keep the copula's rank
                # correlation meaningful (see above). Truncating positions in that list would
                # keep whichever values are numerically smallest and bucket the rest as
                # "other" regardless of how common they actually are, e.g. a value that's 90%
                # of the data landing in "other" just because it's numerically the largest.
                # Pick the truncation by frequency instead, then keep the result in value order.
                by_frequency = sorted(counts.index, key=lambda c: -counts[c])
                keep_set = set(by_frequency[: max_categories - 1])
                kept = [c for c in ordered if c in keep_set]
                tail = [c for c in ordered if c not in keep_set]
            else:
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
        cumulative[-1] = 1.0  # guard against float drift at the top edge
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

        # __other__ has no single real value to cast back to; leave those as NaN and cast
        # the rest normally.
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
    """Convert a datetime Series to whole epoch seconds (int64).

    Handles two wrinkles in one place: pandas' datetime64 unit isn't fixed (ns historically,
    us/s increasingly), so a bare cast to int64 without normalizing the unit first can compare
    nanoseconds against seconds elsewhere in the pipeline. And a timezone-aware Series has no
    direct cast to datetime64[s] at all; this converts to naive UTC explicitly instead of
    relying on numpy's implicit (and noisy) fallback.
    """
    dt = pd.DatetimeIndex(pd.to_datetime(values))
    if dt.tz is not None:
        dt = dt.tz_convert("UTC").tz_localize(None)
    return dt.to_numpy().astype("datetime64[s]").astype("int64")


# Largest first: prefer the coarsest granularity every timestamp is an exact multiple of, so
# daily data emits at midnight rather than an arbitrary second.
GRANULARITY_CANDIDATES_SECONDS = [86400, 3600, 60, 1]


def _infer_granularity(epoch_seconds: np.ndarray) -> tuple[int, int]:
    """The coarsest granularity every timestamp is a multiple of, and the phase offset it's
    relative to (rather than the Unix epoch, so a fixed-UTC-offset timezone doesn't get
    misdetected as finer-grained than it is)."""
    anchor = int(epoch_seconds.min())
    for granularity in GRANULARITY_CANDIDATES_SECONDS:
        if np.all((epoch_seconds - anchor) % granularity == 0):
            return granularity, anchor % granularity
    # Unreachable: GRANULARITY_CANDIDATES_SECONDS ends in 1, and every integer is a multiple
    # of 1, so the loop above always returns before falling off the end.
    raise AssertionError("no granularity candidate matched; this should never happen")


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
    """A regenerated identifier format. Values are never modeled statistically, only the
    shape (prefix + digit width, or a random token of similar length)."""

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
            # Stay within the fitted digit width whenever n allows, instead of a uniformly
            # random start that could blow past it (e.g. CUST001 becoming CUST483920).
            max_value = 10**self.digit_width - 1
            highest_start = max(max_value - n + 1, 0)
            start = int(rng.integers(0, highest_start + 1))
            numbers = start + np.arange(n)
            width = max(self.digit_width, len(str(numbers[-1])) if n else self.digit_width)
            return np.array([f"{self.prefix}{num:0{width}d}" for num in numbers], dtype=object)

        charset = np.array(list(DEFAULT_TOKEN_CHARSET))
        token_length = self.token_length
        # Collision-rejection sampling below hangs forever once n approaches or exceeds the
        # keyspace (len(charset) ** token_length): near exhaustion, almost every draw is a
        # repeat. Widen the token length until the keyspace comfortably exceeds n instead of
        # letting a short fitted length (e.g. a 4-char SKU) silently freeze a large emit().
        while len(charset) ** token_length < max(n * 2, 1):
            token_length += 1

        seen: set[str] = set()
        tokens: list[str] = []
        while len(tokens) < n:
            batch = rng.choice(charset, size=(n - len(tokens), token_length))
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

    Not a language model and not a verbatim bootstrap of real strings; it preserves rough
    vocabulary and sentence length, nothing more. Review any free-text column for PII before
    sharing a profile built from it, since word-level resampling isn't a privacy guarantee.
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
