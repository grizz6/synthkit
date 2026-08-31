import numpy as np
import pandas as pd
import pytest

from synthkit.marginals import OTHER_CATEGORY, CategoricalMarginal


def test_categories_ordered_by_descending_frequency():
    values = pd.Series(["a"] * 10 + ["b"] * 30 + ["c"] * 5)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.categories == ["b", "a", "c"]
    assert marginal.probabilities[0] > marginal.probabilities[1] > marginal.probabilities[2]


def test_numeric_categories_ordered_by_value_not_frequency():
    # Regression test: a low-cardinality numeric column (a rating, a small count) still has a
    # natural order that the copula's rank correlation with other numeric columns depends on.
    # Ordering by descending frequency instead, correct for a genuinely nominal categorical
    # like a color, silently scrambled that order: a wine-quality-style rating peaking at
    # 5 and 6 used to order as [5, 6, 7, 4, 8, 3] instead of [3, 4, 5, 6, 7, 8].
    values = pd.Series([5] * 68 + [6] * 64 + [7] * 20 + [4] * 5 + [8] * 2 + [3] * 1)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.categories == ["3", "4", "5", "6", "7", "8"]


def test_numeric_tail_bucketing_truncates_by_frequency_not_by_value():
    # Regression test: numeric categoricals are ordered by value (not frequency) so the
    # copula's rank correlation stays meaningful, but truncation used to slice positions out
    # of that same value-sorted list. That kept whichever values were numerically smallest and
    # bucketed the rest as "other" regardless of frequency, so a value making up 90% of the
    # data got bucketed into __other__ purely because it was numerically the largest, while
    # four values appearing 5 times each were kept as their own categories.
    common = [900] * 900
    rare = [v for v in range(1, 21) for _ in range(5)]
    values = pd.Series(common + rare)

    marginal = CategoricalMarginal.fit(values, max_categories=5)
    assert "900" in marginal.categories
    dominant_index = marginal.categories.index("900")
    assert marginal.probabilities[dominant_index] == pytest.approx(0.9)


def test_boolean_dtype_categories_ordered_false_before_true():
    values = pd.Series([True, True, True, False])
    marginal = CategoricalMarginal.fit(values)
    assert marginal.categories == ["False", "True"]


def test_probabilities_sum_to_one():
    values = pd.Series(np.random.default_rng(0).choice(["x", "y", "z"], size=1000))
    marginal = CategoricalMarginal.fit(values)
    assert abs(sum(marginal.probabilities) - 1.0) < 1e-9


def test_tail_bucketed_as_other_beyond_max_categories():
    values = pd.Series([f"cat_{i}" for i in range(100) for _ in range(1)])
    marginal = CategoricalMarginal.fit(values, max_categories=10)
    assert OTHER_CATEGORY in marginal.categories
    assert marginal.other_mass > 0
    assert len(marginal.categories) == 10


def test_sample_recovers_approximate_frequencies():
    values = pd.Series(["a"] * 800 + ["b"] * 200)
    marginal = CategoricalMarginal.fit(values)
    u = np.random.default_rng(0).uniform(0, 1, 10000)
    sampled = marginal.sample(u)
    frac_a = (sampled == "a").mean()
    assert abs(frac_a - 0.8) < 0.02


def test_round_trip():
    values = pd.Series(["a", "b", "a", "c", "b", "a"])
    marginal = CategoricalMarginal.fit(values)
    restored = CategoricalMarginal.from_dict(marginal.to_dict())
    assert restored.categories == marginal.categories
    assert restored.probabilities == marginal.probabilities


def test_stable_tiebreak_uses_first_occurrence():
    values = pd.Series(["z", "a", "z", "a"])
    marginal = CategoricalMarginal.fit(values)
    assert marginal.categories == ["z", "a"]


def test_low_cardinality_integer_column_samples_back_as_int():
    values = pd.Series([0, 1, 0, 0, 1] * 20)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.value_dtype == "int"
    sampled = marginal.sample(np.linspace(0, 1, 50))
    assert sampled.dtype.kind in "iu"


def test_numeric_column_with_other_bucket_still_casts_non_other_rows():
    # Regression test: when tail-bucketing produced an __other__ category (only reachable via
    # an explicit column_types override forcing a high-cardinality numeric column into
    # CATEGORICAL), the dtype cast used to be skipped for the *entire* array, leaving every
    # row, not just the __other__ ones, as a stringified numeral instead of a real int.
    values = pd.Series(range(200))  # 200 distinct ints, forces tail-bucketing at max_categories
    marginal = CategoricalMarginal.fit(values, max_categories=10)
    assert marginal.value_dtype == "int"
    assert OTHER_CATEGORY in marginal.categories

    sampled = marginal.sample(np.linspace(0, 1, 100))
    assert sampled.dtype.kind == "f"  # NaN-capable, since __other__ has no single real value
    non_other = sampled[~np.isnan(sampled)]
    assert len(non_other) > 0
    kept_categories = {float(c) for c in marginal.categories if c != OTHER_CATEGORY}
    assert set(non_other).issubset(kept_categories)


def test_low_cardinality_float_column_samples_back_as_float():
    values = pd.Series([1.5, 2.5, 1.5, 2.5] * 20)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.value_dtype == "float"
    sampled = marginal.sample(np.linspace(0, 1, 50))
    assert sampled.dtype.kind == "f"


def test_string_column_still_samples_as_object():
    values = pd.Series(["a", "b", "c"] * 20)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.value_dtype == "str"
    sampled = marginal.sample(np.linspace(0, 1, 10))
    assert sampled.dtype == object


def test_rejects_all_null_column():
    values = pd.Series([None, None, None], dtype=object)
    with pytest.raises(ValueError):
        CategoricalMarginal.fit(values)
