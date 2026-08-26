import numpy as np
import pandas as pd

from synthkit.marginals import OTHER_CATEGORY, CategoricalMarginal


def test_categories_ordered_by_descending_frequency():
    values = pd.Series(["a"] * 10 + ["b"] * 30 + ["c"] * 5)
    marginal = CategoricalMarginal.fit(values)
    assert marginal.categories == ["b", "a", "c"]
    assert marginal.probabilities[0] > marginal.probabilities[1] > marginal.probabilities[2]


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
