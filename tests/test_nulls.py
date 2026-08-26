import numpy as np
import pandas as pd

from synthkit.nulls import NullModel


def test_fits_single_column_null_rate():
    df = pd.DataFrame({"a": [1, None, 3, None, 5, 6, 7, 8, 9, 10]})
    model = NullModel.fit(df, ["a"])
    rng = np.random.default_rng(0)
    sampled = model.sample(10000, rng)
    assert abs(sampled["a"].mean() - 0.2) < 0.02


def test_captures_null_co_occurrence():
    # b is null exactly when a is null: an independent per-column model would get the joint
    # rate of "both null" badly wrong, but the joint pattern model should reproduce it exactly.
    a = [1, None, 3, None, 5] * 20
    b = [10, None, 30, None, 50] * 20
    df = pd.DataFrame({"a": a, "b": b})
    model = NullModel.fit(df, ["a", "b"])
    rng = np.random.default_rng(0)
    sampled = model.sample(10000, rng)

    both_null = (sampled["a"] & sampled["b"]).mean()
    only_one_null = (sampled["a"] ^ sampled["b"]).mean()

    assert abs(both_null - 0.4) < 0.02
    assert only_one_null < 0.02


def test_no_nulls_produces_all_false():
    df = pd.DataFrame({"a": [1, 2, 3]})
    model = NullModel.fit(df, ["a"])
    rng = np.random.default_rng(0)
    sampled = model.sample(100, rng)
    assert not sampled["a"].any()


def test_empty_columns_list():
    df = pd.DataFrame({"a": [1, 2, 3]})
    model = NullModel.fit(df, [])
    rng = np.random.default_rng(0)
    sampled = model.sample(10, rng)
    assert len(sampled) == 10
    assert sampled.shape[1] == 0


def test_round_trip():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
    model = NullModel.fit(df, ["a", "b"])
    restored = NullModel.from_dict(model.to_dict())
    assert restored.patterns == model.patterns
    assert restored.probabilities == model.probabilities
