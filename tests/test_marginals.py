import numpy as np
import pandas as pd

from synthkit.marginals import NumericMarginal


def test_numeric_marginal_recovers_median():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(loc=50, scale=10, size=5000))
    marginal = NumericMarginal.fit(values)
    sampled_median = marginal.sample(np.array([0.5]))[0]
    assert abs(sampled_median - values.median()) < 1.0


def test_numeric_marginal_detects_integer_column():
    values = pd.Series([1, 2, 3, 4, 5] * 20, dtype=float)
    marginal = NumericMarginal.fit(values)
    assert marginal.is_integer is True
    sampled = marginal.sample(np.linspace(0, 1, 50))
    assert np.all(sampled == np.round(sampled))


def test_numeric_marginal_detects_continuous_column():
    values = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5] * 20)
    marginal = NumericMarginal.fit(values)
    assert marginal.is_integer is False


def test_numeric_marginal_round_trip():
    values = pd.Series(np.random.default_rng(1).uniform(0, 100, 500))
    marginal = NumericMarginal.fit(values)
    restored = NumericMarginal.from_dict(marginal.to_dict())
    u = np.linspace(0, 1, 25)
    assert np.allclose(marginal.sample(u), restored.sample(u))


def test_numeric_marginal_rejects_all_null_column():
    values = pd.Series([None, None, None], dtype=float)
    try:
        NumericMarginal.fit(values)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_numeric_marginal_ignores_nulls_when_fitting():
    values = pd.Series([1.0, 2.0, 3.0, None, 4.0, 5.0] * 20)
    marginal = NumericMarginal.fit(values)
    assert marginal.quantile_values[0] >= 1.0
    assert marginal.quantile_values[-1] <= 5.0
