import numpy as np
import pandas as pd
import pytest

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
    with pytest.raises(ValueError):
        NumericMarginal.fit(values)


def test_numeric_marginal_ignores_nulls_when_fitting():
    values = pd.Series([1.0, 2.0, 3.0, None, 4.0, 5.0] * 20)
    marginal = NumericMarginal.fit(values)
    assert marginal.quantile_values[0] >= 1.0
    assert marginal.quantile_values[-1] <= 5.0


def test_numeric_marginal_handles_positive_infinity_without_nan():
    # Regression test: np.quantile's linear interpolation hits `0 * inf = nan` for any knot
    # landing exactly on a sample index adjacent to +inf, and np.maximum.accumulate then
    # propagates that NaN through every subsequent (higher) knot, so sample() would silently
    # emit NaN for a fraction of rows instead of a number.
    values = pd.Series([1.0, 2.0, np.inf, 4.0, 5.0] * 20)
    marginal = NumericMarginal.fit(values)
    assert not any(np.isnan(v) for v in marginal.quantile_values)
    sampled = marginal.sample(np.linspace(0, 1, 1000))
    assert not np.isnan(sampled).any()


def test_numeric_marginal_handles_negative_infinity_without_nan():
    values = pd.Series([-np.inf, 2.0, 3.0, 4.0, 5.0] * 20)
    marginal = NumericMarginal.fit(values)
    assert not any(np.isnan(v) for v in marginal.quantile_values)
    sampled = marginal.sample(np.linspace(0, 1, 1000))
    assert not np.isnan(sampled).any()


def test_numeric_marginal_handles_infinity_on_both_ends_without_nan():
    values = pd.Series([-np.inf, 2.0, 3.0, 4.0, np.inf] * 20)
    marginal = NumericMarginal.fit(values)
    assert not any(np.isnan(v) for v in marginal.quantile_values)
    sampled = marginal.sample(np.linspace(0, 1, 1000))
    assert not np.isnan(sampled).any()
