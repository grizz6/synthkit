import numpy as np
import pandas as pd
import pytest

from synthkit.marginals import BooleanMarginal, DatetimeMarginal


def test_boolean_marginal_fits_probability():
    values = pd.Series([True] * 30 + [False] * 70)
    marginal = BooleanMarginal.fit(values)
    assert abs(marginal.probability_true - 0.3) < 1e-9


def test_boolean_marginal_sample_recovers_rate():
    values = pd.Series([True] * 700 + [False] * 300)
    marginal = BooleanMarginal.fit(values)
    u = np.random.default_rng(0).uniform(0, 1, 10000)
    assert abs(marginal.sample(u).mean() - 0.7) < 0.02


def test_boolean_round_trip():
    marginal = BooleanMarginal.fit(pd.Series([True, False, True]))
    restored = BooleanMarginal.from_dict(marginal.to_dict())
    assert restored.probability_true == marginal.probability_true


def test_datetime_marginal_infers_daily_granularity():
    dates = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D"))
    marginal = DatetimeMarginal.fit(dates)
    assert marginal.granularity_seconds == 86400


def test_datetime_marginal_infers_second_granularity():
    dates = pd.Series(pd.date_range("2024-01-01", periods=100, freq="s"))
    marginal = DatetimeMarginal.fit(dates)
    assert marginal.granularity_seconds == 1


def test_datetime_marginal_sample_stays_on_grid():
    dates = pd.Series(pd.date_range("2024-01-01", periods=200, freq="D"))
    marginal = DatetimeMarginal.fit(dates)
    u = np.random.default_rng(0).uniform(0, 1, 50)
    sampled = marginal.sample(u)
    epoch = sampled.to_numpy().astype("datetime64[s]").astype("int64")
    assert np.all(epoch % 86400 == 0)


def test_datetime_marginal_sample_within_observed_range():
    dates = pd.Series(pd.date_range("2024-01-01", periods=365, freq="D"))
    marginal = DatetimeMarginal.fit(dates)
    u = np.linspace(0, 1, 50)
    sampled = marginal.sample(u)
    assert sampled.min() >= dates.min()
    assert sampled.max() <= dates.max()


def test_datetime_round_trip():
    dates = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    marginal = DatetimeMarginal.fit(dates)
    restored = DatetimeMarginal.from_dict(marginal.to_dict())
    u = np.linspace(0, 1, 10)
    assert (marginal.sample(u) == restored.sample(u)).all()


def test_boolean_marginal_rejects_all_null_column():
    values = pd.Series([None, None], dtype=object)
    with pytest.raises(ValueError):
        BooleanMarginal.fit(values)


def test_datetime_marginal_rejects_all_null_column():
    values = pd.Series([None, None], dtype="datetime64[ns]")
    with pytest.raises(ValueError):
        DatetimeMarginal.fit(values)
