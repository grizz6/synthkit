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


def test_timezone_aware_datetime_fits_without_warning():
    # Regression test: casting a tz-aware Series straight to np.datetime64 raises a
    # UserWarning and silently normalizes to UTC. Confirmed directly with -W error::UserWarning
    # before the fix; fitting should now be silent (the conversion is deliberate, not
    # accidental).
    dates = pd.Series(pd.date_range("2024-01-01", periods=100, freq="D", tz="US/Eastern"))
    marginal = DatetimeMarginal.fit(dates)
    assert marginal.granularity_seconds > 0


def test_timezone_aware_datetime_with_fixed_offset_detects_daily_granularity():
    # Regression test: a fixed-UTC-offset timezone (no DST transition in range) still records
    # genuinely daily data, every value is "local midnight," just not at a UTC-epoch daily
    # boundary. The naive `epoch_seconds % 86400 == 0` check used to call this hourly (the
    # constant ~5-hour offset from UTC midnight isn't itself a multiple of 86400), even though
    # nothing about the underlying data is actually hourly.
    dates = pd.Series(pd.date_range("2024-11-05", periods=30, freq="D", tz="US/Eastern"))
    marginal = DatetimeMarginal.fit(dates)
    assert marginal.granularity_seconds == 86400
    assert marginal.phase_seconds == 18000  # US/Eastern is UTC-5 (EST) in November


def test_timezone_aware_datetime_sample_reproduces_the_phase_offset():
    dates = pd.Series(pd.date_range("2024-11-05", periods=30, freq="D", tz="US/Eastern"))
    marginal = DatetimeMarginal.fit(dates)
    sampled = marginal.sample(np.linspace(0, 1, 50))
    epoch = sampled.to_numpy().astype("datetime64[s]").astype("int64")
    # Every sampled value should land exactly on a daily boundary shifted by the phase, not on
    # a bare UTC-midnight boundary (which would silently move the data by 5 hours).
    assert np.all((epoch - marginal.phase_seconds) % 86400 == 0)
    assert not np.all(epoch % 86400 == 0)


def test_datetime_index_input_does_not_crash():
    # Regression test: to_epoch_seconds used to call pd.to_datetime(values).dt.tz, but a
    # DatetimeIndex (what DatetimeMarginal.sample returns, and what a caller might reasonably
    # pass to drift/privacy checks) has no .dt accessor and raised AttributeError.
    index = pd.date_range("2024-01-01", periods=10, freq="D")
    marginal = DatetimeMarginal.fit(pd.Series(index))
    restored_index = marginal.sample(np.linspace(0, 1, 5))
    assert isinstance(restored_index, pd.DatetimeIndex)
    # Fitting directly on a DatetimeIndex (not wrapped in a Series) must also work.
    DatetimeMarginal.fit(pd.Series(index).astype("datetime64[ns]"))


def test_datetime_phase_defaults_to_zero_for_older_serialized_profiles():
    # A profile saved before phase_seconds existed won't have the key; from_dict should treat
    # that as phase 0 (the old, pre-fix behavior) rather than raising a KeyError.
    dates = pd.Series(pd.date_range("2024-01-01", periods=50, freq="D"))
    marginal = DatetimeMarginal.fit(dates)
    data = marginal.to_dict()
    del data["phase_seconds"]
    restored = DatetimeMarginal.from_dict(data)
    assert restored.phase_seconds == 0
