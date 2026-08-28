import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from synthkit.constraints import Inequality
from synthkit.profile import Profile
from synthkit.types import ColumnType


def make_correlated_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n).clip(18, 90)
    # hours worked correlates with age via a shared latent factor
    hours = 0.5 * age + rng.normal(0, 8, n)
    status = rng.choice(["active", "cancelled", "pending"], size=n, p=[0.7, 0.2, 0.1])
    cancelled_at = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    cancelled_at = cancelled_at.where(pd.Series(status) == "cancelled")
    return pd.DataFrame(
        {
            "age": age,
            "hours": hours,
            "status": status,
            "cancelled_at": cancelled_at,
            "is_active": pd.Series(status) == "active",
        }
    )


def test_fit_emit_round_trip_preserves_row_count():
    df = make_correlated_df()
    profile = Profile.fit(df)
    synthetic = profile.emit(n=500, seed=0)
    assert len(synthetic) == 500
    assert list(synthetic.columns) == list(df.columns)


def test_marginal_fidelity_via_ks_statistic():
    df = make_correlated_df()
    profile = Profile.fit(df)
    synthetic = profile.emit(n=len(df), seed=0)

    statistic, _ = ks_2samp(df["age"], synthetic["age"].astype(float))
    assert statistic < 0.1


def test_correlation_structure_is_preserved():
    df = make_correlated_df(n=5000)
    profile = Profile.fit(df)
    synthetic = profile.emit(n=5000, seed=0)

    real_corr = np.corrcoef(df["age"], df["hours"])[0, 1]
    synth_corr = np.corrcoef(synthetic["age"].astype(float), synthetic["hours"].astype(float))[0, 1]
    assert abs(real_corr - synth_corr) < 0.1


def test_correlation_with_low_cardinality_numeric_column_is_preserved():
    # Regression test: a low-cardinality numeric column (classified categorical by types.py)
    # used to be ordered by descending frequency for the copula, which scrambles its natural
    # numeric order whenever the value counts don't happen to already be monotonic -- silently
    # destroying its correlation with everything else. A "rating" correlated with "amount",
    # with counts peaking in the middle (order [3,4,2,5,1], not [1,2,3,4,5]), exercises exactly
    # that case.
    rng = np.random.default_rng(0)
    n = 3000
    rating = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.05, 0.15, 0.4, 0.3, 0.1])
    amount = rating * 10 + rng.normal(0, 3, n)
    df = pd.DataFrame({"rating": rating, "amount": amount})

    profile = Profile.fit(df)
    synthetic = profile.emit(n=n, seed=0)

    real_corr = np.corrcoef(df["rating"], df["amount"])[0, 1]
    synth_corr = np.corrcoef(synthetic["rating"].astype(float), synthetic["amount"].astype(float))[
        0, 1
    ]
    assert real_corr > 0.5  # sanity: the constructed correlation really is strong
    assert abs(real_corr - synth_corr) < 0.15


def test_conditional_null_pattern_is_roughly_preserved():
    df = make_correlated_df(n=5000)
    profile = Profile.fit(df)
    synthetic = profile.emit(n=5000, seed=0)

    real_null_rate = df["cancelled_at"].isnull().mean()
    synth_null_rate = synthetic["cancelled_at"].isnull().mean()
    assert abs(real_null_rate - synth_null_rate) < 0.05


def test_emit_is_deterministic_given_same_seed():
    df = make_correlated_df()
    profile = Profile.fit(df)
    first = profile.emit(n=200, seed=42)
    second = profile.emit(n=200, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_emit_differs_across_seeds():
    df = make_correlated_df()
    profile = Profile.fit(df)
    first = profile.emit(n=200, seed=1)
    second = profile.emit(n=200, seed=2)
    assert not first["age"].equals(second["age"])


def test_save_load_round_trip_produces_identical_output(tmp_path):
    df = make_correlated_df()
    profile = Profile.fit(df)
    path = tmp_path / "profile.json"
    profile.save(path)

    reloaded = Profile.load(path)
    before = profile.emit(n=100, seed=7)
    after = reloaded.emit(n=100, seed=7)
    pd.testing.assert_frame_equal(before, after)


def test_profile_contains_no_real_values():
    df = make_correlated_df(n=200)
    profile = Profile.fit(df)
    dumped = str(profile.to_dict())
    # Ages are floats with enough entropy that a literal match would be suspicious;
    # spot check a handful of exact real values are not embedded verbatim.
    for value in df["age"].head(20):
        assert f"{value:.10f}" not in dumped


def test_small_all_unique_string_column_does_not_leak_real_values():
    # Regression test: a small (<10 row) all-unique string column used to be classified
    # categorical, storing every real value verbatim and reproducing it in emitted output.
    df = pd.DataFrame(
        {
            "customer_id": ["a1b2c3", "x9y8z7", "q4w5e6", "m1n2b3", "zzz111", "ccc222", "ddd333"],
            "amount": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
        }
    )
    profile = Profile.fit(df)
    assert profile.column_types["customer_id"] == "identifier"
    synthetic = profile.emit(n=20, seed=0)
    assert not set(synthetic["customer_id"]) & set(df["customer_id"])


def test_all_null_column_stays_null_after_emit():
    df = make_correlated_df(n=200)
    df["always_null"] = None
    profile = Profile.fit(df)
    synthetic = profile.emit(n=50, seed=0)
    assert synthetic["always_null"].isnull().all()


def test_constant_column_does_not_crash_copula_fitting():
    # Regression test: a zero-variance column used to poison the entire correlation matrix
    # with NaN and crash Profile.fit with a LinAlgError from np.linalg.cholesky/eigh.
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, n),
            "hours": rng.normal(40, 5, n),
            "exchange_rate": [1.5] * n,
        }
    )
    profile = Profile.fit(df)
    assert "exchange_rate" not in profile.copula_columns
    synthetic = profile.emit(n=50, seed=0)
    assert (synthetic["exchange_rate"] == 1.5).all()


def test_handles_small_dataset_below_copula_threshold():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    profile = Profile.fit(df)
    synthetic = profile.emit(n=10, seed=0)
    assert len(synthetic) == 10


def test_emit_zero_rows_returns_empty_frame_with_correct_columns():
    df = pd.DataFrame({"a": range(50), "b": ["x", "y"] * 25})
    profile = Profile.fit(df)
    synthetic = profile.emit(n=0, seed=0)
    assert len(synthetic) == 0
    assert list(synthetic.columns) == ["a", "b"]


def test_emit_rejects_negative_n_with_a_clear_error():
    df = pd.DataFrame({"a": range(50)})
    profile = Profile.fit(df)
    with pytest.raises(ValueError, match="non-negative"):
        profile.emit(n=-5, seed=0)


def test_fit_with_constraints_enforces_them_on_emit():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "created_at": rng.integers(0, 1000, n).astype(float),
            "updated_at": rng.integers(0, 1000, n).astype(float),
        }
    )
    profile = Profile.fit(df, constraints=[Inequality("created_at", "<=", "updated_at")])
    synthetic = profile.emit(n=500, seed=0)
    assert (synthetic["created_at"] <= synthetic["updated_at"]).all()


def test_profile_without_constraints_has_empty_constraint_list():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10})
    profile = Profile.fit(df)
    assert profile.constraints == []


def test_duplicate_column_names_raise_a_clear_error():
    df = pd.DataFrame(np.random.default_rng(0).random((20, 2)), columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column name"):
        Profile.fit(df)


def test_integer_column_names_survive_save_load_round_trip(tmp_path):
    # Regression test: JSON object keys are always strings, so a profile fit on a dataframe
    # with integer column names used to save/reload with column_types keyed by "0"/"1"/...
    # while `columns` stayed [0, 1, ...] -- emit() then KeyError'd looking up an int in a
    # str-keyed dict. Confirmed directly before this fix.
    df = pd.DataFrame(np.random.default_rng(0).random((50, 3)), columns=[0, 1, 2])
    profile = Profile.fit(df)
    path = tmp_path / "profile.json"
    profile.save(path)

    reloaded = Profile.load(path)
    assert reloaded.columns == ["0", "1", "2"]
    synthetic = reloaded.emit(n=10, seed=0)
    assert list(synthetic.columns) == ["0", "1", "2"]


def test_column_types_override_matches_by_original_non_string_key():
    df = pd.DataFrame({0: range(30), 1: ["a", "b"] * 15})
    profile = Profile.fit(df, column_types={0: ColumnType.TEXT})
    assert profile.column_types["0"] == "text"
