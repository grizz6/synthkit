import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from synthkit.constraints import Inequality
from synthkit.profile import Profile


def make_correlated_df(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 12, n).clip(18, 90)
    # hours worked correlates with age via a shared latent factor
    hours = 0.5 * age + rng.normal(0, 8, n)
    status = rng.choice(["active", "cancelled", "pending"], size=n, p=[0.7, 0.2, 0.1])
    cancelled_at = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        rng.integers(0, 300, n), unit="D"
    )
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
    synth_corr = np.corrcoef(synthetic["age"].astype(float), synthetic["hours"].astype(float))[
        0, 1
    ]
    assert abs(real_corr - synth_corr) < 0.1


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


def test_all_null_column_stays_null_after_emit():
    df = make_correlated_df(n=200)
    df["always_null"] = None
    profile = Profile.fit(df)
    synthetic = profile.emit(n=50, seed=0)
    assert synthetic["always_null"].isnull().all()


def test_handles_small_dataset_below_copula_threshold():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    profile = Profile.fit(df)
    synthetic = profile.emit(n=10, seed=0)
    assert len(synthetic) == 10


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
