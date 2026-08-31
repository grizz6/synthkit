import numpy as np
import pandas as pd

import synthkit as sk


def make_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "created_at": rng.integers(0, 1000, n).astype(float),
            "updated_at": rng.integers(1000, 2000, n).astype(float),
            "amount": rng.normal(100, 20, n),
        }
    )


def test_public_api_matches_documented_usage():
    df = make_df()
    profile = sk.fit(df, constraints=[sk.Inequality("created_at", "<=", "updated_at")])
    synthetic = sk.emit(profile, n=1000, seed=42)

    assert (synthetic["created_at"] <= synthetic["updated_at"]).all()

    report = sk.check(synthetic, profile, real=df)
    assert isinstance(report, sk.CheckReport)
    assert "amount" in report.ks_by_column
    assert report.ks_by_column["amount"] < 0.15


def test_ks_by_column_datetime_fidelity_is_not_spuriously_maximal():
    # Regression test: real datetime data parsed via pd.to_datetime (e.g. from a CSV) is
    # typically datetime64[us] or [ns], while DatetimeMarginal.sample always emits
    # datetime64[s]. _fidelity_by_column used to cast each side to int64 directly without
    # normalizing the unit first, so real epoch values (microseconds) and synthetic epoch
    # values (seconds) differed by a factor of a million, ks_2samp then reported the two
    # distributions as completely disjoint (statistic == 1.0) regardless of actual fidelity.
    rng = np.random.default_rng(0)
    n = 500
    dates = pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1000, n), unit="D")
    df = pd.DataFrame({"event_date": dates, "amount": rng.normal(100, 20, n)})
    assert df["event_date"].dtype != "datetime64[s]"  # sanity: a realistic non-second unit

    profile = sk.fit(df)
    synthetic = sk.emit(profile, n=n, seed=0)
    report = sk.check(synthetic, profile, real=df)

    assert report.ks_by_column["event_date"] < 0.2


def test_ks_by_column_skips_non_numeric_non_datetime_columns():
    # A categorical column has no natural KS statistic; _fidelity_by_column should silently
    # skip it rather than crash trying to cast category labels to float.
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "amount": rng.normal(100, 20, n),
            "tier": rng.choice(["free", "pro", "enterprise"], size=n),
        }
    )
    profile = sk.fit(df)
    synthetic = sk.emit(profile, n=n, seed=0)
    report = sk.check(synthetic, profile, real=df)

    assert "amount" in report.ks_by_column
    assert "tier" not in report.ks_by_column


def test_ks_by_column_skips_a_column_that_is_all_null_at_check_time():
    # check() accepts any real dataframe, not necessarily the one the profile was fit on. If a
    # column classified as continuous at fit time (it had real values then) is entirely null
    # in the dataframe passed to check(), _fidelity_by_column should skip it after dropna()
    # leaves nothing to compare, rather than handing ks_2samp an empty array.
    rng = np.random.default_rng(0)
    fit_df = pd.DataFrame({"amount": rng.normal(100, 20, 300), "note": rng.normal(0, 1, 300)})
    profile = sk.fit(fit_df)
    synthetic = sk.emit(profile, n=300, seed=0)

    check_df = fit_df.copy()
    check_df["note"] = None
    report = sk.check(synthetic, profile, real=check_df)

    assert "amount" in report.ks_by_column
    assert "note" not in report.ks_by_column


def test_ks_by_column_skips_a_column_missing_from_the_check_time_real_dataframe():
    # check() doesn't require the real dataframe passed to it to have exactly the columns the
    # profile was fit on. A column dropped from that dataframe should be skipped, not KeyError.
    rng = np.random.default_rng(0)
    fit_df = pd.DataFrame({"amount": rng.normal(100, 20, 300), "note": rng.normal(0, 1, 300)})
    profile = sk.fit(fit_df)
    synthetic = sk.emit(profile, n=300, seed=0)

    check_df = fit_df.drop(columns=["note"])
    report = sk.check(synthetic, profile, real=check_df)

    assert "amount" in report.ks_by_column
    assert "note" not in report.ks_by_column


def test_profile_save_load_accessible_from_top_level(tmp_path):
    df = make_df(n=300)
    profile = sk.fit(df)
    path = tmp_path / "p.json"
    profile.save(path)
    reloaded = sk.Profile.load(path)
    assert reloaded.columns == profile.columns
