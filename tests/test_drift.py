import numpy as np
import pandas as pd

from synthkit.drift import DEFAULT_DRIFT_THRESHOLD, compute_drift
from synthkit.profile import Profile


def make_df(n=2000, seed=0, category_probs=(0.7, 0.2, 0.1)):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amount": rng.normal(100, 20, n),
            "plan_tier": rng.choice(["basic", "pro", "enterprise"], size=n, p=list(category_probs)),
        }
    )


def test_no_drift_when_distribution_is_unchanged():
    df = make_df(seed=0)
    profile = Profile.fit(df)
    fresh = make_df(seed=1)
    report = compute_drift(profile, fresh)
    assert report.passed
    assert report.drifted_columns == []


def test_detects_numeric_drift():
    df = make_df(seed=0)
    profile = Profile.fit(df)

    drifted = df.copy()
    drifted["amount"] = drifted["amount"] + 200  # a large shift

    report = compute_drift(profile, drifted)
    assert not report.passed
    assert "amount" in report.drifted_columns


def test_detects_categorical_drift():
    df = make_df(seed=0, category_probs=(0.7, 0.2, 0.1))
    profile = Profile.fit(df)

    drifted = make_df(seed=1, category_probs=(0.1, 0.1, 0.8))
    report = compute_drift(profile, drifted)
    assert not report.passed
    assert "plan_tier" in report.drifted_columns


def test_max_drift_property_reports_the_worst_column():
    df = make_df(seed=0)
    profile = Profile.fit(df)
    drifted = df.copy()
    drifted["amount"] = drifted["amount"] + 500

    report = compute_drift(profile, drifted)
    assert report.max_drift == max(report.column_drift.values())


def test_other_bucket_does_not_inflate_drift_for_unchanged_tail():
    # Regression test: when the profile's categorical marginal pooled rare categories into
    # __other__, comparing raw current-category frequencies against reference_probs double
    # counted every individual rare category (once as its own missing key, once again as the
    # unmatched slice of __other__'s mass) even when the tail distribution hadn't drifted at
    # all — this could spuriously trip the drift threshold on an unchanged column.
    rng = np.random.default_rng(0)
    n = 5000
    # 60 distinct rare categories (>50, so CategoricalMarginal tail-buckets some into __other__)
    # plus one dominant category, sampled identically for both "fits" and "fresh" data.
    categories = [f"cat_{i}" for i in range(60)]
    weights = np.array([0.4] + [0.6 / 59] * 59)

    def make(seed):
        return pd.DataFrame({"tag": rng.choice(categories, size=n, p=weights)})

    df = make(seed=0)
    profile = Profile.fit(df)
    assert "__other__" in profile.marginals["tag"]["categories"]  # sanity: tail-bucketing fired

    fresh = make(seed=1)
    report = compute_drift(profile, fresh)
    assert report.column_drift["tag"] < DEFAULT_DRIFT_THRESHOLD


def test_ignores_columns_not_present_in_current_data():
    df = make_df(seed=0)
    profile = Profile.fit(df)
    report = compute_drift(profile, df.drop(columns=["plan_tier"]))
    assert "plan_tier" not in report.column_drift


def test_no_drift_for_unchanged_datetime_column():
    rng = np.random.default_rng(0)
    n = 1000
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    df = pd.DataFrame({"signup_date": dates})
    profile = Profile.fit(df)

    fresh_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.default_rng(1).integers(0, 365, n), unit="D"
    )
    fresh = pd.DataFrame({"signup_date": fresh_dates})

    report = compute_drift(profile, fresh)
    assert report.passed
    assert report.column_drift["signup_date"] < DEFAULT_DRIFT_THRESHOLD


def test_detects_datetime_drift():
    rng = np.random.default_rng(0)
    n = 1000
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    df = pd.DataFrame({"signup_date": dates})
    profile = Profile.fit(df)

    shifted = pd.to_datetime("2026-01-01") + pd.to_timedelta(
        np.random.default_rng(1).integers(0, 365, n), unit="D"
    )
    fresh = pd.DataFrame({"signup_date": shifted})

    report = compute_drift(profile, fresh)
    assert not report.passed
    assert "signup_date" in report.drifted_columns


def test_no_drift_for_unchanged_boolean_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"is_active": rng.choice([True, False], size=2000, p=[0.7, 0.3])})
    profile = Profile.fit(df)

    fresh = pd.DataFrame(
        {"is_active": np.random.default_rng(1).choice([True, False], size=2000, p=[0.7, 0.3])}
    )
    report = compute_drift(profile, fresh)
    assert report.column_drift["is_active"] < DEFAULT_DRIFT_THRESHOLD


def test_detects_boolean_drift():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"is_active": rng.choice([True, False], size=2000, p=[0.7, 0.3])})
    profile = Profile.fit(df)

    fresh = pd.DataFrame(
        {"is_active": np.random.default_rng(1).choice([True, False], size=2000, p=[0.1, 0.9])}
    )
    report = compute_drift(profile, fresh)
    assert not report.passed
    assert "is_active" in report.drifted_columns


def test_identifier_and_text_columns_are_skipped_for_drift():
    # user_id is all-unique (identifier); notes has 35 distinct free-text values across 60
    # rows -- high cardinality but with repeats, landing it in TEXT rather than IDENTIFIER.
    notes_values = [f"note number {i} with some words" for i in range(35)]
    df = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(60)],
            "notes": notes_values + notes_values[:25],
        }
    )
    assert df["notes"].nunique() == 35  # sanity: high cardinality, but not all-unique

    profile = Profile.fit(df)
    assert profile.column_types["user_id"] == "identifier"
    assert profile.column_types["notes"] == "text"

    report = compute_drift(profile, df)
    assert report.column_drift == {}


def test_numeric_drift_is_zero_when_current_column_is_all_null():
    df = pd.DataFrame({"amount": np.random.default_rng(0).normal(100, 20, 500)})
    profile = Profile.fit(df)
    fresh = pd.DataFrame({"amount": [None] * 100})
    report = compute_drift(profile, fresh)
    assert report.column_drift["amount"] == 0.0
