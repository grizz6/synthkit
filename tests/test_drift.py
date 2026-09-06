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
    # counted every individual rare category, once as its own missing key and once as the
    # unmatched slice of __other__'s mass, even when the tail hadn't drifted at all.
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


def test_column_absent_from_current_data_is_unscored_but_reported_missing():
    # It cannot be scored -- there are no values to compare against -- but it is not
    # "ignored": a column the profile models and production has dropped is drift, and the
    # loudest kind, since fixtures keep generating something real data no longer has.
    df = make_df(seed=0)
    profile = Profile.fit(df)
    report = compute_drift(profile, df.drop(columns=["plan_tier"]))
    assert "plan_tier" not in report.column_drift
    assert report.missing_columns == ["plan_tier"]
    assert not report.passed


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
    # rows, high cardinality but with repeats, landing it in TEXT rather than IDENTIFIER.
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


def test_datetime_drift_is_zero_when_current_column_is_all_null():
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, 500), unit="D")
    df = pd.DataFrame({"signup_date": dates})
    profile = Profile.fit(df)
    fresh = pd.DataFrame({"signup_date": pd.to_datetime([None] * 100)})
    report = compute_drift(profile, fresh)
    assert report.column_drift["signup_date"] == 0.0


def test_boolean_drift_is_zero_when_current_column_is_all_null():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"is_active": rng.choice([True, False], size=500)})
    profile = Profile.fit(df)
    fresh = pd.DataFrame({"is_active": pd.array([None] * 100, dtype="boolean")})
    report = compute_drift(profile, fresh)
    assert report.column_drift["is_active"] == 0.0


def test_drift_handles_timezone_aware_columns():
    # DatetimeMarginal.sample() returns a tz-aware index for a tz-aware source, and drift feeds
    # that back through to_epoch_seconds against fresh data. Nothing covered the aware path, so
    # a naive/aware mismatch in that comparison would have gone unnoticed.
    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2022-01-01T00:00:00+00:00") + pd.to_timedelta(
        rng.integers(0, 900, 400), unit="D"
    )
    df = pd.DataFrame({"created_at": dates})
    profile = Profile.fit(df)

    report = compute_drift(profile, df)
    assert report.passed
    assert report.column_drift["created_at"] < DEFAULT_DRIFT_THRESHOLD


def test_drift_detects_a_shift_in_a_timezone_aware_column():
    rng = np.random.default_rng(0)
    base = pd.to_datetime("2022-01-01T00:00:00+00:00")
    df = pd.DataFrame({"t": base + pd.to_timedelta(rng.integers(0, 365, 400), unit="D")})
    profile = Profile.fit(df)

    shifted = pd.DataFrame(
        {
            "t": pd.to_datetime("2026-01-01T00:00:00+00:00")
            + pd.to_timedelta(np.random.default_rng(1).integers(0, 365, 400), unit="D")
        }
    )
    report = compute_drift(profile, shifted)
    assert not report.passed
    assert "t" in report.drifted_columns


def test_schema_change_fails_even_when_surviving_distributions_are_identical():
    # Regression test: drift skipped any column missing from the fresh data and never looked
    # at columns the fresh data had gained, so a profile could pass cleanly against data whose
    # schema had moved out from under it -- and `synthkit diff` would exit 0 in CI while the
    # committed fixtures no longer matched production at all.
    rng = np.random.default_rng(0)
    n = 500
    profile = Profile.fit(
        pd.DataFrame({"age": rng.normal(40, 10, n), "income": rng.normal(50000, 10000, n)})
    )

    # Same distribution for the surviving column: only the schema moved.
    changed = pd.DataFrame({"age": rng.normal(40, 10, n), "tenure": rng.normal(5, 2, n)})
    report = compute_drift(profile, changed)

    assert report.column_drift["age"] < DEFAULT_DRIFT_THRESHOLD  # what remains has not drifted
    assert report.missing_columns == ["income"]
    assert report.new_columns == ["tenure"]
    assert report.schema_changed
    assert not report.passed


def test_new_column_alone_is_enough_to_fail():
    # Fixtures built from this profile have no such column, so any test touching it breaks.
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"age": rng.normal(40, 10, n)})
    profile = Profile.fit(df)

    report = compute_drift(profile, df.assign(loyalty=rng.normal(3, 1, n)))
    assert report.new_columns == ["loyalty"]
    assert report.missing_columns == []
    assert not report.passed


def test_unchanged_schema_reports_no_schema_drift():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"age": rng.normal(40, 10, n), "score": rng.normal(10, 2, n)})
    profile = Profile.fit(df)

    report = compute_drift(profile, df)
    assert report.missing_columns == []
    assert report.new_columns == []
    assert not report.schema_changed
    assert report.passed


def test_identifier_and_text_columns_are_unscored_without_counting_as_schema_drift():
    # These types have no drift function, but they are present on both sides, so skipping them
    # for scoring must not be confused with the column being absent.
    notes = [f"note number {i} with words" for i in range(35)]
    df = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(60)],
            "notes": notes + notes[:25],
        }
    )
    profile = Profile.fit(df)

    report = compute_drift(profile, df)
    assert report.column_drift == {}
    assert report.missing_columns == []
    assert report.new_columns == []
    assert report.passed
