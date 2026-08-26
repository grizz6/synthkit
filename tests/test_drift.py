import numpy as np
import pandas as pd

from synthkit.drift import compute_drift
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


def test_ignores_columns_not_present_in_current_data():
    df = make_df(seed=0)
    profile = Profile.fit(df)
    report = compute_drift(profile, df.drop(columns=["plan_tier"]))
    assert "plan_tier" not in report.column_drift
