import re

import numpy as np
import pandas as pd

from synthkit.inspect import summarize_profile
from synthkit.profile import Profile


def test_summarizes_every_column_in_original_order():
    df = pd.DataFrame({"b": [1.0, 2.0, 3.0] * 10, "a": ["x", "y"] * 15})
    profile = Profile.fit(df)
    summaries = summarize_profile(profile)
    assert [s.name for s in summaries] == ["b", "a"]


def test_numeric_column_reports_its_fitted_range():
    df = pd.DataFrame({"amount": np.random.default_rng(0).uniform(10.0, 50.0, 200)})
    profile = Profile.fit(df)
    summary = summarize_profile(profile)[0]
    assert summary.type == "continuous"
    low, high = (float(x) for x in re.findall(r"-?\d+\.?\d*", summary.detail))
    assert 9 < low < 15
    assert 45 < high < 51


def test_categorical_column_reports_category_count_and_other_mass():
    # 60 distinct values (past CategoricalMarginal's own default max_categories=50, so tail-
    # bucketing kicks in) each repeated 10x (past types.py's cardinality-ratio threshold, so
    # type inference still lands on CATEGORICAL rather than TEXT).
    values = pd.Series([f"cat_{i}" for i in range(60) for _ in range(10)])
    profile = Profile.fit(pd.DataFrame({"tier": values}))
    summary = summarize_profile(profile)[0]
    assert summary.type == "categorical"
    assert "categories" in summary.detail
    assert "other_mass" in summary.detail


def test_categorical_column_omits_other_mass_when_there_is_no_tail():
    values = pd.Series(["a", "b", "c"] * 20)
    profile = Profile.fit(pd.DataFrame({"tier": values}))
    summary = summarize_profile(profile)[0]
    assert "other_mass" not in summary.detail


def test_boolean_column_reports_probability_true():
    df = pd.DataFrame({"is_active": [True, True, False] * 20})
    profile = Profile.fit(df)
    summary = summarize_profile(profile)[0]
    assert summary.type == "boolean"
    assert "P(true)" in summary.detail


def test_datetime_column_reports_range_and_granularity():
    dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(range(30), unit="D")
    profile = Profile.fit(pd.DataFrame({"signup_at": dates}))
    summary = summarize_profile(profile)[0]
    assert summary.type == "datetime"
    assert "granularity=86400s" in summary.detail


def test_identifier_column_reports_its_style():
    values = pd.Series([f"CUST{i:05d}" for i in range(1, 60)])
    profile = Profile.fit(pd.DataFrame({"customer_id": values}))
    summary = summarize_profile(profile)[0]
    assert summary.type == "identifier"
    assert "sequential" in summary.detail


def test_text_column_reports_vocabulary_size():
    unique = [f"note number {i} today" for i in range(30)]
    values = unique + unique[:10]
    profile = Profile.fit(pd.DataFrame({"notes": values}))
    summary = summarize_profile(profile)[0]
    assert summary.type == "text"
    assert "vocabulary" in summary.detail


def test_all_null_column_reports_always_null_and_full_null_rate():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0] * 10, "gone": [None] * 30})
    profile = Profile.fit(df)
    summary = next(s for s in summarize_profile(profile) if s.name == "gone")
    assert summary.detail == "always null"
    assert summary.null_rate == 1.0


def test_null_rate_matches_the_column_s_actual_null_fraction():
    rng = np.random.default_rng(0)
    amount = rng.normal(50, 10, 1000)
    amount[:300] = np.nan
    profile = Profile.fit(pd.DataFrame({"amount": amount}))
    summary = summarize_profile(profile)[0]
    assert summary.null_rate == 0.3


def test_column_with_no_nulls_reports_null_rate_of_none():
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0] * 10})
    profile = Profile.fit(df)
    summary = summarize_profile(profile)[0]
    assert summary.null_rate is None


def test_in_copula_reflects_actual_copula_membership():
    df = pd.DataFrame(
        {
            "amount": np.random.default_rng(0).normal(50, 10, 200),
            "customer_id": [f"CUST{i:05d}" for i in range(1, 201)],
            "flat": [1.0] * 200,
        }
    )
    profile = Profile.fit(df)
    by_name = {s.name: s for s in summarize_profile(profile)}
    assert by_name["amount"].in_copula
    assert not by_name["customer_id"].in_copula  # identifiers are never copula-eligible
    assert not by_name["flat"].in_copula  # constant columns are dropped from the copula
