import re

import numpy as np
import pandas as pd

from synthkit.inspect import compare_profiles, summarize_profile
from synthkit.profile import Profile
from synthkit.types import ColumnType


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


def make_numeric_df(n=400, loc=50.0, seed=0):
    return pd.DataFrame({"amount": np.random.default_rng(seed).normal(loc, 10, n)})


def test_compare_reports_no_changes_for_an_identical_profile():
    profile = Profile.fit(make_numeric_df())
    comparison = compare_profiles(profile, profile)
    assert not comparison.any_changes
    assert comparison.unchanged == ["amount"]


def test_compare_detects_added_and_removed_columns():
    old = Profile.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10, "gone": [4.0, 5.0, 6.0] * 10}))
    new = Profile.fit(pd.DataFrame({"a": [1.0, 2.0, 3.0] * 10, "fresh": [7.0, 8.0, 9.0] * 10}))

    comparison = compare_profiles(old, new)
    assert comparison.added == ["fresh"]
    assert comparison.removed == ["gone"]
    assert comparison.any_changes


def test_compare_detects_a_shifted_numeric_range():
    old = Profile.fit(make_numeric_df(loc=50.0))
    new = Profile.fit(make_numeric_df(loc=500.0))

    comparison = compare_profiles(old, new)
    assert [c.name for c in comparison.changed] == ["amount"]
    assert "->" in comparison.changed[0].changes[0]


def test_compare_detects_a_changed_null_rate():
    values = np.random.default_rng(1).normal(50, 10, 500)
    with_nulls = values.copy()
    with_nulls[:150] = np.nan

    comparison = compare_profiles(
        Profile.fit(pd.DataFrame({"amount": values})),
        Profile.fit(pd.DataFrame({"amount": with_nulls})),
    )
    assert any("null_rate" in change for change in comparison.changed[0].changes)


def test_compare_detects_a_changed_column_type():
    old = Profile.fit(pd.DataFrame({"v": [1, 2, 3] * 20}))
    new = Profile.fit(pd.DataFrame({"v": [1, 2, 3] * 20}), column_types={"v": ColumnType.DISCRETE})

    comparison = compare_profiles(old, new)
    assert any("type" in change for change in comparison.changed[0].changes)


def test_compare_reports_row_counts_and_constraint_counts():
    old = Profile.fit(make_numeric_df(n=100))
    new = Profile.fit(make_numeric_df(n=250))

    comparison = compare_profiles(old, new)
    assert comparison.rows_fit == (100, 250)
    assert comparison.constraint_counts == (0, 0)


def test_compare_detects_a_column_leaving_the_copula():
    # A column that becomes constant is dropped from the copula (zero variance would poison
    # the correlation matrix), which is a real modeling change worth surfacing in a review.
    rng = np.random.default_rng(0)
    varying = pd.DataFrame({"a": rng.normal(50, 10, 200), "b": rng.normal(50, 10, 200)})
    constant = pd.DataFrame({"a": rng.normal(50, 10, 200), "b": [1.5] * 200})

    comparison = compare_profiles(Profile.fit(varying), Profile.fit(constant))
    b_change = next(c for c in comparison.changed if c.name == "b")
    assert any("copula" in change for change in b_change.changes)


def test_compare_handles_two_profiles_sharing_no_columns():
    rng = np.random.default_rng(0)
    old = Profile.fit(pd.DataFrame({"x": rng.normal(0, 1, 100)}))
    new = Profile.fit(pd.DataFrame({"y": rng.normal(0, 1, 100)}))

    comparison = compare_profiles(old, new)
    assert comparison.added == ["y"]
    assert comparison.removed == ["x"]
    assert comparison.changed == []
    assert comparison.unchanged == []
    assert comparison.any_changes


def test_summarize_handles_an_all_null_column_alongside_a_normal_one():
    rng = np.random.default_rng(0)
    profile = Profile.fit(pd.DataFrame({"z": [None] * 50, "k": rng.normal(0, 1, 50)}))
    by_name = {s.name: s for s in summarize_profile(profile)}

    assert by_name["z"].type == "all_null"
    assert by_name["z"].null_rate == 1.0
    assert not by_name["z"].in_copula
    assert by_name["k"].type == "continuous"


def test_compare_describes_a_column_that_stopped_being_all_null():
    # A column that was entirely null when first fitted and has real values now is one of the
    # larger changes a re-fit can produce: it gains a type, a shape, and copula membership.
    rng = np.random.default_rng(0)
    old = Profile.fit(pd.DataFrame({"z": [None] * 50, "k": rng.normal(0, 1, 50)}))
    new = Profile.fit(pd.DataFrame({"z": rng.normal(0, 1, 50), "k": rng.normal(0, 1, 50)}))

    change = next(c for c in compare_profiles(old, new).changed if c.name == "z")
    joined = " ".join(change.changes)
    assert "all_null -> continuous" in joined
    assert "null_rate" in joined
    assert "copula" in joined
