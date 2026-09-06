import numpy as np
import pandas as pd
import pytest

from synthkit.privacy import (
    _finite_range,
    check,
    compute_value_ranges,
    count_exact_matches,
    count_identifying_matches,
    count_rare_combination_leaks,
    distance_to_closest_record,
    explain_identifying_matches,
    gower_distance_matrix,
)


def test_gower_distance_zero_for_identical_rows():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    column_types = {"a": "continuous", "b": "categorical"}
    distances = gower_distance_matrix(df, df, column_types)
    assert np.allclose(np.diag(distances), 0.0)


def test_finite_range_ignores_infinity():
    # A genuine +inf value (a division result, a sentinel) used to make value_range infinite,
    # which silently turned every per-row distance into 0 or nan/huge-finite and flooded the
    # terminal with numpy RuntimeWarnings.
    assert _finite_range(np.array([1.0, 2.0, np.inf, 4.0])) == 3.0


def test_finite_range_all_infinite_falls_back_to_one():
    assert _finite_range(np.array([np.inf, -np.inf, np.inf])) == 1.0


def test_finite_range_all_nan_falls_back_to_one():
    assert _finite_range(np.array([np.nan, np.nan])) == 1.0


def test_finite_range_constant_column_falls_back_to_one():
    assert _finite_range(np.array([5.0, 5.0, 5.0])) == 1.0


def test_gower_distance_handles_infinity_without_warning_or_blowup():
    # Regression test: an unguarded `nanmax - nanmin` with +inf in the column made the
    # normalized distance either 0 (finite-vs-finite, divided by an infinite range) or, via
    # nan_to_num's default posinf fill, numpy's largest finite float for pairs touching the
    # inf value itself, which would dominate every other column's contribution once
    # averaged. Every entry should stay within the intended [0, 1] Gower-distance range.
    df = pd.DataFrame({"a": [1.0, 2.0, np.inf, 4.0, 5.0]})
    distances = gower_distance_matrix(df, df, {"a": "continuous"})
    assert np.isfinite(distances).all()
    assert (distances >= 0).all() and (distances <= 1).all()


def test_compute_value_ranges_covers_datetime_columns():
    query = pd.DataFrame({"a": pd.to_datetime(["2024-01-01", "2024-01-05"])})
    reference = pd.DataFrame({"a": pd.to_datetime(["2024-01-01", "2024-01-10"])})
    ranges = compute_value_ranges(query, reference, {"a": "datetime"})
    assert ranges["a"] == pytest.approx(9 * 86400)  # 9 days, in seconds


def test_compute_value_ranges_skips_a_column_missing_from_either_side():
    query = pd.DataFrame({"a": [1.0, 2.0]})
    reference = pd.DataFrame({"a": [1.0], "b": [5.0]})
    ranges = compute_value_ranges(query, reference, {"a": "continuous", "b": "continuous"})
    assert "a" in ranges
    assert "b" not in ranges


def test_gower_distance_skips_a_column_missing_from_reference_instead_of_crashing():
    # Regression test: columns were selected from query.columns intersected with column_types,
    # but never checked against reference.columns. check()'s real dataframe doesn't have to
    # carry every column the profile was fit on (see api.py's _fidelity_by_column, which
    # already guards this the same way); a column missing from reference used to raise a bare
    # pandas KeyError instead of just being excluded from the distance calculation.
    query = pd.DataFrame({"a": [1.0], "b": [2.0]})
    reference = pd.DataFrame({"a": [1.0]})
    distances = gower_distance_matrix(query, reference, {"a": "continuous", "b": "continuous"})
    assert distances.shape == (1, 1)
    assert distances[0, 0] == 0.0


def test_gower_distance_categorical_mismatch_contributes_one():
    query = pd.DataFrame({"b": ["x"]})
    reference = pd.DataFrame({"b": ["y"]})
    distances = gower_distance_matrix(query, reference, {"b": "categorical"})
    assert distances[0, 0] == 1.0


def test_distance_to_closest_record_picks_nearest():
    query = pd.DataFrame({"a": [5.0]})
    reference = pd.DataFrame({"a": [0.0, 4.9, 100.0]})
    dcr = distance_to_closest_record(query, reference, {"a": "continuous"})
    # nearest reference value is 4.9, at distance 0.1 out of a 100-wide combined range
    assert dcr[0] < 0.01


def test_distance_to_closest_record_batching_matches_unbatched_result():
    # Regression test: batching both the query and reference sides to bound peak memory
    # (measured to reach 2+ GB at 10,000 query rows against an 8,000-row reference set before
    # this existed) must be an exact computation, not an approximation, the per-row
    # nearest-reference distance found in small batches has to be identical to computing the
    # whole matrix at once. Both sides here (250, 300) exceed the batch_size (37), so this
    # exercises the nested query/reference batching, not just one axis.
    rng = np.random.default_rng(0)
    query = pd.DataFrame({"a": rng.normal(0, 10, 250), "b": rng.choice(["x", "y", "z"], 250)})
    reference = pd.DataFrame({"a": rng.normal(0, 10, 300), "b": rng.choice(["x", "y", "z"], 300)})
    column_types = {"a": "continuous", "b": "categorical"}

    unbatched = gower_distance_matrix(query, reference, column_types).min(axis=1)
    batched = distance_to_closest_record(query, reference, column_types, batch_size=37)

    assert np.allclose(unbatched, batched)


def test_distance_to_closest_record_handles_large_reference_with_small_query():
    # The reference side (typically the real dataset) can be much larger than the query side
    # (typically the synthetic output). Before two-sided batching, only a large query was
    # bounded; a large reference against a small query still materialized the whole matrix.
    rng = np.random.default_rng(0)
    query = pd.DataFrame({"a": rng.normal(0, 10, 20)})
    reference = pd.DataFrame({"a": rng.normal(0, 10, 5000)})
    column_types = {"a": "continuous"}

    unbatched = gower_distance_matrix(query, reference, column_types).min(axis=1)
    batched = distance_to_closest_record(query, reference, column_types, batch_size=64)

    assert np.allclose(unbatched, batched)


def test_distance_to_closest_record_batch_size_smaller_than_query_is_used():
    # Sanity check that the batching branch actually runs (not just the batch_size >= len(query)
    # fast path) by using a query larger than a tiny batch size.
    rng = np.random.default_rng(0)
    query = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    reference = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    result = distance_to_closest_record(query, reference, {"a": "continuous"}, batch_size=10)
    assert result.shape == (100,)
    assert np.isfinite(result).all()


def test_count_exact_matches_detects_duplicated_rows():
    real = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    synthetic = pd.DataFrame({"a": [2, 9], "b": ["y", "q"]})
    assert count_exact_matches(synthetic, real) == 1


def test_count_exact_matches_zero_when_no_overlap():
    real = pd.DataFrame({"a": [1, 2, 3]})
    synthetic = pd.DataFrame({"a": [4, 5, 6]})
    assert count_exact_matches(synthetic, real) == 0


def test_rare_combination_leak_detected():
    real = pd.DataFrame(
        {
            "a": ["common"] * 100 + ["rare"],
            "b": ["common"] * 100 + ["combo"],
        }
    )
    synthetic = pd.DataFrame({"a": ["rare"], "b": ["combo"]})
    leaks = count_rare_combination_leaks(synthetic, real, ["a", "b"], threshold=5)
    assert leaks == 1


def test_rare_combination_no_leak_for_common_combo():
    real = pd.DataFrame({"a": ["common"] * 100, "b": ["common"] * 100})
    synthetic = pd.DataFrame({"a": ["common"], "b": ["common"]})
    leaks = count_rare_combination_leaks(synthetic, real, ["a", "b"], threshold=5)
    assert leaks == 0


def test_check_passes_for_well_separated_synthetic_data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(0, 10, 500)})
    # Independently drawn from the same distribution: should look "as far" as a holdout does.
    synthetic = pd.DataFrame({"a": rng.normal(0, 10, 500)})
    report = check(synthetic, real, {"a": "continuous"}, min_dcr_ratio=0.5)
    assert report.passed
    assert report.exact_matches == 0


def test_check_fails_when_synthetic_data_is_literally_the_training_data():
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(0, 10, 200), "b": ["x"] * 200})
    synthetic = real.copy()
    report = check(synthetic, real, {"a": "continuous", "b": "categorical"}, min_dcr_ratio=1.0)
    assert not report.passed
    assert report.exact_matches > 0


def test_check_scores_rare_combinations_pairwise_not_jointly():
    # Regression test: scoring every categorical column jointly makes almost every combination
    # "rare" purely from dimensionality (4 columns of 3 categories each spread over 300 rows
    # gives up to 81 joint cells, most with a handful of rows, even though no single pair is
    # remotely rare). check() should score pairs of columns, not the full cross product.
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({f"cat{i}": rng.choice(["a", "b", "c"], size=n) for i in range(4)})
    column_types = {f"cat{i}": "categorical" for i in range(4)}

    joint_leaks = count_rare_combination_leaks(df, df, list(column_types), threshold=5)
    assert joint_leaks > 0  # sanity: the joint check really is this aggressive

    report = check(df, df, column_types, min_dcr_ratio=0.0)
    assert report.rare_combination_leaks == 0


def test_dcr_ratio_is_neutral_when_holdout_baseline_is_degenerately_zero():
    # Regression test: with too few columns / too little entropy relative to the row count,
    # real holdout rows naturally land exactly on some training row (duplicates happen), so
    # the holdout's own 5th-percentile DCR is 0, not because of anything synthetic did, just
    # because the dataset doesn't have enough columns to make rows distinct. Confirmed
    # directly on a 5-column slice of Adult Census (32,561 rows): both holdout_p and
    # synthetic_p were exactly 0, and dividing by a fallback epsilon reported dcr_ratio=0.0,
    # which reads as a hard privacy failure it isn't.
    rng = np.random.default_rng(0)
    n = 5000
    # Two binary columns and nothing else: with 5000 rows and only 4 possible combinations,
    # both real and synthetic data are guaranteed to be full of exact duplicates.
    df = pd.DataFrame(
        {
            "a": rng.choice([0, 1], size=n),
            "b": rng.choice([0, 1], size=n),
        }
    )
    column_types = {"a": "categorical", "b": "categorical"}

    # Using df as its own "synthetic" isolates the degenerate-baseline case: if the fix works,
    # a dataset this duplicate-heavy reports the neutral ratio rather than 0.0.
    report = check(df, df, column_types, min_dcr_ratio=0.5)
    assert report.dcr_ratio == 1.0


def test_check_rejects_single_row_real_dataset_with_a_clear_error():
    # Regression test: n_holdout is at least 1 by construction, so a 1-row real dataset left
    # the training set empty and distance_to_closest_record's .min(axis=1) on a zero-size
    # array raised a bare numpy error instead of a message that mentions the real dataset.
    synthetic = pd.DataFrame({"a": [1.0, 2.0]})
    real = pd.DataFrame({"a": [5.0]})
    with pytest.raises(ValueError, match="at least 2"):
        check(synthetic, real, {"a": "continuous"})


def test_check_handles_holdout_fraction_of_one():
    # Regression test: holdout_fraction=1.0 put every real row into the holdout regardless of
    # dataset size, leaving training empty and hitting the same zero-size-array crash as the
    # single-row case above.
    rng = np.random.default_rng(0)
    real = pd.DataFrame({"a": rng.normal(0, 1, 100)})
    synthetic = pd.DataFrame({"a": rng.normal(0, 1, 50)})
    report = check(synthetic, real, {"a": "continuous"}, holdout_fraction=1.0)
    assert np.isfinite(report.dcr_ratio)


def _identifier_dataset(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "user_id": [f"U{i:06d}" for i in range(n)],
            "diagnosis": rng.choice(["A", "B", "C"], size=n),
            "zip": rng.choice(["02139", "10001", "94103"], size=n),
            "age": rng.integers(18, 90, n),
        }
    )


_IDENTIFIER_TYPES = {
    "user_id": "identifier",
    "diagnosis": "categorical",
    "zip": "categorical",
    "age": "categorical",
}


def test_identifier_column_no_longer_masks_verbatim_leakage():
    # Regression test, and the most serious one here: identifiers are regenerated from a
    # detected format and can never match, so including them in a whole-row comparison could
    # only ever hide matches in the columns that carry real content. Confirmed directly --
    # a frame copying every sensitive column verbatim from real rows reported exact_matches=0
    # and passed the privacy check outright, purely because its id column differed.
    real = _identifier_dataset()
    leaked = real.copy()
    leaked["user_id"] = [f"X{i:06d}" for i in range(len(real))]

    report = check(leaked, real, _IDENTIFIER_TYPES, min_dcr_ratio=1.0)
    assert report.exact_matches == len(real)
    assert report.identifying_matches == len(real)
    assert not report.passed


def test_a_low_entropy_dataset_is_not_reported_as_leaking():
    # The other direction, and why a raw match count cannot drive the verdict: two binary
    # columns over a thousand rows allow four distinct records, so every synthetic row
    # necessarily reproduces a real one. That is arithmetic, not leakage -- no one can be
    # singled out from a combination that hundreds of people share.
    rng = np.random.default_rng(0)
    n = 1000
    real = pd.DataFrame(
        {
            "record_id": [f"R{i:06d}" for i in range(n)],
            "segment": rng.choice(["a", "b"], size=n),
            "flag": rng.choice(["y", "n"], size=n),
        }
    )
    synthetic = real.copy()
    synthetic["record_id"] = [f"Z{i:06d}" for i in range(n)]

    types = {"record_id": "identifier", "segment": "categorical", "flag": "categorical"}
    report = check(synthetic, real, types, min_dcr_ratio=0.0)

    assert report.exact_matches == n  # every row matches, as it must
    assert report.identifying_matches == 0  # but none of them identify anyone
    assert report.passed


def test_count_exact_matches_can_be_restricted_to_chosen_columns():
    real = pd.DataFrame({"id": ["a", "b"], "v": [1, 2]})
    synthetic = pd.DataFrame({"id": ["x", "y"], "v": [1, 2]})

    assert count_exact_matches(synthetic, real) == 0  # ids differ, so no whole row matches
    assert count_exact_matches(synthetic, real, ["v"]) == 2  # the content is identical


def test_count_identifying_matches_only_counts_rare_records():
    real = pd.DataFrame({"v": ["common"] * 100 + ["rare"]})
    synthetic = pd.DataFrame({"v": ["common", "rare"]})

    assert count_identifying_matches(synthetic, real, ["v"], threshold=5) == 1


def test_count_identifying_matches_is_zero_without_columns_to_compare():
    real = pd.DataFrame({"id": ["a", "b"]})
    synthetic = pd.DataFrame({"id": ["a", "b"]})
    assert count_identifying_matches(synthetic, real, [], threshold=5) == 0


def test_explain_attributes_risk_to_the_column_driving_it():
    # A leave-one-out count is only useful if it separates the column making records unique
    # from one that merely looks sensitive: dropping a coarse column should change nothing.
    rng = np.random.default_rng(1)
    n = 400
    real = pd.DataFrame(
        {
            "segment": rng.choice(["a", "b"], size=n),  # coarse: two values over 400 rows
            "zip": rng.choice([f"{z:05d}" for z in range(60)], size=n),
            "age": rng.integers(18, 90, n),  # highest cardinality of the three
        }
    )
    columns = list(real.columns)
    baseline = count_identifying_matches(real, real, columns)
    remaining = explain_identifying_matches(real, real, columns)

    assert baseline > 0
    assert remaining["segment"] == baseline  # dropping it exposes exactly as many people
    assert remaining["age"] < baseline  # dropping it genuinely reduces the risk
    assert remaining["age"] < remaining["segment"]


def test_explain_covers_every_column_it_was_given():
    real = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"]})
    assert set(explain_identifying_matches(real, real, ["a", "b"])) == {"a", "b"}


def test_explain_on_a_single_column_reports_the_count_with_nothing_left():
    # Dropping the only column leaves no columns to compare, which is zero matches by
    # definition rather than an error.
    real = pd.DataFrame({"a": ["x", "y", "z"]})
    assert explain_identifying_matches(real, real, ["a"]) == {"a": 0}
