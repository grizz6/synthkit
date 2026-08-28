import numpy as np
import pandas as pd
import pytest

from synthkit.privacy import (
    _finite_range,
    check,
    count_exact_matches,
    count_rare_combination_leaks,
    distance_to_closest_record,
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
    reference = pd.DataFrame(
        {"a": rng.normal(0, 10, 300), "b": rng.choice(["x", "y", "z"], 300)}
    )
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
