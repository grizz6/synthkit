import numpy as np
import pandas as pd

from synthkit.privacy import (
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
