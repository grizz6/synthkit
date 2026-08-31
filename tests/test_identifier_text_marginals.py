import numpy as np
import pandas as pd
import pytest

from synthkit.marginals import DEFAULT_MAX_WORD_POOL, IdentifierMarginal, TextMarginal


def test_identifier_detects_sequential_pattern():
    values = pd.Series([f"CUST{i:05d}" for i in range(1, 200)])
    marginal = IdentifierMarginal.fit(values)
    assert marginal.style == "sequential"
    assert marginal.prefix == "CUST"
    assert marginal.digit_width == 5


def test_identifier_sequential_sample_is_unique_and_formatted():
    values = pd.Series([f"CUST{i:05d}" for i in range(1, 200)])
    marginal = IdentifierMarginal.fit(values)
    sampled = marginal.sample(100, np.random.default_rng(0))
    assert len(set(sampled)) == 100
    assert all(s.startswith("CUST") for s in sampled)


def test_identifier_sequential_sample_stays_within_fitted_digit_width():
    # Regression test: the random start offset used to be drawn uniformly from [0, 1_000_000)
    # regardless of the fitted digit width, so fitted 3-digit IDs like CUST001-CUST050 could
    # come back out as CUST483920-CUST483969, a completely different visual shape.
    values = pd.Series([f"CUST{i:03d}" for i in range(1, 51)])
    marginal = IdentifierMarginal.fit(values)
    for seed in range(20):
        sampled = marginal.sample(30, np.random.default_rng(seed))
        assert all(len(s) == len("CUST001") for s in sampled)


def test_identifier_falls_back_to_random_token_for_mixed_formats():
    values = pd.Series(["abc123", "xyz", "999", "hello-world"])
    marginal = IdentifierMarginal.fit(values)
    assert marginal.style == "random_token"


def test_identifier_random_token_sample_is_unique():
    values = pd.Series(["abc123", "xyz-9", "qqq", "zzz111"])
    marginal = IdentifierMarginal.fit(values)
    sampled = marginal.sample(200, np.random.default_rng(0))
    assert len(set(sampled)) == 200


def test_identifier_random_token_widens_length_instead_of_hanging_when_n_exceeds_keyspace():
    # Regression test: rejection-sampling for random tokens used to loop forever once n
    # approached or exceeded the keyspace (36 ** token_length). A 2-char token has only 1296
    # possible values; requesting 2000 unique ones used to hang indefinitely instead of
    # widening the token length to make that many unique values possible.
    marginal = IdentifierMarginal(style="random_token", token_length=2)
    sampled = marginal.sample(2000, np.random.default_rng(0))
    assert len(set(sampled)) == 2000


def test_identifier_round_trip():
    values = pd.Series([f"ID{i:03d}" for i in range(1, 50)])
    marginal = IdentifierMarginal.fit(values)
    restored = IdentifierMarginal.from_dict(marginal.to_dict())
    assert restored == marginal


def test_text_marginal_never_reproduces_exact_original_strings():
    values = pd.Series(
        [
            "the quick brown fox jumps over the lazy dog",
            "a rolling stone gathers no moss at all",
            "pack my box with five dozen liquor jugs today",
        ]
        * 10
    )
    marginal = TextMarginal.fit(values)
    sampled = marginal.sample(50, np.random.default_rng(0))
    assert not set(sampled) & set(values)


def test_text_marginal_preserves_rough_word_count():
    values = pd.Series(["one two three four five"] * 100)
    marginal = TextMarginal.fit(values)
    sampled = marginal.sample(200, np.random.default_rng(0))
    lengths = [len(s.split()) for s in sampled]
    assert abs(np.mean(lengths) - 5) < 0.5


def test_text_marginal_round_trip():
    values = pd.Series(["hello world", "hello there", "world peace"])
    marginal = TextMarginal.fit(values)
    restored = TextMarginal.from_dict(marginal.to_dict())
    assert restored == marginal


def test_identifier_marginal_rejects_all_null_column():
    values = pd.Series([None, None], dtype=object)
    with pytest.raises(ValueError):
        IdentifierMarginal.fit(values)


def test_text_marginal_rejects_all_null_column():
    values = pd.Series([None, None], dtype=object)
    with pytest.raises(ValueError):
        TextMarginal.fit(values)


def test_text_marginal_handles_a_column_of_only_blank_strings():
    # Not all-null (that's rejected above), but every value splits to zero words, leaving no
    # vocabulary to sample from. A single "" placeholder keeps sample() from crashing on an
    # empty word_pool instead of reproducing the exact all-blank column.
    values = pd.Series(["", "   ", ""])
    marginal = TextMarginal.fit(values)
    assert marginal.word_pool == [""]

    sampled = marginal.sample(5, np.random.default_rng(0))
    assert list(sampled) == [""] * 5


def test_text_marginal_caps_word_pool_size():
    # A column with far more distinct words than DEFAULT_MAX_WORD_POOL should have its vocabulary
    # subsampled rather than growing the profile unboundedly.
    values = pd.Series([f"word{i} word{i + 1} word{i + 2}" for i in range(DEFAULT_MAX_WORD_POOL)])
    marginal = TextMarginal.fit(values)
    assert len(marginal.word_pool) == DEFAULT_MAX_WORD_POOL
