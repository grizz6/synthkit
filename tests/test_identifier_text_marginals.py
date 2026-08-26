import numpy as np
import pandas as pd

from synthkit.marginals import IdentifierMarginal, TextMarginal


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


def test_identifier_falls_back_to_random_token_for_mixed_formats():
    values = pd.Series(["abc123", "xyz", "999", "hello-world"])
    marginal = IdentifierMarginal.fit(values)
    assert marginal.style == "random_token"


def test_identifier_random_token_sample_is_unique():
    values = pd.Series(["abc123", "xyz-9", "qqq", "zzz111"])
    marginal = IdentifierMarginal.fit(values)
    sampled = marginal.sample(200, np.random.default_rng(0))
    assert len(set(sampled)) == 200


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
