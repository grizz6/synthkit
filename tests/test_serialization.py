import numpy as np
import pytest

from synthkit.serialization import canonicalize, dump, dumps, load


def test_canonicalize_sorts_keys():
    data = {"b": 1, "a": 2}
    assert list(canonicalize(data).keys()) == ["a", "b"]


def test_canonicalize_converts_numpy_scalars():
    data = {"x": np.float64(1.5), "y": np.int64(3), "z": np.bool_(True)}
    result = canonicalize(data)
    assert result == {"x": 1.5, "y": 3, "z": True}
    assert isinstance(result["x"], float)
    assert isinstance(result["y"], int)
    assert isinstance(result["z"], bool)


def test_canonicalize_rounds_floats_to_significant_digits():
    data = {"x": 1.0 / 3.0}
    result = canonicalize(data)
    assert result["x"] == float(f"{1.0 / 3.0:.15g}")
    # Float noise from arithmetic is rounded away, so the same value always writes the same.
    assert canonicalize(0.1 + 0.2) == 0.3


@pytest.mark.parametrize("scale", [1e-15, 1e-9, 1.0, 1e9, 1e18])
def test_canonicalize_keeps_relative_precision_at_any_scale(scale):
    # Regression test: floats were rounded to 12 decimal places, so a value like 3e-13 became
    # 0.0 and a column measured on that scale collapsed to a handful of distinct values.
    value = scale * 1.234567890123
    assert canonicalize(value) == pytest.approx(value, rel=1e-14, abs=0)


def test_canonicalize_passes_non_finite_floats_through():
    assert canonicalize(float("inf")) == float("inf")
    assert canonicalize(float("-inf")) == float("-inf")
    assert np.isnan(canonicalize(float("nan")))


def test_canonicalize_handles_nested_arrays():
    data = {"matrix": np.array([[1.0, 2.0], [3.0, 4.0]])}
    result = canonicalize(data)
    assert result == {"matrix": [[1.0, 2.0], [3.0, 4.0]]}


def test_dumps_is_deterministic_across_calls():
    data = {"b": [1, 2, 3], "a": {"nested": 1}}
    assert dumps(data) == dumps(data)


def test_dump_and_load_round_trip(tmp_path):
    data = {"a": 1, "b": [1.5, 2.5], "c": {"d": True}}
    path = tmp_path / "profile.json"
    dump(data, path)
    restored = load(path)
    assert restored == canonicalize(data)
