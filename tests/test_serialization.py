import numpy as np

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


def test_canonicalize_rounds_floats():
    data = {"x": 1.0 / 3.0}
    result = canonicalize(data)
    assert result["x"] == round(1.0 / 3.0, 12)


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
