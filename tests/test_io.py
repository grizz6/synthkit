import pandas as pd
import pytest

from synthkit.io import read_table, write_table


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.parquet"
    write_table(df, path)
    restored = read_table(path)
    pd.testing.assert_frame_equal(df, restored)


def test_csv_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "data.csv"
    write_table(df, path)
    restored = read_table(path)
    pd.testing.assert_frame_equal(df, restored)


def test_write_table_creates_parent_directories(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "nested" / "dir" / "data.csv"
    write_table(df, path)
    assert path.exists()


def test_unsupported_extension_raises_on_read(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("hello")
    with pytest.raises(ValueError):
        read_table(path)


def test_unsupported_extension_raises_on_write(tmp_path):
    df = pd.DataFrame({"a": [1]})
    with pytest.raises(ValueError):
        write_table(df, tmp_path / "data.txt")
