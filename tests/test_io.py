import pandas as pd
import pytest

from synthkit.io import read_table, write_table


def test_csv_round_trip_preserves_datetime_columns(tmp_path):
    # Regression test: CSV has no dtype metadata, unlike parquet. A datetime column written
    # via to_csv() comes back as plain ISO-8601 strings on a bare pd.read_csv, and type
    # inference then misclassifies it as a garbled high-cardinality categorical instead of a
    # proper datetime marginal, silently destroying its chronological order and any
    # correlation with other columns through the copula.
    dates = pd.to_datetime(["2022-01-01", "2023-06-15", "2024-03-03"])
    df = pd.DataFrame({"signup_at": dates})
    path = tmp_path / "data.csv"
    write_table(df, path)

    restored = read_table(path)
    assert pd.api.types.is_datetime64_any_dtype(restored["signup_at"])
    assert list(restored["signup_at"]) == list(dates)


def test_csv_does_not_misparse_a_zip_code_like_column_as_a_date(tmp_path):
    # pd.to_datetime's lenient freeform parser happily misreads "02139" as the year 2139.
    # Detection is restricted to a strict ISO-8601 pattern precisely to avoid that: a 5-digit
    # zip code never matches it, so this column is left alone (and pandas' own CSV type
    # inference reads a leading-zero numeric string as int64 before detection even runs).
    df = pd.DataFrame({"zip": ["02139", "10001", "94103"]})
    path = tmp_path / "data.csv"
    write_table(df, path)

    restored = read_table(path)
    assert not pd.api.types.is_datetime64_any_dtype(restored["zip"])


def test_csv_leaves_non_date_string_columns_as_strings(tmp_path):
    df = pd.DataFrame({"customer_id": ["CUST001", "CUST002", "CUST003"]})
    path = tmp_path / "data.csv"
    write_table(df, path)

    restored = read_table(path)
    assert not pd.api.types.is_datetime64_any_dtype(restored["customer_id"])
    assert list(restored["customer_id"]) == ["CUST001", "CUST002", "CUST003"]


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
