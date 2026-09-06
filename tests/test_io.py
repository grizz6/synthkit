import numpy as np
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


def test_csv_round_trip_preserves_timezone_aware_datetimes(tmp_path):
    # Regression test: the first version of the ISO pattern had no timezone group, so a
    # created_at column carrying UTC offsets -- how essentially any real database serializes
    # one -- fell straight through detection and stayed text, which is the exact
    # misclassification the pattern exists to prevent.
    stamps = pd.to_datetime(["2024-01-01 00:00:00+00:00", "2024-06-15 12:30:00+00:00"])
    path = tmp_path / "data.csv"
    write_table(pd.DataFrame({"created_at": stamps}), path)

    restored = read_table(path)
    assert pd.api.types.is_datetime64_any_dtype(restored["created_at"])


def test_csv_detects_a_z_suffixed_utc_timestamp(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text("created_at\n2024-01-01T00:00:00Z\n2024-06-15T12:30:00Z\n")

    restored = read_table(path)
    assert pd.api.types.is_datetime64_any_dtype(restored["created_at"])


def test_csv_leaves_mixed_offset_timestamps_as_text_rather_than_failing(tmp_path):
    # Offsets that differ row to row have no single timezone to resolve to, and pandas rejects
    # them outright instead of coercing. Leaving that one column as text beats failing the
    # whole read.
    path = tmp_path / "data.csv"
    path.write_text("created_at\n2024-01-01T00:00:00+00:00\n2024-06-15T12:30:00-05:00\n")

    restored = read_table(path)
    assert not pd.api.types.is_datetime64_any_dtype(restored["created_at"])
    assert len(restored) == 2


def test_timezone_survives_the_whole_csv_fit_emit_csv_loop(tmp_path):
    # Spans three modules that each handle the zone separately: io detects it in text, the
    # datetime marginal records it, and emit rebuilds the column from it. Each has its own
    # unit tests; this is the loop a user actually runs, where a drop at any step would mean
    # fixtures whose dtype no longer matches the real data they stand in for.
    from synthkit.profile import Profile

    rng = np.random.default_rng(0)
    dates = pd.to_datetime("2022-01-01T00:00:00+00:00") + pd.to_timedelta(
        rng.integers(0, 900, 300), unit="D"
    )
    real = pd.DataFrame({"created_at": dates, "amount": rng.normal(50, 10, 300)})

    real_path = tmp_path / "real.csv"
    write_table(real, real_path)
    loaded = read_table(real_path)
    assert str(loaded["created_at"].dt.tz) == "UTC"

    synthetic = Profile.fit(loaded).emit(n=20, seed=0)
    assert str(synthetic["created_at"].dt.tz) == "UTC"

    synthetic_path = tmp_path / "synthetic.csv"
    write_table(synthetic, synthetic_path)
    assert str(read_table(synthetic_path)["created_at"].dt.tz) == "UTC"


def test_fixed_utc_offset_in_csv_is_preserved_not_normalized_to_utc(tmp_path):
    # A database export often carries a fixed offset rather than a named zone; it must come
    # back as that offset, not silently rewritten to UTC.
    from synthkit.profile import Profile

    path = tmp_path / "x.csv"
    rows = "\n".join(f"2024-01-{day:02d}T00:00:00-05:00" for day in range(1, 29))
    path.write_text("t\n" + rows + "\n")

    loaded = read_table(path)
    assert str(loaded["t"].dt.tz) == "UTC-05:00"

    emitted = Profile.fit(loaded).emit(n=5, seed=0)
    assert str(emitted["t"].dt.tz) == "UTC-05:00"
