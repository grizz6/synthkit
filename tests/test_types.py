import pandas as pd

from synthkit.types import ColumnType, infer_all_types, infer_column_type


def test_integer_low_cardinality_is_categorical():
    s = pd.Series([1, 2, 3, 1, 2, 3, 1] * 5)
    assert infer_column_type(s) == ColumnType.CATEGORICAL


def test_integer_high_cardinality_is_discrete():
    s = pd.Series(range(1000))
    assert infer_column_type(s) == ColumnType.DISCRETE


def test_float_with_fractional_values_is_continuous():
    s = pd.Series([1.1, 2.2, 3.3, 4.4, 5.5] * 20)
    assert infer_column_type(s) == ColumnType.CONTINUOUS


def test_float_all_whole_numbers_is_discrete():
    s = pd.Series([float(i % 1000) for i in range(2000)])
    assert infer_column_type(s) == ColumnType.DISCRETE


def test_boolean_dtype():
    s = pd.Series([True, False, True, True])
    assert infer_column_type(s) == ColumnType.BOOLEAN


def test_datetime_dtype():
    s = pd.to_datetime(pd.Series(["2024-01-01", "2024-01-02", "2024-01-03"]))
    assert infer_column_type(s) == ColumnType.DATETIME


def test_all_unique_strings_is_identifier():
    s = pd.Series([f"user_{i}" for i in range(50)])
    assert infer_column_type(s) == ColumnType.IDENTIFIER


def test_all_unique_strings_below_ten_rows_is_still_identifier():
    # Regression test: a row-count floor used to let a small all-unique string column fall
    # through to CATEGORICAL, which stores every distinct value verbatim as a "category",
    # leaking every real value into the profile.
    s = pd.Series(["a1b2c3", "x9y8z7", "q4w5e6", "m1n2b3", "zzz111"])
    assert infer_column_type(s) == ColumnType.IDENTIFIER


def test_low_cardinality_strings_is_categorical():
    s = pd.Series(["red", "green", "blue", "red", "green"] * 20)
    assert infer_column_type(s) == ColumnType.CATEGORICAL


def test_high_cardinality_non_unique_strings_is_text():
    s = pd.Series([f"a rambling free-text comment number {i % 40}, with words" for i in range(50)])
    assert infer_column_type(s) == ColumnType.TEXT


def test_empty_series_defaults_to_categorical():
    s = pd.Series([None, None], dtype=object)
    assert infer_column_type(s) == ColumnType.CATEGORICAL


def test_infer_all_types_respects_overrides():
    df = pd.DataFrame({"a": [1, 2, 3, 1], "b": ["x", "y", "z", "x"]})
    types = infer_all_types(df, overrides={"a": ColumnType.TEXT})
    assert types["a"] == ColumnType.TEXT
    assert types["b"] == ColumnType.CATEGORICAL
