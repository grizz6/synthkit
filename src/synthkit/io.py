"""Reading and writing tabular data by file extension.

Kept deliberately tiny: the CLI needs to accept both `.parquet` and `.csv` for every data
argument, and this is the one place that decision lives.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".parquet", ".csv"}

# Unlike parquet, CSV has no dtype metadata: a datetime column written by pandas' own
# to_csv() (or any ISO-8601 export) comes back as plain strings, and type inference would
# otherwise misclassify it as a garbled high-cardinality categorical. Restricting detection to
# a strict ISO-8601 pattern (rather than pandas' lenient freeform date parser) avoids false
# positives like a "02139" zip code, which pd.to_datetime happily misreads as the year 2139.
_ISO_DATETIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}:\d{2}(\.\d+)?)?$")


def _parse_iso_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in df.columns:
        series = df[column]
        if not pd.api.types.is_string_dtype(series):
            continue
        non_null = series.dropna().astype(str)
        if non_null.empty or not non_null.str.match(_ISO_DATETIME_PATTERN).all():
            continue
        df[column] = pd.to_datetime(series, errors="coerce")
    return df


def _unsupported(suffix: str) -> ValueError:
    supported = sorted(SUPPORTED_SUFFIXES)
    return ValueError(f"unsupported file extension: {suffix!r} (expected one of {supported})")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return _parse_iso_datetime_columns(pd.read_csv(path))
    raise _unsupported(path.suffix)


def write_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise _unsupported(path.suffix)
