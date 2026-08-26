"""Reading and writing tabular data by file extension.

Kept deliberately tiny: the CLI needs to accept both `.parquet` and `.csv` for every data
argument, and this is the one place that decision lives.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_SUFFIXES = {".parquet", ".csv"}


def _unsupported(suffix: str) -> ValueError:
    supported = sorted(SUPPORTED_SUFFIXES)
    return ValueError(f"unsupported file extension: {suffix!r} (expected one of {supported})")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
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
