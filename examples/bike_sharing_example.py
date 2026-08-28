"""Bike Sharing example: a real datetime column, plus a real derived-column relationship
(`cnt` is always `casual + registered`) used to demonstrate the `Derived` constraint.

Run with:

    python examples/bike_sharing_example.py
"""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import synthkit as sk

DATA_DIR = Path(__file__).parent / "data"
ZIP_PATH = DATA_DIR / "bike-sharing.zip"
CSV_PATH = DATA_DIR / "day.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
)


def load_bike_sharing() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists():
        subprocess.run(["curl", "-sL", "-o", str(ZIP_PATH), DATA_URL], check=True)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extract("day.csv", DATA_DIR)

    df = pd.read_csv(CSV_PATH)
    df["dteday"] = pd.to_datetime(df["dteday"])
    return df.drop(columns=["instant"])  # a bare row counter


def main() -> None:
    df = load_bike_sharing()

    profile = sk.fit(df)
    print(f"{len(df)} rows, {len(df.columns)} columns")
    print("dteday classified as:", profile.column_types["dteday"])
    print("fitted granularity (seconds):", profile.marginals["dteday"]["granularity_seconds"])
    print()

    synthetic = sk.emit(profile, n=len(df), seed=0)

    print("date range, real vs synthetic:")
    print(f"  real:      {df['dteday'].min().date()} to {df['dteday'].max().date()}")
    print(f"  synthetic: {synthetic['dteday'].min().date()} to {synthetic['dteday'].max().date()}")
    epoch = synthetic["dteday"].to_numpy().astype("datetime64[s]").astype("int64")
    on_midnight = (epoch % 86400 == 0).all()
    print(f"  all synthetic dates land on midnight (daily granularity): {on_midnight}")
    print()

    # dteday correlates with cnt through the yearly trend (bike share usage grew year over
    # year); check that the copula preserves it via the epoch-seconds rank transform.
    real_epoch = df["dteday"].to_numpy().astype("datetime64[s]").astype("float64")
    synth_epoch = synthetic["dteday"].to_numpy().astype("datetime64[s]").astype("float64")
    real_corr = np.corrcoef(real_epoch, df["cnt"])[0, 1]
    synth_corr = np.corrcoef(synth_epoch, synthetic["cnt"].astype(float))[0, 1]
    print(f"corr(date, rental count): real={real_corr:.3f} synthetic={synth_corr:.3f}")
    print()

    real_temp_corr = np.corrcoef(df["temp"], df["cnt"])[0, 1]
    synth_temp_corr = np.corrcoef(synthetic["temp"].astype(float), synthetic["cnt"].astype(float))[
        0, 1
    ]
    print(f"corr(temp, rentals): real={real_temp_corr:.3f} synthetic={synth_temp_corr:.3f}")
    print()

    report = sk.check(synthetic, profile, real=df, min_dcr_ratio=0.5)
    print(f"privacy check: dcr_ratio={report.dcr_ratio:.3f} exact_matches={report.exact_matches}")
    print(f"passed: {report.passed}")
    print()

    # cnt is exactly casual + registered in the real data, every single row -- a genuine
    # derived-column relationship, not an approximate correlation. The copula models all three
    # as separately-correlated numeric columns, so without declaring the relationship, it only
    # holds by chance.
    exact_without_constraint = (df["casual"] + df["registered"] == df["cnt"]).mean()
    without_constraint = (synthetic["casual"] + synthetic["registered"] == synthetic["cnt"]).mean()
    print(f"cnt == casual + registered, real data:              {exact_without_constraint:.3f}")
    print(f"cnt == casual + registered, synthetic (no constraint): {without_constraint:.3f}")

    profile_with_constraint = sk.fit(df, constraints=[sk.Derived("cnt", "casual + registered")])
    synthetic_with_constraint = sk.emit(profile_with_constraint, n=len(df), seed=0)
    with_constraint = (
        synthetic_with_constraint["casual"] + synthetic_with_constraint["registered"]
        == synthetic_with_constraint["cnt"]
    ).mean()
    print(f"cnt == casual + registered, synthetic (Derived constraint): {with_constraint:.3f}")


if __name__ == "__main__":
    main()
