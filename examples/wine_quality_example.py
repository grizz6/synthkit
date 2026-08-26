"""A third real-dataset check: Wine Quality, chosen for what the other two don't exercise.

Adult Census and Titanic both mix numeric and categorical columns. Wine Quality is (almost)
entirely continuous — 11 physicochemical measurements plus an integer quality score — with no
nulls at all. That makes it a clean stress test for the one thing this whole package exists
for: does the Gaussian copula preserve the *entire* correlation matrix across many numeric
columns at once, not just one hand-picked pair?

Run with:

    python examples/wine_quality_example.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import synthkit as sk

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "winequality-red.csv"
DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


def load_wine_quality() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        subprocess.run(["curl", "-sL", "-o", str(DATA_PATH), DATA_URL], check=True)
    return pd.read_csv(DATA_PATH, sep=";")


def main() -> None:
    df = load_wine_quality()

    profile = sk.fit(df)
    print(
        f"{len(df)} rows, {len(df.columns)} columns, all copula-eligible: "
        f"{len(profile.copula_columns) == len(df.columns)}"
    )
    print()

    synthetic = sk.emit(profile, n=len(df), seed=0)

    real_corr = df.corr().to_numpy()
    synth_corr = synthetic.astype(float).corr().to_numpy()

    # Compare only the upper triangle (excluding the diagonal, which is trivially 1.0 for both)
    # to avoid double-counting each pair.
    triu = np.triu_indices_from(real_corr, k=1)
    abs_error = np.abs(real_corr[triu] - synth_corr[triu])

    print("full correlation matrix fidelity, across all C(12,2) = 66 column pairs:")
    print(f"  mean absolute error:   {abs_error.mean():.4f}")
    print(f"  max absolute error:    {abs_error.max():.4f}")
    print(f"  pairs within 0.05:     {(abs_error < 0.05).sum()} / {len(abs_error)}")
    print()

    worst_pair_idx = abs_error.argmax()
    a, b = triu[0][worst_pair_idx], triu[1][worst_pair_idx]
    col_a, col_b = df.columns[a], df.columns[b]
    print(f"worst-preserved pair: {col_a} vs {col_b}")
    print(f"  real corr:      {real_corr[a, b]:.3f}")
    print(f"  synthetic corr: {synth_corr[a, b]:.3f}")
    print()

    report = sk.check(synthetic, profile, real=df, min_dcr_ratio=0.5)
    print(f"privacy check: dcr_ratio={report.dcr_ratio:.3f} exact_matches={report.exact_matches}")
    print(f"passed: {report.passed}")
    print("mean per-column KS statistic:", round(np.mean(list(report.ks_by_column.values())), 4))


if __name__ == "__main__":
    main()
