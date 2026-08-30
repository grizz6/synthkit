"""Downloads the UCI Adult Census Income dataset, fits a synthkit profile on it, and compares
real data, "Faker-style" data (every column shuffled independently), and synthkit's output.
Writes an aggregation whose correctness depends on the joint distribution between
hours-per-week and income, passing on real and synthkit data, failing on Faker-style data.

Run with:

    python examples/adult_census_worked_example.py
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import synthkit as sk

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "adult.data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]


def load_adult() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH, names=COLUMNS, skipinitialspace=True)
    df["income_over_50k"] = (df["income"] == ">50K").astype(int)
    # "?" missing values in this dataset only ever land in workclass/occupation/native-country,
    # none of which are kept below, so there's nothing left to filter out here.
    return df[["age", "education-num", "hours-per-week", "income", "income_over_50k"]]


def faker_style(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Every column shuffled independently: individually plausible, jointly meaningless."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({col: rng.permutation(df[col].to_numpy()) for col in df.columns})


def _numeric(series: pd.Series) -> pd.Series:
    # A round trip through a categorical marginal represents everything as strings, so a
    # 0/1 income flag comes back out as "0"/"1" rather than an int; coerce it back.
    return pd.to_numeric(series, errors="coerce")


def correlations(df: pd.DataFrame) -> tuple[float, float]:
    age_edu = np.corrcoef(_numeric(df["age"]), _numeric(df["education-num"]))[0, 1]
    hours_income = np.corrcoef(_numeric(df["hours-per-week"]), _numeric(df["income_over_50k"]))[
        0, 1
    ]
    return age_edu, hours_income


def passes_joint_distribution_test(df: pd.DataFrame) -> bool:
    """A check whose result only makes sense if hours-per-week and income actually correlate:
    mean hours-per-week among the high-income group should exceed the low-income group by a
    real margin. Faker-style shuffled data collapses this gap to approximately zero."""
    income = _numeric(df["income_over_50k"])
    hours = _numeric(df["hours-per-week"])
    high = hours[income == 1].mean()
    low = hours[income == 0].mean()
    return (high - low) > 2.0


def main() -> None:
    real = load_adult()
    real_age_edu, real_hours_income = correlations(real)

    faker = faker_style(real, seed=0)
    faker_age_edu, faker_hours_income = correlations(faker)

    profile = sk.fit(real)

    start = time.perf_counter()
    synthetic = sk.emit(profile, n=10_000, seed=0)
    elapsed = time.perf_counter() - start
    rows_per_sec = 10_000 / elapsed

    synth_age_edu, synth_hours_income = correlations(synthetic)

    print("| | corr(age, education-num) | corr(hours, income) | Rows/sec |")
    print("|---|---|---|---|")
    print(f"| Real data | {real_age_edu:.3f} | {real_hours_income:.3f} | n/a |")
    print(f"| Faker-style (shuffled) | {faker_age_edu:.3f} | {faker_hours_income:.3f} | fast |")
    print(f"| synthkit | {synth_age_edu:.3f} | {synth_hours_income:.3f} | {rows_per_sec:,.0f} |")
    print()

    print("Joint-distribution test (mean hours-per-week, high income vs low income):")
    for name, dataset in [("real", real), ("synthkit", synthetic), ("Faker-style", faker)]:
        result = "PASS" if passes_joint_distribution_test(dataset) else "FAIL"
        print(f"  {name:12s} {result}")


if __name__ == "__main__":
    main()
