"""A second real-dataset check: Titanic, chosen for what Adult Census doesn't exercise.

Adult Census (see adult_census_worked_example.py) has no nulls and a handful of well-behaved
columns. Titanic adds: null co-occurrence (Cabin is null ~77% of the time, correlated with
passenger class), a free-text-ish Name column with no repeats, a messy Ticket identifier
column, and a small dataset (891 rows) relative to Adult Census's 32,561.

Run with:

    python examples/titanic_example.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd

import synthkit as sk

DATA_DIR = Path(__file__).parent / "data"
DATA_PATH = DATA_DIR / "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def load_titanic() -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        # Not urllib: some Python installs (notably python.org's macOS framework builds) ship
        # without a configured CA bundle, so urlretrieve fails with CERTIFICATE_VERIFY_FAILED
        # where the system's own curl succeeds.
        subprocess.run(["curl", "-sL", "-o", str(DATA_PATH), DATA_URL], check=True)
    return pd.read_csv(DATA_PATH)


def main() -> None:
    df = load_titanic()
    df = df.drop(columns=["PassengerId"])  # a bare row counter, not interesting to model

    profile = sk.fit(df)
    print("column types:")
    for column, ctype in profile.column_types.items():
        print(f"  {column:12s} {ctype}")
    print()

    synthetic = sk.emit(profile, n=len(df), seed=0)

    print("null rate, real vs synthetic:")
    for column in ["Age", "Cabin", "Embarked"]:
        real_rate = df[column].isna().mean()
        synth_rate = synthetic[column].isna().mean()
        print(f"  {column:10s} real={real_rate:.3f} synthetic={synth_rate:.3f}")
    print()

    print("category frequencies, real vs synthetic:")
    for column in ["Sex", "Pclass", "Embarked"]:
        real_freq = df[column].value_counts(normalize=True).round(3).to_dict()
        synth_freq = synthetic[column].value_counts(normalize=True).round(3).to_dict()
        print(f"  {column}: real={real_freq}")
        print(f"  {' ' * len(column)}  synthetic={synth_freq}")
    print()

    print("does the survival rate by class survive the round trip?")
    for name, dataset in [("real", df), ("synthetic", synthetic)]:
        rate = dataset.groupby("Pclass")["Survived"].mean().round(3).to_dict()
        print(f"  {name:10s} {rate}")
    print(
        "  (both Survived and Pclass are categorical, so their association goes through the\n"
        "  copula's frequency-ordered category mapping rather than exact rank correlation —\n"
        "  see 'Categorical association is approximate' in docs/LIMITATIONS.md. The direction\n"
        "  survives (class 1 > class 2 > class 3), the magnitude is compressed.)"
    )
    print()

    report = sk.check(synthetic, profile, real=df, min_dcr_ratio=0.5)
    print(
        f"privacy check: dcr_ratio={report.dcr_ratio:.3f} exact_matches={report.exact_matches} "
        f"rare_combination_leaks={report.rare_combination_leaks}"
    )
    print(f"passed: {report.passed}")
    print(
        "  (dcr_ratio > 1 means synthetic rows sit farther from real training rows than a real\n"
        "  holdout does — good. The nonzero rare-combination count is expected on a small,\n"
        "  several-categorical-column dataset like this one; see docs/LIMITATIONS.md.)"
    )

    # Name is all-unique -> should be classified an identifier and never leak a real name.
    assert profile.column_types["Name"] == "identifier"
    assert not set(synthetic["Name"]) & set(df["Name"])
    print("\nno real passenger name appears in the synthetic output: confirmed")


if __name__ == "__main__":
    main()
