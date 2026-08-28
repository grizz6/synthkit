"""Benchmark: rows/sec, profile size, and peak memory, on a realistic column mix.

Run with:

    python scripts/benchmark.py
"""

from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np
import pandas as pd

import synthkit as sk

ROW_COUNTS = [1_000, 10_000, 100_000]


def make_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "customer_id": [f"CUST{i:07d}" for i in range(n)],
            "age": rng.normal(40, 12, n).clip(18, 90),
            "plan_tier": rng.choice(["free", "pro", "enterprise"], size=n, p=[0.6, 0.3, 0.1]),
            "is_active": rng.random(n) < 0.85,
            "signup_at": pd.to_datetime("2022-01-01")
            + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
            "monthly_spend": rng.gamma(2.0, 40.0, n),
        }
    )


def main() -> None:
    df = make_dataset(50_000)

    tracemalloc.start()
    fit_start = time.perf_counter()
    profile = sk.fit(df)
    fit_elapsed = time.perf_counter() - fit_start
    _, fit_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    profile_json = json.dumps(profile.to_dict())
    profile_kb = len(profile_json.encode("utf-8")) / 1024

    print(f"fit: {len(df):,} rows, {len(df.columns)} columns in {fit_elapsed * 1000:.1f} ms")
    print(f"fit peak memory: {fit_peak_bytes / 1024 / 1024:.1f} MB")
    print(f"profile size: {profile_kb:.1f} KB")
    print()
    print("| Rows requested | Time (s) | Rows/sec | Peak memory (MB) |")
    print("|---|---|---|---|")

    for n in ROW_COUNTS:
        tracemalloc.start()
        start = time.perf_counter()
        sk.emit(profile, n=n, seed=0)
        elapsed = time.perf_counter() - start
        _, emit_peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        emit_peak_mb = emit_peak_bytes / 1024 / 1024
        print(f"| {n:,} | {elapsed:.4f} | {n / elapsed:,.0f} | {emit_peak_mb:.1f} |")


if __name__ == "__main__":
    main()
