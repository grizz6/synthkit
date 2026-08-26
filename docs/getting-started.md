# Getting started

## Install

Not yet published to PyPI. For now, install from a checkout:

```bash
git clone https://github.com/grizz6/synthkit.git
cd synthkit
pip install -e .
```

## Fit a profile

Read a real dataset once. This never writes any real rows to disk — the profile is
statistics only.

```python
import pandas as pd
import synthkit as sk

df = pd.read_csv("customers.csv")
profile = sk.fit(df)
profile.save("profiles/customers.json")
```

## Emit synthetic rows

From here on, you never need the real data again. `emit` is deterministic: the same
profile and seed always produce byte-identical rows.

```python
profile = sk.Profile.load("profiles/customers.json")
synthetic = sk.emit(profile, n=10_000, seed=42)
```

## Declare business rules

The copula preserves statistical structure, not business rules — those are declared
separately and enforced after sampling:

```python
profile = sk.fit(
    df,
    constraints=[
        sk.Inequality("created_at", "<=", "updated_at"),
        sk.Derived("total", "subtotal + tax"),
        sk.ConditionalNull("cancelled_at", "status != 'cancelled'"),
    ],
)
```

## Verify it's actually safe to share

```python
synthetic = sk.emit(profile, n=len(df), seed=0)
report = sk.check(synthetic, profile, real=df)

print(report.dcr_ratio)  # >= 1.0 means synthetic rows sit at least as far from
# real training rows as a real holdout does
print(report.exact_matches)  # should be 0
print(report.passed)
```

## From the command line

The same three steps as a CI-friendly pipeline — fit once locally, emit and check on
every run:

```bash
synthkit fit data/customers.parquet -o profiles/customers.json
synthkit emit profiles/customers.json -n 10000 --seed 42 -o fixtures/customers.parquet
synthkit check fixtures/customers.parquet \
  --profile profiles/customers.json \
  --real data/customers.parquet
```

## In pytest

```bash
pip install -e ./pytest-synthkit
```

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

## Next steps

- [Limitations](LIMITATIONS.md) — what synthkit deliberately does not do.
- [examples/](https://github.com/grizz6/synthkit/tree/main/examples) — four worked examples
  on real public datasets, each chosen to exercise a different part of the pipeline.
- [Plan](PLAN.md) — the original build plan this project follows.
