# synthkit

**pytest fixtures that look like your production data, without your production data.**

> Status: core pipeline implemented and tested (fit, emit, constraints, privacy check, drift
> detection, CLI). Not yet published to PyPI. See [docs/PLAN.md](docs/PLAN.md) for the full
> build plan and [CHANGELOG.md](CHANGELOG.md) for what's landed.

## The problem

Teams hold data with real people in it and can't put those rows in a test suite, a demo
environment, or a bug report. The usual fallbacks are all bad: Faker (statistically
meaningless — every column drawn independently, so `age` and `income` don't correlate),
hand-written fixtures (stale in a month), scrubbed production copies (legally fraught and
rarely actually anonymous), or nothing at all.

## What this does

Read a real dataset once. Learn its statistical shape — each column's distribution, the
dependence structure between columns, null patterns, business rules. Save that shape as a
small JSON profile containing **no real records**, which you commit to your repo. Generate as
many synthetic rows as you want from that profile, forever, without touching the real data
again.

```
real data  ──fit──►  profile.json  ──emit──►  synthetic rows
(once, locally)      (committed)              (in CI, forever)
```

## The argument, measured

Run against the real [UCI Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/) dataset (see [examples/](examples/) to reproduce):

| | corr(age, education-num) | corr(hours, income) |
|---|---|---|
| Real data | 0.037 | 0.230 |
| Faker-style (independently shuffled) | 0.001 | 0.000 |
| synthkit | 0.020 | 0.177 |

A test whose correctness depends on that correlation — mean hours worked, high income vs low
income — **passes on real data and on synthkit's output, and fails on Faker-style data.** That's
the whole pitch: synthkit preserves the joint structure that independent-column sampling
throws away.

synthkit also runs fast enough for a pull-request check: ~2,000,000 rows/sec emitted from a
committed profile on a 6-column dataset (see [scripts/benchmark.py](scripts/benchmark.py)).

## Install

Not yet published to PyPI. For now, install from a checkout:

```bash
pip install -e .
pip install -e ./pytest-synthkit  # optional pytest fixture plugin
```

## Usage

```python
import synthkit as sk

profile = sk.fit(df, constraints=[sk.Inequality("created_at", "<=", "updated_at")])
profile.save("profiles/customers.json")

synthetic = sk.emit(profile, n=10_000, seed=42)
report = sk.check(synthetic, profile, real=df)
print(report.dcr_ratio, report.ks_by_column)
```

```bash
# once, locally, on real data
synthkit fit data/customers.parquet -o profiles/customers.json --constraints constraints.yaml

# forever after, in CI, from the committed profile
synthkit emit profiles/customers.json -n 10000 --seed 42 -o fixtures/customers.parquet

# verify what you generated; exits non-zero on failure
synthkit check fixtures/customers.parquet --profile profiles/customers.json --real data/customers.parquet

# has production drifted away from the profile your fixtures are built on? exits non-zero past a threshold
synthkit diff profiles/customers.json data/customers_2026Q3.parquet
```

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

## How it works

1. **Type inference** — continuous, discrete, categorical, boolean, datetime, identifier, or
   free text, per column ([src/synthkit/types.py](src/synthkit/types.py)).
2. **Non-parametric marginals** — an empirical CDF for numeric columns, a frequency table for
   categorical ones, no assumed named distribution
   ([src/synthkit/marginals.py](src/synthkit/marginals.py)).
3. **A Gaussian copula** ties the marginals together via rank correlation, so joint structure
   survives sampling ([src/synthkit/copula.py](src/synthkit/copula.py)).
4. **Constraints and repair** — business rules (`created_at <= updated_at`, derived columns,
   conditional nulls, unique/foreign keys) declared separately and enforced after sampling
   ([src/synthkit/constraints.py](src/synthkit/constraints.py),
   [src/synthkit/repair.py](src/synthkit/repair.py)).
5. **A privacy check** — distance-to-closest-record against a real holdout baseline, so "no
   real records" is measured, not asserted
   ([src/synthkit/privacy.py](src/synthkit/privacy.py)).

See [docs/getting-started.md](docs/getting-started.md) for a full walkthrough and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for what this deliberately does not do.

## Relationship to SDV

[SDV](https://github.com/sdv-dev/SDV) already exists and does this well for research-grade
synthetic datasets. synthkit is aimed at a narrower job — test fixtures — which means
deterministic output, sub-second generation, a small diffable profile format, and a built-in
privacy check, rather than trained generative models.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy src/synthkit
```

## License

MIT — see [LICENSE](LICENSE).
