# synthkit

pytest fixtures that look like your production data, without your production data.

## About

synthkit is a Python library and CLI that learns the statistical shape of a real dataset and generates synthetic rows with that shape, so tests, demos, and bug reports never need real records.

- **Fit once, generate forever:** a small JSON profile holds column distributions, cross-column correlation, null patterns, and business rules, but no real rows. Commit it and emit fixtures from it in CI.
- **Keeps relationships:** a Gaussian copula over non-parametric marginals preserves the correlations that independent, Faker-style generation loses.
- **Measures privacy instead of assuming it:** `check` compares synthetic-to-real distances against a real holdout and can name the column that makes records identifiable.
- **Fits into a workflow:** a pytest fixture plugin, deterministic seeds, drift detection against fresh production data, and a readable `compare` for reviewing re-fitted profiles in PRs.

**Status:** pre-alpha (`0.0.1`), not yet on PyPI. 316 tests; CI runs pytest on Linux, macOS, and Windows with Python 3.10–3.13, plus ruff and mypy. Worked examples use the Adult Census, Titanic, Wine Quality, and Bike Sharing datasets (`examples/`).

## The problem

Teams hold data with real people in it and can't put those rows in a test suite, a demo
environment, or a bug report. The usual options are all bad: Faker generates data that's
statistically meaningless (every column drawn independently, so age and income don't
correlate), hand-written fixtures go stale within a month, scrubbed production copies are
legally risky and rarely actually anonymous, and "nothing" just means you get surprised in
production.

## What this does

Read a real dataset once. Learn its statistical shape: each column's distribution, the
dependence between columns, null patterns, business rules. Save that shape as a small JSON
profile with no real records in it, commit that to your repo, and generate as many synthetic
rows as you want from it, forever, without touching the real data again.

```
real data  --fit-->  profile.json  --emit-->  synthetic rows
(once, locally)      (committed)              (in CI, forever)
```

## Does it actually work?

Fit against the real [UCI Adult Census Income](https://archive.ics.uci.edu/ml/machine-learning-databases/adult/)
dataset (see [examples/](examples/) to reproduce this):

| | corr(age, education-num) | corr(hours, income) |
|---|---|---|
| Real data | 0.037 | 0.230 |
| Faker-style (independently shuffled) | 0.001 | 0.000 |
| synthkit | 0.065 | 0.183 |

A test whose correctness depends on that correlation (mean hours worked, high income vs low
income) passes on real data and on synthkit's output, and fails on Faker-style data. That's
the whole pitch: it preserves the joint structure that sampling each column independently
throws away.

It's also fast enough for a pull-request check: roughly 2 million rows/sec emitted from a
committed profile on a 6-column dataset (see [scripts/benchmark.py](scripts/benchmark.py)).
The benchmark script's own rows/sec column comes out several times lower than that, because
it times `emit()` with `tracemalloc` running to record peak memory. Time `emit()` on its own
to see the raw throughput.

## Install

Not published to PyPI yet. Install from a checkout:

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

# if it fails, find out which column is making records identifiable
synthkit check fixtures/customers.parquet --profile profiles/customers.json \
    --real data/customers.parquet --explain

# has production drifted away from the profile your fixtures are built on?
# fails on a shifted distribution or a changed schema (a column added or dropped)
synthkit diff profiles/customers.json data/customers_2026Q3.parquet

# what's actually in a committed profile, without writing code to find out
synthkit inspect profiles/customers.json

# type inference is a heuristic; override it when it guesses wrong
synthkit fit data/customers.parquet -o profiles/customers.json --column-type rating=discrete

# reviewing a re-fitted profile in a PR: what actually changed, not a wall of shifted floats
synthkit compare profiles/customers.json profiles/customers.new.json

# both inspect and compare speak JSON, for scripting against in CI
synthkit inspect profiles/customers.json --json | jq '.column_summaries[] | select(.null_rate > 0.5)'

# gate a build: fail if a fresh fit no longer matches the committed profile
synthkit compare profiles/customers.json profiles/customers.new.json --fail-on-change

# a foreign_key constraint needs real keys to point at; give it a pool to draw from
synthkit emit profiles/orders.json -n 10000 --seed 42 -o fixtures/orders.parquet \
    --key-pool customer_id=fixtures/customers.parquet:id
```

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

## How it works

1. **Type inference** classifies each column as continuous, discrete, categorical, boolean,
   datetime, identifier, or free text (`src/synthkit/types.py`).
2. **Marginals** are fit non-parametrically: an empirical CDF for numeric columns, a
   frequency table for categorical ones, no assumed named distribution
   (`src/synthkit/marginals.py`).
3. A **Gaussian copula** ties the marginals together via rank correlation so joint structure
   survives sampling (`src/synthkit/copula.py`).
4. **Constraints** (`created_at <= updated_at`, derived columns, conditional nulls, unique or
   foreign keys) are declared separately and enforced after sampling
   (`src/synthkit/constraints.py`, `src/synthkit/repair.py`).
5. A **privacy check** compares distance-to-closest-record against a real holdout baseline,
   so "no real records" is measured rather than assumed. Identifier columns are excluded from
   match detection (they are regenerated, so they can never match and would only mask
   matches elsewhere), and a reproduced record counts against you only when it was rare
   enough in the real data to narrow someone down (`src/synthkit/privacy.py`).

## Known limitations

- The copula captures rank correlation, not arbitrary dependence. It won't reproduce a tight
  nonlinear relationship or a conditional mode.
- Associating two nominal (unordered) categorical columns requires picking some ordering for
  them; synthkit orders by frequency, which is an approximation, not exact.
- Free text columns are approximated by resampling words from the observed vocabulary, not
  generated. Review any free-text column for PII before sharing a profile built from it.
- Identifiers are regenerated from a detected format and never modeled statistically, so two
  identifier columns that were correlated in the original data won't be in the output.
- `check()`'s privacy comparison is O(n * m) in row count on each side; both sides are
  batched to keep memory bounded, but a million-row check still takes real time.
- `synthkit` is already taken on PyPI by an unrelated package. Not a blocker for local
  development; matters only once this actually publishes.

## Relationship to SDV

[SDV](https://github.com/sdv-dev/SDV) already does this well for research-grade synthetic
datasets. synthkit is aimed at a narrower job, test fixtures, which means deterministic
output, sub-second generation, a small diffable profile format, and a built-in privacy check
rather than trained generative models.

## Development

```bash
pip install -e ".[dev]"
pytest tests/
ruff check .
mypy src/synthkit
```

## License

MIT, see [LICENSE](LICENSE).
