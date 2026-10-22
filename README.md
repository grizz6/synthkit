# synthkit

pytest fixtures that look like your production data, without your production data.

## The goal

Tests, demos, and bug reports need realistic data, but real data has real people in it.
Faker-style generators are safe and statistically meaningless: every column is drawn on its
own, so age stops predicting income. I wanted synthetic data that keeps the shape of the real
thing, and a way to prove it carries no real records.

## What it achieves

- **Fit once, generate forever.** `fit` reads real data once and saves a small JSON profile
  with no real rows in it. Commit the profile, and `emit` produces any number of rows from it
  in CI, with the same output for the same seed.
- **Keeps the relationships between columns.** On the UCI Adult Census dataset:

  | | corr(age, education) | corr(hours, income) |
  |---|---|---|
  | Real data | 0.037 | 0.230 |
  | Faker-style | 0.001 | 0.000 |
  | synthkit | 0.065 | 0.183 |

- **Enforces business rules.** Declared constraints (`created_at <= updated_at`,
  `total = subtotal + tax`, unique keys, and foreign keys drawn from a pool you supply) are
  applied to every generated row.
- **Measures privacy.** `check` compares how close synthetic rows sit to real ones against a
  real holdout, counts rows that reproduce a rare real record, and `--explain` names the
  column to coarsen. On a 5-column slice of Adult Census it flags 2,032 such rows out of
  10,000 and points at `age`.
- **Catches drift.** `diff` fails when production data moves away from the committed profile:
  shifted values, shifted null rates, or added and dropped columns.
- **Fast enough for every pull request.** About 3 million rows per second for a 6-column table
  on a laptop.

Status: pre-alpha, not on PyPI. 385 tests run in CI on Linux, macOS, and Windows with Python
3.10 to 3.13. Worked examples on four public datasets are in [examples/](examples/).

## Quick start

```bash
pip install -e .
pip install -e ./pytest-synthkit  # optional pytest fixture

synthkit fit data/customers.parquet -o profiles/customers.json   # once, on real data
synthkit emit profiles/customers.json -n 10000 --seed 42 -o fixtures/customers.parquet
synthkit check fixtures/customers.parquet --profile profiles/customers.json \
    --real data/customers.parquet   # exits non-zero on failure
```

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

`synthkit --help` lists the rest: `diff`, `inspect`, `compare`, and `--json` output for CI.

## Limits

- The model captures rank correlation, not every kind of dependence. A tight nonlinear
  relationship won't survive.
- Free text is resampled from the observed vocabulary. Check text columns for personal data
  before sharing a profile.
- `check` compares every synthetic row with every real row, so a million-row check takes
  real time.

## License

MIT, see [LICENSE](LICENSE).
