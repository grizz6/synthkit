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

Status: pre-alpha, not on PyPI. 387 tests run in CI on Linux, macOS, and Windows with Python
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

## How it works

1. **Type inference.** Each column is classified as one of the types below. Inference is a
   heuristic, so `fit --column-type NAME=TYPE` overrides it.
2. **Marginals.** Each column's distribution is stored non-parametrically: quantiles for
   numbers and dates, a frequency table for categories. No named distribution is assumed.
3. **Gaussian copula.** The rank correlation between columns is stored as one matrix, and
   sampling through it keeps the columns related the way they were.
4. **Null patterns.** Which columns are missing together is modeled jointly, so a column
   that is only filled in when another one is keeps that pattern.
5. **Business rules.** Declared constraints are applied after sampling, in a fixed order.
6. **Profile.** Everything above is saved as canonical JSON with sorted keys and a fixed
   float precision, so a re-fit shows up as a readable diff in a pull request.

| Type | Detected when | Generated as |
|---|---|---|
| `boolean` | The column holds true and false | True at the observed rate |
| `datetime` | The column holds dates or timestamps | Dates at the observed granularity and timezone |
| `categorical` | About 20 or fewer distinct values, whole numbers or text | Categories at their observed frequencies |
| `discrete` | Whole numbers with 20 or more distinct values | Values from the fitted quantiles, as integers |
| `continuous` | Numbers with decimals | Values from the fitted quantiles |
| `identifier` | Text where every value is unique, such as IDs or names | New values in the same detected format |
| `text` | Text where most values are distinct | Words resampled from the observed vocabulary |

## Business rules

The copula gets the statistics right but knows nothing about rules like "a total is the
subtotal plus tax". Declare them in a YAML file and pass it to `fit --constraints`:

```yaml
constraints:
  - type: inequality          # created_at <= updated_at on every row
    left: created_at
    op: "<="                  # also <, >=, >
    right: updated_at
  - type: derived             # recomputed, never sampled
    column: total
    expr: subtotal + tax
  - type: conditional_null    # null exactly when the condition holds
    column: cancelled_at
    null_when: status != 'cancelled'
  - type: unique
    columns: [order_id]       # list several columns for a unique combination
  - type: foreign_key         # drawn from a pool you pass to emit
    column: customer_id
    references: customers.id
```

`expr` and `null_when` use pandas `eval` syntax; wrap a column name containing spaces or
hyphens in backticks. A foreign key takes its values from `emit --key-pool
customer_id=fixtures/customers.parquet:id`, so child fixtures join to parent fixtures.

## Commands

| Command | What it does | Useful options |
|---|---|---|
| `fit DATA -o PROFILE` | Learns a profile from a `.csv` or `.parquet` file | `--constraints`, `--column-type NAME=TYPE`, `--holdout` to run a privacy self-check right away |
| `emit PROFILE -n N --seed S -o OUT` | Generates rows; the same seed always gives the same rows | `--key-pool NAME=FILE:COLUMN` |
| `check FIXTURES --profile P --real DATA` | Privacy check; exits 1 if it fails | `--explain`, `--min-dcr-ratio` (default 0.9), `--json` |
| `diff PROFILE DATA` | Drift check against fresh data; exits 1 on drift | `--threshold` (default 0.1), `--json` |
| `inspect PROFILE` | Summarizes a profile: columns, types, rules | `--json` |
| `compare OLD NEW` | Shows what changed between two profiles | `--fail-on-change` to exit 1 on any change, `--json` |

`--json` keeps the same exit codes, so CI can gate on the exit code and parse the details.

## What the checks measure

**`check`** passes only if all three of these hold:

- **Distance to real rows.** Synthetic rows must sit at least as far from the real training
  rows as a held-out slice of real rows does (`dcr_ratio`, 0.9 or above by default). A
  generator that memorizes rows scores well below 1.
- **No identifying matches.** No synthetic row may reproduce a combination of values that
  fewer than 5 real rows share. Matching a common combination reveals nothing, so those
  count only as `exact_matches`, for information.
- **No rare category pairs.** No synthetic row may reproduce a pair of category values seen
  fewer than 5 times in the real data.

The cutoff of 5 applies to both rare-value rules. Change it with
`--rare-combination-threshold` on the command line or `rare_combination_threshold=` in
`sk.check()`.

When it finds identifying matches, `--explain` reports how many would remain if each column
were dropped, so the column worth coarsening (bucketing ages, say) is at the top.

**`diff`** fails when fresh data has moved away from the profile: a column's values shifted
(KS statistic for numbers and dates, total variation distance for categories), its null rate
moved by more than the threshold, or a column was added or dropped.

## Python API

```python
import synthkit as sk

profile = sk.fit(real_df, constraints="constraints.yaml")  # or a list of sk.Inequality(...) etc.
profile.save("profiles/orders.json")

rows = sk.emit(profile, n=1_000, seed=42, key_pools={"customer_id": customer_ids})
report = sk.check(rows, profile, real=real_df)
report.passed, report.dcr_ratio, report.identifying_matches, report.ks_by_column
```

## pytest fixture

With `pytest-synthkit` installed, `synth_frame` emits rows from a committed profile. Profiles
are cached for the test session and reloaded if the file changes.

```python
def test_orders_join_to_customers(synth_frame):
    customers = synth_frame("profiles/customers.json", n=100, seed=0)
    orders = synth_frame(
        "profiles/orders.json", n=1000, seed=0, key_pools={"customer_id": customers.id.tolist()}
    )
    assert orders.customer_id.isin(customers.id).all()
```

## Examples

Each script in [examples/](examples/) downloads a public dataset, fits a profile, and checks
what survives. The [notebooks](examples/notebooks/) show the same runs with their output.

| Example | What it shows |
|---|---|
| `adult_census_worked_example.py` | Correlations kept where Faker-style generation loses them |
| `titanic_example.py` | Null patterns, identifier columns, and a small dataset |
| `wine_quality_example.py` | A full 12-column correlation matrix |
| `bike_sharing_example.py` | Dates, and `Derived` rules for exact relationships |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ pytest-synthkit/tests/
ruff check . && ruff format --check .
mypy src/synthkit
```

## Limits

- The model captures rank correlation, not every kind of dependence. A tight nonlinear
  relationship won't survive.
- Free text is resampled from the observed vocabulary. Check text columns for personal data
  before sharing a profile.
- `check` compares every synthetic row with every real row, so a million-row check takes
  real time.

## License

MIT, see [LICENSE](LICENSE).
