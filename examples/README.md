# Worked example: UCI Adult Census Income

```bash
python examples/adult_census_worked_example.py
```

Downloads the dataset on first run (cached under `examples/data/`, gitignored) and prints a
comparison of real data, Faker-style independently-shuffled data, and synthkit's output.

## Measured results (2026-08-25, 10,000 synthetic rows, seed 0)

| | corr(age, education-num) | corr(hours, income) | Rows/sec |
|---|---|---|---|
| Real data | 0.037 | 0.230 | n/a |
| Faker-style (shuffled) | 0.001 | 0.000 | fast |
| synthkit | 0.020 | 0.177 | ~3,000,000 |

Joint-distribution test (mean `hours-per-week`, high income vs low income group):

| Dataset | Result |
|---|---|
| real | PASS |
| synthkit | PASS |
| Faker-style | FAIL |

That's the whole argument: a test whose correctness depends on the correlation between two
columns passes against real data and against synthkit's output, and fails against
independently-shuffled data — which is what sampling each column's marginal with no joint
model produces.

## Titanic: nulls, free text, and messy identifiers

```bash
python examples/titanic_example.py
```

A smaller (891-row), messier dataset: `Cabin` is null ~77% of the time, `Name` is entirely
unique (an identifier), and there are six categorical columns at once. This is also what
surfaced two real bugs during development — a small all-unique-string privacy leak and a
combinatorial blowup in the rare-combination privacy check — both fixed and covered by
regression tests; see [CHANGELOG.md](../CHANGELOG.md) and [docs/LIMITATIONS.md](../docs/LIMITATIONS.md).

## Wine Quality: correlation fidelity across many numeric columns at once

```bash
python examples/wine_quality_example.py
```

1,599 rows, 12 almost-entirely-continuous columns, no nulls — a clean stress test for whether
the copula preserves the *whole* correlation matrix (66 pairs), not just one hand-picked pair.
Running this during development caught a severe bug: a low-cardinality numeric column (the
`quality` rating) was ordered by frequency instead of by value for the copula, which silently
destroyed its correlation with every other column (a constructed worst case went from a real
correlation of 0.96 to a synthetic 0.02). Fixed; see CHANGELOG.md.

## Bike Sharing: the one with a real datetime column

```bash
python examples/bike_sharing_example.py
```

731 daily rows with a genuine date column (`dteday`) correlated with a numeric target (rental
count). None of the other three examples have a datetime column at all, so this was the first
time `DatetimeMarginal`'s path through the copula ran against real data — which caught another
bug: the privacy check's per-column fidelity score compared real and synthetic dates in
different units (microseconds vs seconds), reporting them as completely unrelated regardless
of actual fidelity. Fixed; see CHANGELOG.md.
