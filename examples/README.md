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
