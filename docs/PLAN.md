# Synthetic test data package: zero to one

**Three weeks at 20+ hours, from an empty repo to a published package with real users.**
Working name below is `synthkit`. Check PyPI availability before you commit to it; other
candidates are `fixtura`, `shapeshift`, `replikit`, `mimicdata`.

---

## Part 1: what it is and why anyone cares

### The problem, stated precisely

A team holds data with real people in it: customers, patients, transactions, applicants. They
cannot put those rows in a test suite, a demo environment, a staging database, or a bug report.
So they reach for one of four bad options.

**Faker.** Generates rows that are individually plausible and collectively meaningless. Every
column is drawn independently, so age and income are uncorrelated, `created_at` sometimes falls
after `updated_at`, and 40 percent of your customers live in Wyoming. Tests pass against this
data and then fail in production, because production data has structure and the fixture does not.

**Hand-written fixtures.** Five rows someone wrote in 2023. They cover the happy path, they
drift from reality every month, and nobody remembers why row three has a null in it.

**Scrubbed production copies.** Legally fraught, and scrubbing is much harder than it looks.
Remove the name and the email and you still have a row with a birth date, a zip code, and a
gender, which the Sweeney result says identifies most Americans uniquely.

**Nothing.** Test against toy data, get surprised in production.

### What this package does

Read a real dataset once. Learn its statistical shape: each column's distribution, the
dependence structure between columns, the null patterns, and the business rules. Save that
shape as a small JSON file containing **no real records**, which you commit to your repository.
Then generate as many synthetic rows as you want from that file, forever, without ever touching
the real data again.

```
real data  ──fit──►  profile.json  ──emit──►  synthetic rows
(once, locally)      (committed)              (in CI, forever)
```

The profile is the product. It is portable, reviewable in a pull request, diffable, and safe to
share publicly.

### The honest competitive picture

**SDV (Synthetic Data Vault) already exists**, and it is the obvious thing anyone will bring up
in an interview or in your launch thread. Say it first, before they do.

SDV is a research-derived platform: it trains generative models (copulas, CTGAN, TVAE), it is
heavy, it is slow enough that you would not put it in a pull-request check, and its output is
not reproducible run to run without care. It is aimed at producing *research-grade* synthetic
datasets.

**Your wedge is a different job entirely: test fixtures.** That job has four requirements SDV
does not prioritize, and each one is a design decision you can defend.

| Requirement | Why it matters for fixtures | What it forces |
|---|---|---|
| **Deterministic** | A test that generates different data each run is a flaky test | Same profile, same seed, same row count, byte-identical output. Non-negotiable |
| **Fast** | It runs on every pull request | Sub-second for 10,000 rows. No training loop, no GPU, no torch |
| **Committable** | The fixture definition belongs in the repo, next to the tests | A small canonical JSON profile, not a pickled model |
| **Provably safe** | Someone has to sign off on committing it | A built-in privacy check that exits non-zero, not a footnote |

One sentence for the README: **"pytest fixtures that look like your production data, without
your production data."**

### Who uses it, concretely

- A data team that needs CI fixtures and cannot use PII.
- Anyone building a demo or sandbox environment that should not look obviously fake.
- Load testing, where the *distribution* of the input changes the performance you measure.
- **Bug reports.** You can attach `profile.json` to a public GitHub issue and the maintainer can
  regenerate data with your exact shape. Today people either leak data or say "it's a big table
  with some nulls."
- Onboarding a new engineer without granting production access on day one.

---

## Part 2: the technical core

This is the part you must be able to explain without notes. It is also the part that makes the
project mathematically interesting rather than plumbing.

### Step 1: type inference

Per column, classify into: continuous numeric, discrete numeric, categorical, boolean,
datetime, identifier (high-cardinality string), or free text. Getting this wrong poisons
everything downstream, so make it inspectable and overridable in the profile.

Heuristics that work: integer dtype with fewer than ~20 distinct values is categorical, not
numeric. A string column with cardinality equal to row count is an identifier, not a category.
A numeric column where every value is a whole number should emit whole numbers.

### Step 2: marginals, non-parametrically

Do not fit named distributions. Real columns are not normal, not lognormal, and not gamma;
they are bimodal, zero-inflated, and truncated at values someone chose in 2019.

- **Numeric:** store the **empirical CDF** as roughly 100 quantile knots. Sampling is inverse
  transform with linear interpolation between knots. Robust, assumption-free, and small.
- **Categorical:** store the category to frequency table. Cap cardinality at some N and bucket
  the tail as `__other__`, recording how much mass that is.
- **Datetime:** convert to epoch seconds and treat as numeric, but record granularity (daily,
  secondly) and re-quantize on output so you do not emit timestamps at 03:47:13 when every real
  value is midnight.
- **Boolean:** a single probability.
- **Identifier:** infer a format pattern and regenerate. Never model it.
- **Nulls:** store a null rate per column, applied after sampling. **Also store null
  co-occurrence**, because nulls are almost never independent. If `cancelled_at` is null exactly
  when `status != 'cancelled'`, a per-column null rate gets that catastrophically wrong.

### Step 3: dependence, via a Gaussian copula

This is the heart of it. The idea: separate *what each column looks like* from *how the columns
move together*, model each independently, then recombine.

**Fitting.**

1. Rank-transform each column to uniform: `u_i = rank(x_i) / (n + 1)`.
2. Map to standard normal scores: `z_i = Φ⁻¹(u_i)`.
3. Compute the correlation matrix `Σ` of the `z` columns.
4. **Project `Σ` to the nearest positive definite matrix.** Ties in the ranks and pairwise
   estimation mean the empirical matrix often is not PD, and Cholesky will fail. Clip negative
   eigenvalues to a small positive floor and renormalize the diagonal to 1.
5. Store `Σ` and the per-column quantile knots.

**Sampling.**

1. Draw `z ~ N(0, Σ)` using the Cholesky factor `L`, so `z = L @ standard_normal(k)`.
2. `u = Φ(z)`.
3. `x_i = F_i⁻¹(u_i)` using the stored empirical quantile function.

The result preserves each column's marginal shape *exactly as observed* while reproducing the
rank correlation structure. That is the whole trick, and it is why the output looks like the
real thing in a way Faker never will.

**Categoricals in the copula.** Order the categories (by descending frequency, with a stable
tiebreak), assign each an interval on `[0, 1]` proportional to its frequency, and map `u` into
whichever interval it lands in. This preserves marginal frequencies and captures association
with other columns through the copula.

**State this limitation out loud in your docs**, because it is real and an interviewer will find
it: imposing an ordering on an unordered category is an assumption. Associations between two
nominal categoricals are captured only approximately, and the frequency ordering is an arbitrary
choice that happens to work well in practice. Documenting a known weakness is a credibility
signal; discovering it during a demo is not.

### Step 4: constraints and repair

The copula gets the statistics right and the business rules wrong. Declare them separately:

```yaml
constraints:
  - type: inequality        # created_at <= updated_at
    left: created_at
    op: "<="
    right: updated_at
  - type: derived           # total is not sampled, it is computed
    column: total
    expr: "subtotal + tax"
  - type: conditional_null  # cancelled_at is null unless cancelled
    column: cancelled_at
    null_when: "status != 'cancelled'"
  - type: unique
    columns: [customer_id]
  - type: foreign_key
    column: customer_id
    references: customers.id
```

Repair strategies, applied in this order, and the choice per constraint type matters:

- **Derived columns:** recompute. Never sample them at all; drop them before fitting.
- **Inequality:** swap the two values if that fixes it, since swapping preserves both marginals
  exactly. This is a nicer fix than clamping and most implementations miss it.
- **Conditional null:** apply the rule directly after sampling.
- **Unique and foreign key:** generate from a key pool rather than repairing.
- **Anything else:** reject and resample the offending row, with a max-attempts budget and a
  loud error when it is exhausted, because an unsatisfiable constraint set should fail fast
  rather than loop.

### Step 5: the privacy check

Without this, nobody with a compliance function will adopt it, and "the profile contains no real
records" is an assertion rather than a demonstration.

**Distance to Closest Record.** For each synthetic row, compute the distance to its nearest real
row (Gower distance handles mixed types). Now the key move: **compare that distribution against a
baseline**. Hold out some real rows when fitting, and compute the DCR of the *holdout* rows to
the *training* rows. That is what "as close as real data naturally is to itself" looks like.

If your synthetic rows are systematically closer to training rows than the holdout is, you are
memorizing. Report the ratio of the 5th percentile of synthetic DCR to the 5th percentile of
holdout DCR. Above 1.0 is comfortable. Below some threshold, fail.

Also check: no synthetic row exactly matching a real row, and no rare category combination
(appearing fewer than k times in real data) reproduced in the output.

### Step 6: determinism

Seed a `numpy.random.Generator` explicitly, never global state. Canonicalize the profile JSON
with sorted keys and fixed float precision. Then write the test that matters: fit, save, load,
emit twice with the same seed, assert byte-identical output. Also assert that a profile saved
today loads and produces identical output after a round trip.

---

## Part 3: the interface

### CLI

```bash
# once, locally, on real data
synthkit fit data/customers.parquet -o profiles/customers.json
synthkit fit data/customers.parquet --constraints constraints.yaml --holdout 0.2

# forever after, in CI, from the committed profile
synthkit emit profiles/customers.json -n 10000 --seed 42 -o fixtures/customers.parquet

# verify what you generated
synthkit check fixtures/customers.parquet \
  --profile profiles/customers.json \
  --real data/customers.parquet \
  --min-dcr-ratio 1.0
# exits non-zero on failure

# has production drifted away from the profile your fixtures are built on?
synthkit diff profiles/customers.json data/customers_2026Q3.parquet
# exits non-zero past a threshold
```

That last command is worth building even though nobody asked for it. **Test fixtures silently
rot as production data drifts**, and no tool tells you. A CI job that fails with "your fixtures
no longer resemble production, the `plan_tier` distribution has moved" is a genuinely novel and
useful thing to have shipped.

### Python API

```python
import synthkit as sk

profile = sk.fit(df, constraints=[sk.Inequality("created_at", "<=", "updated_at")])
profile.save("profiles/customers.json")

synthetic = sk.emit(profile, n=10_000, seed=42)
report = sk.check(synthetic, profile, real=df)
print(report.dcr_ratio, report.ks_by_column)
```

### pytest plugin

Ship `pytest-synthkit` as a second small package. It is a distribution multiplier: people find
pytest plugins by searching for pytest plugins.

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

---

## Part 4: the worked example that sells it

Build this first, before the launch post, and put it at the top of the README. It is the
argument.

Take a public dataset where two columns are genuinely correlated. Adult Census Income works:
`age` and `hours-per-week` and `education-num` all move together, and it is public and famous.

Then show three numbers:

| | corr(age, education-num) | corr(hours, income) | Rows/sec |
|---|---|---|---|
| Real data | 0.037 | 0.23 | n/a |
| Faker fixtures | ~0.00 | ~0.00 | fast |
| synthkit | within 0.02 of real | within 0.02 of real | measured |

Then the punchline: **write a test that only passes when the correlation is present.** An
aggregation, a bucketing function, a model-scoring path, anything whose behavior depends on the
joint distribution. Show it passing on real data, passing on synthkit data, and failing on Faker
data. That single example is more persuasive than any amount of README prose, and it is exactly
what gets a Show HN thread engaged.

---

## Part 5: fifteen working days

### Week 0, two days: validate before you build

Do not skip this. It is the difference between a package with users and a package with a
README.

**Day -2.** Read the competition properly and write down what each one is for: SDV, Faker,
Mimesis, ydata-synthetic, Gretel, Tonic. Write your wedge as one sentence. If you cannot, the
project is not ready.

**Day -1.** **Write the README first, before any code.** Installation, the worked example, the
three commands, the limitations section. Then show it to five people who would plausibly use it:
r/dataengineering, r/Python, your Webster cohort, anyone you know on a data team. Ask one
question: "would you use this, and what would stop you?" The answers reshape the scope. Also
check the name on PyPI and register it.

README-driven development is also a good story on its own. Most people cannot describe what they
built until after they built it.

### Week 1, days 1 to 5: the core, one table

**Day 1.** Repo scaffolding. `src/` layout, `uv` and `hatchling`, ruff, mypy, pytest,
pre-commit, GitHub Actions matrix across Python 3.10 through 3.13 on Linux, macOS, Windows. A
release workflow using PyPI trusted publishing, wired up but not fired yet.

**Day 2.** Type inference and marginals. Empirical CDF for numerics, frequency tables for
categoricals, null rates. **Round-trip test:** fit on a real dataset, emit `n = len(real)`,
assert a per-column Kolmogorov-Smirnov statistic below a threshold. That test is your fidelity
harness and everything after it leans on it.

**Day 3.** The copula. Rank transform, normal scores, correlation matrix, nearest-PD
projection, Cholesky, sampling. **Test:** correlation matrix of synthetic within tolerance of
real, elementwise.

**Day 4.** Categoricals into the copula, datetimes, integer preservation, null co-occurrence.
The determinism test.

**Day 5.** CLI with Typer, `fit` and `emit`. Parquet and CSV IO. First real end-to-end run on
the Adult dataset. Take the benchmark numbers now and write them down.

### Week 2, days 6 to 10: make it trustworthy

**Day 6.** Constraints DSL and the repair engine, including the swap trick for inequalities.

**Day 7.** The privacy check. Gower distance, DCR, holdout baseline, exact-match and rare-combo
checks. Wire the `check` command with real exit codes.

**Day 8.** The `diff` drift command.

**Day 9.** MkDocs Material site. README with the worked example. Three example notebooks on
three public datasets. A limitations page that is honest about the categorical ordering
assumption, high-cardinality behavior, and what the copula cannot capture (nonlinear dependence
beyond rank correlation, most notably).

**Day 10.** Publish. TestPyPI first, verify install in a clean venv, then PyPI. Tag `v0.1.0`,
write the CHANGELOG, turn on branch protection.

### Week 3, days 11 to 15: users

**Day 11.** Ship `pytest-synthkit`.

**Day 12.** Write the launch post around the worked example, not around the features. Lead with
the failing Faker test.

**Day 13.** Launch: Show HN, r/Python, r/dataengineering, r/datascience, Lobsters, the pytest
and data-engineering Discords, and pull requests adding it to two or three awesome-lists.
Space these across the day rather than firing everything at once.

**Days 14 and 15.** Respond to everything, fast. Triage every issue. Ship `v0.1.1` with the
first round of fixes **within 48 hours of launch**, because visible responsiveness is what turns
a spike of attention into actual adoption.

---

## Part 6: what to measure

Instrument from day one; you cannot reconstruct these later.

- PyPI downloads per day and per version, via `pypistats`
- GitHub stars, forks, issues opened and closed, **median time to first response**
- Fidelity: per-column KS statistic, correlation mean absolute error, on three reference datasets
- Performance: rows per second, profile size in kilobytes, memory ceiling
- Privacy: DCR ratio on the reference datasets

### The bullets this produces

- Published and maintained an open-source Python package with N monthly downloads, preserving
  statistical fidelity within X KS across mixed-type datasets while generating 10,000 rows in
  under Y milliseconds.
- Designed a Gaussian-copula sampler with empirical marginals and a constraint-repair layer,
  reproducing observed rank correlation within Z and enforcing declared business rules.
- Built a distance-to-closest-record privacy check benchmarked against a real holdout, gating
  releases in CI.
- Triaged and resolved N issues from external users, shipping a patch release within 48 hours of
  launch.

Notice that every one of those has a number in it, and every number comes from something you
measured rather than something you asserted.

---

## Part 7: the honest risks

**SDV exists and is well known.** Mitigation: say so first, in your README and your launch post,
and be precise about the different job. A launch that pretends the incumbent does not exist gets
taken apart in the comments within an hour.

**The copula has real limits.** It captures rank correlation, not arbitrary dependence. It will
not reproduce a tight nonlinear relationship or a conditional mode. Document this on the
limitations page rather than letting a user discover it.

**Nobody uses it.** The week 0 validation is the mitigation, and it is why week 0 is not
optional. If five people say "I would not use this because X," you learn X for the price of two
days instead of three weeks.

**Scope creep.** You will be tempted toward multi-table relational synthesis, differential
privacy guarantees, and a web UI. All three are real products and none of them is this one.
The entire value proposition is that it is small enough to run on every pull request. Multi-table
support is the only one worth considering, and only for `v0.2` after real users ask for it.
