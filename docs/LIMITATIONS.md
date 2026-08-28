# Limitations

Honest documentation of a known weakness is a credibility signal; discovering it during a
demo is not. This page collects what synthkit does not do well.

## Timezone-aware datetime columns lose their timezone

A datetime column's timezone is converted to UTC and discarded at fit time; every emitted
value comes back as a naive UTC timestamp, never in the original zone. For a fixed-UTC-offset
timezone this is otherwise harmless — daily data recorded at local midnight is still detected
and reproduced as daily, just at whatever UTC time that midnight falls at (e.g. 05:00 UTC for
US/Eastern in winter). It stops being harmless across a DST transition: local midnight lands
at a different UTC time on either side of the transition, so the granularity detection
correctly (if disappointingly) falls back to hourly rather than daily for a range that spans
one, because the data genuinely isn't daily-and-only-daily once expressed in UTC. If your
fixtures need the original timezone preserved, this is a real gap, not a subtle one.

## The copula captures rank correlation, not arbitrary dependence

The Gaussian copula reproduces the *rank* correlation structure between columns. It will not
reproduce a tight nonlinear relationship (`y = x^2`), a conditional mode (a mixture that only
appears within one subgroup), or any dependence structure that rank correlation doesn't
capture. If a downstream test depends on a nonlinear relationship between two columns, verify
it explicitly rather than assuming the copula preserved it.

## Categorical association is approximate

Associating two unordered categorical columns requires imposing *some* ordering on their
categories so each can be mapped onto a copula interval. synthkit orders by descending
frequency with a stable tiebreak. This is an arbitrary choice that works well in practice, but
it is an assumption: the true association between two nominal categoricals is captured only
approximately, not exactly, the way it is for two numeric columns.

## Free text is word-bag resampling, not language modeling

`TEXT`-classified columns are approximated by resampling words from the observed vocabulary,
preserving rough sentence length and nothing else — no grammar, no semantics, no per-row
coherence. This is a deliberate choice: reproducing an exact original sentence would defeat
the point of the tool, and generating real free text is out of scope for a fixture library
built around numeric/categorical marginals. Free-text columns containing PII (names, notes,
complaint text) should be reviewed by a human before a profile fit on them is shared —
word-level resampling is *not* a privacy guarantee on its own, only a rough shape-preserving
approximation.

## Identifiers are regenerated, not modeled — by design

Per the project's own rule, an identifier's *value* is never statistically modeled, only its
*format*. A uniform sequential or alphanumeric pattern is detected and regenerated; a mixed or
inconsistent identifier format falls back to random tokens of similar length. This means two
identifier columns that were correlated in the original data (a customer ID that determines an
order-number prefix, say) will not be correlated in the output — identifiers are sampled
independently of everything else, including each other.

## Foreign keys require an explicit key pool

The `foreign_key` constraint validates against a `key_pools` dict supplied at emit time; without
one, the column is left as whatever the copula/marginal happened to sample and is *not*
guaranteed to reference a valid parent key. True multi-table relational integrity (a `Profile`
for `customers` linked to a `Profile` for `orders`) is out of scope for `v0.1` — see the
project's own scope notes on multi-table support being the only extension worth considering,
and only after real users ask for it.

## `Derived` and `ConditionalNull` expressions follow pandas' eval syntax, not yours

Both are evaluated with `DataFrame.eval`, which parses `expr`/`null_when` as a Python-like
expression rather than treating column names as opaque strings. A column named `sub-total`
in `Derived("total", "sub-total + tax")` parses as the subtraction `sub - total` — not a
reference to the `sub-total` column — and raises `UndefinedVariableError` if there's no
column literally named `sub`. `Inequality` constraints are unaffected (they compare
`df[left]`/`df[right]` directly, never through `eval`), but any column name that isn't a
valid Python identifier (a hyphen, a space, a leading digit) needs pandas' own escape hatch
when it appears inside a `Derived` or `ConditionalNull` expression: wrap it in backticks,
e.g. `` "`sub-total` + tax" ``.

## Rare-combination leaks are scored pairwise, and small datasets still show plenty

`privacy.check()` flags a synthetic row when *some pair* of categorical columns reproduces a
combination that was rare (fewer than `--rare-combination-threshold`, default 5) in the real
data. Pairs, not the full cross product of every categorical column: scoring all categorical
columns jointly makes almost every combination "rare" purely from dimensionality — on the
Titanic dataset (6 categorical columns, 891 rows), 123 of 160 distinct 6-way combinations
occur fewer than 5 times, which would flag most rows regardless of actual re-identification
risk (confirmed empirically: see [examples/titanic_example.py](../examples/titanic_example.py)).

Pairwise scoring is a real fix for that combinatorial explosion, but it doesn't eliminate the
underlying tension: a small real dataset with several categorical columns often *genuinely*
has many rare pairwise strata (age bracket × embarkation port with only 3 real passengers, say),
and a profile that preserves those strata's true frequency will keep reproducing them in
synthetic output. On Titanic, this still flags roughly 9-10% of synthetic rows at the default
threshold. That's an honest measurement of a real dataset's sparsity, not a false positive to
suppress — loosen `rare_combination_threshold` (or judge by `dcr_ratio` alone) if that's an
expected trade-off for your data, rather than treating a nonzero count as automatically wrong.

## Small datasets skip the copula entirely

If fewer than 10 complete rows remain after dropping nulls across the copula-eligible columns,
the copula is skipped and every column is sampled independently instead. Correlations will not
be preserved for such a small or sparse dataset; this is a deliberate fallback rather than a
bug, since fitting a correlation matrix on a handful of rows would be closer to noise than
signal.

## Round-tripping through a low-cardinality numeric column

A numeric column with fewer than 20 distinct values (a 0/1 flag, a 1-5 rating) is classified as
categorical rather than continuous, per `types.py`'s heuristics. Its emitted dtype is restored
to int/float/bool automatically, but the *value set* is fixed to exactly what was observed —
a rating column that never saw a `5` in the fitted data will never emit a `5`.

## Strict inequality repair can't nudge every dtype apart

A strict constraint (`<` or `>`) that still has a tied row after the swap-based repair gets
the tie broken by nudging one side by the smallest step the dtype supports: ±1 for integers,
`np.nextafter` for floats, one second for datetimes. For any other dtype (most notably
strings), there's no well-defined "smallest step," so a tie is left unresolved rather than
guessed at — a strict inequality constraint declared between two string columns may still
contain equal values after repair.

## `check()`'s memory scales with dataset size — measured, not assumed

`fit()` and `emit()` are fast and memory-light at real scale (50,000 rows, 6 columns: ~325ms
to fit, 10.2 MB peak — see `scripts/benchmark.py`). `check()` is the one that costs something,
because its Gower distance computation is a pairwise comparison across every query row and
every reference row.

The query side is batched (`distance_to_closest_record`'s `batch_size`, default 500) so peak
memory no longer scales with the *synthetic* row count — before that existed, `check()` was
measured hitting **2.2 GB at a mere 10,000 synthetic rows** against an 8,000-row real holdout
(three columns), which would have made "tens of thousands of rows" (the earlier, wrong,
unmeasured claim on this page) cost tens of gigabytes. With batching, the same comparison
across several sizes:

| Rows (synthetic ≈ real) | `check()` time | Peak memory |
|---|---|---|
| 5,000 | 0.5 s | 56 MB |
| 10,000 | 1.7 s | 111 MB |
| 20,000 | 6.5 s | 222 MB |
| 50,000 | 40 s | 556 MB |

Memory now scales linearly with row count (not quadratically) because peak usage is
`batch_size × len(reference)`, and `len(reference)` itself grows with the dataset. Reference
side is *not* batched — a `check()` against a real dataset of a few hundred thousand rows will
still take real memory and time, and a million-row `check()` needs a nearest-neighbor index
(not implemented) to be practical, not just query-side batching.
