# Limitations

Honest documentation of a known weakness is a credibility signal; discovering it during a
demo is not. This page collects what synthkit does not do well.

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

## Performance is not benchmarked past hundreds of thousands of rows

The privacy check's Gower distance computation is a full pairwise comparison — O(n × m) in the
number of rows on each side — which is fine for the fixture-sized data (thousands to tens of
thousands of rows) this package targets, but will not scale to a million-row `check` without a
nearest-neighbor index, which is not implemented.
