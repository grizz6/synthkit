# Changelog

## Unreleased

Initial implementation of the core pipeline described in [docs/PLAN.md](docs/PLAN.md):

- Column type inference (`types.py`).
- Non-parametric marginals for numeric, categorical, boolean, datetime, identifier, and text
  columns (`marginals.py`).
- Joint null-pattern modeling to capture null co-occurrence (`nulls.py`).
- A Gaussian copula tying marginals together via rank correlation (`copula.py`).
- `Profile.fit` / `Profile.emit`, with canonical JSON serialization for a diffable,
  byte-identical-round-trip profile format (`profile.py`, `serialization.py`).
- A constraints DSL and repair engine, including the inequality swap trick
  (`constraints.py`, `repair.py`).
- A privacy check based on distance-to-closest-record against a holdout baseline
  (`privacy.py`).
- Drift detection comparing fresh data against a profile's stored marginals (`drift.py`).
- A CLI (`synthkit fit / emit / check / diff`) and a top-level Python API
  (`synthkit.fit / emit / check`).
- The `pytest-synthkit` companion plugin with a `synth_frame` fixture.
- A worked example on the UCI Adult Census dataset (`examples/`) and a benchmark script
  (`scripts/benchmark.py`).

### Fixed

- `synthkit check --rare-combination-threshold` was parsed by the CLI but never reached
  `privacy.check()`, so it silently had no effect regardless of the value passed.
- `pytest-synthkit`'s `synth_frame` fixture cached a loaded `Profile` by file path alone for
  the whole test session; a profile re-fit and overwritten mid-session was served stale from
  cache. The cache now also keys on the file's mtime.
- A small (<10 row) all-unique string column was classified categorical instead of
  identifier, storing every real value verbatim as a category and reproducing it in emitted
  output — a privacy leak on small datasets. The row-count floor on identifier detection was
  removed.
- A constant (zero-variance) column included in copula fitting drove the correlation matrix
  to NaN and crashed `Profile.fit` with a `LinAlgError`. Such columns are now excluded from
  the copula and sampled independently instead.
- A strict inequality constraint (`<` / `>`) didn't flag a tie (equal values) as a violation,
  so ties passed through unrepaired. Ties are now detected and nudged apart by the smallest
  step the column's dtype supports.
- A multi-column `Unique` constraint overwrote every listed column for every row with a
  fabricated token, discarding realistic sampled values wholesale. It now only disambiguates
  rows whose combination actually collides, by suffixing the last column.
- `IdentifierMarginal`'s sequential style drew a random start offset with no regard for the
  fitted digit width, so a 3-digit source format (`CUST001`) could emit as a 6-digit blowout
  (`CUST483920`). The start is now bounded to keep the original width whenever `n` allows.
- Drift detection double-counted rare categories that a profile had pooled into `__other__`,
  inflating the drift score for an unchanged tail distribution and risking a false-positive
  drift alert. Current data is now bucketed into `__other__` the same way before comparing.
- A categorical marginal's tail-bucketing (`__other__`) disabled the int/float/bool dtype
  cast for the *entire* sampled column, not just the bucketed rows, when reachable via an
  explicit `column_types` override on a high-cardinality numeric column. Non-`__other__` rows
  are now cast correctly; `__other__` rows become `NaN`.
- `privacy.check()`'s rare-combination check scored every categorical column jointly, which
  makes almost every combination "rare" purely from dimensionality (confirmed on Titanic: 123
  of 160 six-way combinations occur under the default threshold). Now scored pairwise instead.
- **Severe:** a low-cardinality numeric column (a 1-5 rating, a small count — classified
  categorical per `types.py`'s heuristics) was ordered by descending frequency for the
  copula, same as a genuinely nominal categorical. Since frequency order rarely matches
  numeric order, this silently destroyed the column's correlation with everything else —
  confirmed on a constructed case where a real correlation of 0.96 came back as 0.02, and on
  Wine Quality's `quality` rating (see `examples/wine_quality_example.py`). Numeric-valued
  categoricals are now ordered by their actual value; genuinely nominal ones (strings) are
  unaffected and still ordered by frequency.

- `synthkit.check()`'s per-column KS fidelity score for datetime columns cast each side to
  `int64` directly with no unit normalization; real data parsed via `pd.to_datetime` (e.g.
  from a CSV) is typically `datetime64[us]` while `DatetimeMarginal.sample` always emits
  `datetime64[s]`, so the two sides differed by a factor of a million and `ks_2samp` reported
  the distributions as completely disjoint (statistic `1.0`) regardless of actual fidelity.
  Confirmed on the Bike Sharing dataset's `dteday` column. Both sides now normalize through
  `datetime64[s]` first, matching the pattern already used correctly elsewhere in the
  codebase (`profile.py`, `drift.py`, `privacy.py`).
- `privacy.check()`'s `dcr_ratio` divided by a fallback epsilon whenever the holdout
  baseline's 5th-percentile DCR was exactly 0 — which happens naturally on a dataset without
  enough entropy across its columns for even real, never-trained-on holdout rows to come out
  distinct (a handful of low-cardinality columns over many rows, e.g. a 5-column slice of
  Adult Census: exact duplicate rows occur in real data too). When synthetic data hit the
  same natural floor, the reported ratio was `0.0`, reading as a hard privacy failure it
  wasn't. Now reports the neutral `1.0` ("exactly as duplicated as real data already is")
  when both sides are degenerately zero.

Also added `examples/notebooks/` — the same four worked examples as executed Jupyter
notebooks (self-contained, real output cells), completing the "three example notebooks" item
from `docs/PLAN.md`'s Day 9. Building and actually executing them (rather than just writing
them) is what surfaced the `dcr_ratio` bug above.

### Performance

- `privacy.check()`'s Gower distance computation materialized the full query × reference
  matrix at once. Measured directly: 10,000 synthetic rows against an 8,000-row real holdout
  (three columns) peaked at **2.2 GB** — `docs/LIMITATIONS.md` previously (and wrongly, since
  it was never actually measured) claimed this was fine up to "tens of thousands of rows,"
  which would have meant tens of gigabytes. `distance_to_closest_record` now batches the
  query side (`batch_size`, default 500), cutting the same 10,000-row case to 111 MB — a 20x
  reduction — with memory now scaling linearly with row count instead of quadratically.

  Batching surfaced a second, more serious bug before it shipped: each batch's Gower distance
  was normalizing against a range computed from just that batch's slice of data, not the full
  dataset, so batched and unbatched results differed by a few percent — a real correctness
  bug, not a rounding difference (caught by a test that compares the two directly). Fixed by
  computing each column's normalization range once, over the full query + reference data,
  and passing it into every batch (`compute_value_ranges`).

### Noted, not yet resolved

- `synthkit` is already taken on PyPI (an unrelated package, currently at v0.6.2). This is
  exactly the check `docs/PLAN.md`'s own Day -1 calls for before writing any code, done late.
  `pytest-synthkit` and the plan's other candidate names (`shapeshift`, `replikit`,
  `mimicdata`) are all still available. Decision: keep the name for now and revisit before an
  actual PyPI publish, rather than rename mid-development.
- Added `.github/workflows/docs.yml` to build and deploy the MkDocs site to GitHub Pages —
  wired up but not fired, same as `release.yml`: it only actually publishes once GitHub Pages
  is turned on for this repo (Settings → Pages → Source: "GitHub Actions"), which is a repo
  setting change left for a deliberate decision rather than flipped automatically.

Not yet published to PyPI. No version has been tagged.
