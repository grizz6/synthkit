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

Not yet published to PyPI. No version has been tagged.
