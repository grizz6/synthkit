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

Not yet published to PyPI. No version has been tagged.
