# Contributing

```bash
git clone https://github.com/grizz6/synthkit
cd synthkit
python -m venv .venv && source .venv/bin/activate
make install
make check   # lint + typecheck + both test suites
```

## Before opening a PR

- `make check` passes.
- New behavior has a test. This project leans on its test suite instead of manual
  verification — see `tests/` for the style (small, focused, one behavior per test).
- If you touched a public-facing surface (`synthkit.fit/emit/check`, the CLI, the profile JSON
  schema), update the relevant section of [README.md](README.md).
- If you're documenting a real limitation rather than fixing it, it belongs in
  [docs/LIMITATIONS.md](docs/LIMITATIONS.md), not a code comment.

## Project layout

- `src/synthkit/` — the library.
- `pytest-synthkit/` — the companion pytest plugin, a separate installable package.
- `examples/` — runnable, narrative examples against real public data.
- `scripts/` — dev tooling (currently just the benchmark).
- `docs/` — the MkDocs site source, plus `docs/PLAN.md`, the original build plan.
