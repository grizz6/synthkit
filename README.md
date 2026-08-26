# synthkit

**pytest fixtures that look like your production data, without your production data.**

> Status: early scaffolding. Not yet installable. See [`docs/PLAN.md`](docs/PLAN.md) for the
> full build plan.

## The problem

Teams hold data with real people in it and can't put those rows in a test suite, a demo
environment, or a bug report. The usual fallbacks are all bad: Faker (statistically
meaningless), hand-written fixtures (stale in a month), scrubbed production copies (legally
fraught and rarely actually anonymous), or nothing at all.

## What this does

Read a real dataset once. Learn its statistical shape — each column's distribution, the
dependence structure between columns, null patterns, business rules. Save that shape as a
small JSON profile containing **no real records**, which you commit to your repo. Generate as
many synthetic rows as you want from that profile, forever, without touching the real data
again.

```
real data  ──fit──►  profile.json  ──emit──►  synthetic rows
(once, locally)      (committed)              (in CI, forever)
```

## Relationship to SDV

[SDV](https://github.com/sdv-dev/SDV) already exists and does this well for research-grade
synthetic datasets. synthkit is aimed at a narrower job — test fixtures — which means
deterministic output, sub-second generation, a small diffable profile format, and a built-in
privacy check, rather than trained generative models.

## Status

Nothing is implemented yet. This repo currently holds scaffolding only.

## License

MIT — see [LICENSE](LICENSE).
