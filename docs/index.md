# synthkit

**pytest fixtures that look like your production data, without your production data.**

Read a real dataset once. Learn its statistical shape — each column's distribution, the
dependence structure between columns, null patterns, business rules. Save that shape as a
small JSON profile containing no real records, which you commit to your repo. Generate as many
synthetic rows as you want from that profile, forever, without touching the real data again.

See the [project README](https://github.com/grizz6/synthkit#readme) for install instructions,
usage examples, and measured results, [Limitations](LIMITATIONS.md) for known weaknesses, and
[Plan](PLAN.md) for the original build plan this project follows.
