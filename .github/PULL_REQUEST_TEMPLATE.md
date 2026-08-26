**What this changes and why**

**How it was tested**

- [ ] `pytest tests/` passes
- [ ] `ruff check .` and `mypy src/synthkit` pass
- [ ] If this touches fitting or sampling: a fidelity check (KS statistic, correlation error)
  showing the change doesn't regress output quality

**Checklist**

- [ ] New behavior has a test
- [ ] `CHANGELOG.md` updated for a user-visible change
- [ ] Docs (`README.md` / `docs/LIMITATIONS.md`) updated if this changes a documented
  guarantee or adds a new limitation
