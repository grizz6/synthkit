# pytest-synthkit

A pytest plugin exposing a `synth_frame` fixture backed by [synthkit](../README.md) profiles.

```python
def test_billing_rollup(synth_frame):
    df = synth_frame("profiles/customers.json", n=500, seed=0)
    assert rollup(df).total.sum() == pytest.approx(df.total.sum())
```

No installation configuration needed beyond `pip install pytest-synthkit` — pytest discovers
the fixture automatically via its plugin entry point.

## Status

Early scaffolding, shipped alongside synthkit itself. See the parent
[README](../README.md) and [docs/PLAN.md](../docs/PLAN.md) for context.
