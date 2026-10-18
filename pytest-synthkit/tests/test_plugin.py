import os

import pandas as pd
from pytest_synthkit.plugin import _load_profile

import synthkit as sk


def _write_profile(path):
    df = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0, 5.0] * 20})
    profile = sk.fit(df)
    profile.save(path)


def test_synth_frame_fixture_is_discovered_and_emits_rows(pytester):
    profile_path = pytester.path / "profile.json"
    _write_profile(profile_path)

    pytester.makepyfile(
        f"""
        def test_uses_synth_frame(synth_frame):
            df = synth_frame({str(profile_path)!r}, n=25, seed=0)
            assert len(df) == 25
            assert "amount" in df.columns
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_load_profile_cache_invalidates_when_file_is_overwritten(tmp_path):
    # Regression test: _load_profile used to be cached on path alone, so a profile re-fit and
    # saved over the same path mid test-session would silently keep serving the stale one.
    path = tmp_path / "profile.json"

    _write_profile(path)
    first = _load_profile(str(path))
    assert first.columns == ["amount"]

    other_df = pd.DataFrame({"other_col": ["x", "y", "z"] * 20})
    sk.fit(other_df).save(path)
    # Ensure the mtime actually advances even on filesystems with coarse timestamp resolution.
    stat = os.stat(path)
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))

    second = _load_profile(str(path))
    assert second.columns == ["other_col"]


def test_synth_frame_is_deterministic_given_same_seed(pytester):
    profile_path = pytester.path / "profile.json"
    _write_profile(profile_path)

    pytester.makepyfile(
        f"""
        def test_same_seed_same_data(synth_frame):
            a = synth_frame({str(profile_path)!r}, n=10, seed=1)
            b = synth_frame({str(profile_path)!r}, n=10, seed=1)
            assert a.equals(b)
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_synth_frame_draws_foreign_keys_from_a_parent_fixture(pytester):
    # Child rows should join to the parent fixture's ids, not to whatever ids the real data
    # happened to contain when the child profile was fit.
    customers_path = pytester.path / "customers.json"
    sk.fit(pd.DataFrame({"id": range(1, 101), "age": [30, 40, 50, 60] * 25})).save(customers_path)
    orders = pd.DataFrame({"customer_id": [1, 2, 3, 4] * 25, "total": [9.5, 20.0] * 50})
    orders_path = pytester.path / "orders.json"
    sk.fit(orders, constraints=[sk.ForeignKey("customer_id", "customers.id")]).save(orders_path)

    pytester.makepyfile(
        f"""
        def test_orders_join_to_customers(synth_frame):
            customers = synth_frame({str(customers_path)!r}, n=20, seed=0)
            pool = customers["id"].tolist()
            orders = synth_frame(
                {str(orders_path)!r}, n=200, seed=0, key_pools={{"customer_id": pool}}
            )
            assert set(orders["customer_id"]) <= set(pool)
        """
    )

    result = pytester.runpytest()
    result.assert_outcomes(passed=1)
