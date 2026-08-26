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
