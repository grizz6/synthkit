import pandas as pd

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
