"""The `synth_frame` fixture."""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from synthkit.profile import Profile


@cache
def _load_profile_cached(path: str, mtime_ns: int) -> Profile:
    return Profile.load(path)


def _load_profile(path: str) -> Profile:
    # Keying on mtime as well as path means a profile overwritten mid test-session
    # invalidates the cache instead of serving stale data.
    mtime_ns = Path(path).stat().st_mtime_ns
    return _load_profile_cached(path, mtime_ns)


@pytest.fixture
def synth_frame():
    """Returns a callable: `synth_frame(profile_path, n, seed) -> pd.DataFrame`.

    Profiles are cached per (path, mtime) for the test session, so calling this fixture
    repeatedly with the same untouched profile across many tests doesn't re-parse the same
    JSON file each time, while a profile overwritten mid-session is picked up fresh.
    """

    def _emit(profile_path: str | Path, n: int, seed: int) -> pd.DataFrame:
        profile = _load_profile(str(profile_path))
        return profile.emit(n=n, seed=seed)

    return _emit
