"""The `synth_frame` fixture.

Distribution multiplier, per the project plan: people find pytest plugins by searching for
pytest plugins, not by searching for the library that happens to back them.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import pandas as pd
import pytest

from synthkit.profile import Profile


@cache
def _load_profile(path: str) -> Profile:
    return Profile.load(path)


@pytest.fixture
def synth_frame():
    """Returns a callable: `synth_frame(profile_path, n, seed) -> pd.DataFrame`.

    Profiles are cached per path for the test session, so calling this fixture repeatedly
    with the same profile across many tests doesn't re-parse the same JSON file each time.
    """

    def _emit(profile_path: str | Path, n: int, seed: int) -> pd.DataFrame:
        profile = _load_profile(str(profile_path))
        return profile.emit(n=n, seed=seed)

    return _emit
