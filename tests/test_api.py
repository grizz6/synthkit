import numpy as np
import pandas as pd

import synthkit as sk


def make_df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "created_at": rng.integers(0, 1000, n).astype(float),
            "updated_at": rng.integers(1000, 2000, n).astype(float),
            "amount": rng.normal(100, 20, n),
        }
    )


def test_public_api_matches_documented_usage():
    df = make_df()
    profile = sk.fit(df, constraints=[sk.Inequality("created_at", "<=", "updated_at")])
    synthetic = sk.emit(profile, n=1000, seed=42)

    assert (synthetic["created_at"] <= synthetic["updated_at"]).all()

    report = sk.check(synthetic, profile, real=df)
    assert isinstance(report, sk.CheckReport)
    assert "amount" in report.ks_by_column
    assert report.ks_by_column["amount"] < 0.15


def test_profile_save_load_accessible_from_top_level(tmp_path):
    df = make_df(n=300)
    profile = sk.fit(df)
    path = tmp_path / "p.json"
    profile.save(path)
    reloaded = sk.Profile.load(path)
    assert reloaded.columns == profile.columns
