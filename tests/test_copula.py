import numpy as np
import pandas as pd

from synthkit.copula import (
    GaussianCopula,
    category_pseudo_uniform,
    nearest_pd_correlation,
    rank_transform_to_uniform,
)
from synthkit.marginals import CategoricalMarginal


def test_rank_transform_is_monotonic_and_bounded():
    values = np.array([5, 1, 3, 2, 4])
    u = rank_transform_to_uniform(values)
    assert (u > 0).all() and (u < 1).all()
    order = np.argsort(values)
    assert np.all(np.diff(u[order]) > 0)


def test_nearest_pd_correlation_fixes_indefinite_matrix():
    # A matrix that is symmetric with unit diagonal but not positive definite.
    bad = np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]])
    fixed = nearest_pd_correlation(bad)
    eigenvalues = np.linalg.eigvalsh(fixed)
    assert (eigenvalues > 0).all()
    assert np.allclose(np.diag(fixed), 1.0)
    # Must not raise.
    np.linalg.cholesky(fixed)


def test_nearest_pd_correlation_is_near_identity_for_already_pd_matrix():
    good = np.array([[1.0, 0.3], [0.3, 1.0]])
    fixed = nearest_pd_correlation(good)
    assert np.allclose(fixed, good, atol=1e-8)


def test_category_pseudo_uniform_orders_by_frequency():
    values = pd.Series(["a"] * 10 + ["b"] * 60 + ["c"] * 30)
    marginal = CategoricalMarginal.fit(values)
    u = category_pseudo_uniform(values.to_numpy(), marginal)
    # b is the most frequent category and should sit first in [0, 1).
    assert u[values == "b"][0] < u[values == "c"][0] < u[values == "a"][0]


def test_copula_recovers_correlation_structure():
    rng = np.random.default_rng(42)
    n = 5000
    true_corr = np.array([[1.0, 0.7], [0.7, 1.0]])
    z = rng.multivariate_normal(mean=[0, 0], cov=true_corr, size=n)

    u1 = rank_transform_to_uniform(z[:, 0])
    u2 = rank_transform_to_uniform(z[:, 1])

    copula = GaussianCopula.fit({"x": u1, "y": u2})
    fitted_corr = np.array(copula.correlation)
    assert abs(fitted_corr[0, 1] - 0.7) < 0.05

    sampled = copula.sample(n, np.random.default_rng(1))
    sampled_z = np.column_stack(
        [norm_ppf_safe(sampled["x"]), norm_ppf_safe(sampled["y"])]
    )
    sampled_corr = np.corrcoef(sampled_z, rowvar=False)
    assert abs(sampled_corr[0, 1] - 0.7) < 0.07


def test_copula_handles_single_column():
    u = rank_transform_to_uniform(np.arange(100))
    copula = GaussianCopula.fit({"only": u})
    sampled = copula.sample(50, np.random.default_rng(0))
    assert "only" in sampled
    assert len(sampled["only"]) == 50


def test_copula_round_trip():
    rng = np.random.default_rng(0)
    u1 = rank_transform_to_uniform(rng.normal(size=200))
    u2 = rank_transform_to_uniform(rng.normal(size=200))
    copula = GaussianCopula.fit({"a": u1, "b": u2})
    restored = GaussianCopula.from_dict(copula.to_dict())
    assert restored.correlation == copula.correlation
    assert restored.columns == copula.columns


def norm_ppf_safe(u: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
