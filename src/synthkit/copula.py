"""The Gaussian copula: ties per-column marginals together via rank correlation.

Separates what each column looks like (handled by marginals.py) from how the columns move
together, models the latter with a single correlation matrix over normal scores, and
recombines at sampling time. This is what makes the output preserve joint structure that
sampling each column independently would throw away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata

from synthkit.marginals import CategoricalMarginal

EIGENVALUE_FLOOR = 1e-6
UNIFORM_EPSILON = 1e-6


def rank_transform_to_uniform(values: np.ndarray) -> np.ndarray:
    """Convert a numeric array to uniform pseudo-observations via `rank / (n + 1)`.

    Dividing by `n + 1` rather than `n` keeps every value strictly inside `(0, 1)`, which
    matters because the next step takes `Phi^-1` of it and `Phi^-1(1)` is infinite.
    """
    n = len(values)
    ranks = rankdata(values, method="average")
    return ranks / (n + 1)


def category_pseudo_uniform(values: np.ndarray, marginal: CategoricalMarginal) -> np.ndarray:
    """Map categorical values to a point inside their frequency-ordered interval.

    Sampling later goes uniform -> category by walking the same ordered intervals
    (`CategoricalMarginal.sample`), so fitting must go category -> uniform through the
    interval's midpoint to be the consistent inverse of that mapping.
    """
    cumulative = np.cumsum([0.0, *marginal.probabilities])
    midpoints = {
        category: (cumulative[i] + cumulative[i + 1]) / 2
        for i, category in enumerate(marginal.categories)
    }
    other_midpoint = midpoints.get("__other__", 0.5)
    return np.array(
        [midpoints.get(str(v), other_midpoint) for v in values],
        dtype=float,
    )


def nearest_pd_correlation(corr: np.ndarray) -> np.ndarray:
    """Project a correlation matrix to the nearest positive-definite one.

    Ties in ranks and pairwise (rather than listwise) estimation routinely leave the raw
    empirical correlation matrix not quite positive definite, which makes the Cholesky
    factorization used for sampling fail. Clipping negative eigenvalues to a small positive
    floor and renormalizing the diagonal back to 1 fixes that with a minimal perturbation.
    """
    symmetric = (corr + corr.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    clipped = np.clip(eigenvalues, EIGENVALUE_FLOOR, None)
    reconstructed = eigenvectors @ np.diag(clipped) @ eigenvectors.T

    diagonal_sqrt = np.sqrt(np.diag(reconstructed))
    normalized = reconstructed / np.outer(diagonal_sqrt, diagonal_sqrt)
    np.fill_diagonal(normalized, 1.0)
    return normalized


@dataclass
class GaussianCopula:
    """A correlation matrix over normal scores, fit on and sampling uniform pseudo-observations."""

    columns: list[str]
    correlation: list[list[float]]

    @classmethod
    def fit(cls, uniform_columns: dict[str, np.ndarray]) -> GaussianCopula:
        columns = list(uniform_columns)
        clipped = {
            col: np.clip(u, UNIFORM_EPSILON, 1 - UNIFORM_EPSILON)
            for col, u in uniform_columns.items()
        }
        z = np.column_stack([norm.ppf(clipped[col]) for col in columns])

        if len(columns) == 1:
            correlation = np.array([[1.0]])
        else:
            correlation = np.corrcoef(z, rowvar=False)
            correlation = nearest_pd_correlation(correlation)

        return cls(columns=columns, correlation=correlation.tolist())

    def sample(self, n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        correlation = np.array(self.correlation)
        k = len(self.columns)
        cholesky = np.linalg.cholesky(correlation)

        z = rng.standard_normal((n, k)) @ cholesky.T
        u = norm.cdf(z)

        return {col: u[:, i] for i, col in enumerate(self.columns)}

    def to_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "correlation": self.correlation}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GaussianCopula:
        return cls(columns=data["columns"], correlation=data["correlation"])
