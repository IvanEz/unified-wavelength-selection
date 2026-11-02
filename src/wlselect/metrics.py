"""Error and orthogonality metrics used by the selection criteria."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "nrmse",
    "rmse",
    "log_sv_product",
    "sv_product",
    "min_singular_value",
    "sv_sum",
    "condition_number",
]


def nrmse(y_true: ArrayLike, y_pred: ArrayLike, *, eps: float = 1e-12) -> float:
    """Normalised root-mean-squared error, as in Arifler et al. (2015).

    For each chromophore (row) the error is normalised by the RMS of the
    reference time course; the result is averaged over chromophores::

        NRMSE = mean_k sqrt( Σ_t (ŷ_kt - y_kt)² / Σ_t y_kt² )

    Args:
        y_true: Reference concentrations ``(n_chromophores, n_timepoints)``.
        y_pred: Predicted concentrations, same shape.
    """
    y_true = np.atleast_2d(np.asarray(y_true, dtype=float))
    y_pred = np.atleast_2d(np.asarray(y_pred, dtype=float))
    num = np.sum((y_pred - y_true) ** 2, axis=-1)
    den = np.maximum(np.sum(y_true**2, axis=-1), eps)
    return float(np.mean(np.sqrt(num / den)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root-mean-squared error over all entries."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _svdvals(M: ArrayLike) -> NDArray:
    return np.linalg.svd(np.asarray(M, dtype=float), compute_uv=False)


def log_sv_product(M: ArrayLike, *, eps: float = 1e-300) -> float:
    """Natural log of the product of singular values (numerically stable)."""
    return float(np.sum(np.log(np.maximum(_svdvals(M), eps))))


def sv_product(M: ArrayLike) -> float:
    """Product of singular values (Marois et al., 2018). May overflow for big matrices."""
    return float(np.prod(_svdvals(M)))


def min_singular_value(M: ArrayLike) -> float:
    """Smallest singular value."""
    return float(np.min(_svdvals(M)))


def sv_sum(M: ArrayLike) -> float:
    """Sum of singular values (nuclear norm)."""
    return float(np.sum(_svdvals(M)))


def condition_number(M: ArrayLike) -> float:
    """2-norm condition number."""
    return float(np.linalg.cond(np.asarray(M, dtype=float)))
