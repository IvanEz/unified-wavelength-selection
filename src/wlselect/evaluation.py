"""Utilities to evaluate and compare selected wavelength subsets."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .criteria import (
    ConcentrationPreservation,
    DataOrthogonality,
    ModelOrthogonality,
    ProjectionMaximization,
    SpectralFitPreservation,
    _as_groups,
)

__all__ = ["evaluate_subset", "subset_fit", "max_wavelength_distance"]


def evaluate_subset(
    subset: ArrayLike,
    M: ArrayLike,
    delta_A: ArrayLike | Sequence[ArrayLike],
    *,
    metric: str = "log_sv_product",
) -> dict[str, float]:
    """Score one subset under all five criteria.

    Useful to see how a subset optimised for one criterion performs under the
    others (e.g. Fig. 3/5 of the paper report concentration NRMSE *and*
    spectral RMSE for the same subsets).

    Returns:
        ``{"cp": NRMSE, "sfp": RMSE, "pm": RMSE, "mo": O(Σ), "do": O(Σ)}``.
    """
    subset = np.asarray(subset, dtype=int)
    return {
        "cp": ConcentrationPreservation(M, delta_A)(subset),
        "sfp": SpectralFitPreservation(M, delta_A)(subset),
        "pm": ProjectionMaximization(M, delta_A)(subset),
        "mo": ModelOrthogonality(M, metric=metric)(subset),
        "do": DataOrthogonality(delta_A=delta_A, metric=metric)(subset),
    }


def subset_fit(
    subset: ArrayLike,
    M: ArrayLike,
    delta_A: ArrayLike | Sequence[ArrayLike],
) -> dict[str, NDArray]:
    """Concentrations and modelled attenuations from all vs. selected wavelengths.

    Returns a dict with

    * ``c_full``  – ``M⁺ ΔA`` (all wavelengths), shape ``(m, T)``
    * ``c_subset`` – ``M_Z⁺ ΔA_Z``, shape ``(m, T)``
    * ``A_full``  – ``M c_full``, shape ``(n, T)``
    * ``A_subset`` – ``M c_subset`` (full-range reconstruction), shape ``(n, T)``
    * ``A_measured`` – the measured ``ΔA``, shape ``(n, T)``
    """
    M = np.asarray(M, dtype=float)
    A, _ = _as_groups(delta_A)
    subset = np.asarray(subset, dtype=int)
    c_full = np.linalg.pinv(M) @ A
    c_sub = np.linalg.pinv(M[subset]) @ A[subset]
    return {
        "c_full": c_full,
        "c_subset": c_sub,
        "A_full": M @ c_full,
        "A_subset": M @ c_sub,
        "A_measured": A,
    }


def max_wavelength_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Largest distance between matched wavelengths of two sorted, equal-size sets.

    The paper uses this to compare sets found by different criteria
    (e.g. "absolute maximal distance of not greater than 3.4 nm").
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.shape != b.shape:
        raise ValueError("Sets must have the same size.")
    return float(np.max(np.abs(a - b))) if a.size else 0.0
