"""Differential modified Beer-Lambert law (dMBLL).

Conventions used throughout the package
---------------------------------------
* ``M``       – absorption matrix of shape ``(n_wavelengths, n_chromophores)``.
  Row ``i`` holds the specific absorption coefficients of all chromophores at
  wavelength ``i`` (optionally multiplied by the optical pathlength ``PL(λ)``).
* ``delta_A`` – attenuation changes of shape ``(n_wavelengths, n_timepoints)``.
  Every column is one measured spectrum ``ΔA(t)``.
* ``delta_c`` – concentration changes of shape ``(n_chromophores, n_timepoints)``.

With these conventions the dMBLL reads ``ΔA = M · Δc`` (paper, Eq. 2) and the
least-squares solution is ``Δc = M⁺ · ΔA`` (paper, Eq. 3).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "attenuation_change",
    "infer_concentrations",
    "forward_model",
]


def attenuation_change(
    intensities: ArrayLike,
    reference: ArrayLike | None = None,
    *,
    eps: float = 1e-12,
) -> NDArray:
    """Compute attenuation changes ``ΔA = -log(I(t) / I(t0))`` (paper, Eq. 1).

    Args:
        intensities: Measured (dark/white corrected) intensities of shape
            ``(n_wavelengths, n_timepoints)``.
        reference: Baseline spectrum ``I(t0)`` of shape ``(n_wavelengths,)``.
            Defaults to the first column of ``intensities``.
        eps: Lower clip to avoid ``log(0)``.

    Returns:
        ``ΔA`` with the same shape as ``intensities`` (natural logarithm).
    """
    intensities = np.asarray(intensities, dtype=float)
    if intensities.ndim == 1:
        intensities = intensities[:, None]
    if reference is None:
        reference = intensities[:, 0]
    reference = np.asarray(reference, dtype=float).reshape(-1, 1)
    ratio = np.clip(intensities, eps, None) / np.clip(reference, eps, None)
    return -np.log(ratio)


def infer_concentrations(delta_A: ArrayLike, M: ArrayLike) -> NDArray:
    """Infer concentration changes ``Δc = M⁺ ΔA`` via the Moore-Penrose pseudoinverse.

    Args:
        delta_A: Attenuation changes ``(n_wavelengths, n_timepoints)`` or ``(n_wavelengths,)``.
        M: Absorption matrix ``(n_wavelengths, n_chromophores)``.

    Returns:
        ``Δc`` of shape ``(n_chromophores, n_timepoints)`` (or ``(n_chromophores,)``).
    """
    return np.linalg.pinv(np.asarray(M, dtype=float)) @ np.asarray(delta_A, dtype=float)


def forward_model(delta_c: ArrayLike, M: ArrayLike) -> NDArray:
    """Model attenuation changes ``ΔÂ = M Δc``."""
    return np.asarray(M, dtype=float) @ np.asarray(delta_c, dtype=float)
