"""Synthetic broadband NIRS data for demos and tests.

The generator mimics a hypoxia-ischaemia (HI) experiment: during the insult
HbO2 drops, HHb rises and diffCCO decreases, followed by recovery. Optionally,
an *unmodelled* absorber confined to short wavelengths can be added, which
reproduces the situation analysed in Sec. 3.4 of the paper where the linear
absorption model cannot describe the 740–780 nm range.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chromophores import DEFAULT_CHROMOPHORES, absorption_matrix

__all__ = ["SyntheticDataset", "make_synthetic_dataset", "hi_time_course"]


@dataclass
class SyntheticDataset:
    """Container returned by :func:`make_synthetic_dataset`."""

    wavelengths: NDArray
    M: NDArray
    delta_A: list[NDArray]  # one (n_wavelengths, T) array per subject
    delta_c: list[NDArray]  # ground-truth (n_chromophores, T) per subject
    chromophores: tuple[str, ...]
    info: dict = field(default_factory=dict)

    @property
    def n_subjects(self) -> int:
        return len(self.delta_A)


def hi_time_course(
    n_timepoints: int,
    onset: float = 0.3,
    duration: float = 0.15,
    steepness: float = 0.01,
) -> NDArray:
    """Smooth 0→1→0 profile modelling an HI insult and recovery (values in [0, 1])."""
    t = np.linspace(0.0, 1.0, n_timepoints)
    up = 1.0 / (1.0 + np.exp(-(t - onset) / steepness))
    down = 1.0 / (1.0 + np.exp(-(t - onset - duration) / (3 * steepness)))
    return up * (1.0 - down)


def make_synthetic_dataset(
    wavelengths: ArrayLike | None = None,
    *,
    n_subjects: int = 4,
    n_timepoints: int = 200,
    chromophores: Sequence[str] = DEFAULT_CHROMOPHORES,
    amplitudes: Sequence[float] = (-0.45, 0.40, -0.06),
    noise_std: float = 0.004,
    unmodelled_band: tuple[float, float] | None = None,
    unmodelled_amplitude: float = 0.25,
    seed: int | None = 0,
) -> SyntheticDataset:
    """Generate synthetic attenuation changes following the dMBLL.

    Args:
        wavelengths: Wavelengths in nm. Default: 740–900 nm with 1 nm spacing.
        n_subjects: Number of independent subjects ("piglets").
        n_timepoints: Timepoints per subject (first one is the baseline).
        chromophores: Chromophores in the model ``M``.
        amplitudes: Peak concentration change per chromophore (mM·cm) during HI.
        noise_std: Std. of additive Gaussian measurement noise on ``ΔA``.
        unmodelled_band: ``(center_nm, width_nm)`` of an extra absorber that is
            *not* part of ``M``; ``None`` disables it.
        unmodelled_amplitude: Peak attenuation change of that absorber.
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    wl = np.arange(740.0, 901.0, 1.0) if wavelengths is None else np.asarray(wavelengths, float)
    M = absorption_matrix(wl, chromophores)
    m = M.shape[1]
    amps = np.resize(np.asarray(amplitudes, dtype=float), m)

    delta_A, delta_c = [], []
    for _ in range(n_subjects):
        onset = rng.uniform(0.25, 0.4)
        duration = rng.uniform(0.1, 0.2)
        profile = hi_time_course(n_timepoints, onset, duration)
        scale = rng.uniform(0.7, 1.3, size=m)
        drift = np.cumsum(rng.normal(0, 0.002, size=(m, n_timepoints)), axis=1)
        c = (amps * scale)[:, None] * profile[None, :] + drift * np.abs(amps)[:, None]
        c[:, 0] = 0.0  # baseline
        A = M @ c
        if unmodelled_band is not None:
            center, width = unmodelled_band
            shape = np.exp(-0.5 * ((wl - center) / width) ** 2)
            A = A + unmodelled_amplitude * rng.uniform(0.7, 1.3) * shape[:, None] * profile[None, :]
        A = A + rng.normal(0.0, noise_std, size=A.shape)
        A[:, 0] = 0.0
        delta_A.append(A)
        delta_c.append(c)

    return SyntheticDataset(
        wavelengths=wl,
        M=M,
        delta_A=delta_A,
        delta_c=delta_c,
        chromophores=tuple(chromophores),
        info={
            "noise_std": noise_std,
            "unmodelled_band": unmodelled_band,
            "unmodelled_amplitude": unmodelled_amplitude,
            "seed": seed,
        },
    )
