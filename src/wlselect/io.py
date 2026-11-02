"""Reading and writing datasets in a simple, tool-agnostic ``.npz`` format.

File layout (all arrays optional except ``wavelengths`` and at least one ``delta_A_*``)::

    wavelengths      (n,)        wavelengths in nm
    M                (n, m)      absorption matrix (optional – can be rebuilt)
    chromophores     (m,)        chromophore names for M
    subjects         (S,)        subject names
    delta_A_0 …      (n, T_s)    attenuation changes for subject s
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .chromophores import DEFAULT_CHROMOPHORES, absorption_matrix

__all__ = ["Dataset", "save_dataset", "load_dataset", "subsample"]


@dataclass
class Dataset:
    """Wavelengths, absorption matrix and per-subject attenuation changes."""

    wavelengths: NDArray
    delta_A: list[NDArray]
    M: NDArray | None = None
    chromophores: tuple[str, ...] = tuple(DEFAULT_CHROMOPHORES)
    subjects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.delta_A = [np.asarray(a, dtype=float) for a in self.delta_A]
        if not self.subjects:
            self.subjects = [f"subject_{i}" for i in range(len(self.delta_A))]
        if self.M is None:
            self.M = absorption_matrix(self.wavelengths, self.chromophores)
        self.M = np.asarray(self.M, dtype=float)
        for a in self.delta_A:
            if a.shape[0] != len(self.wavelengths):
                raise ValueError("Each ΔA must have shape (n_wavelengths, n_timepoints).")

    @property
    def n_wavelengths(self) -> int:
        return len(self.wavelengths)

    def subsample(
        self,
        *,
        n_wavelengths: int | None = None,
        wl_step: int | None = None,
        n_timepoints: int | None = None,
        time_step: int | None = None,
        wl_range: tuple[float, float] | None = None,
    ) -> Dataset:
        """Return a coarser copy (see :func:`subsample`)."""
        wl_idx, t_idx = subsample(
            self.wavelengths,
            [a.shape[1] for a in self.delta_A],
            n_wavelengths=n_wavelengths,
            wl_step=wl_step,
            n_timepoints=n_timepoints,
            time_step=time_step,
            wl_range=wl_range,
        )
        return Dataset(
            wavelengths=self.wavelengths[wl_idx],
            delta_A=[a[wl_idx][:, t] for a, t in zip(self.delta_A, t_idx)],
            M=self.M[wl_idx],
            chromophores=self.chromophores,
            subjects=list(self.subjects),
        )


def _even(n: int, k: int) -> NDArray:
    if k >= n:
        return np.arange(n)
    return np.unique(np.round(np.linspace(0, n - 1, k)).astype(int))


def subsample(
    wavelengths: ArrayLike,
    lengths: Sequence[int],
    *,
    n_wavelengths: int | None = None,
    wl_step: int | None = None,
    n_timepoints: int | None = None,
    time_step: int | None = None,
    wl_range: tuple[float, float] | None = None,
) -> tuple[NDArray, list[NDArray]]:
    """Compute wavelength and time indices for a coarser dataset.

    Args:
        wavelengths: All wavelengths.
        lengths: Number of timepoints of each subject.
        n_wavelengths: Keep this many equally spaced wavelengths
            (e.g. 22 for the coarse experiment in the paper).
        wl_step: Alternatively keep every ``wl_step``-th wavelength.
        n_timepoints: Keep this many equally spaced timepoints per subject.
        time_step: Alternatively keep every ``time_step``-th timepoint.
        wl_range: Restrict to ``[min, max]`` nm before subsampling.

    Returns:
        ``(wavelength_indices, [time_indices per subject])``
    """
    wavelengths = np.asarray(wavelengths, dtype=float)
    wl_idx = np.arange(len(wavelengths))
    if wl_range is not None:
        lo, hi = wl_range
        wl_idx = wl_idx[(wavelengths >= lo) & (wavelengths <= hi)]
    if n_wavelengths is not None:
        wl_idx = wl_idx[_even(len(wl_idx), n_wavelengths)]
    elif wl_step:
        wl_idx = wl_idx[::wl_step]

    t_idx = []
    for T in lengths:
        if n_timepoints is not None:
            t_idx.append(_even(T, n_timepoints))
        elif time_step:
            t_idx.append(np.arange(0, T, time_step))
        else:
            t_idx.append(np.arange(T))
    return wl_idx, t_idx


def save_dataset(path: str | Path, dataset: Dataset) -> None:
    """Save a :class:`Dataset` to ``.npz``."""
    arrays = {
        "wavelengths": dataset.wavelengths,
        "M": dataset.M,
        "chromophores": np.array(dataset.chromophores),
        "subjects": np.array(dataset.subjects),
    }
    for i, a in enumerate(dataset.delta_A):
        arrays[f"delta_A_{i}"] = a
    np.savez_compressed(path, **arrays)


def load_dataset(
    path: str | Path,
    *,
    chromophores: Sequence[str] | None = None,
) -> Dataset:
    """Load a dataset written by :func:`save_dataset` (or any compatible ``.npz``).

    A single 2-D ``delta_A`` array (``(n_wavelengths, T)``) or a 3-D one
    (``(n_subjects, n_wavelengths, T)``) is accepted as well. If
    ``chromophores`` is given, ``M`` is rebuilt from the bundled spectra.
    """
    with np.load(path, allow_pickle=False) as f:
        wl = f["wavelengths"]
        keys = sorted(
            (k for k in f.files if k.startswith("delta_A_")), key=lambda k: int(k.split("_")[-1])
        )
        if keys:
            delta_A = [f[k] for k in keys]
        elif "delta_A" in f.files:
            A = f["delta_A"]
            delta_A = list(A) if A.ndim == 3 else [A]
        else:
            raise KeyError(f"{path} contains no 'delta_A' arrays.")
        stored_chrom = (
            tuple(str(c) for c in f["chromophores"]) if "chromophores" in f.files else None
        )
        subjects = [str(s) for s in f["subjects"]] if "subjects" in f.files else []
        M = f["M"] if ("M" in f.files and chromophores is None) else None

    chrom = tuple(chromophores) if chromophores else (stored_chrom or tuple(DEFAULT_CHROMOPHORES))
    return Dataset(wavelengths=wl, delta_A=delta_A, M=M, chromophores=chrom, subjects=subjects)
