"""Loader for the UCL miniCYRIL broadband NIRS piglet hypoxia-ischaemia dataset.

The raw data are **not** distributed with this package. Request access from the
UCL Biomedical Optics Research Laboratory and arrange the files as::

    <root>/                       # e.g. "miniCYRIL Piglet Data"
        LWP480/refSpectrum.mat                    # white reference
        LWP480/LWP480_DarkCount_12Dec2016.mat     # dark count
        lwp475/LWP475_Ws_14Nov2016_1.mat
        lwp478/LWP474_Ws_07Nov2016.mat
        ...

Pre-processing follows the paper: the first 1000 timepoints are used, spectra
are dark/white corrected and converted to attenuation changes w.r.t. the first
timepoint, ``ΔA(t) = -ln(I(t) / I(t0))``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..chromophores import DEFAULT_CHROMOPHORES, absorption_matrix
from ..io import Dataset

__all__ = [
    "PIGLET_FILES",
    "HELD_OUT_PIGLETS",
    "WHITE_REFERENCE",
    "DARK_REFERENCE",
    "load_piglet",
    "load_piglets",
]

PIGLET_FILES: dict[str, str] = {
    "lwp475": "LWP475_Ws_14Nov2016_1.mat",
    "lwp478": "LWP474_Ws_07Nov2016.mat",  # file of LWP478 is named LWP474
    "lwp479-HI+saline": "LWP479_Ws_05Dec2016.mat",
    "lwp494": "LWP494_Ws_27Mar2017_17  21.mat",
    "lwp499": "LWP499_Ws_02_May_2017.mat",
    "LWP480": "LWP480_Ws_12Dec2016.mat",
    "LWP481": "LWP481_Ws_03Jan2017.mat",
    "LWP484": "LWP484_Ws_23Jan2017.mat",
    "LWP485": "LWP485_Ws_30Jan2017.mat",
    "LWP489": "LWP489_Ws_20Feb2017.mat",
    "LWP488": "LWP488_Ws_13Feb2017.mat",
    "LWP490": "LWP490_Ws_27Feb2017.mat",
    "LWP491": "ResultsLWP491_06Mar2017.mat",  # different internal structure
    "LWP492": "LWP492_Ws_13Mar2017.mat",
    "LWP495": "LWP495_Ws_03Apr2017_14  23.mat",
    "LWP498": "LWP498_Ws_24Apr2017_15.mat",
    "LWP500": "LWP500_Ws_15_May_201716.mat",
    "LWP501": "LWP501_Ws_22_May_201715.mat",
    "LWP502": "LWP502_Ws_30_May_201716.mat",
    "LWP503": "LWP503_Ws_05_Jun_2017_15   8.mat",
    "LWP504": "LWP504_Ws_12Jun_2017_17   1.mat",
    "LWP507": "LWP507_Ws_10Jul_2017_16  55.mat",
    "LWP509": "LWP509_Ws_17Jul_2017_11  21.mat",
    "LWP511": "LWP511_Ws_25Jul_2017_16   2.mat",
    "LWP512": "LWP512_Ws_31Jul_2017_17  16.mat",
}
"""Subject folder → spectra file name (file names contain literal double spaces)."""

HELD_OUT_PIGLETS: tuple[str, ...] = ("LWP503", "LWP504", "LWP507", "LWP509", "LWP511", "LWP512")
"""Subjects excluded by default (held out in the original code)."""

WHITE_REFERENCE = ("LWP480", "refSpectrum.mat", "refSpectrum")
DARK_REFERENCE = ("LWP480", "LWP480_DarkCount_12Dec2016.mat", "DarkCount")


def _loadmat(path: Path) -> dict:
    from scipy.io import loadmat

    if not path.exists():
        raise FileNotFoundError(f"Missing piglet data file: {path}")
    return loadmat(str(path))


def _read_raw(root: Path, piglet: str) -> tuple[NDArray, NDArray]:
    """Return raw ``(wavelengths, spectra)`` with spectra shaped ``(n_wl, T)``."""
    mat = _loadmat(root / piglet / PIGLET_FILES[piglet])
    if "Rawdata" in mat:  # LWP491
        raw = mat["Rawdata"]
        wavelengths = raw["wavelengths"][0, 0][0][0].astype(float)
        spectra = raw["spectralDataAll"][0, 0][0][0].astype(float)
    else:
        wavelengths = mat["wavelengths"].astype(float)
        spectra = mat["spectralDataAll"].astype(float)
    return np.ravel(wavelengths), spectra


def _reference(root: Path, spec: tuple[str, str, str]) -> NDArray:
    folder, fname, key = spec
    return np.ravel(_loadmat(root / folder / fname)[key].astype(float)[:, 0])


def load_piglet(
    root: str | Path,
    piglet: str,
    *,
    wl_range: tuple[float, float] = (780.0, 900.0),
    max_timepoints: int = 1000,
    white: NDArray | None = None,
    dark: NDArray | None = None,
) -> tuple[NDArray, NDArray]:
    """Load one piglet and return ``(wavelengths, ΔA)`` with ``ΔA`` of shape ``(n_wl, T)``.

    Args:
        root: Dataset root folder.
        piglet: Subject key, see :data:`PIGLET_FILES`.
        wl_range: Inclusive wavelength range in nm.
        max_timepoints: Use only the first ``max_timepoints`` measurements.
        white / dark: Full-range white / dark reference spectra; loaded from
            ``LWP480`` if not given.
    """
    root = Path(root)
    if piglet not in PIGLET_FILES:
        raise KeyError(f"Unknown piglet {piglet!r}. Known: {sorted(PIGLET_FILES)}")
    wavelengths, spectra = _read_raw(root, piglet)
    white = _reference(root, WHITE_REFERENCE) if white is None else white
    dark = _reference(root, DARK_REFERENCE) if dark is None else dark
    if not (len(white) == len(dark) == len(wavelengths) == spectra.shape[0]):
        raise ValueError(f"{piglet}: spectra and white/dark references have different lengths.")

    keep = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    wavelengths = wavelengths[keep]
    spectra = spectra[keep][:, :max_timepoints]
    spectra[spectra <= 0] = 1e-4
    white, dark = white[keep, None], dark[keep, None]

    reflectance = (spectra - dark) / (white - dark)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_A = np.log(reflectance[:, :1] / reflectance)
    if not np.all(np.isfinite(delta_A)):
        warnings.warn(
            f"{piglet}: non-finite ΔA values after dark/white correction; replaced by 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        delta_A = np.nan_to_num(delta_A, nan=0.0, posinf=0.0, neginf=0.0)
    return wavelengths, delta_A


def load_piglets(
    root: str | Path,
    piglets: Iterable[str] | None = None,
    *,
    wl_range: tuple[float, float] = (780.0, 900.0),
    max_timepoints: int = 1000,
    exclude: Sequence[str] = HELD_OUT_PIGLETS,
    chromophores: Sequence[str] = DEFAULT_CHROMOPHORES,
    skip_missing: bool = False,
) -> Dataset:
    """Load several piglets into a :class:`~wlselect.io.Dataset`.

    Args:
        root: Dataset root folder.
        piglets: Subjects to load. Default: all of :data:`PIGLET_FILES` minus ``exclude``.
        wl_range: Inclusive wavelength range in nm (paper: 780–900, 740–900, 730–930).
        max_timepoints: Number of initial timepoints per piglet (paper: 1000).
        exclude: Subjects to skip when ``piglets`` is ``None``.
        chromophores: Chromophores for the absorption matrix.
        skip_missing: Skip subjects whose files are missing instead of raising.

    Use :meth:`Dataset.subsample` afterwards to reproduce the coarser settings,
    e.g. ``ds.subsample(n_wavelengths=22, n_timepoints=10)``.
    """
    root = Path(root)
    names = [p for p in PIGLET_FILES if p not in set(exclude)] if piglets is None else list(piglets)
    white = _reference(root, WHITE_REFERENCE)
    dark = _reference(root, DARK_REFERENCE)

    wavelengths, arrays, loaded = None, [], []
    for name in names:
        try:
            wl, dA = load_piglet(
                root, name, wl_range=wl_range, max_timepoints=max_timepoints, white=white, dark=dark
            )
        except FileNotFoundError:
            if skip_missing:
                warnings.warn(f"Skipping {name}: file not found.", stacklevel=2)
                continue
            raise
        if wavelengths is None:
            wavelengths = wl
        elif len(wl) != len(wavelengths) or not np.allclose(wl, wavelengths, atol=1e-3):
            raise ValueError(f"{name}: wavelength grid differs from the first subject.")
        arrays.append(dA)
        loaded.append(name)
    if not arrays:
        raise FileNotFoundError(f"No piglet data could be loaded from {root}")

    M = absorption_matrix(wavelengths, chromophores)
    return Dataset(
        wavelengths=wavelengths,
        delta_A=arrays,
        M=M,
        chromophores=tuple(chromophores),
        subjects=loaded,
    )
