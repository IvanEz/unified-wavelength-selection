"""Chromophore absorption spectra and construction of the absorption matrix ``M``."""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from importlib import resources

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "AVAILABLE_CHROMOPHORES",
    "DEFAULT_CHROMOPHORES",
    "load_spectra",
    "absorption_matrix",
]

AVAILABLE_CHROMOPHORES = ("HbO2", "HHb", "oxCCO", "redCCO", "diffCCO")
"""Chromophores shipped in ``resources/chromophores_nir.csv``."""

DEFAULT_CHROMOPHORES = ("HbO2", "HHb", "diffCCO")
"""Chromophores used in the paper (oxy-/deoxy-haemoglobin and differential CCO)."""

_ALIASES = {
    "hbo2": "HbO2",
    "oxyhemoglobin": "HbO2",
    "oxyhaemoglobin": "HbO2",
    "hb": "HHb",
    "hhb": "HHb",
    "hbr": "HHb",
    "deoxyhemoglobin": "HHb",
    "deoxyhaemoglobin": "HHb",
    "oxcco": "oxCCO",
    "redcco": "redCCO",
    "diffcco": "diffCCO",
    "cco": "diffCCO",
    "oxcco-redcco": "diffCCO",
}


def _canonical(name: str) -> str:
    if name in AVAILABLE_CHROMOPHORES:
        return name
    try:
        return _ALIASES[name.lower()]
    except KeyError as err:
        raise KeyError(
            f"Unknown chromophore {name!r}. Available: {', '.join(AVAILABLE_CHROMOPHORES)}"
        ) from err


@lru_cache(maxsize=1)
def load_spectra() -> tuple[NDArray, dict[str, NDArray]]:
    """Load the bundled NIR chromophore spectra.

    Returns:
        ``(wavelengths_nm, spectra)`` where ``spectra[name]`` holds the specific
        absorption coefficient in ``cm⁻¹ mM⁻¹`` on a 1 nm grid (650–999 nm).
    """
    path = resources.files("wlselect.resources").joinpath("chromophores_nir.csv")
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    header = lines[0].split(",")
    values = np.array([[float(v) for v in ln.split(",")] for ln in lines[1:]])
    table = dict(zip(header, values.T))
    wavelengths = table["wavelength_nm"]
    spectra = {name: table[name] for name in AVAILABLE_CHROMOPHORES}
    return wavelengths, spectra


def absorption_matrix(
    wavelengths: ArrayLike,
    chromophores: Sequence[str] = DEFAULT_CHROMOPHORES,
    *,
    pathlength: ArrayLike | float | None = None,
    spectra: dict[str, tuple[ArrayLike, ArrayLike]] | None = None,
) -> NDArray:
    """Build the absorption matrix ``M`` (paper, Eq. 2) at arbitrary wavelengths.

    Args:
        wavelengths: Target wavelengths in nm, shape ``(n,)``.
        chromophores: Names of the chromophores (columns of ``M``).
        pathlength: Optional optical pathlength ``PL(λ) = DPF(λ)·d`` (scalar or
            per-wavelength array). If ``None`` a unit pathlength is assumed, so
            inferred concentrations are in ``mM·cm``.
        spectra: Optional custom spectra ``{name: (wavelengths_nm, coefficients)}``
            that override / extend the bundled ones.

    Returns:
        ``M`` of shape ``(n, len(chromophores))``.
    """
    wavelengths = np.asarray(wavelengths, dtype=float).ravel()
    base_wl, base = load_spectra()
    columns = []
    for name in chromophores:
        if spectra is not None and name in spectra:
            src_wl, src = (np.asarray(a, dtype=float) for a in spectra[name])
        else:
            src_wl, src = base_wl, base[_canonical(name)]
        if wavelengths.min() < src_wl.min() or wavelengths.max() > src_wl.max():
            raise ValueError(
                f"Wavelengths [{wavelengths.min():.1f}, {wavelengths.max():.1f}] nm are "
                f"outside the tabulated range of {name!r} "
                f"[{src_wl.min():.1f}, {src_wl.max():.1f}] nm."
            )
        columns.append(np.interp(wavelengths, src_wl, src))
    M = np.stack(columns, axis=1)
    if pathlength is not None:
        M = M * np.asarray(pathlength, dtype=float).reshape(-1, 1)
    return M
