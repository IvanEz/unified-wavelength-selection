"""Matplotlib helpers to visualise selections (requires ``pip install wlselect[plot]``)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike

from .search import SelectionPath, SelectionResult

__all__ = ["SERIES_COLORS", "plot_selection", "plot_score_curves", "plot_spectral_fit"]

#: Fixed categorical order (colour-blind validated); assigned in order, never cycled.
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3df"


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:  # pragma: no cover
        raise ImportError("Plotting requires matplotlib: pip install 'wlselect[plot]'") from err
    return plt


def _style(ax) -> None:
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2)
    ax.xaxis.label.set_color(_INK)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def plot_selection(
    wavelengths: ArrayLike,
    M: ArrayLike,
    selections: Mapping[str, ArrayLike | SelectionResult],
    *,
    chromophores: Sequence[str] = ("HbO2", "HHb", "diffCCO"),
    ax=None,
    title: str | None = None,
):
    """Chromophore spectra with the selected wavelengths as vertical lines.

    Args:
        wavelengths: Wavelengths (nm) of the rows of ``M``.
        M: Absorption matrix.
        selections: ``{label: indices or SelectionResult}``; each gets its own line style.
    """
    plt = _plt()
    wavelengths = np.asarray(wavelengths, dtype=float)
    M = np.asarray(M, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for j in range(M.shape[1]):
        name = chromophores[j] if j < len(chromophores) else f"chromophore {j}"
        ax.plot(wavelengths, M[:, j], color=SERIES_COLORS[j], lw=2, label=name)
    styles = ["-", "--", ":", "-."]
    for k, (label, sel) in enumerate(selections.items()):
        idx = sel.indices if isinstance(sel, SelectionResult) else np.asarray(sel, dtype=int)
        for i, w in enumerate(wavelengths[idx]):
            ax.axvline(
                w,
                color=_INK_2,
                ls=styles[k % len(styles)],
                lw=1.2,
                alpha=0.8,
                label=label if i == 0 else None,
            )
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Absorption coefficient [cm⁻¹ mM⁻¹]")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return ax


def plot_score_curves(
    results: Mapping[str, SelectionPath | Mapping[int, SelectionResult]],
    *,
    ax=None,
    log: bool = True,
    ylabel: str = "Criterion value",
    title: str | None = None,
):
    """Criterion value vs. number of wavelengths for several algorithms (cf. paper Fig. 2/3)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    styles = ["-", "--", ":", "-."]
    for k, (label, res) in enumerate(results.items()):
        steps = list(res) if isinstance(res, SelectionPath) else list(res.values())
        steps = sorted(steps, key=lambda s: s.d)
        ax.plot(
            [s.d for s in steps],
            [s.score for s in steps],
            color=SERIES_COLORS[k],
            ls=styles[k % len(styles)],
            lw=2,
            marker="o",
            ms=4,
            label=label,
        )
    if log:
        ax.set_yscale("log")
    from matplotlib.ticker import MaxNLocator

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Number of wavelengths d")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, loc="left")
    if len(results) > 1:
        ax.legend(frameon=False)
    _style(ax)
    return ax


def plot_spectral_fit(
    wavelengths: ArrayLike,
    measured: ArrayLike,
    fits: Mapping[str, ArrayLike],
    *,
    ax=None,
    title: str | None = None,
):
    """Measured ``ΔA`` at one timepoint vs. model reconstructions (cf. paper Fig. 6/7)."""
    plt = _plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wavelengths, measured, color=_INK, lw=1, label="Measured")
    for k, (label, fit) in enumerate(fits.items()):
        ax.plot(wavelengths, fit, color=SERIES_COLORS[k], lw=2, label=label)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("ΔAttenuation")
    if title:
        ax.set_title(title, loc="left")
    ax.legend(frameon=False)
    _style(ax)
    return ax
