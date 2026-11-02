"""Wavelength-selection criteria ``J(Z)`` from the unified framework (paper, Fig. 1).

Five criteria are implemented:

=========================  =====================================================  ==========
Criterion                  ``J(Z)``                                               Direction
=========================  =====================================================  ==========
ConcentrationPreservation  ``NRMSE(M⁺ ΔA − M_Z⁺ ΔA_Z)``                           minimise
SpectralFitPreservation    ``RMSE(M (M_Z⁺ ΔA_Z) − ΔA)``                           minimise
ModelOrthogonality         ``O(Σ_Z)`` with ``M_Z = U Σ_Z Vᵀ``                     maximise
DataOrthogonality          ``O(Σ_Z)`` with ``ΔA_Z = U Σ_Z Vᵀ``                    maximise
ProjectionMaximization     ``RMSE(M_Z (M_Z⁺ ΔA_Z) − ΔA_Z)``                       minimise
=========================  =====================================================  ==========

``Z`` is a subset of wavelength *indices*; ``M_Z`` / ``ΔA_Z`` are the rows of
``M`` / ``ΔA`` belonging to ``Z``. Every criterion supports *batched*
evaluation of many subsets of equal size at once (``evaluate``), which is what
makes brute-force search and differential evolution fast.

Multiple subjects (e.g. piglets) are supported by passing a list of ``ΔA``
arrays; errors are computed per subject and averaged, exactly as in the paper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Criterion",
    "ConcentrationPreservation",
    "SpectralFitPreservation",
    "ModelOrthogonality",
    "DataOrthogonality",
    "ProjectionMaximization",
    "ORTHOGONALITY_METRICS",
    "CRITERIA",
    "get_criterion",
]

# Max. number of float64 entries materialised per batch (~200 MB).
_MAX_BATCH_ENTRIES = 25_000_000

ORTHOGONALITY_METRICS = {
    # name: maximise?
    "log_sv_product": True,
    "sv_product": True,
    "min_sv": True,
    "sv_sum": True,
    "condition_number": False,
}
"""Supported singular-value metrics ``O(Σ)`` for the orthogonality criteria."""


def _as_groups(delta_A: ArrayLike | Sequence[ArrayLike]) -> tuple[NDArray, NDArray]:
    """Concatenate one or several ``(n_wavelengths, T_s)`` arrays along time."""
    if isinstance(delta_A, np.ndarray) and delta_A.ndim <= 2:
        arrays = [delta_A]
    elif isinstance(delta_A, np.ndarray) and delta_A.ndim == 3:
        arrays = list(delta_A)  # (n_subjects, n_wavelengths, T)
    else:
        arrays = list(delta_A)
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
    n = {a.shape[0] for a in arrays}
    if len(n) != 1:
        raise ValueError(f"All ΔA arrays need the same number of wavelengths, got {sorted(n)}")
    lengths = np.array([a.shape[1] for a in arrays])
    if np.any(lengths == 0):
        raise ValueError("Empty ΔA array given.")
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.concatenate(arrays, axis=1), starts


def _svd_metric(sv: NDArray, metric: str) -> NDArray:
    """Reduce singular values ``sv`` (…, k) to the requested scalar metric."""
    if metric == "log_sv_product":
        return np.sum(np.log(np.maximum(sv, 1e-300)), axis=-1)
    if metric == "sv_product":
        return np.prod(sv, axis=-1)
    if metric == "min_sv":
        return np.min(sv, axis=-1)
    if metric == "sv_sum":
        return np.sum(sv, axis=-1)
    if metric == "condition_number":
        smin = np.min(sv, axis=-1)
        with np.errstate(divide="ignore"):
            return np.where(smin > 0, np.max(sv, axis=-1) / smin, np.inf)
    raise ValueError(f"Unknown metric {metric!r}. Choose from {sorted(ORTHOGONALITY_METRICS)}")


class Criterion(ABC):
    """Base class of a wavelength-selection criterion ``J(Z)``.

    Args:
        M: Absorption matrix ``(n_wavelengths, n_chromophores)``. Required for
            model-based criteria.
        delta_A: Attenuation changes ``(n_wavelengths, n_timepoints)``, or a list
            of such arrays (one per subject), or a 3-D array
            ``(n_subjects, n_wavelengths, n_timepoints)``. Required for
            data-based criteria.
        wavelengths: Optional wavelength values (nm), used for reporting only.
        strategy: How subsets are evaluated for the model-based criteria (CP, SFP,
            PM). ``"direct"`` works on ``ΔA`` itself (cost ∝ #timepoints);
            ``"gram"`` uses per-subject Gram matrices ``ΔA ΔAᵀ`` (cost independent
            of #timepoints, mathematically identical). ``"auto"`` (default) picks
            the cheaper one.
    """

    #: Short identifier, e.g. ``"cp"``.
    key: str = ""
    #: Human readable name.
    name: str = ""
    #: ``True`` if larger values of ``J`` are better.
    maximize: bool = False
    requires_model: bool = True
    requires_data: bool = True
    #: Whether the criterion implements :meth:`_evaluate_batch_gram`.
    supports_gram: bool = False

    def __init__(
        self,
        M: ArrayLike | None = None,
        delta_A: ArrayLike | Sequence[ArrayLike] | None = None,
        *,
        wavelengths: ArrayLike | None = None,
        strategy: str = "auto",
    ) -> None:
        if strategy not in ("auto", "direct", "gram"):
            raise ValueError("strategy must be 'auto', 'direct' or 'gram'")
        self.strategy = strategy
        self._grams_cache: NDArray | None = None
        if self.requires_model and M is None:
            raise ValueError(f"{type(self).__name__} requires the absorption matrix M.")
        if self.requires_data and delta_A is None:
            raise ValueError(f"{type(self).__name__} requires measured attenuations ΔA.")

        self.M = None if M is None else np.asarray(M, dtype=float)
        if self.M is not None and self.M.ndim != 2:
            raise ValueError("M must be 2-D (n_wavelengths, n_chromophores).")

        if delta_A is not None:
            self.A, self.starts = _as_groups(delta_A)
        else:
            self.A, self.starts = None, np.array([0])

        n = {x.shape[0] for x in (self.M, self.A) if x is not None}
        if len(n) != 1:
            raise ValueError(f"M and ΔA disagree on the number of wavelengths: {sorted(n)}")
        self.n_wavelengths: int = n.pop()
        self.wavelengths = (
            None if wavelengths is None else np.asarray(wavelengths, dtype=float).ravel()
        )
        if self.wavelengths is not None and len(self.wavelengths) != self.n_wavelengths:
            raise ValueError("Length of `wavelengths` does not match M / ΔA.")
        self._setup()

    # ------------------------------------------------------------------ API
    def _setup(self) -> None:  # noqa: B027 - optional hook
        """Precompute quantities that do not depend on the subset."""

    @property
    def n_groups(self) -> int:
        return len(self.starts)

    @property
    def n_timepoints(self) -> int:
        return 0 if self.A is None else self.A.shape[1]

    def __call__(self, subset: Iterable[int]) -> float:
        """Evaluate ``J`` for a single subset of wavelength indices."""
        subset = np.asarray(list(subset), dtype=np.intp)
        return float(self.evaluate(subset[None, :])[0])

    def evaluate(self, subsets: ArrayLike) -> NDArray:
        """Evaluate ``J`` for a batch of subsets of equal size.

        Args:
            subsets: Integer array ``(n_subsets, d)`` of wavelength indices.

        Returns:
            Array ``(n_subsets,)`` with the criterion values.
        """
        subsets = np.asarray(subsets, dtype=np.intp)
        if subsets.ndim == 1:
            subsets = subsets[None, :]
        if subsets.size and (subsets.min() < 0 or subsets.max() >= self.n_wavelengths):
            raise IndexError("Subset index out of range.")
        d = subsets.shape[1]
        if self._use_gram(d):
            func = self._evaluate_batch_gram
            per_subset = max(
                1, self.n_groups * d * (d + 2 * (self.M.shape[1] if self.M is not None else 1))
            )
        else:
            func = self._evaluate_batch
            per_subset = max(1, d * max(self.n_timepoints, 1) * 3)
        chunk = max(1, _MAX_BATCH_ENTRIES // per_subset)
        out = np.empty(len(subsets), dtype=float)
        for s in range(0, len(subsets), chunk):
            out[s : s + chunk] = func(subsets[s : s + chunk])
        return out

    def _use_gram(self, d: int) -> bool:
        if not self.supports_gram or self.strategy == "direct":
            return False
        if self.strategy == "gram":
            return True
        # direct ~ d·T_total ; gram ~ G·d² (+ one-off Gram matrices)
        return self.n_groups * d < 0.5 * self.n_timepoints

    @property
    def _grams(self) -> NDArray:
        """Per-subject Gram matrices ``ΔA_s ΔA_sᵀ``, shape ``(G, n, n)`` (lazy)."""
        if self._grams_cache is None:
            bounds = np.append(self.starts, self.n_timepoints)
            self._grams_cache = np.stack(
                [self.A[:, s:e] @ self.A[:, s:e].T for s, e in zip(bounds[:-1], bounds[1:])]
            )
        return self._grams_cache

    def _gram_blocks(self, subsets: NDArray) -> NDArray:
        """``G_s[Z, Z]`` for all subjects and subsets → ``(G, B, d, d)``."""
        return self._grams[:, subsets[:, :, None], subsets[:, None, :]]

    def _evaluate_batch_gram(self, subsets: NDArray) -> NDArray:
        raise NotImplementedError

    def objective(self, subsets: ArrayLike) -> NDArray:
        """Values to *minimise* (``J`` or ``-J`` depending on :attr:`maximize`)."""
        values = self.evaluate(subsets)
        return -values if self.maximize else values

    def is_better(self, a: float, b: float) -> bool:
        """Whether score ``a`` is better than ``b``."""
        return a > b if self.maximize else a < b

    @abstractmethod
    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        """Evaluate a (memory-bounded) batch of subsets."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_wavelengths={self.n_wavelengths})"

    # -------------------------------------------------------------- helpers
    def _reduce_groups(self, x: NDArray) -> NDArray:
        """Sum ``x`` (…, T) over time within each subject → (…, n_groups)."""
        return np.add.reduceat(x, self.starts, axis=-1)

    def _group_lengths(self) -> NDArray:
        return np.diff(np.append(self.starts, self.n_timepoints))

    def _subset_concentrations(self, subsets: NDArray) -> tuple[NDArray, NDArray]:
        """Return ``(M_Z, Δc_Z = M_Z⁺ ΔA_Z)`` for a batch of subsets."""
        M_Z = self.M[subsets]  # (B, d, m)
        A_Z = self.A[subsets]  # (B, d, T)
        return M_Z, np.linalg.pinv(M_Z) @ A_Z  # (B, m, T)


class ConcentrationPreservation(Criterion):
    """Preserve concentrations inferred from the full range (Arifler et al. 2015).

    ``J(Z) = NRMSE(M⁺ ΔA − M_Z⁺ ΔA_Z)``, minimised. The NRMSE is computed per
    chromophore and subject over time, then averaged.
    """

    key = "cp"
    name = "Concentration Preservation"
    maximize = False
    supports_gram = True

    def _setup(self) -> None:
        self._P = np.linalg.pinv(self.M)  # (m, n)
        self.c_ref = self._P @ self.A  # (m, T)
        self._c_ref_energy = np.maximum(self._reduce_groups(self.c_ref**2), 1e-24)  # (m, G)

    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        _, c_Z = self._subset_concentrations(subsets)
        err = self._reduce_groups((c_Z - self.c_ref) ** 2)  # (B, m, G)
        return np.mean(np.sqrt(err / self._c_ref_energy), axis=(1, 2))

    def _evaluate_batch_gram(self, subsets: NDArray) -> NDArray:
        # Σ_t (c_Z - c)² = diag(P_Z G_ZZ P_Zᵀ) - 2 diag(P_Z (G P ᵀ)_Z) + diag(P G Pᵀ)
        if not hasattr(self, "_GPt"):
            self._GPt = self._grams @ self._P.T  # (G, n, m)
        P_Z = np.linalg.pinv(self.M[subsets])  # (B, m, d)
        quad = np.einsum("bkd,gbde,bke->bkg", P_Z, self._gram_blocks(subsets), P_Z, optimize=True)
        cross = np.einsum("bkd,gbdk->bkg", P_Z, self._GPt[:, subsets], optimize=True)
        err = np.maximum(quad - 2.0 * cross + self._c_ref_energy, 0.0)  # (B, m, G)
        return np.mean(np.sqrt(err / self._c_ref_energy), axis=(1, 2))


class SpectralFitPreservation(Criterion):
    """Preserve the spectral fit over the full measured range.

    ``J(Z) = RMSE(M (M_Z⁺ ΔA_Z) − ΔA)``, minimised (per subject, then averaged).
    """

    key = "sfp"
    name = "Spectral Fit Preservation"
    maximize = False
    supports_gram = True

    def _evaluate_batch_gram(self, subsets: NDArray) -> NDArray:
        # Σ_t ||M c_Z - a||² = tr(MᵀM P_Z G_ZZ P_Zᵀ) - 2 tr(P_Z (G M)_Z) + tr(G)
        if not hasattr(self, "_GM"):
            self._GM = self._grams @ self.M  # (G, n, m)
            self._trG = np.trace(self._grams, axis1=1, axis2=2)  # (G,)
        P_Z = np.linalg.pinv(self.M[subsets])  # (B, m, d)
        X = np.einsum("bkd,gbde,ble->gbkl", P_Z, self._gram_blocks(subsets), P_Z, optimize=True)
        quad = np.einsum("kl,gbkl->bg", self._G, X)
        cross = np.einsum("bkd,gbdk->bg", P_Z, self._GM[:, subsets], optimize=True)
        sse = np.maximum(quad - 2.0 * cross + self._trG, 0.0)
        return np.mean(np.sqrt(sse / self._n_entries), axis=1)

    def _setup(self) -> None:
        # ||M c - a||² = cᵀ G c - 2 cᵀ h + aᵀa  → avoids materialising (B, n, T)
        self._G = self.M.T @ self.M  # (m, m)
        self._H = self.M.T @ self.A  # (m, T)
        self._a2 = np.sum(self.A**2, axis=0)  # (T,)
        self._n_entries = self.n_wavelengths * self._group_lengths()  # (G,)

    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        _, c_Z = self._subset_concentrations(subsets)  # (B, m, T)
        quad = np.einsum("bmt,mk,bkt->bt", c_Z, self._G, c_Z, optimize=True)
        lin = np.einsum("bmt,mt->bt", c_Z, self._H, optimize=True)
        sse_t = np.maximum(quad - 2.0 * lin + self._a2, 0.0)
        sse = self._reduce_groups(sse_t)  # (B, G)
        return np.mean(np.sqrt(sse / self._n_entries), axis=1)


class ProjectionMaximization(Criterion):
    """Projection maximisation (proposed in the paper, Sec. 2.2.2).

    ``J(Z) = RMSE(M_Z (M_Z⁺ ΔA_Z) − ΔA_Z)``, minimised. Removes the wavelengths
    the model ``M`` describes worst, i.e. it identifies the band in which the
    chosen model can explain the data. Typically used with SBS to select a
    *band* (large ``d``) rather than a minimal set.
    """

    key = "pm"
    name = "Projection Maximization"
    maximize = False
    supports_gram = True

    def _evaluate_batch_gram(self, subsets: NDArray) -> NDArray:
        # ||(I - H) ΔA_Z||² = tr(G_ZZ) - tr(H G_ZZ),  H = M_Z M_Z⁺ (orthogonal projector)
        if not hasattr(self, "_diagG"):
            self._diagG = np.diagonal(self._grams, axis1=1, axis2=2)  # (G, n)
        M_Z = self.M[subsets]  # (B, d, m)
        P_Z = np.linalg.pinv(M_Z)  # (B, m, d)
        Y = self._gram_blocks(subsets) @ M_Z  # (G, B, d, m)
        tr_HG = np.einsum("bkd,gbdk->bg", P_Z, Y, optimize=True)
        tr_G = self._diagG[:, subsets].sum(-1).T  # (B, G)
        sse = np.maximum(tr_G - tr_HG, 0.0)
        n_entries = subsets.shape[1] * self._group_lengths()
        return np.mean(np.sqrt(sse / n_entries), axis=1)

    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        M_Z, c_Z = self._subset_concentrations(subsets)
        resid = M_Z @ c_Z - self.A[subsets]  # (B, d, T)
        sse = self._reduce_groups(np.sum(resid**2, axis=1))  # (B, G)
        n_entries = subsets.shape[1] * self._group_lengths()
        return np.mean(np.sqrt(sse / n_entries), axis=1)


class _OrthogonalityCriterion(Criterion):
    def __init__(self, *args, metric: str = "log_sv_product", **kwargs) -> None:
        if metric not in ORTHOGONALITY_METRICS:
            raise ValueError(
                f"Unknown metric {metric!r}. Choose from {sorted(ORTHOGONALITY_METRICS)}"
            )
        self.metric = metric
        self.maximize = ORTHOGONALITY_METRICS[metric]
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_wavelengths={self.n_wavelengths}, metric={self.metric!r})"


class ModelOrthogonality(_OrthogonalityCriterion):
    """Maximise orthogonality of the model features (rows of ``M_Z``).

    ``J(Z) = O(Σ_Z)`` with ``M_Z = U Σ_Z Vᵀ``. The default metric is the
    (log-)product of singular values (Marois et al. 2018); alternatives are
    ``"min_sv"``, ``"sv_sum"`` and ``"condition_number"`` (minimised).
    Only ``M`` is needed.
    """

    key = "mo"
    name = "Model Orthogonality"
    requires_data = False

    def __init__(
        self,
        M: ArrayLike,
        delta_A: ArrayLike | Sequence[ArrayLike] | None = None,
        *,
        metric: str = "log_sv_product",
        wavelengths: ArrayLike | None = None,
    ) -> None:
        # ΔA is irrelevant for this criterion; it is accepted for a uniform API.
        super().__init__(M, None, metric=metric, wavelengths=wavelengths)

    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        sv = np.linalg.svd(self.M[subsets], compute_uv=False)  # (B, min(d, m))
        return _svd_metric(sv, self.metric)


class DataOrthogonality(_OrthogonalityCriterion):
    """Maximise orthogonality of the measured data ``ΔA_Z`` (filter method).

    ``J(Z) = O(Σ_Z)`` with ``ΔA_Z = U Σ_Z Vᵀ``, computed per subject and
    averaged. Model independent: only ``ΔA`` is needed.

    Singular values are obtained from the per-subject Gram matrices
    ``ΔA ΔAᵀ``, so evaluation cost does not depend on the number of timepoints.
    """

    key = "do"
    name = "Data Orthogonality"
    requires_model = False

    def __init__(
        self,
        M: ArrayLike | None = None,
        delta_A: ArrayLike | Sequence[ArrayLike] | None = None,
        *,
        metric: str = "log_sv_product",
        wavelengths: ArrayLike | None = None,
    ) -> None:
        # M is irrelevant for this criterion; it is accepted for a uniform API.
        super().__init__(None, delta_A, metric=metric, wavelengths=wavelengths)

    def _setup(self) -> None:
        self._lengths = self._group_lengths()

    def evaluate(self, subsets: ArrayLike) -> NDArray:  # cost independent of T
        subsets = np.asarray(subsets, dtype=np.intp)
        if subsets.ndim == 1:
            subsets = subsets[None, :]
        if subsets.size and (subsets.min() < 0 or subsets.max() >= self.n_wavelengths):
            raise IndexError("Subset index out of range.")
        chunk = max(1, _MAX_BATCH_ENTRIES // max(1, subsets.shape[1] ** 2))
        out = np.empty(len(subsets), dtype=float)
        for s in range(0, len(subsets), chunk):
            out[s : s + chunk] = self._evaluate_batch(subsets[s : s + chunk])
        return out

    def _evaluate_batch(self, subsets: NDArray) -> NDArray:
        d = subsets.shape[1]
        values = np.zeros(len(subsets))
        for gram, T in zip(self._grams, self._lengths):
            sub = gram[subsets[:, :, None], subsets[:, None, :]]  # (B, d, d)
            eig = np.linalg.eigvalsh(sub)[:, ::-1][:, : min(d, T)]  # descending
            sv = np.sqrt(np.maximum(eig, 0.0))
            values += _svd_metric(sv, self.metric)
        return values / len(self._grams)


CRITERIA: dict[str, type[Criterion]] = {
    "cp": ConcentrationPreservation,
    "sfp": SpectralFitPreservation,
    "mo": ModelOrthogonality,
    "do": DataOrthogonality,
    "pm": ProjectionMaximization,
}
"""Registry mapping short keys to criterion classes."""

_CRITERION_ALIASES = {
    "concentration": "cp",
    "concentration_preservation": "cp",
    "spectral_fit": "sfp",
    "spectral_fit_preservation": "sfp",
    "model_orthogonality": "mo",
    "data_orthogonality": "do",
    "projection": "pm",
    "projection_maximization": "pm",
}


def get_criterion(name: str) -> type[Criterion]:
    """Look up a criterion class by key (``cp``, ``sfp``, ``mo``, ``do``, ``pm``) or long name."""
    key = name.lower().replace("-", "_").replace(" ", "_")
    key = _CRITERION_ALIASES.get(key, key)
    try:
        return CRITERIA[key]
    except KeyError as err:
        raise KeyError(f"Unknown criterion {name!r}. Choose from {sorted(CRITERIA)}") from err
