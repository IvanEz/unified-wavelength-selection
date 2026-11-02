"""Search algorithms that optimise a :class:`~wlselect.criteria.Criterion`.

* :func:`brute_force` – exhaustive search over all ``C(n, d)`` subsets.
* :func:`sequential_backward_selection` (SBS) – start from all wavelengths and
  iteratively remove the least relevant one(s).
* :func:`sequential_forward_selection` (SFS) – start from the empty set and
  iteratively add the most relevant wavelength.
* :func:`differential_evolution` – stochastic search via SciPy's integer DE.

All functions return :class:`SelectionResult` objects (SBS/SFS return a
:class:`SelectionPath`, i.e. one result per visited subset size).
"""

from __future__ import annotations

import itertools
import json
import math
import os
import time
import warnings
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .criteria import Criterion

__all__ = [
    "SelectionResult",
    "SelectionPath",
    "brute_force",
    "sequential_backward_selection",
    "sequential_forward_selection",
    "differential_evolution",
    "select",
    "ALGORITHMS",
]


# --------------------------------------------------------------------------- results
@dataclass
class SelectionResult:
    """Outcome of a wavelength selection for one subset size ``d``."""

    indices: NDArray
    score: float
    criterion: str
    algorithm: str
    wavelengths: NDArray | None = None
    runtime_s: float = 0.0
    n_evaluations: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def d(self) -> int:
        return len(self.indices)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["indices"] = [int(i) for i in self.indices]
        out["wavelengths"] = (
            None if self.wavelengths is None else [float(w) for w in self.wavelengths]
        )
        out["score"] = float(self.score)
        out["d"] = self.d
        return out

    def __repr__(self) -> str:
        wl = (
            ""
            if self.wavelengths is None
            else ", wavelengths=[" + ", ".join(f"{w:.1f}" for w in self.wavelengths) + "]"
        )
        return (
            f"SelectionResult(algorithm={self.algorithm!r}, criterion={self.criterion!r}, "
            f"d={self.d}, score={self.score:.6g}{wl})"
        )


@dataclass
class SelectionPath:
    """Sequence of results produced by a sequential method (one per subset size)."""

    steps: list[SelectionResult]
    criterion: str
    algorithm: str
    runtime_s: float = 0.0

    def __getitem__(self, d: int) -> SelectionResult:
        """Return the result with exactly ``d`` wavelengths."""
        for step in self.steps:
            if step.d == d:
                return step
        raise KeyError(f"No subset of size {d} in path (sizes: {self.sizes})")

    def __iter__(self) -> Iterator[SelectionResult]:
        return iter(self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def sizes(self) -> list[int]:
        return [s.d for s in self.steps]

    @property
    def scores(self) -> NDArray:
        return np.array([s.score for s in self.steps])

    @property
    def final(self) -> SelectionResult:
        return self.steps[-1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "algorithm": self.algorithm,
            "runtime_s": self.runtime_s,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def __repr__(self) -> str:
        return (
            f"SelectionPath(algorithm={self.algorithm!r}, criterion={self.criterion!r}, "
            f"sizes={self.sizes[0]}→{self.sizes[-1]}, n_steps={len(self)})"
        )


def _result(
    criterion: Criterion,
    indices: ArrayLike,
    score: float,
    algorithm: str,
    runtime: float,
    n_evals: int,
    **extra: Any,
) -> SelectionResult:
    idx = np.sort(np.asarray(indices, dtype=int))
    wl = None if criterion.wavelengths is None else criterion.wavelengths[idx]
    return SelectionResult(
        indices=idx,
        score=float(score),
        criterion=criterion.key,
        algorithm=algorithm,
        wavelengths=wl,
        runtime_s=runtime,
        n_evaluations=n_evals,
        extra=extra,
    )


def _candidates(criterion: Criterion, candidates: ArrayLike | None) -> NDArray:
    if candidates is None:
        return np.arange(criterion.n_wavelengths)
    cand = np.unique(np.asarray(candidates, dtype=int))
    if cand.size and (cand.min() < 0 or cand.max() >= criterion.n_wavelengths):
        raise IndexError("Candidate index out of range.")
    return cand


def _resolve_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


# ---------------------------------------------------------------- brute force
_WORKER_CRITERION: Criterion | None = None


def _init_worker(criterion: Criterion) -> None:
    global _WORKER_CRITERION
    _WORKER_CRITERION = criterion


def _best_in_chunk(chunk: NDArray) -> tuple[float, NDArray]:
    obj = _WORKER_CRITERION.objective(chunk)
    i = int(np.argmin(obj))
    return float(obj[i]), chunk[i]


def _combination_chunks(pool: NDArray, d: int, size: int) -> Iterator[NDArray]:
    it = itertools.combinations(range(len(pool)), d)
    while True:
        block = list(itertools.islice(it, size))
        if not block:
            return
        yield pool[np.asarray(block, dtype=np.intp)]


def brute_force(
    criterion: Criterion,
    d: int,
    *,
    candidates: ArrayLike | None = None,
    chunk_size: int = 20_000,
    n_jobs: int | None = 1,
    max_combinations: float = 5e9,
    progress: bool = False,
) -> SelectionResult:
    """Exhaustively evaluate all subsets of size ``d``.

    Args:
        criterion: The criterion to optimise.
        d: Number of wavelengths to select.
        candidates: Optional subset of wavelength indices to choose from
            (e.g. the band found by projection maximisation).
        chunk_size: Number of subsets evaluated per vectorised batch.
        n_jobs: Number of worker processes (``-1`` = all cores).
        max_combinations: Safety limit; raise if ``C(n, d)`` exceeds it.
        progress: Show a ``tqdm`` progress bar.
    """
    pool = _candidates(criterion, candidates)
    if not 1 <= d <= len(pool):
        raise ValueError(f"d must be in [1, {len(pool)}], got {d}")
    total = math.comb(len(pool), d)
    if total > max_combinations:
        raise ValueError(
            f"C({len(pool)}, {d}) = {total:.3g} combinations exceeds max_combinations="
            f"{max_combinations:.3g}. Use SBS/DE or raise the limit explicitly."
        )
    n_jobs = _resolve_jobs(n_jobs)
    n_chunks = math.ceil(total / chunk_size)
    chunks = _combination_chunks(pool, d, chunk_size)

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=n_chunks, desc=f"brute force d={d}", unit="chunk")

    start = time.perf_counter()
    best_obj, best_idx = np.inf, None
    if n_jobs == 1:
        _init_worker(criterion)
        results = map(_best_in_chunk, chunks)
        for obj, idx in results:
            if obj < best_obj:
                best_obj, best_idx = obj, idx
            if bar is not None:
                bar.update()
    else:
        with ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker, initargs=(criterion,)
        ) as ex:
            for obj, idx in ex.map(_best_in_chunk, chunks, chunksize=4):
                if obj < best_obj:
                    best_obj, best_idx = obj, idx
                if bar is not None:
                    bar.update()
    if bar is not None:
        bar.close()
    runtime = time.perf_counter() - start
    score = -best_obj if criterion.maximize else best_obj
    return _result(criterion, best_idx, score, "brute_force", runtime, total)


# ----------------------------------------------------------- sequential methods
def sequential_backward_selection(
    criterion: Criterion,
    d: int = 1,
    *,
    initial: ArrayLike | None = None,
    remove_fraction: float | None = None,
    progress: bool = False,
) -> SelectionPath:
    """Sequential Backward Selection (SBS).

    Starting from all wavelengths (or ``initial``), the wavelength whose removal
    yields the best criterion value is removed until ``d`` wavelengths remain.
    One SBS run therefore yields subsets for *all* cardinalities ``n-1 … d``.

    Args:
        criterion: Criterion to optimise.
        d: Target number of wavelengths.
        initial: Optional starting set of wavelength indices (e.g. a band from
            projection maximisation).
        remove_fraction: If set (e.g. ``0.05``), remove this fraction of the
            current set per iteration (at least one) instead of a single
            wavelength. Speeds up band selection on large ranges.
        progress: Show a ``tqdm`` progress bar.
    """
    current = _candidates(criterion, initial)
    if not 1 <= d <= len(current):
        raise ValueError(f"d must be in [1, {len(current)}], got {d}")

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=len(current) - d, desc="SBS", unit="wl")

    steps: list[SelectionResult] = []
    start = time.perf_counter()
    while len(current) > d:
        t0 = time.perf_counter()
        k = len(current)
        # all subsets "current without element i"
        mask = ~np.eye(k, dtype=bool)
        subsets = np.broadcast_to(current, (k, k))[mask].reshape(k, k - 1)
        obj = criterion.objective(subsets)

        n_remove = 1
        if remove_fraction:
            n_remove = int(np.clip(int(k * remove_fraction), 1, k - d))
        # removing i gives obj[i]; remove the elements whose removal hurts least
        to_remove = np.argsort(obj, kind="stable")[:n_remove]
        current = np.delete(current, to_remove)
        score = (
            float(obj[to_remove[0]])
            if n_remove == 1
            else float(criterion.objective(current[None])[0])
        )
        score = -score if criterion.maximize else score
        steps.append(_result(criterion, current, score, "sbs", time.perf_counter() - t0, k))
        if bar is not None:
            bar.update(n_remove)
    if bar is not None:
        bar.close()
    return SelectionPath(steps, criterion.key, "sbs", time.perf_counter() - start)


def sequential_forward_selection(
    criterion: Criterion,
    d: int,
    *,
    initial: ArrayLike | None = None,
    candidates: ArrayLike | None = None,
    progress: bool = False,
) -> SelectionPath:
    """Sequential Forward Selection (SFS).

    Starting from the empty set (or ``initial``), iteratively add the wavelength
    that gives the best criterion value until ``d`` wavelengths are selected.
    """
    pool = _candidates(criterion, candidates)
    current = np.array([], dtype=int) if initial is None else _candidates(criterion, initial)
    available = np.setdiff1d(pool, current)
    if not len(current) < d <= len(current) + len(available):
        raise ValueError(f"d={d} is not reachable from |initial|={len(current)}")

    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=d - len(current), desc="SFS", unit="wl")

    steps: list[SelectionResult] = []
    start = time.perf_counter()
    while len(current) < d:
        t0 = time.perf_counter()
        subsets = np.column_stack(
            [np.broadcast_to(current, (len(available), len(current))), available]
        )
        obj = criterion.objective(subsets)
        best = int(np.argmin(obj))
        current = np.append(current, available[best])
        available = np.delete(available, best)
        score = -obj[best] if criterion.maximize else obj[best]
        steps.append(
            _result(criterion, current, score, "sfs", time.perf_counter() - t0, len(subsets))
        )
        if bar is not None:
            bar.update()
    if bar is not None:
        bar.close()
    return SelectionPath(steps, criterion.key, "sfs", time.perf_counter() - start)


# ------------------------------------------------------- differential evolution
def _repair(rows: NDArray, n: int) -> NDArray:
    """Replace duplicate indices in each row by the nearest unused index."""
    rows = rows.copy()
    for r in rows:
        if len(np.unique(r)) == len(r):
            continue
        used: set[int] = set()
        for j, v in enumerate(r):
            v = int(v)
            if v in used:
                for off in range(1, n):
                    for cand in (v - off, v + off):
                        if 0 <= cand < n and cand not in used:
                            v = cand
                            break
                    else:
                        continue
                    break
            used.add(v)
            r[j] = v
    return rows


def differential_evolution(
    criterion: Criterion,
    d: int,
    *,
    candidates: ArrayLike | None = None,
    popsize: int = 15,
    maxiter: int = 40,
    strategy: str = "best1bin",
    mutation: float | tuple[float, float] = (0.5, 1.0),
    recombination: float = 0.7,
    seed: int | None = 42,
    polish: bool = False,
    **kwargs: Any,
) -> SelectionResult:
    """Integer Differential Evolution (Storn & Price, 1997) via SciPy.

    Each individual is a vector of ``d`` integer positions into the candidate
    pool. Duplicate positions are repaired to the nearest unused position.
    Defaults (``best1bin``, popsize 15, 40 generations, seed 42) follow the
    paper's implementation. Extra ``kwargs`` are passed to
    :func:`scipy.optimize.differential_evolution`.
    """
    from scipy.optimize import differential_evolution as _de

    pool = _candidates(criterion, candidates)
    n = len(pool)
    if not 1 <= d <= n:
        raise ValueError(f"d must be in [1, {n}], got {d}")

    n_evals = 0

    def objective(x: NDArray) -> NDArray:
        nonlocal n_evals
        pos = np.clip(np.rint(np.atleast_2d(x.T if x.ndim == 2 else x)), 0, n - 1).astype(int)
        pos = _repair(pos, n)
        n_evals += len(pos)
        return criterion.objective(pool[pos])

    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        res = _de(
            objective,
            bounds=[(0, n - 1)] * d,
            strategy=strategy,
            popsize=popsize,
            maxiter=maxiter,
            mutation=mutation,
            recombination=recombination,
            seed=seed,
            polish=polish,
            integrality=np.ones(d, dtype=bool),
            vectorized=True,
            updating="deferred",
            **kwargs,
        )
    runtime = time.perf_counter() - start
    best_pos = _repair(np.clip(np.rint(res.x), 0, n - 1).astype(int)[None], n)[0]
    idx = pool[best_pos]
    score = float(criterion(idx))
    return _result(
        criterion,
        idx,
        score,
        "differential_evolution",
        runtime,
        n_evals,
        nit=int(res.nit),
        converged=bool(res.success),
    )


# -------------------------------------------------------------------- dispatcher
ALGORITHMS = {
    "brute_force": brute_force,
    "sbs": sequential_backward_selection,
    "sfs": sequential_forward_selection,
    "de": differential_evolution,
}
_ALGO_ALIASES = {
    "brute": "brute_force",
    "bruteforce": "brute_force",
    "exhaustive": "brute_force",
    "differential_evolution": "de",
    "backward": "sbs",
    "forward": "sfs",
}


def select(
    criterion: Criterion,
    d: int | Sequence[int],
    algorithm: str = "sbs",
    **kwargs: Any,
) -> dict[int, SelectionResult]:
    """Select wavelengths for one or several cardinalities ``d``.

    Sequential methods are run once down/up to the extreme ``d``; brute force
    and DE are run separately for each ``d``.

    Returns:
        Mapping ``d → SelectionResult``.
    """
    key = _ALGO_ALIASES.get(algorithm.lower(), algorithm.lower())
    if key not in ALGORITHMS:
        raise KeyError(f"Unknown algorithm {algorithm!r}. Choose from {sorted(ALGORITHMS)}")
    ds = sorted({int(d)} if np.isscalar(d) else {int(x) for x in d})
    if key == "sbs":
        start_set = _candidates(criterion, kwargs.get("initial"))
        path = sequential_backward_selection(criterion, min(ds), **kwargs)
        out = {}
        for x in ds:
            if x == len(start_set):  # the starting set itself
                out[x] = _result(criterion, start_set, criterion(start_set), "sbs", 0.0, 1)
            else:
                out[x] = path[x]  # KeyError if skipped via remove_fraction
        return out
    if key == "sfs":
        path = sequential_forward_selection(criterion, max(ds), **kwargs)
        return {x: path[x] for x in ds}
    func = ALGORITHMS[key]
    return {x: func(criterion, x, **kwargs) for x in ds}
