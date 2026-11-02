"""Command-line interface: ``wlselect <command> ...``.

Commands
--------
select           Run a wavelength selection on a dataset (.npz).
evaluate         Score given wavelengths under all five criteria.
prepare-piglets  Convert the raw miniCYRIL piglet .mat files into a .npz dataset.
make-synthetic   Write a synthetic demo dataset.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .criteria import CRITERIA, ORTHOGONALITY_METRICS, get_criterion
from .evaluation import evaluate_subset
from .io import Dataset, load_dataset, save_dataset
from .search import select


def _subsample_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("subsampling")
    g.add_argument(
        "--wl-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="restrict to a wavelength range in nm",
    )
    g.add_argument("--n-wavelengths", type=int, help="keep N equally spaced wavelengths")
    g.add_argument("--wl-step", type=int, help="keep every k-th wavelength")
    g.add_argument("--n-timepoints", type=int, help="keep N equally spaced timepoints per subject")
    g.add_argument("--time-step", type=int, help="keep every k-th timepoint")


def _load(args: argparse.Namespace) -> Dataset:
    ds = load_dataset(args.data, chromophores=args.chromophores)
    if any(
        getattr(args, k) is not None
        for k in ("wl_range", "n_wavelengths", "wl_step", "n_timepoints", "time_step")
    ):
        ds = ds.subsample(
            wl_range=tuple(args.wl_range) if args.wl_range else None,
            n_wavelengths=args.n_wavelengths,
            wl_step=args.wl_step,
            n_timepoints=args.n_timepoints,
            time_step=args.time_step,
        )
    return ds


def _build_criterion(name: str, ds: Dataset, metric: str):
    cls = get_criterion(name)
    kwargs = {"wavelengths": ds.wavelengths}
    if cls.key in ("mo", "do"):
        kwargs["metric"] = metric
    return cls(ds.M, ds.delta_A, **kwargs)


def _read_band(path: Path) -> np.ndarray:
    """Wavelengths from ``{"wavelengths": [...]}`` or a ``wlselect select`` result file."""
    payload = json.loads(Path(path).read_text())
    if "wavelengths" in payload:
        return np.asarray(payload["wavelengths"], dtype=float)
    if "results" in payload and payload["results"]:
        first = next(iter(payload["results"].values()))
        return np.asarray(first["wavelengths"], dtype=float)
    raise SystemExit(f"{path}: no 'wavelengths' found.")


def _cmd_select(args: argparse.Namespace) -> int:
    ds = _load(args)
    crit = _build_criterion(args.criterion, ds, args.metric)

    kwargs: dict = {}
    if args.band:
        band_wl = _read_band(args.band)
        idx = np.flatnonzero(
            np.isclose(ds.wavelengths[:, None], band_wl[None, :], atol=1e-3).any(1)
        )
        if len(idx) != len(band_wl):
            raise SystemExit("Some band wavelengths are not present in the dataset.")
        # SBS starts from the band; the other algorithms choose within it
        kwargs["initial" if args.algorithm == "sbs" else "candidates"] = idx
    if args.algorithm == "sbs" and args.remove_fraction:
        kwargs["remove_fraction"] = args.remove_fraction
    if args.algorithm in ("brute_force", "brute"):
        kwargs["n_jobs"] = args.n_jobs
    if args.algorithm == "de":
        kwargs.update(seed=args.seed, popsize=args.popsize, maxiter=args.maxiter)
    else:
        kwargs["progress"] = not args.quiet

    print(
        f"[wlselect] {crit.name} | {args.algorithm} | n={crit.n_wavelengths} wavelengths "
        f"({ds.wavelengths.min():.1f}-{ds.wavelengths.max():.1f} nm), "
        f"{len(ds.delta_A)} subject(s), {sum(a.shape[1] for a in ds.delta_A)} spectra",
        file=sys.stderr,
    )
    results = select(crit, args.d, args.algorithm, **kwargs)
    for d, res in sorted(results.items()):
        wl = ", ".join(f"{w:.1f}" for w in res.wavelengths)
        print(f"d={d:>3}  J={res.score:.6g}  t={res.runtime_s:.2f}s  [{wl}]")

    if args.out:
        payload = {
            "criterion": crit.key,
            "criterion_name": crit.name,
            "algorithm": args.algorithm,
            "data": str(args.data),
            "results": {str(d): r.to_dict() for d, r in sorted(results.items())},
        }
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"[wlselect] results written to {args.out}", file=sys.stderr)
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    ds = _load(args)
    target = np.asarray(args.wavelengths, dtype=float)
    idx = np.array([int(np.argmin(np.abs(ds.wavelengths - w))) for w in target])
    scores = evaluate_subset(idx, ds.M, ds.delta_A, metric=args.metric)
    print("selected:", ", ".join(f"{w:.2f}" for w in ds.wavelengths[idx]), "nm")
    for key, val in scores.items():
        print(f"  {CRITERIA[key].name:<27} {val:.6g}")
    return 0


def _cmd_prepare_piglets(args: argparse.Namespace) -> int:
    from .datasets.piglets import HELD_OUT_PIGLETS, load_piglets

    ds = load_piglets(
        args.root,
        piglets=args.piglets,
        wl_range=tuple(args.wl_range),
        max_timepoints=args.max_timepoints,
        exclude=() if args.no_exclude else HELD_OUT_PIGLETS,
        chromophores=args.chromophores or ("HbO2", "HHb", "diffCCO"),
        skip_missing=args.skip_missing,
    )
    save_dataset(args.out, ds)
    print(
        f"Saved {len(ds.subjects)} piglets, {ds.n_wavelengths} wavelengths "
        f"({ds.wavelengths.min():.1f}-{ds.wavelengths.max():.1f} nm) to {args.out}"
    )
    return 0


def _cmd_make_synthetic(args: argparse.Namespace) -> int:
    from .synthetic import make_synthetic_dataset

    band = None if args.no_unmodelled else (args.band_center, args.band_width)
    syn = make_synthetic_dataset(
        np.arange(args.wl_range[0], args.wl_range[1] + 1e-9, args.wl_spacing),
        n_subjects=args.subjects,
        n_timepoints=args.timepoints,
        noise_std=args.noise,
        unmodelled_band=band,
        seed=args.seed,
    )
    save_dataset(
        args.out,
        Dataset(syn.wavelengths, syn.delta_A, syn.M, syn.chromophores),
    )
    print(f"Saved synthetic dataset ({syn.n_subjects} subjects) to {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wlselect",
        description="Unified wavelength selection for diffuse spectroscopy (arXiv:2606.23223).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    # select ---------------------------------------------------------------
    p = sub.add_parser("select", help="run a wavelength selection")
    p.add_argument("--data", required=True, type=Path, help="dataset .npz")
    p.add_argument(
        "-c", "--criterion", required=True, help=f"one of {sorted(CRITERIA)} (or long name)"
    )
    p.add_argument(
        "-a", "--algorithm", default="sbs", choices=["sbs", "sfs", "brute_force", "brute", "de"]
    )
    p.add_argument("-d", nargs="+", type=int, required=True, help="subset size(s)")
    p.add_argument(
        "--metric",
        default="log_sv_product",
        choices=sorted(ORTHOGONALITY_METRICS),
        help="singular-value metric for mo/do",
    )
    p.add_argument("--chromophores", nargs="+", help="rebuild M from these chromophores")
    p.add_argument(
        "--band",
        type=Path,
        help="JSON result file whose 'wavelengths' restrict the search "
        "(e.g. a projection-maximisation band)",
    )
    p.add_argument("--remove-fraction", type=float, help="SBS: remove this fraction per step")
    p.add_argument("--n-jobs", type=int, default=1, help="brute force: worker processes (-1 = all)")
    p.add_argument("--seed", type=int, default=42, help="DE: random seed")
    p.add_argument("--popsize", type=int, default=15, help="DE: population size multiplier")
    p.add_argument("--maxiter", type=int, default=40, help="DE: generations")
    p.add_argument("-o", "--out", type=Path, help="write results to JSON")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    _subsample_args(p)
    p.set_defaults(func=_cmd_select)

    # evaluate -------------------------------------------------------------
    p = sub.add_parser("evaluate", help="score given wavelengths under all criteria")
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("-w", "--wavelengths", nargs="+", type=float, required=True)
    p.add_argument("--metric", default="log_sv_product", choices=sorted(ORTHOGONALITY_METRICS))
    p.add_argument("--chromophores", nargs="+")
    _subsample_args(p)
    p.set_defaults(func=_cmd_evaluate)

    # prepare-piglets ---------------------------------------------------------
    p = sub.add_parser("prepare-piglets", help="convert raw miniCYRIL .mat files to .npz")
    p.add_argument("--root", required=True, type=Path, help="'miniCYRIL Piglet Data' folder")
    p.add_argument("-o", "--out", required=True, type=Path)
    p.add_argument("--wl-range", nargs=2, type=float, default=(780.0, 900.0))
    p.add_argument("--max-timepoints", type=int, default=1000)
    p.add_argument("--piglets", nargs="+", help="subset of subjects (default: all but held-out)")
    p.add_argument("--no-exclude", action="store_true", help="also load held-out subjects")
    p.add_argument("--chromophores", nargs="+")
    p.add_argument("--skip-missing", action="store_true")
    p.set_defaults(func=_cmd_prepare_piglets)

    # make-synthetic -------------------------------------------------------
    p = sub.add_parser("make-synthetic", help="write a synthetic demo dataset")
    p.add_argument("-o", "--out", required=True, type=Path)
    p.add_argument("--wl-range", nargs=2, type=float, default=(740.0, 900.0))
    p.add_argument("--wl-spacing", type=float, default=1.0)
    p.add_argument("--subjects", type=int, default=4)
    p.add_argument("--timepoints", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.004)
    p.add_argument(
        "--band-center", type=float, default=750.0, help="centre of the unmodelled absorber (nm)"
    )
    p.add_argument("--band-width", type=float, default=12.0)
    p.add_argument("--no-unmodelled", action="store_true", help="disable the unmodelled absorber")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_make_synthetic)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
