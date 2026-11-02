"""Reproduce the piglet experiments of arXiv:2606.23223.

Requires the UCL miniCYRIL piglet data (not public, see README). Either point to
the raw folder with ``--root`` or to a pre-converted ``.npz`` with ``--data``
(created by ``wlselect prepare-piglets``; it must cover the needed wavelength range).

Experiments
-----------
coarse      Fig. 2   – CP, 22 wavelengths in 780–900 nm, 10 timepoints/piglet,
                       brute force / SBS / DE for d = 3 … 21
fine        Fig. 3–5, Table 2 – CP and SFP, 37 wavelengths in 780–900 nm,
                       40 timepoints/piglet, d = 3 … 10, plus runtimes (Fig. 4)
ortho       Table 2  – model orthogonality (37 wl) and data orthogonality
                       (37 wl, 100 timepoints/piglet) with SBS
projection  Fig. 12  – PM band selection (SBS, d = 153) on 740–900 and 730–930 nm
range740    Fig. 10–11 – CP / MO / DO on the 740–900 nm range

Example::

    python experiments/reproduce_paper.py --root "data/miniCYRIL Piglet Data" \\
        --experiment all --out results/ --n-jobs -1
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

import wlselect as ws
from wlselect.io import Dataset

ALGOS = ("brute_force", "sbs", "de")


# ------------------------------------------------------------------ data access
def get_data(args, wl_range, n_wavelengths=None, n_timepoints=None, n_subjects=None) -> Dataset:
    if args.data is not None:
        ds = ws.load_dataset(args.data)
        if ds.wavelengths.min() > wl_range[0] + 1 or ds.wavelengths.max() < wl_range[1] - 1:
            raise SystemExit(
                f"{args.data} covers {ds.wavelengths.min():.0f}-{ds.wavelengths.max():.0f} nm, "
                f"experiment needs {wl_range[0]:.0f}-{wl_range[1]:.0f} nm."
            )
    else:
        from wlselect.datasets import load_piglets

        ds = load_piglets(args.root, wl_range=wl_range, max_timepoints=1000)
    if n_subjects is not None:
        ds = Dataset(
            ds.wavelengths, ds.delta_A[:n_subjects], ds.M, ds.chromophores, ds.subjects[:n_subjects]
        )
    return ds.subsample(wl_range=wl_range, n_wavelengths=n_wavelengths, n_timepoints=n_timepoints)


def run_algorithms(crit, ds_range, args, algos=ALGOS) -> dict[str, dict[int, ws.SelectionResult]]:
    out = {}
    for algo in algos:
        t0 = time.perf_counter()
        if algo == "sbs":
            out[algo] = ws.select(crit, ds_range, "sbs")
        elif algo == "brute_force":
            out[algo] = {
                d: ws.brute_force(crit, d, n_jobs=args.n_jobs, max_combinations=np.inf)
                for d in ds_range
            }
        elif algo == "de":
            out[algo] = {d: ws.differential_evolution(crit, d, seed=args.seed) for d in ds_range}
        print(f"    {algo:<12} done in {time.perf_counter() - t0:8.1f}s")
    return out


def save(results: dict, name: str, out: Path, ds: Dataset, extra_eval: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for crit_key, by_algo in results.items():
        for algo, by_d in by_algo.items():
            for d, r in sorted(by_d.items()):
                row = {
                    "criterion": crit_key,
                    "algorithm": algo,
                    "d": d,
                    "score": r.score,
                    "runtime_s": r.runtime_s,
                    "wavelengths": " ".join(f"{w:.1f}" for w in r.wavelengths),
                }
                if extra_eval:
                    scores = ws.evaluate_subset(r.indices, ds.M, ds.delta_A)
                    row["concentration_nrmse"] = scores["cp"]
                    row["spectral_rmse"] = scores["sfp"]
                rows.append(row)
    with open(out / f"{name}.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        c: {a: {str(d): r.to_dict() for d, r in byd.items()} for a, byd in ba.items()}
        for c, ba in results.items()
    }
    (out / f"{name}.json").write_text(json.dumps(payload, indent=2))
    print(f"  → {out / name}.csv")


def plot_curves(results: dict, crit_key: str, ylabel: str, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from wlselect.plotting import plot_score_curves
    except ImportError:
        return
    names = {"brute_force": "Brute force", "sbs": "SBS", "de": "Differential Evolution"}
    fig, ax = plt.subplots(figsize=(6, 4))
    plot_score_curves({names[a]: r for a, r in results[crit_key].items()}, ax=ax, ylabel=ylabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ experiments
def exp_coarse(args):
    print("[coarse] CP, 22 wavelengths, 10 timepoints/piglet (Fig. 2)")
    ds = get_data(args, (780, 900), n_wavelengths=22, n_timepoints=10)
    crit = ws.ConcentrationPreservation(ds.M, ds.delta_A, wavelengths=ds.wavelengths)
    res = {"cp": run_algorithms(crit, list(range(3, 22)), args)}
    save(res, "coarse_cp", args.out, ds)
    plot_curves(res, "cp", "Concentration NRMSE", args.out / "coarse_cp.png")


def exp_fine(args):
    print("[fine] CP & SFP, 37 wavelengths, 40 timepoints/piglet (Fig. 3–5, Table 2)")
    ds = get_data(args, (780, 900), n_wavelengths=37, n_timepoints=40)
    res = {}
    for key, cls in (("cp", ws.ConcentrationPreservation), ("sfp", ws.SpectralFitPreservation)):
        print(f"  criterion {key}")
        crit = cls(ds.M, ds.delta_A, wavelengths=ds.wavelengths)
        res[key] = run_algorithms(crit, list(range(3, args.max_d + 1)), args)
        plot_curves(res, key, crit.name, args.out / f"fine_{key}.png")
    save(res, "fine_cp_sfp", args.out, ds)


def exp_ortho(args):
    print("[ortho] model & data orthogonality with SBS (Table 2)")
    ds = get_data(args, (780, 900), n_wavelengths=37, n_timepoints=40)
    mo = ws.ModelOrthogonality(ds.M, wavelengths=ds.wavelengths)
    res = {"mo": {"sbs": ws.select(mo, [3, 4, 5, 8], "sbs")}}
    ds100 = get_data(args, (780, 900), n_wavelengths=37, n_timepoints=100)
    do = ws.DataOrthogonality(delta_A=ds100.delta_A, wavelengths=ds100.wavelengths)
    res["do"] = {"sbs": ws.select(do, [3, 4, 5, 8], "sbs")}
    save(res, "orthogonality", args.out, ds)


def exp_projection(args):
    print("[projection] PM band selection, d = 153 (Fig. 12)")
    res = {}
    for lo, hi in ((740, 900), (730, 930)):
        ds = get_data(args, (lo, hi), n_subjects=args.pm_subjects)
        pm = ws.ProjectionMaximization(ds.M, ds.delta_A, wavelengths=ds.wavelengths)
        print(f"  {lo}-{hi} nm: {pm.n_wavelengths} wavelengths")
        path = ws.sequential_backward_selection(pm, 153, progress=True)
        res[f"pm_{lo}_{hi}"] = {"sbs": {153: path[153]}}
        wl = path[153].wavelengths
        print(f"    kept {wl.min():.1f}-{wl.max():.1f} nm")
        path.save_json(args.out / f"projection_{lo}_{hi}_path.json")
    save(res, "projection", args.out, ds, extra_eval=False)


def exp_range740(args):
    print("[range740] CP (brute force / SBS / DE), MO and DO on 740–900 nm (Fig. 10–11)")
    ds = get_data(args, (740, 900), n_wavelengths=37, n_timepoints=40)
    cp = ws.ConcentrationPreservation(ds.M, ds.delta_A, wavelengths=ds.wavelengths)
    res = {"cp": run_algorithms(cp, [3, 4, 8], args)}
    res["mo"] = {
        "sbs": ws.select(ws.ModelOrthogonality(ds.M, wavelengths=ds.wavelengths), [3, 8], "sbs")
    }
    res["do"] = {
        "sbs": ws.select(
            ws.DataOrthogonality(delta_A=ds.delta_A, wavelengths=ds.wavelengths), [3, 8], "sbs"
        )
    }
    save(res, "range740", args.out, ds)


EXPERIMENTS = {
    "coarse": exp_coarse,
    "fine": exp_fine,
    "ortho": exp_ortho,
    "projection": exp_projection,
    "range740": exp_range740,
}


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--root", type=Path, help="raw 'miniCYRIL Piglet Data' folder")
    src.add_argument("--data", type=Path, help="pre-converted .npz dataset")
    p.add_argument("-e", "--experiment", nargs="+", default=["all"], choices=[*EXPERIMENTS, "all"])
    p.add_argument("-o", "--out", type=Path, default=Path("results"))
    p.add_argument("--n-jobs", type=int, default=-1, help="processes for brute force")
    p.add_argument("--max-d", type=int, default=10, help="largest d in the fine experiment")
    p.add_argument("--pm-subjects", type=int, default=10, help="piglets used for PM (paper: 10)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    names = list(EXPERIMENTS) if "all" in args.experiment else args.experiment
    for name in names:
        EXPERIMENTS[name](args)


if __name__ == "__main__":
    main()
