"""Quick start: compare the five selection criteria on synthetic data.

Run:  python examples/01_quickstart.py
"""

import numpy as np

import wlselect as ws

# 1) Data ---------------------------------------------------------------------
# ΔA has shape (n_wavelengths, n_timepoints); one array per subject.
data = ws.make_synthetic_dataset(
    wavelengths=np.arange(780.0, 901.0, 2.0),  # 61 wavelengths
    n_subjects=4,
    n_timepoints=100,
    unmodelled_band=None,
    seed=0,
)
M, dA, wl = data.M, data.delta_A, data.wavelengths
print(f"M: {M.shape}, subjects: {len(dA)}, ΔA per subject: {dA[0].shape}")

# 2) Criteria -----------------------------------------------------------------
criteria = {
    "Concentration Preservation": ws.ConcentrationPreservation(M, dA, wavelengths=wl),
    "Spectral Fit Preservation": ws.SpectralFitPreservation(M, dA, wavelengths=wl),
    "Model Orthogonality": ws.ModelOrthogonality(M, wavelengths=wl),
    "Data Orthogonality": ws.DataOrthogonality(delta_A=dA, wavelengths=wl),
}

# 3) Select d = 3 wavelengths with SBS (one run yields every cardinality) -------
print("\nSelected wavelengths (SBS):")
for name, crit in criteria.items():
    path = ws.sequential_backward_selection(crit, d=3)
    for d in (3, 5):
        res = path[d]
        print(f"  {name:<27} d={d}: {np.round(res.wavelengths, 1)}")

# 4) Brute force for the preservation criteria --------------------------------
print("\nBrute force (d = 3):")
for name in ("Concentration Preservation", "Spectral Fit Preservation"):
    res = ws.brute_force(criteria[name], d=3)
    print(f"  {name:<27} {np.round(res.wavelengths, 1)}  J={res.score:.4g}  ({res.runtime_s:.2f}s)")

# 5) How does one subset score under all criteria? -----------------------------
subset = ws.brute_force(criteria["Concentration Preservation"], d=3).indices
print("\nCross-evaluation of the CP-optimal set:")
for key, val in ws.evaluate_subset(subset, M, dA).items():
    print(f"  {ws.CRITERIA[key].name:<27} {val:.4g}")
