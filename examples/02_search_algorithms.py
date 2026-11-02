"""Brute force vs. SBS vs. Differential Evolution (cf. paper Fig. 2–4).

Run:  python examples/02_search_algorithms.py   (writes search_algorithms.png)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import wlselect as ws  # noqa: E402
from wlselect.plotting import plot_score_curves, plot_selection  # noqa: E402

# Coarse setting as in the paper: 22 equally spaced wavelengths in 780–900 nm,
# few timepoints per subject so that brute force over all d is feasible.
data = ws.make_synthetic_dataset(np.linspace(780, 900, 22), n_subjects=6, n_timepoints=10, seed=3)
crit = ws.ConcentrationPreservation(data.M, data.delta_A, wavelengths=data.wavelengths)

ds = list(range(3, 21))
results = {
    "Brute force": {d: ws.brute_force(crit, d) for d in ds},
    "SBS": ws.select(crit, ds, "sbs"),
    "Differential Evolution": {d: ws.differential_evolution(crit, d, seed=42) for d in ds},
}
for name, res in results.items():
    t = sum(r.runtime_s for r in res.values())
    print(
        f"{name:<23} d=3: {np.round(res[3].wavelengths, 1)}  NRMSE={res[3].score:.4f}  total {t:.1f}s"
    )

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
plot_score_curves(results, ax=axes[0], ylabel="Concentration NRMSE", title="Concentration error")
plot_selection(
    data.wavelengths,
    data.M,
    {k: v[3] for k, v in results.items()},
    ax=axes[1],
    title="Selected 3 wavelengths",
)
fig.tight_layout()
fig.savefig("search_algorithms.png", dpi=150)
print("saved search_algorithms.png")
