"""The workflow proposed in the paper: band selection with projection maximisation,
then a minimal wavelength set inside that band.

The synthetic data contain an absorber at ~750 nm that is *not* in the model M
(mimicking the poorly modelled 740–780 nm range of the piglet data, Sec. 3.4).

Run:  python examples/03_projection_maximization.py   (writes projection_maximization.png)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import wlselect as ws  # noqa: E402
from wlselect.plotting import plot_selection, plot_spectral_fit  # noqa: E402

data = ws.make_synthetic_dataset(
    np.arange(740.0, 901.0, 1.0),
    n_subjects=4,
    n_timepoints=120,
    unmodelled_band=(750.0, 12.0),
    seed=7,
)
M, dA, wl = data.M, data.delta_A, data.wavelengths

# Step 1 — band selection: keep half of the wavelengths the model explains best
pm = ws.ProjectionMaximization(M, dA, wavelengths=wl)
band = ws.sequential_backward_selection(pm, d=len(wl) // 2, remove_fraction=0.02).final
print(
    f"PM band: {len(band.indices)} of {len(wl)} wavelengths, "
    f"{band.wavelengths.min():.0f}–{band.wavelengths.max():.0f} nm"
)

# Step 2 — minimal set: (a) without and (b) with the PM band as candidate pool
cp = ws.ConcentrationPreservation(M, dA, wavelengths=wl)
naive = ws.sequential_backward_selection(cp, d=3).final
within = ws.sequential_backward_selection(cp, d=3, initial=band.indices).final
print("CP on full range :", np.round(naive.wavelengths, 1))
print("CP within PM band:", np.round(within.wavelengths, 1))

# Ground truth check: which set recovers the true concentrations better?
for label, res in (("full range", naive), ("PM band", within)):
    errs = [
        ws.metrics.nrmse(c_true, np.linalg.pinv(M[res.indices]) @ A[res.indices])
        for c_true, A in zip(data.delta_c, dA)
    ]
    print(f"  NRMSE vs. ground-truth concentrations ({label}): {np.mean(errs):.3f}")

# Figure ------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
plot_selection(wl, M, {"PM band": band}, ax=axes[0], title="Projection maximisation band")
t = int(np.argmax(np.abs(dA[0]).sum(0)))
fit = ws.subset_fit(within.indices, M, dA[0])
fit_naive = ws.subset_fit(naive.indices, M, dA[0])
plot_spectral_fit(
    wl,
    dA[0][:, t],
    {
        "True chromophore signal": (M @ data.delta_c[0])[:, t],
        "3 WLs, full range": fit_naive["A_subset"][:, t],
        "3 WLs, PM band": fit["A_subset"][:, t],
    },
    ax=axes[1],
    title=f"Model reconstruction at t={t}",
)
fig.tight_layout()
fig.savefig("projection_maximization.png", dpi=150)
print("saved projection_maximization.png")
