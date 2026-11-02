import numpy as np
import pytest
from scipy.io import savemat

import wlselect as ws
from wlselect import bll
from wlselect.cli import main
from wlselect.datasets import load_piglets
from wlselect.datasets.piglets import load_piglet


# ------------------------------------------------------------------ chromophores
def test_chromophore_values():
    M = ws.absorption_matrix([760.0, 830.0])
    # HHb peak at ~760 nm, diffCCO maximum at ~830 nm (cf. paper Fig. 8)
    assert M[0, 1] == pytest.approx(3.855, abs=1e-3)
    assert M[1, 2] == pytest.approx(5.403, abs=1e-3)
    wl, spectra = ws.chromophores.load_spectra()
    assert wl[0] == 650 and wl[-1] == 999
    np.testing.assert_allclose(spectra["diffCCO"], spectra["oxCCO"] - spectra["redCCO"], atol=2e-6)


def test_chromophore_options():
    M = ws.absorption_matrix([800, 810], ["hbo2", "Hb"], pathlength=[2.0, 4.0])
    M0 = ws.absorption_matrix([800, 810], ["HbO2", "HHb"])
    np.testing.assert_allclose(M, M0 * [[2.0], [4.0]])
    with pytest.raises(ValueError):
        ws.absorption_matrix([600.0])
    with pytest.raises(KeyError):
        ws.absorption_matrix([800.0], ["melanin"])
    custom = ws.absorption_matrix([800.0], ["X"], spectra={"X": ([700, 900], [1.0, 3.0])})
    assert custom[0, 0] == pytest.approx(2.0)


# --------------------------------------------------------------------------- BLL
def test_bll_roundtrip():
    M = ws.absorption_matrix(np.arange(780, 901, 4.0))
    c = np.random.default_rng(0).normal(size=(3, 10))
    np.testing.assert_allclose(bll.infer_concentrations(bll.forward_model(c, M), M), c, atol=1e-10)
    intens = np.exp(-bll.forward_model(c, M))
    dA = bll.attenuation_change(intens * np.exp(0.3), reference=np.exp(0.3) * np.ones(len(M)))
    np.testing.assert_allclose(dA, M @ c, atol=1e-10)


# ------------------------------------------------------------------------- io
def test_dataset_roundtrip(tmp_path, small_data):
    ds = ws.Dataset(small_data.wavelengths, small_data.delta_A, small_data.M)
    ws.save_dataset(tmp_path / "d.npz", ds)
    back = ws.load_dataset(tmp_path / "d.npz")
    np.testing.assert_allclose(back.M, ds.M)
    assert len(back.delta_A) == 3 and back.subjects == ds.subjects
    sub = back.subsample(n_wavelengths=7, n_timepoints=5)
    assert sub.n_wavelengths == 7 and sub.delta_A[0].shape == (7, 5)
    assert sub.wavelengths[0] == 780 and sub.wavelengths[-1] == 900
    rng = back.subsample(wl_range=(800, 850), time_step=10)
    assert rng.wavelengths.min() >= 800 and rng.delta_A[0].shape[1] == 3


# ----------------------------------------------------------------- piglet loader
def _fake_piglet_root(tmp_path):
    root = tmp_path / "miniCYRIL Piglet Data"
    wl = np.linspace(700, 950, 120)
    (root / "LWP480").mkdir(parents=True)
    white = np.full((120, 1), 1000.0)
    dark = np.full((120, 1), 10.0)
    savemat(root / "LWP480" / "refSpectrum.mat", {"refSpectrum": white})
    savemat(root / "LWP480" / "LWP480_DarkCount_12Dec2016.mat", {"DarkCount": dark})
    rng = np.random.default_rng(0)
    spectra = {}
    for name, fname in [
        ("lwp475", "LWP475_Ws_14Nov2016_1.mat"),
        ("LWP480", "LWP480_Ws_12Dec2016.mat"),
    ]:
        (root / name).mkdir(exist_ok=True)
        S = 10 + 500 * rng.uniform(0.5, 1.0, size=(120, 1)) * rng.uniform(0.9, 1.1, size=(1, 1200))
        spectra[name] = S
        savemat(
            root / name / fname,
            {
                "wavelengths": wl[:, None],
                "spectralDataAll": S,
                "AllConcentration": np.zeros((3, 1200)),
            },
        )
    return root, wl, spectra, white, dark


def test_piglet_loader(tmp_path):
    root, wl, spectra, white, dark = _fake_piglet_root(tmp_path)
    w, dA = load_piglet(root, "lwp475", wl_range=(780, 900))
    keep = (wl >= 780) & (wl <= 900)
    S = spectra["lwp475"][keep][:, :1000]
    R = (S - dark[keep]) / (white[keep] - dark[keep])
    np.testing.assert_allclose(dA, np.log(R[:, :1] / R))
    assert dA.shape == (keep.sum(), 1000)
    np.testing.assert_allclose(dA[:, 0], 0)

    ds = load_piglets(root, ["lwp475", "LWP480"], wl_range=(780, 900))
    assert ds.subjects == ["lwp475", "LWP480"] and ds.M.shape == (keep.sum(), 3)
    with pytest.warns(UserWarning, match="Skipping"):
        ds2 = load_piglets(root, None, skip_missing=True)
    assert set(ds2.subjects) == {"lwp475", "LWP480"}


# --------------------------------------------------------------------------- CLI
def test_cli_end_to_end(tmp_path, capsys):
    data = tmp_path / "syn.npz"
    assert main(["make-synthetic", "-o", str(data), "--subjects", "2", "--timepoints", "40"]) == 0
    out = tmp_path / "res.json"
    assert (
        main(
            [
                "select",
                "--data",
                str(data),
                "-c",
                "cp",
                "-a",
                "brute",
                "-d",
                "3",
                "--wl-range",
                "780",
                "900",
                "--n-wavelengths",
                "15",
                "-o",
                str(out),
                "-q",
            ]
        )
        == 0
    )
    assert out.exists()
    band = tmp_path / "band.json"
    assert (
        main(
            [
                "select",
                "--data",
                str(data),
                "-c",
                "pm",
                "-a",
                "sbs",
                "-d",
                "80",
                "--remove-fraction",
                "0.05",
                "-q",
                "-o",
                str(tmp_path / "pm.json"),
            ]
        )
        == 0
    )
    # a `select` result file can be used directly as a band
    cmd = ["select", "--data", str(data), "-c", "cp", "-a", "de", "-d", "3"]
    assert main([*cmd, "--band", str(tmp_path / "pm.json")]) == 0
    # use a PM band as the starting set
    import json

    band.write_text(json.dumps({"wavelengths": list(np.arange(780.0, 901.0, 4.0))}))
    assert (
        main(
            [
                "select",
                "--data",
                str(data),
                "-c",
                "mo",
                "-a",
                "sbs",
                "-d",
                "3",
                "--band",
                str(band),
                "-q",
            ]
        )
        == 0
    )
    assert main(["evaluate", "--data", str(data), "-w", "780", "830", "900"]) == 0
    printed = capsys.readouterr().out
    assert "Concentration Preservation" in printed
