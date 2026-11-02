import itertools
import json

import numpy as np
import pytest

import wlselect as ws


@pytest.fixture
def crit(small_data):
    return ws.ConcentrationPreservation(
        small_data.M, small_data.delta_A, wavelengths=small_data.wavelengths
    )


def test_brute_force_is_global_optimum(crit):
    res = ws.brute_force(crit, 3, chunk_size=97)
    best = min(itertools.combinations(range(crit.n_wavelengths), 3), key=lambda z: crit(z))
    np.testing.assert_array_equal(res.indices, best)
    assert res.score == pytest.approx(crit(best))
    assert res.n_evaluations == 2300  # C(25, 3)
    np.testing.assert_allclose(res.wavelengths, crit.wavelengths[res.indices])


def test_brute_force_parallel_matches_serial(crit):
    a = ws.brute_force(crit, 3, n_jobs=1)
    b = ws.brute_force(crit, 3, n_jobs=2, chunk_size=200)
    np.testing.assert_array_equal(a.indices, b.indices)


def test_brute_force_maximisation(small_data):
    mo = ws.ModelOrthogonality(small_data.M)
    res = ws.brute_force(mo, 3)
    best = max(itertools.combinations(range(mo.n_wavelengths), 3), key=lambda z: mo(z))
    np.testing.assert_array_equal(res.indices, best)


def test_brute_force_candidates_and_limit(crit):
    cand = np.arange(5, 15)
    res = ws.brute_force(crit, 3, candidates=cand)
    assert set(res.indices) <= set(cand)
    with pytest.raises(ValueError):
        ws.brute_force(crit, 12, max_combinations=10)


def test_sbs_path(crit):
    path = ws.sequential_backward_selection(crit, 3)
    assert path.sizes == list(range(24, 2, -1))
    for step in path:
        assert step.score == pytest.approx(crit(step.indices))
    # nested subsets
    for a, b in zip(path.steps[:-1], path.steps[1:]):
        assert set(b.indices) < set(a.indices)
    # SBS can never beat brute force
    assert path[3].score >= ws.brute_force(crit, 3).score - 1e-12


def test_sbs_remove_fraction_and_initial(crit):
    path = ws.sequential_backward_selection(crit, 5, remove_fraction=0.2)
    assert path.final.d == 5
    assert len(path) < 20
    init = np.arange(3, 20)
    path = ws.sequential_backward_selection(crit, 3, initial=init)
    assert set(path.final.indices) <= set(init)


def test_sfs(crit):
    path = ws.sequential_forward_selection(crit, 4)
    assert path.sizes == [1, 2, 3, 4]
    for a, b in zip(path.steps[:-1], path.steps[1:]):
        assert set(a.indices) < set(b.indices)


def test_differential_evolution_finds_optimum(crit):
    bf = ws.brute_force(crit, 3)
    de = ws.differential_evolution(crit, 3, seed=0, maxiter=60)
    assert de.d == 3 and len(set(de.indices)) == 3
    assert de.score <= bf.score * 1.05
    assert de.n_evaluations > 0


def test_de_repair_removes_duplicates():
    from wlselect.search import _repair

    rows = np.array([[3, 3, 3], [0, 0, 1], [4, 4, 4]])
    out = _repair(rows, 5)
    for r in out:
        assert len(set(r)) == 3 and r.min() >= 0 and r.max() < 5


def test_select_dispatch(crit, tmp_path):
    res = ws.select(crit, [3, 4], "sbs")
    assert set(res) == {3, 4}
    res_full = ws.select(crit, [25, 3], "sbs")
    assert res_full[25].score == pytest.approx(0.0, abs=1e-10)
    res_bf = ws.select(crit, 3, "brute")
    assert res_bf[3].algorithm == "brute_force"
    path = ws.sequential_backward_selection(crit, 20)
    path.save_json(tmp_path / "p.json")
    data = json.loads((tmp_path / "p.json").read_text())
    assert data["steps"][-1]["d"] == 20
    with pytest.raises(KeyError):
        ws.select(crit, 3, "magic")


def test_projection_maximization_removes_unmodelled_band(band_data):
    """PM should discard the wavelengths around the unmodelled absorber at 750 nm."""
    pm = ws.ProjectionMaximization(
        band_data.M, band_data.delta_A, wavelengths=band_data.wavelengths
    )
    n = pm.n_wavelengths
    path = ws.sequential_backward_selection(pm, n // 2)
    kept = path.final.wavelengths
    removed = np.setdiff1d(band_data.wavelengths, kept)
    near_band = (band_data.wavelengths >= 740) & (band_data.wavelengths <= 762)
    # (almost) all wavelengths of the contaminated band are removed
    assert np.isin(band_data.wavelengths[near_band], removed).mean() > 0.9
