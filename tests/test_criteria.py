"""Batched criterion implementations must match the straightforward formulas of the paper."""

import numpy as np
import pytest

import wlselect as ws
from wlselect import metrics


def _random_subsets(n, d, k=15, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([np.sort(rng.choice(n, d, replace=False)) for _ in range(k)])


def naive_cp(M, As, Z):
    vals = []
    for A in As:
        c = np.linalg.pinv(M) @ A
        cz = np.linalg.pinv(M[Z]) @ A[Z]
        vals.append(metrics.nrmse(c, cz))
    return np.mean(vals)


def naive_sfp(M, As, Z):
    return np.mean([metrics.rmse(A, M @ (np.linalg.pinv(M[Z]) @ A[Z])) for A in As])


def naive_pm(M, As, Z):
    return np.mean([metrics.rmse(A[Z], M[Z] @ (np.linalg.pinv(M[Z]) @ A[Z])) for A in As])


@pytest.mark.parametrize("d", [2, 3, 5, 9])
@pytest.mark.parametrize("strategy", ["direct", "gram"])
def test_concentration_preservation(small_data, d, strategy):
    crit = ws.ConcentrationPreservation(small_data.M, small_data.delta_A, strategy=strategy)
    subs = _random_subsets(crit.n_wavelengths, d)
    expected = [naive_cp(small_data.M, small_data.delta_A, Z) for Z in subs]
    np.testing.assert_allclose(crit.evaluate(subs), expected, rtol=1e-8)


@pytest.mark.parametrize("d", [2, 3, 5, 9])
@pytest.mark.parametrize("strategy", ["direct", "gram"])
def test_spectral_fit(small_data, d, strategy):
    crit = ws.SpectralFitPreservation(small_data.M, small_data.delta_A, strategy=strategy)
    subs = _random_subsets(crit.n_wavelengths, d)
    expected = [naive_sfp(small_data.M, small_data.delta_A, Z) for Z in subs]
    np.testing.assert_allclose(crit.evaluate(subs), expected, rtol=1e-6)


@pytest.mark.parametrize("d", [4, 7, 12])
@pytest.mark.parametrize("strategy", ["direct", "gram"])
def test_projection_maximization(small_data, d, strategy):
    crit = ws.ProjectionMaximization(small_data.M, small_data.delta_A, strategy=strategy)
    subs = _random_subsets(crit.n_wavelengths, d)
    expected = [naive_pm(small_data.M, small_data.delta_A, Z) for Z in subs]
    np.testing.assert_allclose(crit.evaluate(subs), expected, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("metric", sorted(ws.criteria.ORTHOGONALITY_METRICS))
def test_model_orthogonality(small_data, metric):
    crit = ws.ModelOrthogonality(small_data.M, metric=metric)
    fn = {
        "log_sv_product": metrics.log_sv_product,
        "sv_product": metrics.sv_product,
        "min_sv": metrics.min_singular_value,
        "sv_sum": metrics.sv_sum,
        "condition_number": metrics.condition_number,
    }[metric]
    subs = _random_subsets(crit.n_wavelengths, 5)
    np.testing.assert_allclose(crit.evaluate(subs), [fn(small_data.M[Z]) for Z in subs], rtol=1e-8)
    assert crit.maximize == (metric != "condition_number")


@pytest.mark.parametrize("metric", ["log_sv_product", "min_sv", "sv_sum"])
def test_data_orthogonality(small_data, metric):
    crit = ws.DataOrthogonality(delta_A=small_data.delta_A, metric=metric)
    fn = {
        "log_sv_product": metrics.log_sv_product,
        "min_sv": metrics.min_singular_value,
        "sv_sum": metrics.sv_sum,
    }[metric]
    subs = _random_subsets(crit.n_wavelengths, 6)
    expected = [np.mean([fn(A[Z]) for A in small_data.delta_A]) for Z in subs]
    np.testing.assert_allclose(crit.evaluate(subs), expected, rtol=1e-6)


@pytest.mark.parametrize("strategy", ["direct", "gram"])
def test_full_set_is_perfect_for_preservation(small_data, strategy):
    full = np.arange(small_data.M.shape[0])
    for cls in (ws.ConcentrationPreservation, ws.SpectralFitPreservation):
        ref = cls(small_data.M, small_data.delta_A, strategy="direct")(full)
        assert cls(small_data.M, small_data.delta_A, strategy=strategy)(full) == pytest.approx(
            ref, abs=1e-6
        )
    assert (
        ws.ConcentrationPreservation(small_data.M, small_data.delta_A, strategy="direct")(full)
        < 1e-10
    )


def test_single_and_3d_inputs_agree(small_data):
    A3 = np.stack(small_data.delta_A)
    Z = [0, 7, 20]
    a = ws.ConcentrationPreservation(small_data.M, small_data.delta_A)(Z)
    b = ws.ConcentrationPreservation(small_data.M, A3)(Z)
    assert a == pytest.approx(b)
    single = ws.ConcentrationPreservation(small_data.M, small_data.delta_A[0])(Z)
    assert single == pytest.approx(naive_cp(small_data.M, small_data.delta_A[:1], Z))


def test_input_validation(small_data):
    with pytest.raises(ValueError):
        ws.ConcentrationPreservation(small_data.M, None)
    with pytest.raises(ValueError):
        ws.ConcentrationPreservation(small_data.M, small_data.delta_A[0][:-1])
    with pytest.raises(IndexError):
        ws.ConcentrationPreservation(small_data.M, small_data.delta_A)([0, 999])
    with pytest.raises(ValueError):
        ws.ModelOrthogonality(small_data.M, metric="nope")


def test_registry():
    assert ws.get_criterion("Projection Maximization") is ws.ProjectionMaximization
    assert ws.get_criterion("cp") is ws.ConcentrationPreservation
    with pytest.raises(KeyError):
        ws.get_criterion("unknown")
