import numpy as np
import pytest

import wlselect as ws


@pytest.fixture(scope="session")
def small_data():
    """Small synthetic problem: 25 wavelengths, 3 subjects, 30 timepoints each."""
    return ws.make_synthetic_dataset(
        np.arange(780.0, 901.0, 5.0), n_subjects=3, n_timepoints=30, unmodelled_band=None, seed=1
    )


@pytest.fixture(scope="session")
def band_data():
    """Data with an unmodelled absorber at 750 nm (740–900 nm, 2 nm spacing)."""
    return ws.make_synthetic_dataset(
        np.arange(740.0, 901.0, 2.0),
        n_subjects=2,
        n_timepoints=60,
        unmodelled_band=(750.0, 10.0),
        unmodelled_amplitude=0.3,
        seed=2,
    )
