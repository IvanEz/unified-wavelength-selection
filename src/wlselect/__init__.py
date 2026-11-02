"""wlselect – a unified framework for wavelength selection in diffuse spectroscopy.

Reference implementation of

    K. Scibilia, M. Rosier, L. Giannoni, P. Ricci, F. Lange, F. S. Pavone,
    I. Tachtsidis, D. Rueckert, I. Ezhov. "A unified perspective on wavelength
    selection for molecular composition inference from diffuse spectroscopy",
    arXiv:2606.23223 (2026).

Quick start::

    import wlselect as ws

    data = ws.make_synthetic_dataset()
    crit = ws.ConcentrationPreservation(data.M, data.delta_A, wavelengths=data.wavelengths)
    print(ws.brute_force(crit, d=3))
"""

from . import bll, chromophores, metrics
from .chromophores import AVAILABLE_CHROMOPHORES, DEFAULT_CHROMOPHORES, absorption_matrix
from .criteria import (
    CRITERIA,
    ConcentrationPreservation,
    Criterion,
    DataOrthogonality,
    ModelOrthogonality,
    ProjectionMaximization,
    SpectralFitPreservation,
    get_criterion,
)
from .evaluation import evaluate_subset, max_wavelength_distance, subset_fit
from .io import Dataset, load_dataset, save_dataset
from .search import (
    SelectionPath,
    SelectionResult,
    brute_force,
    differential_evolution,
    select,
    sequential_backward_selection,
    sequential_forward_selection,
)
from .synthetic import SyntheticDataset, make_synthetic_dataset

__version__ = "1.0.0"

__all__ = [
    "AVAILABLE_CHROMOPHORES",
    "CRITERIA",
    "DEFAULT_CHROMOPHORES",
    "ConcentrationPreservation",
    "Criterion",
    "DataOrthogonality",
    "Dataset",
    "ModelOrthogonality",
    "ProjectionMaximization",
    "SelectionPath",
    "SelectionResult",
    "SpectralFitPreservation",
    "SyntheticDataset",
    "absorption_matrix",
    "bll",
    "brute_force",
    "chromophores",
    "differential_evolution",
    "evaluate_subset",
    "get_criterion",
    "load_dataset",
    "make_synthetic_dataset",
    "max_wavelength_distance",
    "metrics",
    "save_dataset",
    "select",
    "sequential_backward_selection",
    "sequential_forward_selection",
    "subset_fit",
    "__version__",
]
