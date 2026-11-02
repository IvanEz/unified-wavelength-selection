"""Rebuild ``src/wlselect/resources/chromophores_nir.csv`` from the raw UCL spectra.

The raw text files are published by the UCL Biomedical Optics Research
Laboratory (https://www.ucl.ac.uk/medphys/research/borl/intro/spectra).
Download them into one folder and run::

    python scripts/build_chromophore_table.py path/to/spectra

Unit conversion (identical to the processing used for the paper):

* HbO2, HHb  (``hb02.txt``, ``hb.txt``): × 1e4  → cm⁻¹ mM⁻¹
* oxCCO, redCCO (``moody cyt aa3 oxidised/reduced.txt``): × ln(10)
* diffCCO = oxCCO − redCCO
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

LN10 = 2.3025851
OUT = (
    Path(__file__).resolve().parents[1] / "src" / "wlselect" / "resources" / "chromophores_nir.csv"
)

HEADER = """# Specific absorption coefficients of NIR tissue chromophores, 650-999 nm, 1 nm grid.
# Units: cm^-1 mM^-1 (natural-log base, i.e. ln(10) x molar extinction for the CCO spectra).
# Derived from the UCL Biomedical Optics Research Laboratory (BORL) spectra collection:
#   https://www.ucl.ac.uk/medphys/research/borl/intro/spectra
#   HbO2 / HHb : files 'hb02.txt' / 'hb.txt' (x 1e4 to convert from mm^-1 uM^-1)
#   oxCCO/redCCO: Moody cytochrome aa3 oxidised / reduced (x ln 10)
#   diffCCO    : oxCCO - redCCO
# Conversion identical to the processing used in the paper (see scripts/build_chromophore_table.py).
wavelength_nm,HbO2,HHb,oxCCO,redCCO,diffCCO"""


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if parts and parts[0] != "\x00":
            x.append(float(parts[0]))
            y.append(float(parts[1]))
    return np.array(x), np.array(y)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("spectra_dir", type=Path)
    p.add_argument("-o", "--out", type=Path, default=OUT)
    args = p.parse_args()

    grid = np.arange(650, 1000)

    def load(name: str, scale: float) -> np.ndarray:
        x, y = read_spectrum(args.spectra_dir / name)
        return np.interp(grid, x, y) * scale

    hbo2 = load("hb02.txt", 1e4)
    hhb = load("hb.txt", 1e4)
    ox = load("moody cyt aa3 oxidised.txt", LN10)
    red = load("moody cyt aa3 reduced.txt", LN10)
    table = np.column_stack([grid, hbo2, hhb, ox, red, ox - red])
    np.savetxt(
        args.out, table, delimiter=",", header=HEADER, comments="", fmt=["%d"] + ["%.6f"] * 5
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
