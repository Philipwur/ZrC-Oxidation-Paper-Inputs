"""Shared locations for the analysis scripts.

The scripts read the trajectory files of the 16 simulations (`<T>_<cell>.xyz`)
and their gas-exchange logs (`RateAnalysis_<T>_<cell>.csv`). Set the
environment variable ZRC_DATA_DIR to the folder that holds them. Without it
the scripts look in `analysis/data/`.

Scripts in a subfolder import this module as

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import Paths as RP

Run this file directly to check that the data are found.
"""

import os
import sys
from pathlib import Path

REVIEW_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("ZRC_DATA_DIR", REVIEW_DIR / "data"))
REWRITE_DIR = REVIEW_DIR / "core"

TEMPERATURES = (873, 973, 1073, 1273)
CELLS = (1, 2, 3, 4)
HORIZON_PS = 140.0
SEED = 20260715

LIB_DIR = REVIEW_DIR / "lib"

# The scripts import the modules in core/ and the shared modules in lib/ by bare
# name, so both folders have to be importable. Appended (not inserted) so that
# local scripts win on a clash.
for _folder in (REWRITE_DIR, LIB_DIR):
    if str(_folder) not in sys.path:
        sys.path.append(str(_folder))


def trajectory(temperature, cell):
    """Path of one trajectory, e.g. trajectory(973, 1) -> .../973_1.xyz."""
    return DATA_DIR / f"{temperature}_{cell}.xyz"


def gas_log(temperature, cell):
    """Path of one gas-exchange log, e.g. .../RateAnalysis_973_1.csv."""
    return DATA_DIR / f"RateAnalysis_{temperature}_{cell}.csv"


def results_dir(script_file, name=None):
    """Results folder beside the calling script: <ScriptName>Results/."""
    script = Path(script_file).resolve()
    out = script.parent / (name or f"{script.stem}Results")
    out.mkdir(exist_ok=True)
    return out


if __name__ == "__main__":
    for label, path in (("data", DATA_DIR), ("core scripts", REWRITE_DIR)):
        print(f"{label:13s} {'ok     ' if path.is_dir() else 'MISSING'} {path}")
    missing = [p.name for t in TEMPERATURES for c in CELLS
               for p in (trajectory(t, c), gas_log(t, c)) if not p.is_file()]
    print(f"trajectories and gas logs: {32 - len(missing)}/32 found")
    if missing:
        print("missing:", ", ".join(missing))
