"""One pass over the raw trajectories: Zr and C neighbour counts of every
tracked carbon at every MD step.

Why: the state definitions (thresholds, cutoffs, the 0.2 ps filter) can then
be applied and varied exactly, at full time resolution, without reading the
950 MB trajectory files again. The cached 50 fs frames only allow the filter
to be applied on that grid, which lets through more short excursions.

For each trajectory this writes StateModelResults/RawCoordination/<name>.npz:
  Times_fs      (n_frames,)          frame times
  NZr           (3, n_frames, 48)    Zr neighbours within 0.9, 1.0, 1.1 x 3.33 A
  NC            (3, n_frames, 48)    C neighbours within 0.9, 1.0, 1.1 x 2.31 A
  ZrCutoffs_A, CCutoffs_A            the cutoffs used
Counts are -1 once a carbon has been removed. Carbon identities are tracked
with the production routine (BootstrapCalculation.MapAtomIdentities), so
column k is the same carbon as AtomID k everywhere else.

Reading uses the production frame reader. Trajectories are read-only.

Run:  python RawCoordinationPass.py              all sixteen, 8 processes
      python RawCoordinationPass.py --test 2000  first 2000 frames of 973_1, timing only
"""

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP  # noqa: E402

import BootstrapCalculation as Bc  # noqa: E402

OUT = Path(__file__).resolve().parent / "StateModelResults" / "RawCoordination"
SCALES = (0.9, 1.0, 1.1)
HORIZON_FS = RP.HORIZON_PS * 1000.0 + 0.5


def run(task):
    temperature, cell, max_frames = task
    name = f"{temperature}_{cell}"
    target = OUT / f"{name}.npz"
    if target.exists() and max_frames is None:
        return name, 0, 0.0, "already done"
    started = time.time()
    zr_cut = np.array(SCALES) * Bc.CutoffZrC
    c_cut = np.array(SCALES) * Bc.CutoffCC
    times, n_zr, n_c = [], [], []
    tracked = np.arange(Bc.TotalStartC)
    previous = None
    for time_fs, atoms, coords, lattice in Bc.ReadTrajectoryGenerator(str(RP.trajectory(temperature, cell)), max_frames):
        if time_fs > HORIZON_FS:
            break
        frac = Bc.CartesianToFractional(coords, lattice)
        frac_c = frac[atoms == "C"]
        if previous is not None:
            tracked = Bc.MapAtomIdentities(previous, frac_c, tracked, lattice)
        elif len(frac_c) < len(tracked):
            tracked = tracked[:len(frac_c)]
        previous = frac_c
        row_zr = np.full((3, Bc.TotalStartC), -1, dtype=np.int8)
        row_c = np.full((3, Bc.TotalStartC), -1, dtype=np.int8)
        if len(frac_c):
            d_zr = Bc.MinimumImageCrossDistances(frac_c, frac[atoms == "Zr"], lattice)
            d_c = Bc.MinimumImageDistances(frac_c, lattice)
            np.fill_diagonal(d_c, np.inf)
            for k in range(3):
                row_zr[k, tracked] = (d_zr < zr_cut[k]).sum(1)
                row_c[k, tracked] = (d_c < c_cut[k]).sum(1)
        times.append(time_fs)
        n_zr.append(row_zr)
        n_c.append(row_c)
    elapsed = time.time() - started
    if max_frames is None:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, Times_fs=np.array(times), NZr=np.stack(n_zr, axis=1), NC=np.stack(n_c, axis=1),
                            ZrCutoffs_A=zr_cut, CCutoffs_A=c_cut)
    return name, len(times), elapsed, "written" if max_frames is None else "test only"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None, help="time the first N frames of 973_1 and write nothing")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.test:
        name, n, elapsed, status = run((973, 1, args.test))
        print(f"{name}: {n} frames in {elapsed:.1f} s ({1e3 * elapsed / n:.2f} ms per frame), {status}")
        return
    tasks = [(t, c, None) for t in RP.TEMPERATURES for c in RP.CELLS]
    with Pool(args.jobs) as pool:
        for name, n, elapsed, status in pool.imap_unordered(run, tasks):
            print(f"{name}: {n} frames, {elapsed / 60:.1f} min, {status}", flush=True)
    print("Raw pass finished:", OUT)


if __name__ == "__main__":
    main()
