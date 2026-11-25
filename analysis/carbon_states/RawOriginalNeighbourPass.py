"""One pass over the raw trajectories: for every tracked carbon at every MD
step, how many of its ORIGINAL Zr neighbours are still within the cutoff.

Why: counting any Zr within 3.33 A cannot tell a bonded carbon that is still
in its lattice site from one that has left it and sits near other Zr atoms.
Counting only the Zr atoms that were the carbon's neighbours at the start can.

Original neighbours of a carbon: the Zr atoms within 3.33 A in the first
frame (five for a surface carbon, six below the surface). The nearest Zr
shell of ZrC is near 2.35 A and the next near 4.1 A, so this set is the same
for any cutoff between 3.00 and 3.66 A.

Zr atoms are followed from frame to frame with the same routine that follows
carbon (BootstrapCalculation.MapAtomIdentities), so the result does not rely
on the order of atoms in the file.

For each trajectory this writes StateModelResults/RawOriginalNeighbours/<name>.npz:
  Times_fs      (n_frames,)          frame times, identical to RawCoordination
  NOriginal     (3, n_frames, 48)    original Zr neighbours within 0.9, 1.0, 1.1 x 3.33 A
  NStart        (48,)                number of original neighbours of each carbon
  ZrCutoffs_A                        the cutoffs used
  LargestZrStep_A                    largest move of one Zr between two MD steps
Counts are -1 once a carbon has been removed.

Run:  python RawOriginalNeighbourPass.py              all sixteen, 8 processes
      python RawOriginalNeighbourPass.py --test 2000  first 2000 frames of 973_1, timing only
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

OUT = Path(__file__).resolve().parent / "StateModelResults" / "RawOriginalNeighbours"
SCALES = (0.9, 1.0, 1.1)
HORIZON_FS = RP.HORIZON_PS * 1000.0 + 0.5


def run(task):
    temperature, cell, max_frames = task
    name = f"{temperature}_{cell}"
    target = OUT / f"{name}.npz"
    if target.exists() and max_frames is None:
        return name, 0, 0.0, "already done", 0.0
    started = time.time()
    zr_cut = np.array(SCALES) * Bc.CutoffZrC
    times, rows = [], []
    tracked = np.arange(Bc.TotalStartC)
    prev_c = prev_zr = zr_ids = original = None
    largest = 0.0
    for time_fs, atoms, coords, lattice in Bc.ReadTrajectoryGenerator(str(RP.trajectory(temperature, cell)), max_frames):
        if time_fs > HORIZON_FS:
            break
        frac = Bc.CartesianToFractional(coords, lattice)
        frac_c, frac_zr = frac[atoms == "C"], frac[atoms == "Zr"]
        if prev_c is not None:
            tracked = Bc.MapAtomIdentities(prev_c, frac_c, tracked, lattice)
            if len(frac_zr) != len(prev_zr):
                raise ValueError(f"{name}: number of Zr atoms changed at {time_fs} fs")
            new_ids = Bc.MapAtomIdentities(prev_zr, frac_zr, zr_ids, lattice)
            step = frac_zr[np.argsort(new_ids)] - prev_zr[np.argsort(zr_ids)]
            step -= np.round(step)
            largest = max(largest, float(np.linalg.norm(step @ lattice, axis=1).max()))
            zr_ids = new_ids
        else:
            if len(frac_c) != Bc.TotalStartC:
                raise ValueError(f"{name}: first frame has {len(frac_c)} carbons")
            zr_ids = np.arange(len(frac_zr))
        prev_c, prev_zr = frac_c, frac_zr
        row = np.full((3, Bc.TotalStartC), -1, dtype=np.int8)
        if len(frac_c):
            d_zr = Bc.MinimumImageCrossDistances(frac_c, frac_zr, lattice)
            if original is None:
                original = d_zr < Bc.CutoffZrC                   # (carbon ID, Zr ID), first frame
            mine = original[tracked][:, zr_ids]                  # (current carbon, current Zr)
            for k in range(3):
                row[k, tracked] = ((d_zr < zr_cut[k]) & mine).sum(1)
        times.append(time_fs)
        rows.append(row)
    elapsed = time.time() - started
    if max_frames is None:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, Times_fs=np.array(times), NOriginal=np.stack(rows, axis=1), NStart=original.sum(1).astype(np.int8),
                            ZrCutoffs_A=zr_cut, LargestZrStep_A=largest)
    return name, len(times), elapsed, "written" if max_frames is None else "test only", largest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, default=None, help="time the first N frames of 973_1 and write nothing")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if args.test:
        name, n, elapsed, status, largest = run((973, 1, args.test))
        print(f"{name}: {n} frames in {elapsed:.1f} s ({1e3 * elapsed / n:.2f} ms per frame), largest Zr step {largest:.3f} A, {status}")
        return
    tasks = [(t, c, None) for t in RP.TEMPERATURES for c in RP.CELLS]
    with Pool(args.jobs) as pool:
        for name, n, elapsed, status, largest in pool.imap_unordered(run, tasks):
            print(f"{name}: {n} frames, {elapsed / 60:.1f} min, largest Zr step {largest:.3f} A, {status}", flush=True)
    print("Raw pass finished:", OUT)


if __name__ == "__main__":
    main()
