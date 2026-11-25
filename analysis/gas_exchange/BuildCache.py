"""Parse the 16 xyz trajectories once into compact arrays.

The xyz files are about 1 GB of text each. This script
reads each file once and writes one .npz per trajectory to a cache folder
(default: the system temp folder, about 4 GB in total). The
analysis scripts in this folder read the cache.

Per trajectory:
    pos   float32 [frames, nmax, 3]  Cartesian positions in A, NaN where padded
    el    int8    [frames, nmax]     0 padding, 1 O, 2 Zr, 3 C
    n     int16   [frames]           number of atoms in the frame
    time  float64 [frames]           fs, from the frame header
    step  int32   [frames]           MD step, from the frame header
    lat   float64 [frames, 3, 3]     lattice of each frame, rows are the cell vectors
                                     (it changes slightly from one MD segment to the next)

Usage:
    python BuildCache.py                 all 16, three at a time
    python BuildCache.py --only 873_1
    python BuildCache.py --cache D:/somewhere
"""

import argparse
import os
import re
import sys
import tempfile
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP

DEFAULT_CACHE = Path(os.environ.get("R27_CACHE", Path(tempfile.gettempdir()) / "Paper1_R27_cache"))
CODE = {"O": 1, "Zr": 2, "C": 3}
LATTICE = re.compile(r'Lattice="([^"]+)"')
STEP = re.compile(r"Step=(\d+)")
TIME = re.compile(r"Time=(\S+)")


def cache_file(cache, temperature, cell):
    return Path(cache) / f"{temperature}_{cell}.npz"


def build(args):
    temperature, cell, cache = args
    out = cache_file(cache, temperature, cell)
    if out.is_file():
        return f"{temperature}_{cell}: cached"
    positions, elements, times, steps, lattices = [], [], [], [], []
    lattice = None
    with open(RP.trajectory(temperature, cell)) as handle:
        while True:
            count = handle.readline()
            if not count:
                break
            count = count.strip()
            if not count:
                continue
            natoms = int(count)
            header = handle.readline()
            tokens = "".join(islice(handle, natoms)).split()
            if len(tokens) != 4 * natoms:
                break  # truncated last frame
            match = LATTICE.search(header)
            if match:
                lattice = np.array(match.group(1).split(), dtype=float).reshape(3, 3)
            lattices.append(lattice)
            elements.append(np.array([CODE[s] for s in tokens[0::4]], dtype=np.int8))
            coords = np.array(tokens, dtype=object).reshape(natoms, 4)[:, 1:]
            positions.append(coords.astype(np.float32))
            steps.append(int(STEP.search(header).group(1)))
            times.append(float(TIME.search(header).group(1)))
    nframes = len(positions)
    nmax = max(len(e) for e in elements)
    pos = np.full((nframes, nmax, 3), np.nan, dtype=np.float32)
    el = np.zeros((nframes, nmax), dtype=np.int8)
    n = np.zeros(nframes, dtype=np.int16)
    for i, (p, e) in enumerate(zip(positions, elements)):
        pos[i, : len(e)] = p
        el[i, : len(e)] = e
        n[i] = len(e)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, pos=pos, el=el, n=n, time=np.array(times), step=np.array(steps, dtype=np.int32), lat=np.array(lattices))
    tmp.replace(out)
    return f"{temperature}_{cell}: {nframes} frames, up to {nmax} atoms"


def load(cache, temperature, cell):
    with np.load(cache_file(cache, temperature, cell)) as data:
        return {key: data[key] for key in data.files}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--workers", type=int, default=3)
    options = parser.parse_args()
    cache = Path(options.cache)
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [(t, c, cache) for t in RP.TEMPERATURES for c in RP.CELLS]
    if options.only:
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}" == options.only]
    with Pool(min(options.workers, len(jobs))) as pool:
        for line in pool.imap_unordered(build, jobs):
            print(line, flush=True)
