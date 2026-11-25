"""Loads the spin-polarisation results (Results/results*.json from collect.py) together with
the atom groups written by SelectSnapshots.py, and derives the per-molecule quantities used
by AnalyseSpinTest.py.

Sphere charges and moments come from LORBIT = 11. They count only what lies inside the PAW
sphere of each atom, so they are used as differences against the lone O2 molecule at the same
bond length, never as absolute charges.
"""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1] / "spin_polarisation"  # results and caches are kept beside the scripts in spin_polarisation/
STRUCT = HERE / "Structures"
RESULTS = HERE / "Results"
R_PAIR = 2.10          # two O atoms closer than this are treated as one O2 unit


def load(name="results.json"):
    data = json.loads((RESULTS / name).read_text())
    index = {r["name"]: r for r in csv.DictReader(open(STRUCT / "INDEX.csv"))}
    out = {}
    for s, d in data.items():
        d = dict(d)
        d["index"] = index.get(s, index.get("O2_isolated") if s.startswith("O2_isolated") else {})
        d["lattice"] = np.array(d["lattice"])
        d["frac"] = np.array(d["frac"])
        d["species"] = np.array(sum([[e] * c for e, c in zip(d["elements"], d["counts"])], []))
        groups = STRUCT / s / "atoms.csv"
        if not groups.exists() and s.startswith("O2_isolated"):
            d["group"] = np.array(["O free O2", "O free O2"])
        else:
            d["group"] = np.array([r["group"] for r in csv.DictReader(open(groups))])
        out[s] = d
    return out


def distances(d, a, b):
    x = d["frac"][a][:, None, :] - d["frac"][b][None, :, :]
    x -= np.round(x)
    return np.linalg.norm(x @ d["lattice"], axis=2)


def o2_units(d):
    """Every pair of O atoms that are each other's nearest O within R_PAIR, with its bond length,
    its distance to the nearest Zr, and its sphere charge and moment in both calculations."""
    io = np.where(d["species"] == "O")[0]
    iz = np.where(d["species"] == "Zr")[0]
    doo = distances(d, io, io)
    np.fill_diagonal(doo, 9.0)
    near = doo.argmin(1)
    dzr = distances(d, io, iz).min(1) if len(iz) else np.full(len(io), np.inf)
    units = []
    for k, a in enumerate(io):
        m = near[k]
        if k < m and near[m] == k and doo[k, m] < R_PAIR:
            b = io[m]
            u = dict(atoms=(int(a), int(b)), r=float(doo[k, m]), d_zr=float(min(dzr[k], dzr[m])),
                     free=bool(d["group"][a] == "O free O2" and d["group"][b] == "O free O2"))
            for mode in ("nsp", "sp"):
                r = d.get(mode)
                ok = r and r["charge"]
                u[f"q_{mode}"] = float(r["charge"][a] + r["charge"][b]) if ok else np.nan
                u[f"m_{mode}"] = float(r["moment"][a] + r["moment"][b]) if ok and r["moment"] else (0.0 if mode == "nsp" else np.nan)
            units.append(u)
    return units


def energy(d, mode, key="free_energy"):
    r = d.get(mode)
    return r[key] if r else np.nan


if __name__ == "__main__":
    import sys
    data = load(sys.argv[1] if len(sys.argv) > 1 else "results.json")
    print(f"{'structure':30s} {'conv':5s} {'dE(F)':>8s} {'dE(E0)':>8s} {'M_tot':>7s} | O2 units: r(A) dZr(A) q_nsp q_sp m_sp")
    for s, d in data.items():
        conv = "yes" if d["sp"] and d["sp"]["converged"] else "NO"
        dF = energy(d, "nsp") - energy(d, "sp")
        d0 = energy(d, "nsp", "energy_sigma0") - energy(d, "sp", "energy_sigma0")
        units = sorted(o2_units(d), key=lambda u: -u["d_zr"])
        txt = "  ".join(f"[{u['r']:.2f} {min(u['d_zr'], 99):4.1f} {u['q_nsp']:.2f} {u['q_sp']:.2f} {u['m_sp']:+.2f}{'*' if u['free'] else ''}]" for u in units[:11])
        print(f"{s:30s} {conv:5s} {dF:8.3f} {d0:8.3f} {d['sp']['total_moment'] if d['sp'] else float('nan'):7.2f} | {txt}")
