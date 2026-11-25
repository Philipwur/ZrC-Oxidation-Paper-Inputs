"""How often do gas-phase O2 molecules sit in close contact with each other in the MD?

The spin-polarisation check showed that gas-phase O2 units with a reduced moment have an O
atom of another molecule within about 1.7 to 2.0 A. Real (triplet) O2 molecules do not
approach closer than about 3 A. This script counts such contacts along the trajectories.

For frames sampled every 250 fs:
  gas O atom      an O atom further than 4.0 A from every Zr and not bonded to carbon (C-O > 1.5 A)
  cluster         gas O atoms connected by O-O distances below 2.3 A
A cluster of two atoms is an O2 molecule. Larger clusters are molecules in contact (O4 and up).

Writes Results/GasPhaseClusters.csv (one row per sampled frame) and prints a summary.
Run:  python GasPhaseClusters.py [trajectory names, default one cell per temperature]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import SelectSnapshots as S

OUT = Path(__file__).resolve().parent / "Results"
EVERY_FS = 250.0
R_GAS, R_CO, R_LINK, R_BOND = 4.0, 1.5, 2.3, 1.6


def sampled(path, every):
    """One frame every `every` fs, other frames skipped without parsing."""
    target = 0.0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            n = int(line)
            head = f.readline().decode()
            t = float(S.TIME.search(head).group(1))
            if t + 1e-6 < target:
                for _ in range(n):
                    f.readline()
                continue
            target = t + every
            rows = [f.readline().split() for _ in range(n)]
            yield S.frame_dict(head, t, rows, path.stem)


def analyse(fr):
    sp, pos, lat = fr["species"], fr["pos"], fr["lattice"]
    io, iz, ic = (np.where(sp == s)[0] for s in ("O", "Zr", "C"))
    d_oz = S.distances(pos[io], pos[iz], lat).min(1)
    d_oc = S.distances(pos[io], pos[ic], lat).min(1) if len(ic) else np.full(len(io), 9.0)
    gas = np.where((d_oz > R_GAS) & (d_oc > R_CO))[0]
    out = dict(time_ps=fr["time_fs"] / 1000, gas_O_atoms=len(gas), molecules=0, molecules_in_contact=0, largest_cluster=0,
               clusters_O2=0, clusters_O3=0, clusters_O4=0, clusters_larger=0, shortest_contact_A=np.nan)
    if len(gas) < 2:
        return out
    d = S.distances(pos[io[gas]], pos[io[gas]], lat)
    np.fill_diagonal(d, 9.0)
    # molecules: mutual nearest neighbours within the bond length range seen for vibrating O2
    near = d.argmin(1)
    mol = [(i, near[i]) for i in range(len(gas)) if i < near[i] and near[near[i]] == i and d[i, near[i]] < 2.1]
    out["molecules"] = len(mol)
    seen, sizes = set(), []
    for k in range(len(gas)):
        if k in seen:
            continue
        stack, comp = [k], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += list(np.where(d[x] < R_LINK)[0])
        seen |= comp
        sizes.append(len(comp))
    out["largest_cluster"] = max(sizes)
    out["clusters_O2"] = sizes.count(2)
    out["clusters_O3"] = sizes.count(3)
    out["clusters_O4"] = sizes.count(4)
    out["clusters_larger"] = sum(1 for x in sizes if x > 4)
    contacts = []
    for i, j in mol:
        others = [x for x in range(len(gas)) if x not in (i, j)]
        if others:
            c = min(d[i, others].min(), d[j, others].min())
            contacts.append(c)
    if contacts:
        out["molecules_in_contact"] = int(sum(c < R_LINK for c in contacts))
        out["shortest_contact_A"] = float(min(contacts))
    return out


def main():
    names = sys.argv[1:] or ["873_1", "973_2", "1073_1", "1273_1"]
    rows = []
    for name in names:
        t, c = name.split("_")
        n = 0
        for fr in sampled(S.RP.trajectory(int(t), int(c)), EVERY_FS):
            if fr["time_fs"] > 140000:
                break
            rows.append(dict(trajectory=name, temperature=int(t), **analyse(fr)))
            n += 1
        print(f"{name}: {n} frames", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "GasPhaseClusters.csv", index=False)
    print()
    print(f"{'trajectory':10s} {'frames with gas O2':>18s} {'mean molecules':>14s} {'molecules in contact':>21s} {'frames with a contact':>22s} {'O3':>5s} {'O4':>5s} {'>O4':>5s} {'shortest (A)':>12s}")
    for name, g in df.groupby("trajectory", sort=False):
        w = g[g.molecules > 0]
        two = g[g.molecules > 1]
        print(f"{name:10s} {len(w):18d} {w.molecules.mean():14.2f} {w.molecules_in_contact.sum() / max(w.molecules.sum(), 1):20.1%} "
              f"{(two.molecules_in_contact > 0).mean() if len(two) else float('nan'):21.1%} {g.clusters_O3.sum():5d} {g.clusters_O4.sum():5d} {g.clusters_larger.sum():5d} {g.shortest_contact_A.min():12.2f}")
    w = df[df.molecules > 0]
    two = df[df.molecules > 1]
    print(f"\nAll: {w.molecules_in_contact.sum()} of {w.molecules.sum()} gas-molecule instances ({w.molecules_in_contact.sum() / w.molecules.sum():.1%}) "
          f"have an O atom of another gas species within {R_LINK} A. Among frames with two or more gas molecules "
          f"({len(two)} of {len(df)}), {(two.molecules_in_contact > 0).mean():.1%} contain such a contact.")


if __name__ == "__main__":
    main()
