"""Picks the structures for the spin-polarisation check from the AIMD trajectories and writes them as VASP POSCAR files.

Every structure is a real frame of a trajectory, unchanged. Each is to be run
twice as a static calculation, with and without spin polarisation.

Sets written to Structures/:
  O2   one O2 molecule alone in the simulation cell (not a trajectory frame)
  S    four snapshots across the run (20, 60, 100 and 140 ps, one per temperature)
  C    two end-of-run snapshots with the longest carbon chains
  E1   one O2 molecule followed from the gas to dissociation on the pristine
       surface (873 K cell 1, start of the run)
  E2   the same on an oxidised surface (1273 K cell 1, after 100 ps)

Beside each POSCAR:
  MAGMOM.txt   initial moments for the spin-polarised run, in POSCAR atom order
  atoms.csv    POSCAR index, element, index in the trajectory frame and a group
               label, for comparing forces by group afterwards
Structures/INDEX.csv lists every structure with its source frame.

Atom order in every POSCAR is O, Zr, C, the block order of the trajectory
files. The POTCAR has to follow the same order.

Run:  python SelectSnapshots.py        (a few minutes, reads four trajectories)
The trajectories are only read.
"""

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import Paths as RP  # noqa: E402

OUT = HERE / "Structures"
ORDER = ("O", "Zr", "C")
R_OO = 1.60          # O-O bonded on the surface (adsorbed peroxo about 1.5 A)
R_OO_FREE = 2.10     # O-O of a molecule in the gas, see classify()
R_ZRO_FREE = 3.17    # the paper's free-molecule convention for gas-phase O2
R_ZRC = 3.33         # the paper's Zr-C cutoff
R_CC = 2.31          # the paper's C-C cutoff
R_CO = 1.50          # C-O bonded
EXCLUDED = {"973_1", "1273_3"}       # trajectories with erroneous removals, not used here

TIME = re.compile(r"Time=([-\d.eE+]+)")
LATT = re.compile(r'Lattice="([^"]+)"')
EFREE = re.compile(r"EFree_eV=([-\d.eE+]+)")
STEP = re.compile(r"Step=(\d+)")


# ------------------------------------------------------------------ reading
def frames(path, t0=0.0, t1=np.inf, stop_after=None):
    """Frames with t0 <= time < t1 (fs). Frames before t0 are skipped without parsing."""
    found = 0
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                return
            n = int(line)
            head = f.readline().decode()
            t = float(TIME.search(head).group(1))
            if t < t0:
                for _ in range(n):
                    f.readline()
                continue
            if t >= t1:
                return
            rows = [f.readline().split() for _ in range(n)]
            yield frame_dict(head, t, rows, path.stem)
            found += 1
            if stop_after and found >= stop_after:
                return


def frame_dict(head, t, rows, source):
    e = EFREE.search(head)
    return dict(source=source, time_fs=t, step=int(STEP.search(head).group(1)),
                lattice=np.array(LATT.search(head).group(1).split(), float).reshape(3, 3),
                species=np.array([r[0].decode() for r in rows]),
                pos=np.array([[float(x) for x in r[1:4]] for r in rows]),
                efree=float(e.group(1)) if e else None)


def last_frame(path):
    """The last complete frame of a file, read from its end."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        f.seek(max(0, size - 80000))
        tail = f.read().decode(errors="ignore").split("\n")
    heads = [i for i, l in enumerate(tail) if "Lattice=" in l]
    for i in reversed(heads):
        n = int(tail[i - 1])
        rows = [l.encode().split() for l in tail[i + 1:i + 1 + n] if l.strip()]
        if len(rows) == n:
            return frame_dict(tail[i], float(TIME.search(tail[i]).group(1)), rows, path.stem)
    raise RuntimeError(f"no complete frame at the end of {path}")


# ------------------------------------------------------------------ geometry
def distances(pos_a, pos_b, lattice):
    """Minimum-image distances between two sets of positions."""
    inv = np.linalg.inv(lattice)
    d = (pos_a @ inv)[:, None, :] - (pos_b @ inv)[None, :, :]
    d -= np.round(d)
    return np.linalg.norm(d @ lattice, axis=2)


def classify(fr):
    """Group label per atom, the free O2 molecules, and carbon chain sizes."""
    sp, pos, lat = fr["species"], fr["pos"], fr["lattice"]
    io, iz, ic = (np.where(sp == s)[0] for s in ORDER)
    d_oo, d_oz = distances(pos[io], pos[io], lat), distances(pos[io], pos[iz], lat)
    d_cz, d_cc, d_co = distances(pos[ic], pos[iz], lat), distances(pos[ic], pos[ic], lat), distances(pos[ic], pos[io], lat)
    np.fill_diagonal(d_oo, 9.0)
    np.fill_diagonal(d_cc, 9.0)
    label = np.array(["" for _ in sp], dtype=object)
    label[iz] = "Zr"
    gas_c = d_cz.min(1) > R_ZRC                              # carbon away from every Zr
    chain = (d_cc < R_CC).any(1)
    for k, a in enumerate(ic):
        label[a] = "C gas product" if gas_c[k] and not chain[k] else ("C chain" if chain[k] else "C no chain")
    o_near_zr = d_oz.min(1) <= R_ZRO_FREE
    o_on_c = (d_co < R_CO).any(0)                           # O bonded to a carbon
    o_on_gas_c = (d_co[gas_c & ~chain] < R_CO).any(0) if (gas_c & ~chain).any() else np.zeros(len(io), bool)
    # Gas-phase O2 in these trajectories vibrates strongly (O-O between about 1.0 and 1.9 A), so a
    # free molecule is two O atoms away from the solid that are each other's nearest O within 2.1 A.
    in_gas = ~o_near_zr & ~o_on_c
    nearest = d_oo.argmin(1)
    free_pairs, bound_pairs = [], []
    for k, a in enumerate(io):
        m = nearest[k]
        if o_on_gas_c[k] and not o_near_zr[k]:
            label[a] = "O gas product"
        elif in_gas[k] and in_gas[m] and nearest[m] == k and d_oo[k, m] < R_OO_FREE:
            label[a] = "O free O2"
            if k < m:
                free_pairs.append((int(a), int(io[m])))
        elif d_oo[k, m] < R_OO and nearest[m] == k:          # includes the outer atom of an end-on molecule
            label[a] = "O adsorbed O2"
            if k < m:
                bound_pairs.append((int(a), int(io[m])))
        elif in_gas[k]:
            label[a] = "O unbound atom"
        else:
            label[a] = "O bound"
    # carbon clusters
    sizes, seen = [], set()
    for k in range(len(ic)):
        if k in seen or not chain[k]:
            continue
        stack, comp = [k], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack += list(np.where(d_cc[x] < R_CC)[0])
        seen |= comp
        sizes.append(len(comp))
    return dict(label=label, free_pairs=free_pairs, bound_pairs=bound_pairs, chain_sizes=sorted(sizes, reverse=True))


# ------------------------------------------------------------------ writing
INDEX = []


def magmom_line(values):
    out, run = [], [values[0], 1]
    for v in values[1:]:
        if v == run[0]:
            run[1] += 1
        else:
            out.append(run)
            run = [v, 1]
    out.append(run)
    return " ".join(f"{n}*{v:g}" if n > 1 else f"{v:g}" for v, n in out)


def write_structure(name, fr, note, extra=None, seed_atoms=()):
    """`seed_atoms` are trajectory indices that get an initial moment of 1 whatever their label
    (the two O atoms of a followed molecule, also after it has bound or its bond has stretched)."""
    info = classify(fr)
    sp, pos, lat = fr["species"], fr["pos"], fr["lattice"]
    order = np.concatenate([np.where(sp == s)[0] for s in ORDER])        # stable within each element
    counts = [int((sp == s).sum()) for s in ORDER]
    frac = (pos[order] @ np.linalg.inv(lat)) % 1.0
    folder = OUT / name
    folder.mkdir(parents=True, exist_ok=True)
    title = f"{name} | {fr['source']}.xyz time {fr['time_fs'] / 1000:.4f} ps step {fr['step']} | {note}"
    lines = [title, "1.0"] + ["  " + "  ".join(f"{x:18.12f}" for x in row) for row in lat]
    lines += ["  " + "  ".join(ORDER), "  " + "  ".join(str(c) for c in counts), "Direct"]
    lines += ["  " + "  ".join(f"{x:16.12f}" for x in row) for row in frac]
    with open(folder / "POSCAR", "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    labels = info["label"][order]
    # Initial moments: 1 on every O of an O-O pair (free or adsorbed), 0.5 on chain carbon, 0 elsewhere.
    seed = [1.0 if l in ("O free O2", "O adsorbed O2") else 2.0 if l == "O unbound atom" else 0.5 if l == "C chain" else 0.0 for l in labels]
    seed = [1.0 if a in seed_atoms else s for a, s in zip(order, seed)]
    with open(folder / "MAGMOM.txt", "w", newline="\n") as f:
        f.write("MAGMOM = " + magmom_line(seed) + "\n")
    with open(folder / "atoms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["poscar_index", "element", "trajectory_index", "group", "initial_moment"])
        for i, (a, l, m) in enumerate(zip(order, labels, seed), start=1):
            w.writerow([i, sp[a], int(a) + 1, l, m])
    c = Counter(labels)
    row = dict(name=name, source=fr["source"] + ".xyz", time_ps=round(fr["time_fs"] / 1000, 4), step=fr["step"], atoms=len(sp),
               O=counts[0], Zr=counts[1], C=counts[2], free_O2=len(info["free_pairs"]), adsorbed_O2=len(info["bound_pairs"]),
               gas_product_atoms=c["O gas product"] + c["C gas product"], chain_C=c["C chain"],
               longest_chain=info["chain_sizes"][0] if info["chain_sizes"] else 0,
               md_free_energy_eV=fr["efree"] if fr["efree"] is not None else "", note=note)
    row.update(extra or {})
    INDEX.append(row)
    print(f"  {name:34s} t={row['time_ps']:9.3f} ps  atoms {len(sp)}  free O2 {row['free_O2']}  adsorbed O2 {row['adsorbed_O2']}  "
          f"chain C {row['chain_C']} (longest {row['longest_chain']})")
    return info


# ------------------------------------------------------------------ adsorption events
def find_event(path, t0, t1, need_dissociation=True):
    """The first O2 molecule in [t0, t1) fs that goes from the gas (both O further than 4.5 A from
    every Zr and every solid carbon) to bound (an O within 2.3 A) to dissociated (O-O above 2.4 A),
    with the atom list unchanged in between so that atom indices can be followed. On the oxidised
    surface the molecules adsorb without dissociating promptly, so there the last step is not
    required and the adsorbed molecule is followed for 1 ps instead."""
    window = list(frames(path, t0, t1))
    print(f"  {path.stem}: {len(window)} frames between {t0 / 1000:.0f} and {t1 / 1000:.0f} ps")
    # split where the atom list changes or an atom jumps (indices are only comparable inside a segment)
    segments, start = [], 0
    for k in range(1, len(window) + 1):
        if k < len(window):
            a, b = window[k - 1], window[k]
            same = len(a["species"]) == len(b["species"]) and (a["species"] == b["species"]).all()
            if same:
                d = (b["pos"] - a["pos"]) @ np.linalg.inv(a["lattice"])
                d -= np.round(d)
                same = np.linalg.norm(d @ a["lattice"], axis=1).max() < 1.0
            if same:
                continue
        segments.append((start, k))
        start = k
    best = None
    for s, e in segments:
        if e - s < 50:
            continue
        seg = window[s:e]
        sp = seg[0]["species"]
        io, iz, ic = (np.where(sp == x)[0] for x in ORDER)
        dmin = np.empty((len(seg), len(io)))                 # O to nearest Zr or solid carbon
        for k, fr in enumerate(seg):
            d_oz = distances(fr["pos"][io], fr["pos"][iz], fr["lattice"]).min(1)
            solid_c = distances(fr["pos"][ic], fr["pos"][iz], fr["lattice"]).min(1) <= R_ZRC
            d_oc = distances(fr["pos"][io], fr["pos"][ic][solid_c], fr["lattice"]).min(1) if solid_c.any() else np.full(len(io), 9.0)
            dmin[k] = np.minimum(d_oz, d_oc)
        first = seg[0]
        d0 = distances(first["pos"][io], first["pos"][io], first["lattice"])
        pairs = set()
        for k in range(0, len(seg), 20):                     # O-O pairs seen in the gas
            fr = seg[k]
            d = distances(fr["pos"][io], fr["pos"][io], fr["lattice"])
            for i, j in zip(*np.where(np.triu(d < R_OO, 1))):
                if dmin[k, i] > 4.5 and dmin[k, j] > 4.5:
                    pairs.add((int(i), int(j)))
        for i, j in sorted(pairs):
            doo = np.array([distances(fr["pos"][io[[i]]], fr["pos"][io[[j]]], fr["lattice"])[0, 0] for fr in seg])
            near = np.minimum(dmin[:, i], dmin[:, j])
            gas = np.where((near > 4.5) & (doo < R_OO))[0]
            if not len(gas):
                continue
            bound = np.where((near < 2.3) & (np.arange(len(seg)) > gas[0]))[0]
            if not len(bound):
                continue
            tb = bound[0]
            if (doo[gas[gas < tb][-1]:tb] > R_OO + 0.2).any():   # must arrive as an intact molecule
                continue
            diss = np.where((doo > 2.4) & (np.arange(len(seg)) > tb))[0]
            if not len(diss) and need_dissociation:
                continue
            td = diss[0] if len(diss) else None
            if td is None:                                   # want 1 ps in which the molecule stays adsorbed, not a bounce
                since = np.array([fr["time_fs"] for fr in seg[tb:]]) - seg[tb]["time_fs"]
                if since[-1] < 1000.0 or (near[tb:][since <= 1000.0] > 3.0).any():
                    continue
            if best is None or seg[tb]["time_fs"] < best["t_bind"]:
                best = dict(seg=seg, o=(int(io[i]), int(io[j])), doo=doo, near=near, tb=tb, td=td, t_bind=seg[tb]["time_fs"])
    return best


def event_frames(ev):
    """Frames along one event, from the gas to dissociation."""
    doo, near, tb, td = ev["doo"], ev["near"], ev["tb"], ev["td"]
    picks = []
    for x in (4.5, 3.5, 3.0, 2.6):                           # the approach
        k = np.where(near[:tb] >= x)[0]
        if len(k):
            picks.append((int(k[-1]), f"approach, nearest solid atom {x} A"))
    picks.append((int(tb), "bound, an O within 2.3 A of the surface"))
    if td is None:                                           # adsorbs as a molecule: follow the adsorbed state
        t = np.array([fr["time_fs"] for fr in ev["seg"]]) - ev["seg"][tb]["time_fs"]
        for dt in (250.0, 1000.0):
            picks.append((int(np.where(t >= dt)[0][0]), f"adsorbed molecule, {dt / 1000:g} ps after binding"))
        after = np.arange(len(doo)) > tb
        k = int(np.where(after)[0][np.argmax(doo[after])])
        picks.append((k, f"adsorbed molecule, longest O-O distance ({doo[k]:.2f} A)"))
    else:
        stretched = np.where((doo > 1.8) & (np.arange(len(doo)) > tb))[0]
        if len(stretched) and stretched[0] - tb > 4:
            picks.append((int((tb + stretched[0]) // 2), "adsorbed molecule, halfway to bond stretching"))
        if len(stretched):
            picks.append((int(stretched[0]), "O-O stretched above 1.8 A"))
        picks.append((int(td), "dissociated, O-O above 2.4 A"))
    out, used = [], set()
    for k, what in picks:
        if k not in used:
            used.add(k)
            out.append((k, what))
    return sorted(out)


def main():
    OUT.mkdir(exist_ok=True)

    print("O2 alone in the cell")
    cell = next(frames(RP.trajectory(873, 1), stop_after=1))
    centre = 0.5 * cell["lattice"].sum(0)
    lat = cell["lattice"]
    pos = np.array([centre - [0.615, 0, 0], centre + [0.615, 0, 0]])
    frac = (pos @ np.linalg.inv(lat)) % 1.0
    folder = OUT / "O2_isolated"
    folder.mkdir(parents=True, exist_ok=True)
    note = "one O2, bond 1.23 A along x, in the cell of 873_1.xyz at t = 0"
    with open(folder / "POSCAR", "w", newline="\n") as f:              # written directly, it has no Zr or C block
        f.write(f"O2_isolated | {note}\n1.0\n")
        f.write("\n".join("  " + "  ".join(f"{x:18.12f}" for x in row) for row in lat) + "\n  O\n  2\nDirect\n")
        f.write("\n".join("  " + "  ".join(f"{x:16.12f}" for x in row) for row in frac) + "\n")
    with open(folder / "MAGMOM.txt", "w", newline="\n") as f:
        f.write("MAGMOM = 2*1\n")
    with open(folder / "atoms.csv", "w", newline="") as f:
        f.write("poscar_index,element,trajectory_index,group,initial_moment\n1,O,,O free O2,1.0\n2,O,,O free O2,1.0\n")
    INDEX.append(dict(name="O2_isolated", source="built, not a frame", time_ps="", step="", atoms=2, O=2, Zr=0, C=0, free_O2=1,
                      adsorbed_O2=0, gas_product_atoms=0, chain_C=0, longest_chain=0, md_free_energy_eV="", note=note))
    print("  O2_isolated")
    # The same molecule at other bond lengths, to compare the O-O potential with and without spin.
    for r in (1.15, 1.35, 1.50, 1.70):
        name = f"O2_isolated_r{r:.2f}"
        pos = np.array([centre - [r / 2, 0, 0], centre + [r / 2, 0, 0]])
        frac = (pos @ np.linalg.inv(lat)) % 1.0
        (OUT / name).mkdir(parents=True, exist_ok=True)
        with open(OUT / name / "POSCAR", "w", newline="\n") as f:
            f.write(f"{name} | one O2, bond {r:.2f} A along x, in the cell of 873_1.xyz at t = 0\n1.0\n")
            f.write("\n".join("  " + "  ".join(f"{x:18.12f}" for x in row) for row in lat) + "\n  O\n  2\nDirect\n")
            f.write("\n".join("  " + "  ".join(f"{x:16.12f}" for x in row) for row in frac) + "\n")
        with open(OUT / name / "MAGMOM.txt", "w", newline="\n") as f:
            f.write("MAGMOM = 2*1\n")
        INDEX.append(dict(name=name, source="built, not a frame", time_ps="", step="", atoms=2, O=2, Zr=0, C=0, free_O2=1,
                          adsorbed_O2=0, gas_product_atoms=0, chain_C=0, longest_chain=0, md_free_energy_eV="",
                          note=f"one O2, bond {r:.2f} A (optional, O-O potential)"))
        print(f"  {name}")

    print("Snapshots across the run")
    for tag, (temp, cell_id, t_ps) in zip("1234", ((873, 1, 20), (973, 2, 60), (1073, 1, 100))):
        fr = next(frames(RP.trajectory(temp, cell_id), t0=t_ps * 1000.0, stop_after=1))
        write_structure(f"S{tag}_{temp}K_cell{cell_id}_{t_ps:03d}ps", fr, f"snapshot at {t_ps} ps, {temp} K")
    write_structure("S4_1273K_cell1_140ps", last_frame(RP.trajectory(1273, 1)), "snapshot at the end of the run, 1273 K")

    print("Longest carbon chains at the end of the run")
    ends = []
    for temp in RP.TEMPERATURES:
        for cell_id in RP.CELLS:
            if f"{temp}_{cell_id}" in EXCLUDED or (temp, cell_id) == (1273, 1):
                continue
            fr = last_frame(RP.trajectory(temp, cell_id))
            sizes = classify(fr)["chain_sizes"]
            ends.append((sizes[0] if sizes else 0, sum(sizes), temp, cell_id, fr))
    ends.sort(key=lambda x: (-x[0], -x[1]))
    for tag, (longest, total, temp, cell_id, fr) in zip("12", ends[:2]):
        write_structure(f"C{tag}_{temp}K_cell{cell_id}_140ps_chains", fr, f"end of the run, longest carbon chain {longest} atoms")

    print("Adsorption events")
    for tag, temp, cell_id, t0, t1, what, diss in (("E1", 873, 1, 0.0, 12000.0, "pristine surface", True),
                                                   ("E2", 1273, 1, 100000.0, 130000.0, "oxidised surface", False)):
        ev = find_event(RP.trajectory(temp, cell_id), t0, t1, need_dissociation=diss)
        if ev is None:
            print(f"  no complete event found for {tag}")
            continue
        seg = ev["seg"]
        for n, (k, stage) in enumerate(event_frames(ev), start=1):
            fr = seg[k]
            order = np.concatenate([np.where(fr["species"] == s)[0] for s in ORDER])
            where = [int(np.where(order == a)[0][0]) + 1 for a in ev["o"]]
            write_structure(f"{tag}_{temp}K_cell{cell_id}_f{n}", fr, f"O2 on the {what}: {stage}", seed_atoms=ev["o"],
                            extra=dict(event=tag, event_O_poscar_indices=f"{where[0]} {where[1]}", O_O_distance_A=round(float(ev["doo"][k]), 3),
                                       nearest_solid_atom_A=round(float(ev["near"][k]), 3),
                                       time_from_binding_fs=round(fr["time_fs"] - ev["t_bind"], 1)))

    keys = []
    for row in INDEX:
        keys += [k for k in row if k not in keys]
    with open(OUT / "INDEX.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(INDEX)
    print(f"{len(INDEX)} structures written to {OUT}")


if __name__ == "__main__":
    main()
