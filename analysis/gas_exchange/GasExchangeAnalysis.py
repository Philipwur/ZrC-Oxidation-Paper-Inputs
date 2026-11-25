"""Post hoc analysis of the gas-exchange protocol in the 16 trajectories.

Reads the cache written by BuildCache.py and goes
through every frame of every trajectory once. Nothing is rerun and no input
file is changed.

For every frame the script repeats the gas detection of the production code
(`FindGasesFromArrays` in SGUSCHI/src/workflow/OxidationAnalysis.py): two atoms
are bonded when their distance is below 1.5 x the sum of their covalent radii,
O-O bonds are pruned so that each O has at most one O neighbour, and a
connected group of two or three atoms is a gas molecule. The detection is done
three times, with the cutoffs scaled by 0.9, 1.0 and 1.1.

Per trajectory it writes one pickle to the cache folder with
  removals   one row per molecule that left the cell, with its state in the
             last frame in which it was present
  episodes   every stretch of frames in which a carbon-containing group was
             detached, per cutoff scale, with how it ended
  carbons    the fate of every carbon atom (removed and when, or still there)
  thermo     kinetic energies of the solid and of the gas molecules, from
             finite differences of the positions. The time step of each
             segment is checked first, see true_time_steps
  supply     number of gas O2 molecules per frame and the O2 additions

Summarise.py turns the pickles into tables and a figure.

Usage:
    python GasExchangeAnalysis.py              all 16, three at a time
    python GasExchangeAnalysis.py --only 973_1
"""

import argparse
import pickle
import sys
from collections import Counter, deque
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP
from BuildCache import DEFAULT_CACHE, load

# Covalent radii of the paper (cutoffs 3.17, 3.33, 2.31, 2.15 and 1.98 A at scale 1.0).
RADIUS = np.array([0.0, 0.66, 1.45, 0.77])  # index = element code (1 O, 2 Zr, 3 C)
MASS = np.array([0.0, 15.999, 91.224, 12.011])
TOLERANCE = 1.5
SCALES = (0.9, 1.0, 1.1)
SEGMENT = 80
KE_TO_EV = 103.642697  # amu A^2 fs^-2 in eV
KB = 8.617333e-5
SYMBOL = {1: "O", 2: "Zr", 3: "C"}
WINDOWS = ((0.0, 5.0), (5.0, 40.0), (40.0, 140.0), (0.0, 140.0))


def segment_starts(el, lat, nframes):
    """Index of the first frame of every 80-step MD segment.

    The files were joined from pieces, and one or two extra frames sit at the
    joins. Frames where the composition or the lattice changes are exact
    segment starts, so every nominal start (a multiple of 80) takes its offset
    from the nearest such frame.
    """
    changed = np.any(el[1:] != el[:-1], axis=1) | np.any(lat[1:] != lat[:-1], axis=(1, 2))
    markers = np.where(changed)[0] + 1
    nominal = np.arange(0, nframes, SEGMENT)
    if len(markers) == 0:
        return nominal
    position = np.clip(np.searchsorted(markers, nominal), 0, len(markers) - 1)
    left = np.clip(position - 1, 0, len(markers) - 1)
    nearest = np.where(np.abs(markers[left] - nominal) <= np.abs(markers[position] - nominal),
                       markers[left], markers[position])
    offset = nearest % SEGMENT
    offset = np.where(offset > SEGMENT // 2, offset - SEGMENT, offset)
    starts = np.unique(np.concatenate([nominal + offset, markers]))
    return starts[(starts >= 0) & (starts < nframes)]


def true_time_steps(pos_all, el_all, n_all, lat_all, time, starts, temperature):
    """Time step of every frame pair (f-1, f), NaN where it should not be used.

    In most files the time-step label of an 80-step segment is the time step of
    the segment before it (the labels are shifted by one segment), and in some
    parts it is correct. The thermostat holds the kinetic temperature of all
    atoms within about 1% of the target in every segment, so for each segment
    the label of that segment and of the one before are both tried and the one
    that gives the kinetic temperature closer to the target is kept. The two
    labels differ by 8% or more when they differ at all, which changes the
    temperature by 16% or more, so the choice is not ambiguous.
    """
    nframes = len(time)
    step = np.full(nframes, np.nan)
    choice = Counter()
    ends = np.append(starts[1:], nframes)
    previous_label = None
    for start, end in zip(starts, ends):
        if end - start < 10:
            continue
        n = int(n_all[start])
        frames = np.arange(start + 1, end)
        label = float(np.median(np.diff(time[start:end])))
        box = np.diag(lat_all[start])
        delta = pos_all[frames, :n].astype(np.float64) - pos_all[frames - 1, :n]
        delta -= np.round(delta / box) * box
        mass = MASS[el_all[start, :n]]
        msd = float((mass[None, :] * (delta ** 2).sum(axis=2)).sum(axis=1).mean() / (3 * n - 3))
        best = label
        if previous_label is not None and msd > 0:
            t_label = msd / label ** 2 * KE_TO_EV / KB
            t_previous = msd / previous_label ** 2 * KE_TO_EV / KB
            if abs(t_previous - temperature) < abs(t_label - temperature):
                best = previous_label
        choice["label of the segment" if best == label else "label of the segment before"] += 1
        step[frames] = best
        previous_label = label
    return step, dict(choice)


def distance_matrix(pos, lattice, inverse):
    frac = pos @ inverse
    diff = frac[:, None, :] - frac[None, :, :]
    diff -= np.round(diff)
    return np.linalg.norm(diff @ lattice, axis=2)


def min_image(delta, lattice, inverse):
    frac = delta @ inverse
    frac -= np.round(frac)
    return frac @ lattice


def free_groups(el, dist, scale):
    """Groups of atoms that are not connected to the solid, as index tuples.

    All Zr atoms are within the Zr-Zr cutoff of each other, so the solid is the
    group that contains Zr. An atom that is not bonded to Zr and not bonded,
    directly or through others, to an atom that is, belongs to a free group.
    """
    base = RADIUS[el][:, None] + RADIUS[el][None, :]
    bond = dist < (TOLERANCE * scale) * base
    np.fill_diagonal(bond, False)
    oxygen = np.where(el == 1)[0]
    if len(oxygen) > 1:
        sub = bond[np.ix_(oxygen, oxygen)]
        rows, cols = np.where(np.triu(sub, 1))
        if len(rows):
            order = np.argsort(dist[oxygen[rows], oxygen[cols]])
            taken = set()
            for k in order:
                a, b = oxygen[rows[k]], oxygen[cols[k]]
                if a in taken or b in taken:
                    bond[a, b] = bond[b, a] = False
                else:
                    taken.add(a)
                    taken.add(b)
    zirconium = el == 2
    attached = zirconium | bond[:, zirconium].any(axis=1)
    loose = np.where(~attached)[0]
    groups = []
    if len(loose) == 0:
        return groups
    seen = set()
    for seed in loose:
        if seed in seen:
            continue
        stack, members, tied = [seed], [], False
        seen.add(seed)
        while stack:
            atom = stack.pop()
            members.append(atom)
            for other in np.where(bond[atom])[0]:
                if attached[other]:
                    tied = True
                elif other not in seen:
                    seen.add(other)
                    stack.append(other)
        if not tied:
            groups.append(tuple(sorted(members)))
    return groups


def formula(el, members):
    return "".join(sorted(SYMBOL[el[i]] for i in members))


def match_ids(prev_el, prev_pos, prev_ids, el, pos, lattice, inverse, next_id):
    """Carry atom identities across a frame in which atoms were added or removed."""
    ids = np.full(len(el), -1, dtype=np.int64)
    for code in (1, 2, 3):
        old = np.where(prev_el == code)[0]
        new = np.where(el == code)[0]
        if len(old) and len(new):
            delta = min_image(prev_pos[old][:, None, :] - pos[new][None, :, :], lattice, inverse)
            cost = np.linalg.norm(delta, axis=2)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < 1.0:
                    ids[new[c]] = prev_ids[old[r]]
    fresh = np.where(ids < 0)[0]
    ids[fresh] = np.arange(next_id, next_id + len(fresh))
    return ids, next_id + len(fresh)


def analyse(args):
    temperature, cell, cache = args
    data = load(cache, temperature, cell)
    pos_all, el_all, n_all, time, lat_all = data["pos"], data["el"], data["n"], data["time"], data["lat"]
    nframes = len(time)
    starts = segment_starts(el_all, lat_all, nframes)
    is_boundary = np.zeros(nframes, dtype=bool)  # last frame of a segment, the one the controller looked at
    is_boundary[starts[starts > 0] - 1] = True
    step_fs, step_choice = true_time_steps(pos_all, el_all, n_all, lat_all, time, starts, temperature)

    removals, additions, replays = [], [], []
    episodes = {scale: [] for scale in SCALES}
    active = {scale: {} for scale in SCALES}
    odd_groups = Counter()
    carbon_gone = {}
    n_o2 = np.zeros(nframes, dtype=np.int8)

    nbins = 150
    thermo = {
        "ke_solid": np.zeros(nbins), "n_solid": np.zeros(nbins),
        "ke_all": np.zeros(nbins), "n_all": np.zeros(nbins), "pairs": np.zeros(nbins),
        "ke_trans": np.zeros(nbins), "ke_rot": np.zeros(nbins), "ke_vib": np.zeros(nbins), "n_mol": np.zeros(nbins),
        "ke_product": np.zeros(nbins), "n_product": np.zeros(nbins),
        "by_no2": np.zeros((8, 3)),  # rows: number of gas O2 (7 = seven or more), columns: KE solid, atoms, pairs
    }

    prev = None
    next_id = 0
    history = deque(maxlen=3)
    for f in range(nframes):
        n = int(n_all[f])
        el = el_all[f, :n]
        pos = pos_all[f, :n].astype(np.float64)
        lattice = lat_all[f]
        inverse = np.linalg.inv(lattice)

        velocity = None
        vanished = []
        if prev is None:
            ids = np.arange(n, dtype=np.int64)
            next_id = n
        else:
            same = len(prev["el"]) == n and np.array_equal(prev["el"], el)
            if same:
                # Same atoms in the same order. Atoms are only ever added or removed, never reordered.
                ids = prev["ids"]
                delta = min_image(pos - prev["pos"], lattice, inverse)
                step = np.linalg.norm(delta, axis=1)
                dt = step_fs[f]
                if np.median(step) > 0.05:
                    # All atoms jump at once. An 80-step segment was run a second time from the same start.
                    replays.append({"frame": f, "time_ps": float(time[f]) / 1000.0, "median_jump_A": float(np.median(step))})
                elif np.isfinite(dt) and step.max() > 1e-7:
                    velocity = delta / dt
            else:
                ids, next_id = match_ids(prev["el"], prev["pos"], prev["ids"], el, pos, lattice, inverse, next_id)
                present = set(ids.tolist())
                vanished = [i for i in prev["ids"].tolist() if i not in present]
                new_oxygen = [i for i, e in zip(ids.tolist(), el.tolist()) if e == 1 and i not in prev["idset"]]
                if new_oxygen:
                    additions.append({"frame": f, "time_fs": float(time[f]), "atoms": len(new_oxygen)})
        idset = set(ids.tolist())

        dist = distance_matrix(pos, lattice, inverse)
        groups = {scale: free_groups(el, dist, scale) for scale in SCALES}

        # Removal events: atoms present in the previous frame and gone in this one.
        if vanished:
            p = prev
            index_of = {i: k for k, i in enumerate(p["ids"].tolist())}
            gone = [index_of[i] for i in vanished]
            for i in vanished:
                if p["el"][index_of[i]] == 3:
                    carbon_gone[i] = float(time[f])
            remaining = set(gone)
            molecules = []
            for group in p["groups"][1.0]:
                if set(group) <= remaining:
                    molecules.append(group)
                    remaining -= set(group)
            if remaining:
                molecules.append(tuple(sorted(remaining)))
            for members in molecules:
                members = list(members)
                member_el = p["el"][members]
                others = np.setdiff1d(np.arange(len(p["el"])), members)
                d_other = p["dist"][np.ix_(members, others)]
                zr = np.where(p["el"] == 2)[0]
                d_zr = p["dist"][np.ix_(members, zr)]
                carbon = [m for m in members if p["el"][m] == 3]
                oxygen = [m for m in members if p["el"][m] == 1]
                bond_lengths = [float(p["dist"][c, o]) for c in carbon for o in oxygen]
                key = frozenset(p["ids"][members].tolist())
                record = {
                    "trajectory": f"{temperature}_{cell}", "temperature": temperature, "cell": cell,
                    "frame": f, "time_ps": float(time[f]) / 1000.0,
                    "species": formula(p["el"], members), "natoms": len(members),
                    "co_bonds": bond_lengths,
                    "bonded_molecule": bool(len(carbon) == 1 and len(oxygen) >= 1 and max(bond_lengths) < 1.7),
                    "d_solid": float(d_other.min()), "d_zr": float(d_zr.min()),
                    "on_boundary": bool(is_boundary[f - 1]),
                    "carbon_ids": p["ids"][carbon].tolist(),
                }
                for scale in SCALES:
                    record[f"detected_{scale}"] = tuple(sorted(members)) in set(p["groups"][scale])
                    episode = active[scale].get(key)
                    record[f"free_fs_{scale}"] = (float(time[f - 1] - time[episode["start"]]) if episode else np.nan)
                episode = active[1.0].get(key)
                if episode and len(episode["series"]) >= 2:
                    series = np.array(episode["series"][-12:])
                    record["d_solid_change"] = float(series[-1, 1] - series[0, 1])
                    record["d_solid_change_fs"] = float(series[-1, 0] - series[0, 0])
                    vel = np.array([v for v in episode["vcom"][-10:] if v is not None])
                    if len(vel):
                        mean_v = vel.mean(axis=0)
                        mass = MASS[member_el].sum()
                        record["speed_A_per_ps"] = float(np.linalg.norm(mean_v) * 1000.0)
                        record["ke_trans_eV"] = float(0.5 * mass * (mean_v ** 2).sum() * KE_TO_EV)
                        # Sign of the motion along the surface normal: away from the slab is positive.
                        com = p["pos"][members].mean(axis=0)
                        slab = p["pos"][zr]
                        rel = min_image(com[None, :] - slab, p["lattice"], p["inverse"])
                        nearest = rel[np.argmin(np.linalg.norm(rel, axis=1))]
                        record["v_normal_A_per_ps"] = float(np.sign(nearest[0]) * mean_v[0] * 1000.0)
                removals.append(record)

        # Episodes of detached groups that contain carbon (CO, CO2 and anything else that is not O2).
        for scale in SCALES:
            current = {}
            for group in groups[scale]:
                species = formula(el, group)
                if species == "OO":
                    continue
                current[frozenset(ids[list(group)].tolist())] = (group, species)
            for key in list(active[scale]):
                if key not in current:
                    episode = active[scale].pop(key)
                    episode["end"] = f - 1
                    episode["outcome"] = "removed" if not key <= idset else "rebonded"
                    episode["duration_fs"] = float(time[f - 1] - time[episode["start"]])
                    episode["frames"] = f - episode["start"]
                    episode["spans_boundary"] = bool(is_boundary[episode["start"]:f].any())
                    series = np.array(episode.pop("series"))
                    episode["d_solid_max"] = float(series[:, 1].max())
                    episode["d_solid_last"] = float(series[-1, 1])
                    episode.pop("vcom")
                    episodes[scale].append(episode)
            for key, (group, species) in current.items():
                members = list(group)
                others = np.setdiff1d(np.arange(n), members)
                d_solid = float(dist[np.ix_(members, others)].min())
                vcom = None
                if velocity is not None:
                    w = MASS[el[members]]
                    vcom = (velocity[members] * w[:, None]).sum(axis=0) / w.sum()
                if key not in active[scale]:
                    active[scale][key] = {
                        "trajectory": f"{temperature}_{cell}", "temperature": temperature, "scale": scale,
                        "species": species, "start": f, "start_ps": float(time[f]) / 1000.0,
                        "carbon_ids": [int(i) for i, m in zip(ids[members], members) if el[m] == 3],
                        "series": [], "vcom": [],
                    }
                active[scale][key]["series"].append((float(time[f]), d_solid))
                active[scale][key]["vcom"].append(vcom)

        # Free groups at scale 1.0 that the controller does not treat as gas (one atom, or four and more).
        o2_groups, product_atoms = [], []
        for group in groups[1.0]:
            species = formula(el, group)
            if species == "OO":
                o2_groups.append(group)
            else:
                product_atoms.extend(group)
                if len(group) == 1 or len(group) > 3:
                    odd_groups[species] += 1
        n_o2[f] = len(o2_groups)

        # Kinetic energies from the displacement between the previous frame and this one.
        if velocity is not None and time[f] <= 150000.0:
            b = min(int(time[f] // 1000.0), nbins - 1)
            ke = 0.5 * MASS[el] * (velocity ** 2).sum(axis=1) * KE_TO_EV
            gas = np.zeros(n, dtype=bool)
            for group in o2_groups:
                gas[list(group)] = True
            product = np.zeros(n, dtype=bool)
            product[product_atoms] = True
            solid = ~gas & ~product
            thermo["ke_solid"][b] += ke[solid].sum()
            thermo["n_solid"][b] += solid.sum()
            thermo["ke_all"][b] += ke.sum()
            thermo["n_all"][b] += n
            thermo["pairs"][b] += 1
            thermo["ke_product"][b] += ke[product].sum()
            thermo["n_product"][b] += product.sum()
            row = min(len(o2_groups), 7)
            thermo["by_no2"][row] += (ke[solid].sum(), solid.sum(), 1)
            for a, c in o2_groups:
                vcom = 0.5 * (velocity[a] + velocity[c])
                vrel = velocity[a] - velocity[c]
                axis = min_image((pos[a] - pos[c])[None, :], lattice, inverse)[0]
                axis /= np.linalg.norm(axis)
                along = vrel @ axis
                thermo["ke_trans"][b] += 0.5 * 2 * MASS[1] * (vcom ** 2).sum() * KE_TO_EV
                thermo["ke_vib"][b] += 0.5 * (MASS[1] / 2) * along ** 2 * KE_TO_EV
                thermo["ke_rot"][b] += 0.5 * (MASS[1] / 2) * ((vrel ** 2).sum() - along ** 2) * KE_TO_EV
                thermo["n_mol"][b] += 1

        prev = {"el": el, "pos": pos, "ids": ids, "idset": idset, "dist": dist, "groups": groups,
                "lattice": lattice, "inverse": inverse}

    final_carbon = [int(i) for i, e in zip(prev["ids"], prev["el"]) if e == 3]
    for scale in SCALES:
        for key, episode in active[scale].items():
            episode.pop("series")
            episode.pop("vcom")
            episode.update(end=nframes - 1, outcome="open at end", duration_fs=np.nan, frames=nframes - episode["start"],
                           spans_boundary=False, d_solid_max=np.nan, d_solid_last=np.nan)
            episodes[scale].append(episode)

    result = {
        "trajectory": f"{temperature}_{cell}", "temperature": temperature, "cell": cell,
        "nframes": nframes, "time": time, "starts": starts, "removals": removals, "additions": additions,
        "episodes": episodes, "carbon_gone": carbon_gone, "final_carbon": final_carbon,
        "odd_groups": dict(odd_groups), "n_o2": n_o2, "thermo": thermo, "replays": replays, "step_choice": step_choice,
    }
    with open(Path(cache) / f"analysis_{temperature}_{cell}.pkl", "wb") as handle:
        pickle.dump(result, handle)
    return (f"{temperature}_{cell}: {len(removals)} molecules removed, "
            f"{len(episodes[1.0])} detached episodes at scale 1.0, {len(additions)} additions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--workers", type=int, default=3)
    options = parser.parse_args()
    jobs = [(t, c, options.cache) for t in RP.TEMPERATURES for c in RP.CELLS]
    if options.only:
        jobs = [j for j in jobs if f"{j[0]}_{j[1]}" == options.only]
    if len(jobs) == 1:
        print(analyse(jobs[0]))
    else:
        with Pool(min(options.workers, len(jobs))) as pool:
            for line in pool.imap_unordered(analyse, jobs):
                print(line, flush=True)
