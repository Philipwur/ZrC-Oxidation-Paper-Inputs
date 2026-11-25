"""Tables and figure for the gas-exchange analysis.

Reads the pickles written by GasExchangeAnalysis.py and writes to Results/:
    Removals.csv         every molecule that left a cell
    Episodes.csv         every detached carbon-containing group, three cutoff scales
    Tables.md            all tables
    results.json         the same numbers, machine readable
    Fig_GasExchange.pdf  four panels
    DelayedRemovals.md, ChecksPassed.csv, SeparationAtChecks.csv
                         molecules that were already detached at a check under the
                         bond cutoffs of the paper and were removed at a later one:
                         how many, what they did in the extra time, and the distance
                         that separates them from the molecules removed at once
                         (this part also reads the arrays written by BuildCache.py)

Usage:
    python Summarise.py
"""

import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP
from BuildCache import DEFAULT_CACHE
from GasExchangeAnalysis import RADIUS, SYMBOL, TOLERANCE, distance_matrix

KB = 8.617333e-5
SCALES = (0.9, 1.0, 1.1)
OUT = Path(__file__).resolve().parent / "Results"
OUT.mkdir(exist_ok=True)
HORIZON = RP.HORIZON_PS


def load_all(cache):
    results = {}
    for t in RP.TEMPERATURES:
        for c in RP.CELLS:
            with open(Path(cache) / f"analysis_{t}_{c}.pkl", "rb") as handle:
                results[(t, c)] = pickle.load(handle)
    return results


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and not np.isnan(v)], dtype=float)
    if len(values) == 0:
        return "n/a"
    return f"{np.median(values):.2f} ({values.min():.2f} to {values.max():.2f})"


def main():
    results = load_all(DEFAULT_CACHE)
    lines = ["# Gas-exchange analysis: tables", "",
             "Written by `Summarise.py` from the 16 trajectories. Times are limited to the first "
             f"{HORIZON:.0f} ps unless stated.", ""]
    summary = {}

    # ---------------------------------------------------------------- removals
    removals = pd.DataFrame([r for res in results.values() for r in res["removals"]])
    removals = removals[removals["species"] != "OO"].copy()
    removals["genuine"] = removals["bonded_molecule"] & removals["detected_1.0"]
    removals["within_horizon"] = removals["time_ps"] <= HORIZON
    removals["kT_eV"] = KB * removals["temperature"]
    removals["ke_trans_kT"] = removals["ke_trans_eV"] / removals["kT_eV"]
    removals["co_bonds"] = removals["co_bonds"].apply(lambda b: " ".join(f"{x:.3f}" for x in b))
    removals.drop(columns=["carbon_ids"]).to_csv(OUT / "Removals.csv", index=False)

    genuine = removals[removals["genuine"]]
    lines += ["## 1. Molecules removed from the cells", "",
              f"- Molecules removed in all trajectories: {len(removals)} "
              f"({int(removals['within_horizon'].sum())} within {HORIZON:.0f} ps).",
              f"- Intact CO or CO2 that the detection finds in the last frame before removal: {len(genuine)}.",
              f"- Removed atoms that were not a molecule: {int((~removals['bonded_molecule']).sum())} "
              "(the erroneous deletions known from the July audit).",
              f"- Intact molecules not yet detached in the last stored frame: "
              f"{int((removals['bonded_molecule'] & ~removals['detected_1.0']).sum())} "
              "(the stored frame is one MD step before the structure the controller saw).",
              f"- Removals on the last frame of an 80-step segment: {int(removals['on_boundary'].sum())} of {len(removals)}.",
              "", "State of the intact molecules in the last frame before removal. Median (smallest to largest).", "",
              "| T (K) | CO | CO2 | Distance to nearest atom of the solid (A) | Distance to nearest Zr (A) | "
              "Moving away from the surface | Speed along the normal (A/ps) | Translational energy (kT) | "
              "Time detached before removal (fs) |",
              "|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for t in list(RP.TEMPERATURES) + ["all"]:
        g = genuine if t == "all" else genuine[genuine["temperature"] == t]
        has_v = g["v_normal_A_per_ps"].notna()
        away = int((g.loc[has_v, "v_normal_A_per_ps"] > 0).sum())
        row = {
            "T": t, "CO": int((g["species"] == "CO").sum()), "CO2": int((g["species"] == "COO").sum()),
            "d_solid": quantiles(g["d_solid"]), "d_zr": quantiles(g["d_zr"]),
            "away": f"{away} of {int(has_v.sum())}", "v_normal": quantiles(g["v_normal_A_per_ps"]),
            "ke_kT": quantiles(g["ke_trans_kT"]), "free_fs": quantiles(g["free_fs_1.0"]),
        }
        rows.append(row)
        lines.append(f"| {t} | {row['CO']} | {row['CO2']} | {row['d_solid']} | {row['d_zr']} | {row['away']} | "
                     f"{row['v_normal']} | {row['ke_kT']} | {row['free_fs']} |")
    summary["removals"] = rows
    summary["removal_counts"] = {
        "all": int(len(removals)), "genuine": int(len(genuine)),
        "not_a_molecule": int((~removals["bonded_molecule"]).sum()),
        "mean_ke_trans_kT": float(genuine["ke_trans_kT"].mean()),
        "mean_d_solid": float(genuine["d_solid"].mean()),
        "share_d_solid_below_3": float((genuine["d_solid"] < 3.0).mean()),
    }
    lines += ["", f"- Mean translational energy of a removed molecule: {genuine['ke_trans_kT'].mean():.1f} kT. "
              "A molecule leaving a surface in thermal equilibrium carries 2 kT on average.",
              f"- Share of removed molecules closer than 3 A to the solid: {100 * (genuine['d_solid'] < 3.0).mean():.0f}%.", ""]

    # ---------------------------------------------------------------- episodes
    episodes = pd.DataFrame([e for res in results.values() for s in SCALES for e in res["episodes"][s]])
    episodes["has_carbon"] = episodes["species"].str.contains("C")
    carbon_gone = {(key, cid): t for key, res in results.items() for cid, t in res["carbon_gone"].items()}
    traj_key = {f"{t}_{c}": (t, c) for t in RP.TEMPERATURES for c in RP.CELLS}

    def later_removal(row):
        delays = []
        for cid in row["carbon_ids"]:
            gone = carbon_gone.get((traj_key[row["trajectory"]], cid))
            delays.append(np.nan if gone is None else gone / 1000.0 - row["start_ps"])
        return delays[0] if delays else np.nan

    episodes["removed_later_after_ps"] = episodes.apply(later_removal, axis=1)
    carbon_episodes = episodes[episodes["has_carbon"]].copy()
    carbon_episodes.drop(columns=["carbon_ids"]).to_csv(OUT / "Episodes.csv", index=False)

    lines += ["## 2. Detached carbon-containing groups and how they ended", "",
              "A group is detached when none of its atoms is within the bond cutoff of the solid. "
              "Counted at every stored frame, about 1 fs apart.", "",
              "| Cutoff scale | Detached episodes | Ended by removal | Reattached before the next check | "
              "Longest reattached episode (fs) | Largest distance reached by a reattached group (A) | "
              "Reattached while present at a check |",
              "|---|---|---|---|---|---|---|"]
    episode_rows = []
    for s in SCALES:
        e = carbon_episodes[carbon_episodes["scale"] == s]
        back = e[e["outcome"] == "rebonded"]
        row = {"scale": s, "episodes": int(len(e)), "removed": int((e["outcome"] == "removed").sum()),
               "reattached": int(len(back)),
               "longest_fs": float(back["duration_fs"].max()) if len(back) else 0.0,
               "largest_A": float(back["d_solid_max"].max()) if len(back) else 0.0,
               "reattached_at_check": int(back["spans_boundary"].sum())}
        episode_rows.append(row)
        lines.append(f"| {s} | {row['episodes']} | {row['removed']} | {row['reattached']} | {row['longest_fs']:.0f} | "
                     f"{row['largest_A']:.2f} | {row['reattached_at_check']} |")
    summary["episodes"] = episode_rows

    # Reattached episodes at the production cutoffs: what became of those carbons.
    back = carbon_episodes[(carbon_episodes["scale"] == 1.0) & (carbon_episodes["outcome"] == "rebonded")]
    atoms = back.groupby(["trajectory", back["carbon_ids"].apply(lambda c: c[0] if c else -1)]).agg(
        episodes=("start", "count"), first_ps=("start_ps", "min"), later=("removed_later_after_ps", "max"))
    removed_later = atoms["later"].notna()
    lines += ["", "Carbon atoms behind the reattached episodes under the cutoffs stated in the paper (scale 1.0):", "",
              f"- Carbon atoms involved: {len(atoms)}.",
              f"- Removed as CO or CO2 later in the trajectory: {int(removed_later.sum())}, "
              f"after {quantiles(atoms.loc[removed_later, 'later'])} ps.",
              f"- Still in the cell at the end: {int((~removed_later).sum())}.", ""]
    summary["reattached_carbons"] = {"atoms": int(len(atoms)), "removed_later": int(removed_later.sum()),
                                     "still_present": int((~removed_later).sum()),
                                     "delay_ps_median": float(atoms.loc[removed_later, "later"].median()) if removed_later.any() else None}

    # ---------------------------------------------------------------- sensitivity of the detection
    lines += ["## 3. Detection repeated with the cutoffs changed by 10%", "",
              "Scale 1.0 is the set of cutoffs stated in the paper. Scale 1.1 means longer bond cutoffs, so a molecule has to be further from the surface to count as gas. "
              "Scale 0.9 means shorter cutoffs, so it counts as gas sooner.", ""]
    late = genuine[~genuine["detected_1.1"]]
    has_v = late["v_normal_A_per_ps"].notna()
    lines += ["**Longer cutoffs (scale 1.1).**", "",
              f"- Of the {len(genuine)} intact molecules removed, {int(genuine['detected_1.1'].sum())} are also gas "
              f"under the longer cutoffs at the same check. {len(late)} are not yet.",
              f"- Those {len(late)} sit {quantiles(late['d_solid'])} A from the solid. "
              f"{int((late.loc[has_v, 'v_normal_A_per_ps'] > 0).sum())} of {int(has_v.sum())} move away from the surface, "
              f"at {quantiles(late['v_normal_A_per_ps'])} A/ps along the normal.", ""]
    summary["scale_1.1"] = {"detected": int(genuine["detected_1.1"].sum()), "not_yet": int(len(late)),
                            "not_yet_moving_away": int((late.loc[has_v, "v_normal_A_per_ps"] > 0).sum()),
                            "not_yet_with_velocity": int(has_v.sum())}

    tight = carbon_episodes[(carbon_episodes["scale"] == 0.9) & (carbon_episodes["outcome"] == "rebonded")
                            & carbon_episodes["spans_boundary"]]
    tight_atoms = tight.groupby(["trajectory", tight["carbon_ids"].apply(lambda c: c[0] if c else -1)]).agg(
        temperature=("temperature", "first"), first_ps=("start_ps", "min"), later=("removed_later_after_ps", "max"))
    within = tight_atoms[tight_atoms["first_ps"] <= HORIZON]
    removed_later = within["later"].notna()
    lines += ["**Shorter cutoffs (scale 0.9).**", "",
              f"- Carbon atoms that would have been removed at a check although they were not removed there in the "
              f"production runs: {len(within)}.",
              f"- Of these, removed later in the production run anyway: {int(removed_later.sum())}, "
              f"after {quantiles(within.loc[removed_later, 'later'])} ps.",
              f"- Never removed in the production run: {int((~removed_later).sum())}. "
              "These are the carbon atoms whose fate depends on the cutoff.", "",
              "| T (K) | Removed in production | Extra removals with scale 0.9 | of which removed later anyway | "
              "of which never removed |", "|---|---|---|---|---|"]
    tight_rows = []
    for t in RP.TEMPERATURES:
        w = within[within["temperature"] == t]
        n_removed = int(((removals["temperature"] == t) & removals["within_horizon"]).sum())
        row = {"T": t, "removed": n_removed, "extra": int(len(w)), "later": int(w["later"].notna().sum()),
               "never": int(w["later"].isna().sum())}
        tight_rows.append(row)
        lines.append(f"| {t} | {row['removed']} | {row['extra']} | {row['later']} | {row['never']} |")
    summary["scale_0.9"] = tight_rows
    lines.append("")

    # ---------------------------------------------------------------- other free groups
    odd = Counter()
    for res in results.values():
        odd.update(res["odd_groups"])
    lone = episodes[(episodes["scale"] == 1.0) & (~episodes["has_carbon"])]
    lines += ["## 4. Free groups that the controller does not treat as gas", "",
              "The controller removes groups of two or three atoms. Frames in which a single free atom or a free "
              "group of four or more atoms exists, summed over all trajectories:", ""]
    total_frames = sum(res["nframes"] for res in results.values())
    for species, count in sorted(odd.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {species}: {count} frames of {total_frames} ({100 * count / total_frames:.2f}%).")
    if len(lone):
        lines.append(f"- Episodes of a single free O atom: {len(lone)}, longest {lone['duration_fs'].max():.0f} fs.")
    lines.append("")
    summary["odd_groups"] = {k: int(v) for k, v in odd.items()}
    summary["total_frames"] = int(total_frames)

    # ---------------------------------------------------------------- temperatures
    lines += ["## 5. Kinetic temperatures of the solid and of the gas", "",
              "From finite differences of the positions between consecutive frames. The solid is every atom that "
              "is not in a detached group. Each O2 molecule is split into translation (3 degrees of freedom), "
              "rotation (2) and vibration (1). Mean over the four cells, with the standard deviation between cells.", "",
              "| T target (K) | All atoms (K) | Solid (K) | Solid below all atoms (K) | Solid / all atoms | "
              "O2 translation (K) | O2 rotation (K) | O2 vibration (K) | Mean number of gas O2 |",
              "|---|---|---|---|---|---|---|---|---|"]
    thermo_rows = []
    series = {}
    for t in RP.TEMPERATURES:
        per_cell = defaultdict(list)
        for c in RP.CELLS:
            th = results[(t, c)]["thermo"]
            w = slice(0, int(HORIZON))
            per_cell["all"].append(2 * th["ke_all"][w].sum() / ((3 * th["n_all"][w].sum() - 3 * th["pairs"][w].sum()) * KB))
            per_cell["solid"].append(2 * th["ke_solid"][w].sum() / (3 * th["n_solid"][w].sum() * KB))
            per_cell["trans"].append(2 * th["ke_trans"][w].sum() / (3 * th["n_mol"][w].sum() * KB))
            per_cell["rot"].append(2 * th["ke_rot"][w].sum() / (2 * th["n_mol"][w].sum() * KB))
            per_cell["vib"].append(2 * th["ke_vib"][w].sum() / (th["n_mol"][w].sum() * KB))
            per_cell["n_o2"].append(th["n_mol"][w].sum() / th["pairs"][w].sum())
            with np.errstate(invalid="ignore", divide="ignore"):
                series[(t, c)] = 2 * th["ke_solid"][:int(HORIZON)] / (3 * th["n_solid"][:int(HORIZON)] * KB)
        row = {"T": t}
        for key, values in per_cell.items():
            row[key] = float(np.mean(values))
            row[key + "_sd"] = float(np.std(values, ddof=1))
        thermo_rows.append(row)
        lines.append(f"| {t} | {row['all']:.0f} ± {row['all_sd']:.0f} | {row['solid']:.0f} ± {row['solid_sd']:.0f} | "
                     f"{row['all'] - row['solid']:.0f} | {row['solid'] / row['all']:.3f} | "
                     f"{row['trans']:.0f} ± {row['trans_sd']:.0f} | "
                     f"{row['rot']:.0f} ± {row['rot_sd']:.0f} | {row['vib']:.0f} ± {row['vib_sd']:.0f} | {row['n_o2']:.2f} |")
    summary["temperatures"] = thermo_rows

    lines += ["", "Solid temperature by time window (K, mean of the four cells):", "",
              "| T target (K) | 0 to 5 ps | 5 to 40 ps | 40 to 140 ps |", "|---|---|---|---|"]
    window_rows = []
    for t in RP.TEMPERATURES:
        row = {"T": t}
        for name, (a, b) in (("0-5", (0, 5)), ("5-40", (5, 40)), ("40-140", (40, 140))):
            values = []
            for c in RP.CELLS:
                th = results[(t, c)]["thermo"]
                values.append(2 * th["ke_solid"][a:b].sum() / (3 * th["n_solid"][a:b].sum() * KB))
            row[name] = float(np.mean(values))
        window_rows.append(row)
        lines.append(f"| {t} | {row['0-5']:.0f} | {row['5-40']:.0f} | {row['40-140']:.0f} |")
    summary["solid_by_window"] = window_rows

    lines += ["", "Solid temperature by the number of O2 molecules in the gas at that moment "
              "(K, all frames of the four cells pooled, share of the frames in brackets):", "",
              "| T target (K) | 0 | 1 | 2 | 3 | 4 and more |", "|---|---|---|---|---|---|"]
    by_rows = []
    for t in RP.TEMPERATURES:
        table = sum(results[(t, c)]["thermo"]["by_no2"] for c in RP.CELLS)
        merged = np.vstack([table[:4], table[4:].sum(axis=0)])
        cells_txt, row = [], {"T": t}
        for k, (ke, natoms, pairs) in enumerate(merged):
            if natoms > 0:
                value = 2 * ke / (3 * natoms * KB)
                share = pairs / merged[:, 2].sum()
                cells_txt.append(f"{value:.0f} ({100 * share:.0f}%)")
                row[str(k)] = {"T_solid": float(value), "share": float(share)}
            else:
                cells_txt.append("n/a")
        by_rows.append(row)
        lines.append(f"| {t} | " + " | ".join(cells_txt) + " |")
    summary["solid_by_n_o2"] = by_rows
    lines.append("")

    # ---------------------------------------------------------------- supply
    lines += ["## 6. Oxygen in the gas and replenishment", "",
              "| T (K) | O2 additions per cell | Mean time between additions (ps) | Frames with no gas O2 | "
              "with one | with two | with three or more |", "|---|---|---|---|---|---|---|"]
    supply_rows = []
    for t in RP.TEMPERATURES:
        n_add, gaps, counts = [], [], np.zeros(4)
        for c in RP.CELLS:
            res = results[(t, c)]
            times = np.array([a["time_fs"] for a in res["additions"]]) / 1000.0
            times = times[times <= HORIZON]
            n_add.append(len(times))
            gaps.extend(np.diff(times).tolist())
            mask = res["time"] <= HORIZON * 1000.0
            n = np.clip(res["n_o2"][mask], 0, 3)
            counts += np.bincount(n, minlength=4)
        share = counts / counts.sum()
        row = {"T": t, "additions": float(np.mean(n_add)), "gap_ps": float(np.mean(gaps)),
               "share": [float(x) for x in share]}
        supply_rows.append(row)
        lines.append(f"| {t} | {row['additions']:.1f} | {row['gap_ps']:.1f} | {100 * share[0]:.0f}% | "
                     f"{100 * share[1]:.0f}% | {100 * share[2]:.0f}% | {100 * share[3]:.0f}% |")
    summary["supply"] = supply_rows

    replays = [dict(r, trajectory=res["trajectory"]) for res in results.values() for r in res["replays"]]
    choice = Counter()
    for res in results.values():
        choice.update(res["step_choice"])
    lines += ["", "## 7. Time step labels", "",
              "The stored time of each frame comes from a time step per 80-step segment. The kinetic temperature of "
              "all atoms shows which time step each segment was really run with.", "",
              f"- Segments whose own label fits (or both labels are equal): {choice['label of the segment']}.",
              f"- Segments that were run with the time step labelled on the segment before: "
              f"{choice['label of the segment before']}.", ""]
    summary["step_choice"] = dict(choice)
    lines += ["## 8. Segments that were run twice", "",
              f"In {len(replays)} places an 80-step segment starts again from the positions at which the previous "
              "segment started, so that segment was run twice. The frames of both runs are in the files.", ""]
    for r in replays:
        lines.append(f"- {r['trajectory']} at {r['time_ps']:.2f} ps")
    summary["replays"] = replays
    lines.append("")

    (OUT / "Tables.md").write_text("\n".join(lines), encoding="utf-8")
    with open(OUT / "results.json", "w") as handle:
        json.dump(summary, handle, indent=1, default=float)

    # ---------------------------------------------------------------- figure
    colours = {873: "#2166ac", 973: "#67a9cf", 1073: "#ef8a62", 1273: "#b2182b"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    bins = np.arange(2.0, 8.01, 0.5)
    ax.hist([genuine.loc[genuine["temperature"] == t, "d_solid"].clip(upper=7.99) for t in RP.TEMPERATURES], bins=bins,
            stacked=True, color=[colours[t] for t in RP.TEMPERATURES], label=[f"{t} K" for t in RP.TEMPERATURES])
    ax.set_xlabel("Distance to the solid at removal (Å)")
    ax.set_ylabel("Molecules")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[0, 1]
    bins = np.arange(-10, 40.01, 2.5)
    ax.hist([genuine.loc[genuine["temperature"] == t, "v_normal_A_per_ps"].dropna().clip(-9.99, 39.99) for t in RP.TEMPERATURES],
            bins=bins, stacked=True, color=[colours[t] for t in RP.TEMPERATURES])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Velocity away from the surface (Å/ps)")
    ax.set_ylabel("Molecules")
    ax = axes[1, 0]
    bins = np.arange(0, 201, 10)
    ax.hist([genuine.loc[genuine["temperature"] == t, "free_fs_1.0"].dropna().clip(upper=199) for t in RP.TEMPERATURES],
            bins=bins, stacked=True, color=[colours[t] for t in RP.TEMPERATURES])
    ax.set_xlabel("Time detached before removal (fs)")
    ax.set_ylabel("Molecules")
    ax = axes[1, 1]
    for t in RP.TEMPERATURES:
        mean = np.nanmean([series[(t, c)] for c in RP.CELLS], axis=0)
        smooth = np.convolve(mean, np.ones(5) / 5, mode="valid")
        ax.plot(np.arange(len(smooth)) + 2.5, smooth / t, color=colours[t], lw=1.2)
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Solid temperature / target")
    for label, ax in zip("abcd", axes.ravel()):
        ax.text(-0.14, 1.03, f"({label})", transform=ax.transAxes, fontsize=10, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "Fig_GasExchange.pdf")
    fig.savefig(OUT / "Fig_GasExchange.png", dpi=200)
    print("\n".join(lines))
    delayed_removals()


# -----------------------------------------------------------------------------
# Molecules that were detached at a check and removed at a later one
# -----------------------------------------------------------------------------

# Trajectories with the erroneous deletions described in the Supplementary Information.
KNOWN_FAULTY = ("973_1", "1273_3")
PAIRS = ("C(mol)-Zr", "O(mol)-Zr", "C(mol)-O", "O(mol)-C", "O(mol)-O", "C(mol)-C")


def detached_products(data, frame):
    """Detached CO and CO2 in one frame under the cutoffs of the paper, with the
    shortest distance from the molecule to each kind of atom outside it."""
    n = int(data["n"][frame])
    el = data["el"][frame, :n]
    pos = data["pos"][frame, :n].astype(float)
    lattice = data["lat"][frame]
    dist = distance_matrix(pos, lattice, np.linalg.inv(lattice))
    bond = dist < TOLERANCE * (RADIUS[el][:, None] + RADIUS[el][None, :])
    np.fill_diagonal(bond, False)
    zirconium = el == 2
    attached = zirconium | bond[:, zirconium].any(axis=1)
    found = []
    for carbon in np.where((el == 3) & ~attached)[0]:
        members, stack, tied = {carbon}, [carbon], False
        while stack:
            atom = stack.pop()
            for other in np.where(bond[atom])[0]:
                if attached[other]:
                    tied = True
                elif other not in members:
                    members.add(other)
                    stack.append(other)
        if tied or not 2 <= len(members) <= 3:
            continue
        members = sorted(members)
        others = np.array([i for i in range(n) if i not in members])
        row = {}
        for atom in members:
            for code in (1, 2, 3):
                partners = others[el[others] == code]
                key = f"{SYMBOL[el[atom]]}(mol)-{SYMBOL[code]}"
                row[key] = min(row.get(key, 99.0), float(dist[atom, partners].min()))
        found.append(row)
    return found


def delayed_removals():
    lines = ["# Removal at the checks: tables", "", "Written by `Summarise.py`.", ""]
    passed_rows, separation = [], []
    for t in RP.TEMPERATURES:
        for c in RP.CELLS:
            name = f"{t}_{c}"
            with open(Path(DEFAULT_CACHE) / f"analysis_{name}.pkl", "rb") as handle:
                res = pickle.load(handle)
            with np.load(Path(DEFAULT_CACHE) / f"{name}.npz") as npz:
                data = {k: npz[k] for k in ("pos", "el", "n", "lat")}
            time, starts = res["time"], res["starts"]
            checks = starts[starts > 0] - 1  # last frame of each segment
            gone = res["carbon_gone"]
            removed_episode = {(e["end"], tuple(e["carbon_ids"])): e for e in res["episodes"][1.0]
                               if e["outcome"] == "removed"}
            for r in res["removals"]:
                if r["species"] == "OO" or not r["bonded_molecule"] or not r["detected_1.0"]:
                    continue
                episode = removed_episode.get((r["frame"] - 1, tuple(r["carbon_ids"])))
                earlier = checks[(checks >= episode["start"]) & (checks < r["frame"] - 1)] if episode else []
                passed_rows.append({"trajectory": name, "temperature": t, "time_ps": r["time_ps"],
                                    "species": r["species"], "checks_passed": len(earlier),
                                    "extra_time_fs": float(time[r["frame"] - 1] - time[earlier[0]]) if len(earlier) else 0.0,
                                    "faulty_trajectory": name in KNOWN_FAULTY})
                for row in detached_products(data, r["frame"] - 1):
                    separation.append({"trajectory": name, "frame": r["frame"] - 1, "kind": "removed", **row})
                for check in earlier:
                    for row in detached_products(data, int(check)):
                        separation.append({"trajectory": name, "frame": int(check), "kind": "left in",
                                           "then": "removed at a later check", **row})
            for e in res["episodes"][1.0]:
                if "C" not in e["species"] or e["outcome"] != "rebonded":
                    continue
                for check in checks[(checks >= e["start"]) & (checks <= e["end"])]:
                    delay = [gone[i] - time[check] for i in e["carbon_ids"] if i in gone]
                    for row in detached_products(data, int(check)):
                        separation.append({"trajectory": name, "frame": int(check), "kind": "left in",
                                           "then": "reattached",
                                           "removed_after_ps": delay[0] / 1000.0 if delay else np.nan, **row})

    passed = pd.DataFrame(passed_rows)
    passed.to_csv(OUT / "ChecksPassed.csv", index=False)
    sep = pd.DataFrame(separation)
    sep["faulty_trajectory"] = sep["trajectory"].isin(KNOWN_FAULTY)
    sep.to_csv(OUT / "SeparationAtChecks.csv", index=False)

    lines += ["## 1. Checks at which a detached molecule was left in the cell", "",
              "Detached means detached under the bond cutoffs stated in the paper. "
              "Intact molecules only (150).", "",
              "| T (K) | Removed at the first check | One check later | Two checks later | Three checks later |",
              "|---|---|---|---|---|"]
    for t in list(RP.TEMPERATURES) + ["all"]:
        g = passed if t == "all" else passed[passed["temperature"] == t]
        counts = [int((g["checks_passed"] == k).sum()) for k in range(4)]
        lines.append(f"| {t} | {counts[0]} | {counts[1]} | {counts[2]} | {counts[3]} |")
    late = passed[passed["checks_passed"] > 0]
    clean = passed[~passed["faulty_trajectory"]]
    lines += ["", f"- Removed later than the first check: {len(late)} of {len(passed)} "
              f"({100 * len(late) / len(passed):.0f}%). Without the two trajectories with known faults "
              f"({', '.join(KNOWN_FAULTY)}): {int((clean['checks_passed'] > 0).sum())} of {len(clean)}.",
              f"- Extra time these molecules spent in the cell: median {late['extra_time_fs'].median():.0f} fs, "
              f"longest {late['extra_time_fs'].max():.0f} fs.", ""]

    left = sep[sep["kind"] == "left in"]
    stayed = left[left["then"] == "removed at a later check"]
    back = left[left["then"] == "reattached"]
    lines += ["## 2. What the molecules that were left in did next", "",
              f"- Times a detached molecule was left in the cell at a check: {len(left)}.",
              f"- It stayed detached until it was removed at a later check: {len(stayed)}.",
              f"- It reattached to the surface: {len(back)}. "
              f"Of these, the carbon atom left as CO or CO2 later in {int(back['removed_after_ps'].notna().sum())} cases, "
              f"after {back['removed_after_ps'].median():.2f} ps (median, longest {back['removed_after_ps'].max():.2f} ps).",
              f"- Carbon atoms that were detached at a check and are still in the cell at the end: "
              f"{int(back['removed_after_ps'].isna().sum())}.", ""]

    healthy = sep[~sep["faulty_trajectory"]]
    removed, kept = healthy[healthy["kind"] == "removed"], healthy[healthy["kind"] == "left in"]
    lines += ["## 3. What separates the molecules that were removed from those left in", "",
              "Shortest distance between an atom of the molecule and an atom outside it, at the check. "
              f"The two trajectories with known faults are left out ({len(removed)} removed, {len(kept)} left in). "
              "The last two columns give the single cutoff on that distance that best separates the two groups.", "",
              "| Pair | Cutoff in the paper (A) | Removed: smallest three (A) | Left in: largest three (A) | "
              "Best single cutoff (A) | Correctly separated |", "|---|---|---|---|---|---|"]
    paper = {"C(mol)-Zr": 3.33, "O(mol)-Zr": 3.17, "C(mol)-O": 2.15, "O(mol)-C": 2.15, "O(mol)-O": 1.98, "C(mol)-C": 2.31}
    best_rows = {}
    for pair in PAIRS:
        a, b = removed[pair].dropna().values, kept[pair].dropna().values
        grid = np.arange(1.9, 5.0, 0.01)
        score = [(a > x).sum() + (b <= x).sum() for x in grid]
        cut = grid[int(np.argmax(score))]
        best_rows[pair] = (cut, max(score))
        lines.append(f"| {pair} | {paper[pair]:.2f} | {', '.join(f'{x:.2f}' for x in np.sort(a)[:3])} | "
                     f"{', '.join(f'{x:.2f}' for x in np.sort(b)[::-1][:3])} | {cut:.2f} | "
                     f"{max(score)} of {len(a) + len(b)} |")
    a, b = removed["C(mol)-O"].values, kept["C(mol)-O"].values
    lines += ["", f"- Carbon of the molecule to the nearest oxygen outside it: {int((a > 2.98).sum())} of {len(a)} removed "
              f"molecules are above 2.98 A and {int((b <= 2.98).sum())} of {len(b)} left-in molecules are below.",
              "- With the cutoffs of the paper, none of the left-in molecules is separated from the removed ones, "
              "since all of them count as detached.", ""]

    (OUT / "DelayedRemovals.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
