"""Kinetic model on two carbon properties at once.

Each carbon has two yes/no properties, defined identically for all carbon:

  Zr coordination   five or more of its original Zr neighbours within 3.33 A,
                    or fewer
  carbon bonding    a carbon within 2.31 A, or none

Original Zr neighbours are the Zr atoms within 3.33 A of the carbon in the
first frame. Counting only these separates a bonded carbon that is still in
its lattice site from one that has left it and sits near other Zr atoms.
Counting any Zr (count="any") is the earlier rule, kept for comparison.

No property overrides the other. The four joint states, plus gas:

  Lattice            coordinated,       no carbon bond
  Undercoordinated   undercoordinated,  no carbon bond
  ChainInLattice     coordinated,       carbon bond
  ChainUnder         undercoordinated,  carbon bond
  Gas                removed as CO or CO2

Built from the raw-trajectory pass, so the 0.2 ps filter acts on every MD
step. Each property is filtered separately. A change is accepted once it has
held for 0.2 ps and is stamped at the moment it began (one physical clock).
The default is 0.2 ps for every change (chain_hold=None).
An earlier version required a change of the Zr coordination class of bonded
carbon to hold for 0.5 ps (chain_hold=0.5). It was
dropped because the liberation rate already counts only liberations that
are not reversed, and one persistence time is simpler to state. Outputs in
SquareModelResults/ were written with the earlier default.
Release as gas is recorded from the instantaneous state, which is
undercoordinated and unbonded, so gas is reached from Undercoordinated only.
Five erroneously deleted carbons are censored at deletion.

Outputs in SquareModelResults/: transition counts and rates per temperature,
conversion (first departure from Lattice), coupling between the two
properties, route shares, a test of the constant-rate model against the
measured state fractions, a sensitivity table, plots S1 to S5 and SUMMARY.md.

Run:  python SquareModel.py      (needs RawCoordinationPass.py and
                                  RawOriginalNeighbourPass.py to have run)
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "carbon_states"  # results and caches are kept beside the scripts in carbon_states/
RAW = HERE / "StateModelResults" / "RawCoordination"
ORIGINAL = HERE / "StateModelResults" / "RawOriginalNeighbours"
OUT = HERE / "SquareModelResults"
LEDGER = RP.REWRITE_DIR / "IndependentTrajectoryResults" / "CarbonOrigins.csv"

LAT, UND, CHL, CHU, GAS, CENS = 0, 1, 2, 3, 4, -1
NAMES = ["Lattice", "Undercoordinated", "ChainInLattice", "ChainUnder", "Gas"]
SHORT = ["Lattice", "Undercoord.", "Chain in lattice", "Chain undercoord.", "Gas"]
COLOURS = ["0.45", "#7570b3", "#66c2a5", "#1b9e77", "#d95f02"]
T_COLOURS = {873: "#1f77b4", 973: "#d62728", 1073: "#2ca02c", 1273: "#9467bd"}
KB_EV = 8.617333e-5
INV_T = 1.0 / np.array(RP.TEMPERATURES, float)
N_BOOT = 10000
EXPECT_CHAIN = {873: 35, 973: 31, 1073: 48, 1273: 69}
EXPECT_GAS = {873: 31, 973: 47, 1073: 38, 1273: 36}
EXPECT_UNDER = {873: 5, 973: 3, 1073: 1, 1273: 2}
CELLS = [(t, c) for t in RP.TEMPERATURES for c in RP.CELLS]


def confirm(label, times, hold, flush):
    """Persistence filter on one yes/no property of one carbon. A change is
    accepted once it has held for `hold` ps and is stamped at its onset.
    A change still pending at the 140 ps horizon is not accepted (flush off),
    which is what reproduces the published states at 140 ps."""
    n = len(label)
    out = np.empty(n, dtype=np.int8)
    cuts = np.flatnonzero(np.diff(label)) + 1
    starts, ends = np.concatenate(([0], cuts)), np.concatenate((cuts, [n]))
    current, pos = label[0], 0
    for b, e in zip(starts, ends):
        if label[b] == current:
            continue
        if times[e - 1] - times[b] >= hold - 1e-9 or (flush and e == n):
            out[pos:b] = current
            current, pos = label[b], b
    out[pos:] = current
    return out


def confirm_from(label, times, hold, start):
    """The same filter, starting from a given class instead of the first label."""
    n = len(label)
    out = np.empty(n, dtype=np.int8)
    cuts = np.flatnonzero(np.diff(label)) + 1
    starts, ends = np.concatenate(([0], cuts)), np.concatenate((cuts, [n]))
    current, pos = start, 0
    for b, e in zip(starts, ends):
        if label[b] == current:
            continue
        if times[e - 1] - times[b] >= hold - 1e-9:
            out[pos:b] = current
            current, pos = label[b], b
    out[pos:] = current
    return out


def build(name, artifacts, zr_k=1, c_k=1, min_zr=5, hold=0.2, count="original", chain_hold=None):
    """Joint state of every carbon at every MD step. `count` chooses which Zr
    atoms are counted: the carbon's original neighbours, or any Zr.
    `chain_hold` is the persistence time for the coordination class of a
    carbon while it is bonded; None or a value up to `hold` means no
    separate rule."""
    with np.load(RAW / f"{name}.npz") as z:
        times, n_zr, n_c = z["Times_fs"] / 1000.0, z["NZr"][zr_k], z["NC"][c_k]
    if count == "original":
        with np.load(ORIGINAL / f"{name}.npz") as z:
            if not np.array_equal(z["Times_fs"] / 1000.0, times):
                raise ValueError(f"{name}: the two raw passes do not cover the same frames")
            n_zr = z["NOriginal"][zr_k]
    elif count != "any":
        raise ValueError(count)
    states = np.full(n_zr.shape, GAS, dtype=np.int8)
    odd_gas = 0
    for a in range(48):
        n = int((n_zr[:, a] >= 0).sum())
        removed = n < len(times)
        coord = confirm((n_zr[:n, a] >= min_zr).astype(np.int8), times[:n], hold, flush=False)
        bond = confirm((n_c[:n, a] >= 1).astype(np.int8), times[:n], hold, flush=False)
        chain_coord = coord
        if chain_hold is not None and chain_hold > hold and bond.any():
            chain_coord = coord.copy()
            edges = np.flatnonzero(np.diff(np.concatenate(([0], bond, [0]))))
            for s, e in zip(edges[::2], edges[1::2]):       # each bonded stretch starts from the class it is entered in
                chain_coord[s:e] = confirm_from((n_zr[s:e, a] >= min_zr).astype(np.int8), times[s:e], chain_hold, coord[s])
        st = np.where(bond == 1, np.where(chain_coord == 1, CHL, CHU), np.where(coord == 1, LAT, UND)).astype(np.int8)
        if removed and a in artifacts:
            states[n:, a] = CENS
        elif removed:
            k = n - 1                      # release is recorded from the instantaneous state before removal
            if not (n_c[k, a] == 0 and n_zr[k, a] < min_zr):
                odd_gas += 1
            while k > 0 and n_c[k - 1, a] == 0 and n_zr[k - 1, a] < min_zr:
                k -= 1
            st[k:] = UND
        states[:n, a] = st
    return times, states, odd_gas


def tally(times, states):
    """Episodes, transition counts, time in each state and per-carbon
    conversion records for one cell."""
    counts, time_in, records, episodes = np.zeros((5, 5)), np.zeros(5), [], []
    end_time = RP.HORIZON_PS
    for a in range(48):
        s = states[:, a]
        valid = int((s != CENS).sum())
        stop = times[valid] if valid < len(times) else end_time      # censoring time
        cuts = np.flatnonzero(np.diff(s[:valid])) + 1
        starts, ends = np.concatenate(([0], cuts)), np.concatenate((cuts, [valid]))
        path = []
        for b, e in zip(starts, ends):
            state = int(s[b])
            t0 = times[b] if b else 0.0
            t1 = times[e] if e < valid else stop
            nxt = int(s[e]) if e < valid else None
            if state != GAS:
                time_in[state] += t1 - t0
            if nxt is not None:
                counts[state, nxt] += 1
            episodes.append((a, state, t0, t1, nxt))
            path.append((state, t0))
        left = [p for p in path if p[0] != LAT]
        chain_under = [k for k, p in enumerate(path) if p[0] == CHU]
        records.append(dict(AtomID=a, Convert_ps=left[0][1] if left else stop, Converted=bool(left),
                            ConvertTo=NAMES[left[0][0]] if left else "",
                            EverBonded=any(p[0] in (CHL, CHU) for p in path), EverUnder=any(p[0] in (UND, CHU) for p in path),
                            RouteToChainUnder=NAMES[path[chain_under[0] - 1][0]] if chain_under else "",
                            Final=NAMES[int(s[valid - 1])] if valid == len(times) else "ArtifactCensored"))
    return counts, time_in, records, episodes


def analyse(**rule):
    """Everything for one choice of definitions."""
    ledger = pd.read_csv(LEDGER)
    per_cell, records, state_grid, odd = {}, [], {}, 0
    grid = np.arange(0.0, RP.HORIZON_PS + 1e-9, 0.25)
    for temperature, cell in CELLS:
        name = f"{temperature}_{cell}"
        artifacts = set(ledger[(ledger.Trajectory == name) & (ledger.Kind == "ArtifactRemoval")].AtomID)
        times, states, odd_gas = build(name, artifacts, **rule)
        odd += odd_gas
        counts, time_in, recs, _ = tally(times, states)
        per_cell[(temperature, cell)] = (counts, time_in)
        for r in recs:
            records.append(dict(Temperature=temperature, Trajectory=name, **r))
        state_grid[(temperature, cell)] = states[np.clip(np.searchsorted(times, grid), 0, len(times) - 1)]
    return dict(per_cell=per_cell, atoms=pd.DataFrame(records), grid=grid, state_grid=state_grid, odd_gas=odd)


def pooled(per_cell, temperature=None):
    keys = [k for k in per_cell if temperature is None or k[0] == temperature]
    return sum(per_cell[k][0] for k in keys), sum(per_cell[k][1] for k in keys)


def conversion_rates(atoms):
    return np.array([a.Converted.sum() / a.Convert_ps.sum() for _, a in atoms.groupby("Temperature")])


def ea_of(rates):
    slope, intercept = np.polyfit(INV_T, np.log(rates), 1)
    resid = np.log(rates) - (intercept + slope * INV_T)
    return -slope * KB_EV, 1 - np.sum(resid ** 2) / np.sum((np.log(rates) - np.log(rates).mean()) ** 2)


def bootstrap_conversion(atoms, rng, resample_atoms):
    logs = np.empty((N_BOOT, 4))
    for j, temperature in enumerate(RP.TEMPERATURES):
        sub = atoms[atoms.Temperature == temperature].sort_values(["Trajectory", "AtomID"])
        t, e = sub.Convert_ps.to_numpy().reshape(4, 48), sub.Converted.to_numpy().reshape(4, 48)
        cell = rng.integers(0, 4, size=(N_BOOT, 4))
        atom = rng.integers(0, 48, size=(N_BOOT, 4, 48)) if resample_atoms else np.broadcast_to(np.arange(48), (N_BOOT, 4, 48))
        pick = (cell[:, :, None], atom)
        logs[:, j] = np.log(e[pick].sum((1, 2)) / t[pick].sum((1, 2)))
    x = INV_T - INV_T.mean()
    ea = -(logs - logs.mean(1, keepdims=True)) @ x / np.sum(x ** 2) * KB_EV
    return float(ea.std(ddof=1)), [float(np.percentile(ea, 2.5)), float(np.percentile(ea, 97.5))]


def boot_cells(per_cell, rng, temperature=None):
    """Resampled pooled (counts, time) arrays: whole cells drawn with replacement within each temperature."""
    temps = RP.TEMPERATURES if temperature is None else (temperature,)
    n = np.zeros((N_BOOT, 5, 5))
    t = np.zeros((N_BOOT, 5))
    for temp in temps:
        cn = np.stack([per_cell[(temp, c)][0] for c in RP.CELLS])
        ct = np.stack([per_cell[(temp, c)][1] for c in RP.CELLS])
        pick = rng.integers(0, 4, size=(N_BOOT, 4))
        n += cn[pick].sum(1)
        t += ct[pick].sum(1)
    return n, t


def interval(values):
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


EDGES = [(LAT, UND), (UND, LAT), (CHL, CHU), (CHU, CHL), (LAT, CHL), (CHL, LAT), (UND, CHU), (CHU, UND), (UND, GAS)]
DIAGONALS = [(LAT, CHU), (CHU, LAT), (UND, CHL), (CHL, UND)]
COUPLING = [("Losing Zr coordination", (LAT, UND), (CHL, CHU), "bonded / unbonded"),
            ("Regaining Zr coordination", (UND, LAT), (CHU, CHL), "bonded / unbonded"),
            ("Forming a carbon bond", (LAT, CHL), (UND, CHU), "undercoordinated / coordinated"),
            ("Breaking a carbon bond", (CHL, LAT), (CHU, UND), "undercoordinated / coordinated")]


def main():
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(RP.SEED)
    base = analyse()
    per_cell, atoms = base["per_cell"], base["atoms"]
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    lines = ["# Square model: Zr coordination and carbon bonding tracked together", "",
             "Five or more of a carbon's original Zr neighbours within 3.33 A counts as coordinated, a carbon within 2.31 A as bonded. "
             "Original neighbours are the Zr within 3.33 A in the first frame. Each property is filtered separately "
             "(0.2 ps, every MD step) and stamped at onset. While a carbon is bonded, a change of its coordination class must hold 0.5 ps. Gas is reached from Undercoordinated only. Five erroneous deletions "
             "are censored.", ""]

    # ---- guardrails -------------------------------------------------------
    final = pd.crosstab(atoms.Temperature, atoms.Final)
    for t in RP.TEMPERATURES:
        row = final.loc[t]
        got = (int(row.get("ChainInLattice", 0) + row.get("ChainUnder", 0)), int(row.get("Gas", 0)), int(row.get("Undercoordinated", 0)))
        want = (EXPECT_CHAIN[t], EXPECT_GAS[t], EXPECT_UNDER[t])
        if got != want:
            raise ValueError(f"{t} K: chain, gas, undercoordinated at 140 ps are {got}, published {want}")
    lines += ["Check: chain, gas and undercoordinated counts at 140 ps equal the published values at every temperature. "
              f"Gas carbons not undercoordinated and unbonded at their last MD step: {base['odd_gas']}.", "",
              "## Carbons in each state at 140 ps", "", "| T (K) | " + " | ".join(NAMES) + " | Censored |", "|---|" + "---|" * 6]
    for t in RP.TEMPERATURES:
        lines.append(f"| {t} | " + " | ".join(str(int(final.loc[t].get(n, 0))) for n in NAMES) + f" | {int(final.loc[t].get('ArtifactCensored', 0))} |")

    # ---- transitions and rates ---------------------------------------------
    counts_all, time_all = pooled(per_cell)
    lines += ["", "## Transitions, pooled over temperature (rows from, columns to)", "", "| | " + " | ".join(NAMES) + " | Time in state (ps) |",
              "|---|" + "---|" * 6]
    for i in range(4):
        lines.append(f"| {NAMES[i]} | " + " | ".join("" if i == j else str(int(counts_all[i, j])) for j in range(5)) + f" | {time_all[i]:.0f} |")
    rate_rows = []
    for temperature in RP.TEMPERATURES:
        n, t = pooled(per_cell, temperature)
        bn, bt = boot_cells(per_cell, rng, temperature)
        for i, j in EDGES + DIAGONALS:
            with np.errstate(divide="ignore", invalid="ignore"):
                lo, hi = interval(bn[:, i, j] / bt[:, i])
            rate_rows.append(dict(Temperature=temperature, From=NAMES[i], To=NAMES[j], Events=int(n[i, j]), TimeInFrom_ps=t[i],
                                  Rate_per_ps=n[i, j] / t[i], Low=lo, High=hi))
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(OUT / "Rates.csv", index=False)
    lines += ["", "## Rates along each edge (10^-3 ps^-1 per carbon in the origin state) and four-point activation energy", "",
              "| Transition | " + " | ".join(f"{t} K" for t in RP.TEMPERATURES) + " | Events | Ea (eV) |", "|---|" + "---|" * 6]
    for i, j in EDGES:
        r = rates[(rates.From == NAMES[i]) & (rates.To == NAMES[j])].sort_values("Temperature")
        ok = (r.Events > 0).all()
        ea = f"{ea_of(r.Rate_per_ps.to_numpy())[0]:+.2f}" if ok else "n/a"
        lines.append(f"| {NAMES[i]} to {NAMES[j]} | " + " | ".join(f"{1e3 * v:.2f}" for v in r.Rate_per_ps) + f" | {int(r.Events.sum())} | {ea} |")
    lines.append("")
    lines.append("Diagonal moves (both properties change within the filter): " +
                 ", ".join(f"{NAMES[i]} to {NAMES[j]} {int(counts_all[i, j])}" for i, j in DIAGONALS) + ".")

    # ---- conversion ---------------------------------------------------------
    conv = conversion_rates(atoms)
    ea, r2 = ea_of(conv)
    sd2, ci2 = bootstrap_conversion(atoms, rng, True)
    sd1, ci1 = bootstrap_conversion(atoms, rng, False)
    route = pd.crosstab(atoms.Temperature, atoms.ConvertTo)
    lines += ["", "## Conversion: first departure from the lattice state", "",
              "| T (K) | Converted | Rate (10^-3 ps^-1) | by losing Zr coordination | by bonding in place | both at once |", "|---|---|---|---|---|---|"]
    for t, rate in zip(RP.TEMPERATURES, conv):
        row = route.loc[t]
        lines.append(f"| {t} | {int(atoms[atoms.Temperature == t].Converted.sum())} | {1e3 * rate:.2f} | {int(row.get('Undercoordinated', 0))} | "
                     f"{int(row.get('ChainInLattice', 0))} | {int(row.get('ChainUnder', 0))} |")
    lines += ["", f"Activation energy of conversion: {ea:.3f} eV (R2 {r2:.3f}); two-level bootstrap s.d. {sd2:.3f}, 95% [{ci2[0]:.3f}, {ci2[1]:.3f}]; "
              f"cell-only s.d. {sd1:.3f}, 95% [{ci1[0]:.3f}, {ci1[1]:.3f}]. Submitted commitment value: 0.152 eV."]

    # ---- coupling -----------------------------------------------------------
    bn, bt = boot_cells(per_cell, rng)
    coupling = []
    lines += ["", "## Coupling between the two properties (pooled over temperature)", "",
              "| Step | Rate without (ps^-1) | Rate with (ps^-1) | Ratio | 95% range | Ratio is |", "|---|---|---|---|---|---|"]
    for label, (i0, j0), (i1, j1), meaning in COUPLING:
        r0, r1 = counts_all[i0, j0] / time_all[i0], counts_all[i1, j1] / time_all[i1]
        with np.errstate(divide="ignore", invalid="ignore"):
            lo, hi = interval((bn[:, i1, j1] / bt[:, i1]) / (bn[:, i0, j0] / bt[:, i0]))
        coupling.append(dict(Step=label, RateWithout=r0, RateWith=r1, Ratio=r1 / r0, Low=lo, High=hi, Meaning=meaning,
                             EventsWithout=int(counts_all[i0, j0]), EventsWith=int(counts_all[i1, j1])))
        lines.append(f"| {label} | {r0:.2e} ({int(counts_all[i0, j0])}) | {r1:.2e} ({int(counts_all[i1, j1])}) | {r1 / r0:.2g} | {lo:.2g} to {hi:.2g} | {meaning} |")
    coupling = pd.DataFrame(coupling)
    coupling.to_csv(OUT / "Coupling.csv", index=False)

    # ---- routes -------------------------------------------------------------
    bonded = atoms[atoms.EverBonded]
    route_cu = pd.crosstab(bonded.Temperature, bonded.RouteToChainUnder.replace({"": "Never undercoordinated"}))
    order = ["Undercoordinated", "ChainInLattice", "Lattice", "Never undercoordinated"]
    labels = ["Lost Zr coordination first", "Bonded first", "Both at once", "Bonded, still coordinated"]
    lines += ["", "## Carbons that bonded to carbon, by route to the undercoordinated chain state", "",
              "| T (K) | " + " | ".join(labels) + " | Total |", "|---|" + "---|" * 5]
    for t in RP.TEMPERATURES:
        vals = [int(route_cu.loc[t].get(k, 0)) for k in order]
        lines.append(f"| {t} | " + " | ".join(map(str, vals)) + f" | {sum(vals)} |")
    tot = [int(route_cu.get(k, pd.Series(dtype=float)).sum()) if k in route_cu else 0 for k in order]
    lines.append("| all | " + " | ".join(map(str, tot)) + f" | {sum(tot)} |")

    # ---- model test ---------------------------------------------------------
    grid = base["grid"]
    measured, modelled, misfit = {}, {}, []
    for temperature in RP.TEMPERATURES:
        n, t = pooled(per_cell, temperature)
        q = np.zeros((5, 5))
        q[:4] = n[:4] / t[:4, None]
        np.fill_diagonal(q, 0.0)
        np.fill_diagonal(q, -q.sum(1))
        sg = np.concatenate([base["state_grid"][(temperature, c)] for c in RP.CELLS], axis=1)
        observed = (sg != CENS).sum(1)
        measured[temperature] = np.stack([(sg == k).sum(1) / observed for k in range(5)], axis=1)
        modelled[temperature] = np.stack([expm(q * tt)[0] for tt in grid])
        diff = modelled[temperature] - measured[temperature]
        misfit.append((temperature, np.abs(diff).max(0), np.sqrt((diff ** 2).mean(0))))
    lines += ["", "## Does a constant-rate model reproduce the measured state fractions?", "",
              "Largest gap between model and measurement over 0 to 140 ps, in fraction of all carbon.", "",
              "| T (K) | " + " | ".join(NAMES) + " |", "|---|" + "---|" * 5]
    for temperature, worst, _ in misfit:
        lines.append(f"| {temperature} | " + " | ".join(f"{v:.3f}" for v in worst) + " |")

    # ---- sensitivity --------------------------------------------------------
    variants = [("As defined (5 original Zr, 0.2 ps)", {}), ("Bonded carbon coordination 0.5 ps", dict(chain_hold=0.5)),
                ("Bonded carbon coordination 1.0 ps", dict(chain_hold=1.0)), ("Any Zr counted, 0.2 ps (earliest rule)", dict(count="any", chain_hold=None)),
                ("Coordinated at 4 or more Zr", dict(min_zr=4)), ("Filter 0.5 ps", dict(hold=0.5)),
                ("Filter 0.1 ps", dict(hold=0.1)), ("Carbon cutoff 2.08 A", dict(c_k=0)), ("Carbon cutoff 2.54 A", dict(c_k=2)),
                ("Zr cutoff 3.00 A", dict(zr_k=0)), ("Zr cutoff 3.66 A", dict(zr_k=2))]
    sens = []
    for label, rule in variants:
        res = base if not rule else analyse(**rule)
        a = res["atoms"]
        n, tm = pooled(res["per_cell"])
        r = conversion_rates(a)
        loss_ratio = (n[CHL, CHU] / tm[CHL]) / (n[LAT, UND] / tm[LAT])
        bond_ratio = (n[UND, CHU] / tm[UND]) / (n[LAT, CHL] / tm[LAT])
        fin = a.Final.value_counts()
        b = a[a.EverBonded]
        rt = b.RouteToChainUnder.value_counts()
        sens.append(dict(Rule=label, Converted=int(a.Converted.sum()), **{f"Rate_{t}": v for t, v in zip(RP.TEMPERATURES, r)}, Ea_eV=ea_of(r)[0],
                         ChainInLattice=int(fin.get("ChainInLattice", 0)), ChainUnder=int(fin.get("ChainUnder", 0)), Gas=int(fin.get("Gas", 0)),
                         Undercoordinated=int(fin.get("Undercoordinated", 0)), ChainSwitches=int(n[CHL, CHU] + n[CHU, CHL]),
                         BondedFirst=int(rt.get("ChainInLattice", 0)), CoordinationFirst=int(rt.get("Undercoordinated", 0)),
                         BothAtOnce=int(rt.get("Lattice", 0)), StillCoordinated=int(rt.get("", 0)), OddGas=res["odd_gas"],
                         LossRatio=loss_ratio, BondRatio=bond_ratio))
    sens = pd.DataFrame(sens)
    sens.to_csv(OUT / "Sensitivity.csv", index=False)
    lines += ["", "## Sensitivity to the definitions", "",
              "| Rule | Converted | Conversion rate, 873 to 1273 K (10^-3 ps^-1) | Ea (eV) | Chain in lattice / under at 140 ps | Gas | Chain switches | "
              "Bonded first / coordination first / at once / still coordinated | Coupling: losing Zr / forming bond |", "|---|---|---|---|---|---|---|---|---|"]
    for _, s in sens.iterrows():
        lines.append(f"| {s.Rule} | {s.Converted} | " + ", ".join(f"{1e3 * s[f'Rate_{t}']:.2f}" for t in RP.TEMPERATURES) + f" | {s.Ea_eV:.3f} | "
                     f"{s.ChainInLattice} / {s.ChainUnder} | {s.Gas} | {s.ChainSwitches} | {s.BondedFirst} / {s.CoordinationFirst} / {s.BothAtOnce} / {s.StillCoordinated} | {s.LossRatio:.1f} / {s.BondRatio:.1f} |")
    lines += ["", "## Limits", "",
              "- Coordination is counted over a carbon's original Zr neighbours, so a carbon that leaves its site is not counted as coordinated again at another site.",
              "- Forming a carbon bond needs a partner, so the bonding rates are effective first-order rates.",
              "- Intervals come from resampling four cells per temperature. Edges with few events per temperature are only meaningful pooled.", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    atoms.to_csv(OUT / "Atoms.csv", index=False)
    (OUT / "Run.json").write_text(json.dumps(dict(Seed=RP.SEED, Bootstrap=N_BOOT, MinZrCoordinated=5, ZrCounted="original neighbours", Filter_ps=0.2, BondedCarbonCoordinationFilter_ps=0.5, Clock="onset"), indent=2) + "\n", encoding="utf-8")

    # ======================= plots ==========================================
    # S1: the square
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    pos = {LAT: (0.17, 0.74), UND: (0.55, 0.74), CHL: (0.17, 0.22), CHU: (0.55, 0.22), GAS: (0.885, 0.74)}
    share = time_all / time_all[:4].sum()
    at_end = atoms.Final.value_counts()
    for k, (x, y) in pos.items():
        ax.add_patch(FancyBboxPatch((x - 0.105, y - 0.085), 0.21, 0.17, boxstyle="round,pad=0,rounding_size=0.02", fc="white", ec=COLOURS[k], lw=1.6))
        ax.text(x, y + 0.04, SHORT[k], ha="center", va="center", fontsize=9.5)
        extra = f"{int(at_end.get(NAMES[k], 0))} at 140 ps" + ("" if k == GAS else f"\n{100 * share[k]:.1f}% of carbon time")
        ax.text(x, y - 0.035, extra, ha="center", va="center", fontsize=7.5, color="0.3", linespacing=1.3)
    big = counts_all.max()

    def arrow(i, j, off=(0, 0), text=(0, 0)):
        n = int(counts_all[i, j])
        (x0, y0), (x1, y1) = pos[i], pos[j]
        d = np.array([x1 - x0, y1 - y0]) / np.hypot(x1 - x0, y1 - y0)
        pad = 0.115 if abs(d[0]) > abs(d[1]) else 0.095
        a = (x0 + d[0] * pad + off[0], y0 + d[1] * pad + off[1])
        b = (x1 - d[0] * pad + off[0], y1 - d[1] * pad + off[1])
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7 + 5 * np.sqrt(n / big), lw=0.5 + 4.5 * np.sqrt(n / big),
                                     color="0.3" if n >= 20 else "0.65", shrinkA=0, shrinkB=0))
        ax.text((a[0] + b[0]) / 2 + text[0], (a[1] + b[1]) / 2 + text[1], str(n), ha="center", va="center", fontsize=8.5)

    arrow(LAT, UND, (0, 0.03), (0, 0.035)); arrow(UND, LAT, (0, -0.03), (0, -0.035))
    arrow(CHL, CHU, (0, 0.03), (0, 0.035)); arrow(CHU, CHL, (0, -0.03), (0, -0.035))
    arrow(LAT, CHL, (-0.03, 0), (-0.04, 0)); arrow(CHL, LAT, (0.03, 0), (0.04, 0))
    arrow(UND, CHU, (0.03, 0), (0.04, 0)); arrow(CHU, UND, (-0.03, 0), (-0.04, 0))
    arrow(UND, GAS, (0, 0), (0, 0.035))
    ax.text(0.36, 0.985, "loses or regains Zr coordination", ha="center", va="top", fontsize=8, color="#7570b3")
    ax.text(0.005, 0.48, "forms or breaks a carbon bond", ha="left", va="center", fontsize=8, color="#1b9e77", rotation=90)
    diag = ", ".join(f"{SHORT[i]} to {SHORT[j]}: {int(counts_all[i, j])}" for i, j in DIAGONALS if counts_all[i, j])
    ax.text(0.5, 0.02, ("Diagonal moves, not drawn.  " + diag) if diag else "No move changes both properties at once.",
            ha="center", va="bottom", fontsize=7.5, color="0.3")
    fig.savefig(OUT / "S1_Square.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # S2: edge rates against temperature
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(wspace=0.25))
    groups = [("Zr coordination", [(LAT, UND, "s", "#7570b3", "#7570b3"), (UND, LAT, "s", "#7570b3", "white"),
                                   (CHL, CHU, "^", "#1b9e77", "#1b9e77"), (CHU, CHL, "^", "#1b9e77", "white")]),
              ("Carbon bond and gas", [(LAT, CHL, "s", "0.3", "0.3"), (CHL, LAT, "s", "0.3", "white"), (UND, CHU, "^", "#1b9e77", "#1b9e77"),
                                       (CHU, UND, "^", "#1b9e77", "white"), (UND, GAS, "o", "#d95f02", "#d95f02")])]
    for ax, (title, edges) in zip(axes, groups):
        for (i, j, marker, colour, face), dx in zip(edges, np.linspace(-0.012, 0.012, len(edges))):
            r = rates[(rates.From == NAMES[i]) & (rates.To == NAMES[j])].sort_values("Temperature")
            r = r[r.Events > 0]
            ax.errorbar(1000 / r.Temperature + dx, r.Rate_per_ps, yerr=[r.Rate_per_ps - r.Low, r.High - r.Rate_per_ps], fmt=marker, ms=5,
                        color=colour, mfc=face, capsize=2, lw=1, label=f"{SHORT[i]} to {SHORT[j]}")
        ax.set_yscale("log")
        ax.set_xlabel("1000/T (1/K)")
        ax.set_ylabel("Rate (1/ps per carbon in the origin state)")
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    fig.savefig(OUT / "S2_EdgeRates.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # S3: coupling and routes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(wspace=0.32, width_ratios=[1.15, 1, 1]))
    ax = axes[0]
    y = np.arange(len(coupling))[::-1]
    ax.errorbar(coupling.Ratio, y, xerr=[coupling.Ratio - coupling.Low, coupling.High - coupling.Ratio], fmt="o", color="0.15", capsize=3)
    ax.axvline(1, color="0.6", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s}\n({m})" for s, m in zip(coupling.Step, coupling.Meaning)], fontsize=8)
    ax.set_xlabel("Rate ratio (1 = the other property makes no difference)")
    ax.set_title("Does one property change the rate of the other?", fontsize=9.5, loc="left")
    ax = axes[1]
    x, bottom = np.arange(4), np.zeros(4)
    for key, label, colour in zip(["Undercoordinated", "ChainInLattice", "ChainUnder"], ["Lost Zr coordination", "Bonded in place", "Both at once"],
                                  ["#7570b3", "#66c2a5", "#1b9e77"]):
        vals = np.array([float(route.loc[t].get(key, 0)) for t in RP.TEMPERATURES])
        if vals.sum():
            ax.bar(x, vals, 0.62, bottom=bottom, color=colour, label=label)
            bottom += vals
    ax.set_ylim(0, 1.3 * bottom.max())
    ax.set_xticks(x)
    ax.set_xticklabels(RP.TEMPERATURES)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Converted carbons")
    ax.set_title("How carbon first leaves the lattice state", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[2]
    bottom = np.zeros(4)
    for key, label, colour in zip(order, labels, ["#7570b3", "#66c2a5", "#1b9e77", "0.75"]):
        vals = np.array([float(route_cu.loc[t].get(key, 0)) for t in RP.TEMPERATURES])
        if vals.sum():
            ax.bar(x, vals, 0.62, bottom=bottom, color=colour, label=label)
            bottom += vals
    ax.set_ylim(0, 1.35 * bottom.max())
    ax.set_xticks(x)
    ax.set_xticklabels(RP.TEMPERATURES)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Carbons that bonded to carbon")
    ax.set_title("Chain carbon: which property changed first", fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "S3_CouplingAndRoutes.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # S4: model test
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.9), sharey=True, gridspec_kw=dict(wspace=0.06))
    for ax, temperature in zip(axes, RP.TEMPERATURES):
        for k in range(1, 5):
            ax.plot(grid, measured[temperature][:, k], color=COLOURS[k], lw=1.4)
            ax.plot(grid, modelled[temperature][:, k], color=COLOURS[k], lw=1.2, ls="--")
        ax.set_title(f"{temperature} K", fontsize=9.5)
        ax.set_xlabel("Time (ps)")
        ax.set_xlim(0, RP.HORIZON_PS)
    axes[0].set_ylabel("Fraction of all carbon")
    handles = [plt.Line2D([], [], color=COLOURS[k], lw=1.6, label=SHORT[k]) for k in range(1, 5)]
    handles += [plt.Line2D([], [], color="0.2", lw=1.4, label="measured"), plt.Line2D([], [], color="0.2", lw=1.2, ls="--", label="constant-rate model")]
    fig.legend(handles=handles, ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.07), fontsize=8.5)
    fig.savefig(OUT / "S4_ModelTest.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    # S5: conversion rate
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(1000 * INV_T, 1e3 * conv, "s", color="0.15", ms=6, label=f"Conversion ({ea:.2f} eV)")
    slope, intercept = np.polyfit(INV_T, np.log(conv), 1)
    xs = np.linspace(INV_T.min() * 0.98, INV_T.max() * 1.02, 50)
    ax.plot(1000 * xs, 1e3 * np.exp(intercept + slope * xs), color="0.5", lw=1)
    for key, label, colour, marker in (("Undercoordinated", "by losing Zr coordination", "#7570b3", "o"), ("ChainInLattice", "by bonding in place", "#1b9e77", "^")):
        part = np.array([(atoms[(atoms.Temperature == t)].ConvertTo == key).sum() / atoms[atoms.Temperature == t].Convert_ps.sum() for t in RP.TEMPERATURES])
        ax.plot(1000 * INV_T, 1e3 * part, marker, color=colour, mfc="white", ms=6, label=f"{label} ({ea_of(part)[0]:.2f} eV)")
    ax.set_yscale("log")
    ax.set_xlabel("1000/T (1/K)")
    ax.set_ylabel("Rate (10^-3 /ps per lattice carbon)")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT / "S5_Conversion.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    print("\n".join(lines))
    print("Written:", OUT)


if __name__ == "__main__":
    main()
