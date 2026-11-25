"""Figure of the carbon states, their transitions and the commitment rates.

Each carbon has two yes/no properties, and no property overrides the other:
  Zr coordination   five or more of its ORIGINAL Zr neighbours within 3.33 A
  carbon bonding    a carbon within 2.31 A
giving Lattice, Undercoordinated, Chain in lattice, Chain undercoordinated,
plus Gas. States come from SquareModel.build (0.2 ps persistence for every
change of state).

Two events per carbon:
  liberation   first loss of Zr coordination, bonded or not
  commitment   first carbon bond or release as gas

Panels: (a) the five states with pooled transition counts and atoms per
state at 140 ps, (b) commitment rates with their Arrhenius fit and the gas
and chain channels, (c) cumulative committed fractions at 1273 K.

Reads the carbon ledger (core/IndependentTrajectoryResults/CarbonOrigins.csv)
and, for the event times, ForwardEventResults/Subjects_BothDefinitions.csv
written by ForwardEventAnalyses.py.

Run:  python CarbonStatesFigure.py                    the figure of the paper
      python CarbonStatesFigure.py --with-liberation  also plots the liberation rate in (b)
About a minute, the states are rebuilt from the trajectories.

Output: CarbonStatesFigureResults/Fig_CarbonStates.png/.pdf and the numbers
behind it (CarbonStates_Atoms.csv, CarbonStates_Rates.csv, CarbonStates_Numbers.md).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP  # noqa: E402
import SquareModel as SQ  # noqa: E402
from SurvivalAnalysis import FitExponentialRate  # noqa: E402

OUT = RP.results_dir(__file__)
C_GAS, C_CHAIN, C_UND, C_TOT, C_LAT = "#d95f02", "#1b9e77", "#7570b3", "0.15", "0.45"
C_CHAIN_LAT = "#66c2a5"   # lighter green for bonded carbon still in its site
TEXT_WIDTH_MM = 131.2     # text width of the journal, so the figure prints unscaled
KB_EV = 8.617333e-5
N_BOOT = 10000

# Guardrails: counts this figure must reproduce.
GAS_EVENTS = {873: 28, 973: 44, 1073: 36, 1273: 36}      # gas as first commitment
CHAIN_EVENTS = {873: 39, 973: 35, 1073: 52, 1273: 73}    # first carbon bond
AT_140_PS = {"Lattice": 417, "Undercoordinated": 11, "ChainInLattice": 57, "ChainUnder": 126, "Gas": 152,
             "ArtifactCensored": 5}


def build_tables():
    """Per-carbon liberation and commitment records and pooled transition
    counts, straight from the square-model state builder."""
    ledger = pd.read_csv(SQ.LEDGER)
    counts, rows = np.zeros((5, 5)), []
    for temperature, cell in SQ.CELLS:
        name = f"{temperature}_{cell}"
        artifacts = set(ledger[(ledger.Trajectory == name) & (ledger.Kind == "ArtifactRemoval")].AtomID)
        times, states, _ = SQ.build(name, artifacts)
        c, _, records, episodes = SQ.tally(times, states)
        counts += c
        final = {r["AtomID"]: r["Final"] for r in records}
        by_atom = {}
        for atom, state, t0, t1, _ in episodes:
            by_atom.setdefault(atom, []).append((state, t0, t1))
        for atom, eps in by_atom.items():
            stop = eps[-1][2]
            lib = next((t0 for s, t0, _ in eps if s in (SQ.UND, SQ.CHU)), None)
            com = next(((s, t0) for s, t0, _ in eps if s in (SQ.CHL, SQ.CHU, SQ.GAS)), None)
            rows.append(dict(Temperature=temperature, Trajectory=name, AtomID=atom, Observed_ps=stop,
                             Lib_ps=stop if lib is None else lib, Liberated=lib is not None,
                             Commit_ps=com[1] if com else stop, CommitEvent=com is not None,
                             Channel=("Gas" if com[0] == SQ.GAS else "Chain") if com else "",
                             BondedWhenLiberated=bool(lib is not None and com is not None and com[0] != SQ.GAS and com[1] < lib),
                             Final=final[atom]))
    atoms = pd.DataFrame(rows).sort_values(["Temperature", "Trajectory", "AtomID"]).reset_index(drop=True)
    got = atoms.Final.value_counts().to_dict()
    assert got == AT_140_PS, got
    for temperature, grp in atoms.groupby("Temperature"):
        assert (grp.Channel == "Gas").sum() == GAS_EVENTS[temperature], (temperature, (grp.Channel == "Gas").sum())
        assert (grp.Channel == "Chain").sum() == CHAIN_EVENTS[temperature], (temperature, (grp.Channel == "Chain").sum())
    return atoms, counts


def use_paper_clock(atoms):
    """Commitment times on the paper's clocks (gas at the logged desorption
    time, chain entries at confirmation, 0.2 ps after onset), taken from
    ForwardEventAnalyses.py, so that the plotted rates are the tabulated ones.
    The state model stamps events at onset, which shifts chain times by 0.2 ps."""
    path = Path(__file__).resolve().parent / "ForwardEventResults" / "Subjects_BothDefinitions.csv"
    if not path.exists():
        return atoms
    paper = pd.read_csv(path)[["Trajectory", "AtomID", "ForwardDuration", "ForwardCause"]]
    merged = atoms.merge(paper, on=["Trajectory", "AtomID"], how="left", validate="one_to_one")
    assert merged.ForwardCause.notna().all()
    assert ((merged.ForwardCause > 0) == merged.CommitEvent).all()
    assert ((merged.ForwardCause == 1) == (merged.Channel == "Gas")).all()
    assert (merged.ForwardDuration - merged.Commit_ps).abs().max() < 0.25
    merged["Commit_ps"] = merged.ForwardDuration
    return merged.drop(columns=["ForwardDuration", "ForwardCause"])


def rate_table(atoms):
    rows = []
    for temperature, a in atoms.groupby("Temperature"):
        series = {"Liberation": (a.Lib_ps, a.Liberated), "Total": (a.Commit_ps, a.CommitEvent),
                  "Gas": (a.Commit_ps, a.Channel == "Gas"), "Chain": (a.Commit_ps, a.Channel == "Chain")}
        for label, (durations, events) in series.items():
            fit = FitExponentialRate(durations.to_numpy(float), events.to_numpy(float))
            rows.append(dict(Temperature=temperature, Channel=label, Events=fit["Events"], Exposure_ps=fit["Exposure_ps"],
                             Rate_per_ps=fit["Lambda"], CI_Lower=fit["CI_Lower"], CI_Upper=fit["CI_Upper"]))
    return pd.DataFrame(rows)


def arrhenius(rates):
    inv_t = 1.0 / np.array(RP.TEMPERATURES, float)
    slope, intercept = np.polyfit(inv_t, np.log(rates), 1)
    resid = np.log(rates) - (intercept + slope * inv_t)
    r2 = 1 - np.sum(resid ** 2) / np.sum((np.log(rates) - np.log(rates).mean()) ** 2)
    return dict(Ea_eV=float(-slope * KB_EV), R2=float(r2), Slope=float(slope), Intercept=float(intercept))


def bootstrap(atoms):
    """Two-level bootstrap (cells, then carbons within each drawn cell), the
    same draws for both events so that their difference is paired."""
    rng = np.random.default_rng(RP.SEED)
    inv_t = 1.0 / np.array(RP.TEMPERATURES, float)
    x = inv_t - inv_t.mean()
    logs = {k: np.empty((N_BOOT, 4)) for k in ("Liberation", "Commitment")}
    for j, temperature in enumerate(RP.TEMPERATURES):
        sub = atoms[atoms.Temperature == temperature]
        cell = rng.integers(0, 4, size=(N_BOOT, 4))
        atom = rng.integers(0, 48, size=(N_BOOT, 4, 48))
        pick = (cell[:, :, None], atom)
        for key, (t_col, e_col) in {"Liberation": ("Lib_ps", "Liberated"), "Commitment": ("Commit_ps", "CommitEvent")}.items():
            t, e = sub[t_col].to_numpy(float).reshape(4, 48), sub[e_col].to_numpy(float).reshape(4, 48)
            logs[key][:, j] = np.log(e[pick].sum((1, 2)) / t[pick].sum((1, 2)))
    ea = {k: -(v - v.mean(1, keepdims=True)) @ x / np.sum(x ** 2) * KB_EV for k, v in logs.items()}
    ea["Difference"] = ea["Liberation"] - ea["Commitment"]
    return {k: dict(SD=float(v.std(ddof=1)), CI95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]) for k, v in ea.items()}


def count_key(ax, x, y):
    """Small legend for the state diagram: what the numbers on arrows and in
    boxes are. (x, y) is its left edge and vertical centre, in millimetres."""
    ax.add_patch(FancyArrowPatch((x, y + 2.0), (x + 5.0, y + 2.0), arrowstyle="-|>", mutation_scale=7, lw=1.3,
                                 color="0.28", shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt"))
    ax.text(x + 6.6, y + 2.0, "transitions", fontsize=8, color="0.3", ha="left", va="center")
    ax.add_patch(FancyBboxPatch((x + 0.3, y - 3.3), 4.4, 2.6, boxstyle="round,pad=0.0,rounding_size=0.6",
                                fc="white", ec="0.45", lw=0.9))
    ax.text(x + 6.6, y - 2.0, "atoms at 140 ps", fontsize=8, color="0.3", ha="left", va="center")


def panel_states(ax, counts, final):
    """Five boxes, drawn in millimetres inside a framed panel whose two axes
    are the two properties. Rows are Zr coordination (lattice above,
    non-lattice below) and columns are carbon bonding (non-chain left, chain
    right), so the starting state, Lattice, is at the top left. Vertical
    moves change the first property and horizontal moves the second. Gas is
    reached from Undercoordinated only and sits below it, under the grid.
    Box widths follow the width of the frame."""
    fig = ax.figure
    box = ax.get_position()
    width = box.width * fig.get_figwidth() * 25.4
    height = box.height * fig.get_figheight() * 25.4
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    def n(i, j):
        return int(counts[i, j])

    roomy = width >= 62.0
    tall, margin, run, edge = 12.5, (3.0 if roomy else 2.5), (9.0 if roomy else 6.5), (2.2 if roomy else 1.8)
    room = width - 2 * edge - margin - 2.0 - run                    # shared by the two columns of boxes
    w_non, w_chain = min(22.0, 0.42 * room), min(27.0, 0.58 * room)
    pad = (width - (margin + w_non + 2.0 + run + w_chain)) / 2
    x_non = pad + margin + w_non / 2
    x_chain = x_non + w_non / 2 + 2.0 + run + w_chain / 2
    w_gas = min(20.0, w_non)
    top, gas_y = height - 3.0 - 6.0 - tall / 2, 3.0 + tall / 2
    free = (top - tall / 2) - (gas_y + tall / 2) - tall            # shared between the two vertical gaps
    low = gas_y + tall + 0.46 * free
    ax.set_xticks([x_non, x_chain])
    ax.set_xticklabels(["Non-chain", "Chain"])
    ax.set_yticks([low, top])
    ax.set_yticklabels(["Non-lattice", "Lattice"], rotation=90, va="center")
    ax.tick_params(length=3.5, labelsize=9)

    b_left, b_bottom = x_non - w_non / 2 - margin, low - tall / 2 - 3.2
    ax.add_patch(FancyBboxPatch((b_left, b_bottom), w_non + 2 * margin, top + tall / 2 + 6.0 - b_bottom,
                                boxstyle="round,pad=0.0,rounding_size=2.2", fc="0.94", ec="none", zorder=0))
    ax.text(x_non, top + tall / 2 + 3.1, "uncommitted", fontsize=8.5, color="0.3", va="center", ha="center", style="italic")
    last = "Chain (free)"
    nodes = [("Lattice", x_non, top, w_non, C_LAT, final["Lattice"]),
             ("Chain (in lattice)", x_chain, top, w_chain, C_CHAIN_LAT, final["ChainInLattice"]),
             ("Free carbon", x_non, low, w_non, C_UND, final["Undercoordinated"]),
             (last, x_chain, low, w_chain, C_CHAIN, final["ChainUnder"]),
             ("Gas", x_non, gas_y, w_gas, C_GAS, final["Gas"])]
    for name, x, y, w, colour, count in nodes:
        two = "\n" in name
        ax.add_patch(FancyBboxPatch((x - w / 2, y - tall / 2), w, tall, boxstyle="round,pad=0.0,rounding_size=1.5",
                                    fc="white", ec=colour, lw=1.3, zorder=3))
        ax.text(x, y + (1.9 if two else 2.2), name, ha="center", va="center", fontsize=9, zorder=4, linespacing=1.0)
        ax.text(x, y - (4.0 if two else 2.8), f"{int(count)}", ha="center", va="center", fontsize=8.5, color="0.35", zorder=4)

    def arrow(count, start, end, label_at, ha="center", va="center"):
        minor = count < 20
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=6.5 if minor else 9,
                                     lw=0.6 if minor else 0.9 + 2.6 * np.sqrt(count / 472.0),
                                     color="0.68" if minor else "0.28", shrinkA=0, shrinkB=0, zorder=2,
                                     joinstyle="miter", capstyle="butt"))
        ax.text(*label_at, f"{count}", fontsize=8 if minor else 9, color="0.45" if minor else "0.1", ha=ha, va=va, zorder=5)

    # horizontal moves: forming and breaking a carbon bond
    h0, h1 = x_non + w_non / 2 + 1.0, x_chain - w_chain / 2 - 1.0
    hm = (h0 + h1) / 2 + 0.8
    for y, forward, back in ((top, n(SQ.LAT, SQ.CHL), n(SQ.CHL, SQ.LAT)), (low, n(SQ.UND, SQ.CHU), n(SQ.CHU, SQ.UND))):
        arrow(forward, (h0, y + 2.2), (h1, y + 2.2), (hm, y + 3.7), va="bottom")
        arrow(back, (h1, y - 2.2), (h0, y - 2.2), (hm, y - 3.5), va="top")
    # vertical moves: losing and regaining Zr coordination
    v0, v1 = top - tall / 2 - 1.0, low + tall / 2 + 1.0
    vm = (v0 + v1) / 2
    for x, down, up in ((x_non, n(SQ.LAT, SQ.UND), n(SQ.UND, SQ.LAT)), (x_chain, n(SQ.CHL, SQ.CHU), n(SQ.CHU, SQ.CHL))):
        arrow(down, (x + 2.6, v0), (x + 2.6, v1), (x + 4.5, vm), ha="left")
        arrow(up, (x - 2.6, v1), (x - 2.6, v0), (x - 4.3, vm), ha="right")
    # release as gas, from undercoordinated carbon only
    g0, g1 = low - tall / 2 - 1.0, gas_y + tall / 2 + 1.0
    arrow(n(SQ.UND, SQ.GAS), (x_non, g0), (x_non, g1), (x_non + 2.6, (b_bottom + g1) / 2), ha="left")
    count_key(ax, x_chain + w_chain / 2 - 25.6, gas_y)


def panel_rates(ax, rates, fits, legend_size=8.5, floor=None, liberation=True):
    grid = np.linspace(850, 1300, 200)
    if liberation:
        series = [("Liberation", "s", C_TOT, "white", -0.015, fits["Liberation"], (0, (4, 2))),
                  ("Total", "s", C_TOT, C_TOT, 0.005, fits["Commitment"], "-"),
                  ("Gas", "o", C_GAS, "white", -0.007, None, None),
                  ("Chain", "^", C_CHAIN, "white", 0.021, None, None)]
    else:
        series = [("Total", "s", C_TOT, C_TOT, 0.0, fits["Commitment"], "-"),
                  ("Gas", "o", C_GAS, "white", -0.010, None, None),
                  ("Chain", "^", C_CHAIN, "white", 0.010, None, None)]
    for channel, marker, colour, face, offset, fit, style in series:
        if fit:
            ax.plot(1000 / grid, 1e3 * np.exp(fit["Intercept"] + fit["Slope"] / grid), color="0.5", lw=1.0, ls=style, zorder=1)
        for _, r in rates[rates.Channel == channel].iterrows():
            v = 1e3 * r.Rate_per_ps
            ax.errorbar(1000 / r.Temperature + offset, v, yerr=[[v - 1e3 * r.CI_Lower], [1e3 * r.CI_Upper - v]],
                        fmt=marker, ms=5.0, color=colour, mfc=face, capsize=2.2, lw=1.1, mew=1.1, zorder=3)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 8]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.745, 1.19)
    ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    ax.set_ylim(floor if floor else (0.30 if liberation else 0.42), 9.5)
    ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    ax.set_ylabel(r"Rate ($10^{-3}$ ps$^{-1}$)")
    handles = [Line2D([], [], marker="s", color=C_TOT, mfc="white", ms=5.0, lw=1.0, ls=(0, (4, 2)),
                      label=f"Liberation ({fits['Liberation']['Ea_eV']:.2f} eV)")] if liberation else []
    handles += [Line2D([], [], marker="s", color=C_TOT, ms=5.0, lw=1.0, label=f"Commitment ({fits['Commitment']['Ea_eV']:.2f} eV)"),
                Line2D([], [], marker="o", color=C_GAS, mfc="white", ms=5.0, lw=0, label="Gas"),
                Line2D([], [], marker="^", color=C_CHAIN, mfc="white", ms=5.0, lw=0, label="Chain")]
    ax.legend(handles=handles, frameon=False, loc="lower left", handlelength=2.3, labelspacing=0.25, fontsize=legend_size, borderaxespad=0.3)


def panel_cumulative(ax, atoms, temperature=1273):
    a = atoms[(atoms.Temperature == temperature) & atoms.CommitEvent]
    gas = np.sort(a.Commit_ps[a.Channel == "Gas"].to_numpy())
    chain = np.sort(a.Commit_ps[a.Channel == "Chain"].to_numpy())
    total = np.sort(np.concatenate([gas, chain]))
    n = int((atoms.Temperature == temperature).sum())
    assert n == 192 and len(total) == GAS_EVENTS[temperature] + CHAIN_EVENTS[temperature]
    ax.axvspan(0, 40, facecolor="0.93", edgecolor="none", zorder=0)
    ax.axvline(40, color="0.65", lw=0.7, ls=":", zorder=1)
    for times, colour, lw in ((gas, C_GAS, 1.4), (chain, C_CHAIN, 1.4), (total, C_TOT, 1.8)):
        ax.step(np.concatenate([[0], times, [RP.HORIZON_PS]]),
                np.concatenate([[0], np.arange(1, len(times) + 1), [len(times)]]) / n, where="post", color=colour, lw=lw)
    ax.set_xlim(0, RP.HORIZON_PS)
    ax.set_ylim(0, 0.75)
    ax.set_xticks([0, 40, 80, 120, 140])
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("C fraction committed")
    ax.legend(handles=[Line2D([], [], color=C_TOT, lw=1.8, label="Total"), Line2D([], [], color=C_GAS, lw=1.4, label="Gas"),
                       Line2D([], [], color=C_CHAIN, lw=1.4, label="Chain")],
              frameon=False, loc="upper left", bbox_to_anchor=(40 / RP.HORIZON_PS, 1.0), handlelength=1.3,
              borderaxespad=0.4, borderpad=0.2, labelspacing=0.25, fontsize=8.5)
    ax.text(20, 0.41, "Initial\ntransient", ha="center", va="center", fontsize=8, color="0.4")
    return dict(Gas=len(gas), Chain=len(chain), FirstChain_ps=float(chain[0]), FirstGas_ps=float(gas[0]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=float, default=54.7, help="width in mm of the state diagram")
    parser.add_argument("--with-liberation", action="store_true", help="also plot the liberation rate in panel (b)")
    args = parser.parse_args()
    atoms, counts = build_tables()
    undrawn = {f"{SQ.NAMES[i]} to {SQ.NAMES[j]}": int(counts[i, j]) for i, j in SQ.DIAGONALS if counts[i, j]}
    print("Transitions:", {f"{SQ.NAMES[i]} to {SQ.NAMES[j]}": int(counts[i, j]) for i in range(4) for j in range(5) if counts[i, j]})
    print("Not drawn:", undrawn or "none")
    atoms = use_paper_clock(atoms)
    rates = rate_table(atoms)
    fits = {"Liberation": arrhenius(rates[rates.Channel == "Liberation"].Rate_per_ps.to_numpy()),
            "Commitment": arrhenius(rates[rates.Channel == "Total"].Rate_per_ps.to_numpy())}
    boot = bootstrap(atoms)
    final = atoms.Final.value_counts()

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.labelsize": 10,
                         "legend.fontsize": 8.5, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "mathtext.fontset": "stix", "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42})
    both = args.with_liberation
    name = "Fig_CarbonStates" + ("_WithLiberation" if both else "")
    W, H = TEXT_WIDTH_MM, 108.0
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))

    def axes(left, bottom, w, h):          # millimetres from the lower left corner
        return fig.add_axes([left / W, bottom / H, w / W, h / H])
    ax_net = axes(6.5, 11.0, args.frame, 91.0)
    panel_states(ax_net, counts, final)
    left = 6.5 + args.frame + 12.5         # room for the tick labels and axis label of the plots
    ax_rates, ax_cum = axes(left, 64.0, W - 2.8 - left, 38.0), axes(left, 11.0, W - 2.8 - left, 38.0)
    panel_rates(ax_rates, rates, fits, legend_size=8 if both else 8.5, floor=0.19 if both else None, liberation=both)
    cumulative = panel_cumulative(ax_cum, atoms)
    for ax, letter in zip((ax_net, ax_rates, ax_cum), "abc"):
        ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=10, pad=5)
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)

    atoms.to_csv(OUT / "CarbonStates_Atoms.csv", index=False)
    rates.to_csv(OUT / "CarbonStates_Rates.csv", index=False)
    names = SQ.NAMES
    lines = ["# Numbers drawn in Fig_CarbonStates", "",
             "Two-property states (original Zr neighbours, 0.2 ps for every change of state). "
             "Liberation is the first loss of Zr coordination, bonded or not. Commitment is the first carbon bond or release as gas.", "",
             "## Transitions, pooled", "", "| From | To | Count |", "|---|---|---:|"]
    lines += [f"| {names[i]} | {names[j]} | {int(counts[i, j])} |" for i in range(4) for j in range(5) if counts[i, j]]
    lines += ["", "Not drawn: " + ", ".join(f"{names[i]} to {names[j]} ({int(counts[i, j])})" for i, j in SQ.DIAGONALS if counts[i, j]) + ".",
              "", "## Carbon atoms at 140 ps", ""] + [f"- {k}: {int(final[k])}" for k in AT_140_PS]
    lines += ["", "## Rates (10^-3 ps^-1) with model-based 95% intervals", "",
              "| T (K) | Liberation | Commitment | Gas | Chain | Liberated | of which bonded first |", "|---:|---|---|---|---|---:|---:|"]

    def cell(r):
        return f"{1e3 * r.Rate_per_ps:.2f} [{1e3 * r.CI_Lower:.2f}, {1e3 * r.CI_Upper:.2f}]"
    for temperature in RP.TEMPERATURES:
        c = rates[rates.Temperature == temperature].set_index("Channel")
        a = atoms[atoms.Temperature == temperature]
        lines.append(f"| {temperature} | {cell(c.loc['Liberation'])} | {cell(c.loc['Total'])} | {cell(c.loc['Gas'])} | "
                     f"{cell(c.loc['Chain'])} | {int(a.Liberated.sum())} | {int(a.BondedWhenLiberated.sum())} |")
    committed = atoms[atoms.CommitEvent]
    first_chain = committed[committed.Channel == "Chain"]
    in_place = int((~first_chain.Liberated | (first_chain.Commit_ps < first_chain.Lib_ps)).sum())
    lines += ["", f"Arrhenius, four temperatures, unweighted: liberation {fits['Liberation']['Ea_eV']:.3f} eV (R2 {fits['Liberation']['R2']:.3f}), "
              f"commitment {fits['Commitment']['Ea_eV']:.3f} eV (R2 {fits['Commitment']['R2']:.3f}).",
              f"Two-level bootstrap ({N_BOOT} resamples, seed {RP.SEED}): liberation s.d. {boot['Liberation']['SD']:.3f}, "
              f"95% [{boot['Liberation']['CI95'][0]:.3f}, {boot['Liberation']['CI95'][1]:.3f}]. Commitment s.d. {boot['Commitment']['SD']:.3f}, "
              f"95% [{boot['Commitment']['CI95'][0]:.3f}, {boot['Commitment']['CI95'][1]:.3f}]. Liberation minus commitment "
              f"95% [{boot['Difference']['CI95'][0]:.3f}, {boot['Difference']['CI95'][1]:.3f}].", "",
              f"Of {len(first_chain)} first carbon bonds, {in_place} formed while the carbon still had its Zr coordination.",
              f"Liberated carbons: {int(atoms.Liberated.sum())}. Committed carbons: {int(atoms.CommitEvent.sum())}. "
              f"Committed and never liberated: {int((atoms.CommitEvent & ~atoms.Liberated).sum())}. "
              f"Liberated and never committed: {int((atoms.Liberated & ~atoms.CommitEvent).sum())}.", "",
              f"Cumulative panel, 1273 K: gas {cumulative['Gas']}, chain {cumulative['Chain']} of 192."]
    (OUT / "CarbonStates_Numbers.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("\nFigure written:", OUT / f"{name}.png")


if __name__ == "__main__":
    main()
