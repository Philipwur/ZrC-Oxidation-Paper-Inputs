"""Convergence check: k-point sampling and plane-wave cutoff.

Baseline   static run with the MD settings (special k-point 0.25 0.25 0.25, 400 eV), from the
           spin check: ../spin_polarisation/Results/results.json, entry "nsp".
Variants   Results/results.json here (collect.py on the cluster): k221, k441, e520.

What is compared, per frame and variant
  forces            difference per atom against the baseline, by atom group, and against the
                    typical force on an atom
  energy            free energy per atom (a constant shift does not matter for the dynamics)
  energy changes    along the eight frames of the adsorption event, which have the same
                    composition: F(frame) - F(first frame), variant against baseline. The error
                    of these changes is what enters a Boltzmann factor, so it is compared with
                    the thermal energy.

Writes Results/Tables.md, Results/Frames.csv, Results/EventEnergies.csv, Results/Fig_Convergence.pdf/.png
Run:  python AnalyseConvergence.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SPIN = HERE.parent / "spin_polarisation"
sys.path.insert(0, str(SPIN))
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import SpinData as SD          # noqa: E402
import AnalyseSpinTest as A    # noqa: E402

OUT = HERE / "Results"
KB = 8.617333e-5
LABEL = {"k221": "1×2×2 k-mesh", "k441": "1×4×4 k-mesh", "e520": "520 eV cutoff"}
COLOUR = {"k221": "#386cb0", "k441": "#7570b3", "e520": "#d95f02"}
GROUPS = ["Zr", "C", "O in solid", "O2"]


def groups_of(d):
    spec = d["species"]
    in_unit = np.zeros(len(spec), bool)
    for u in SD.o2_units(d):
        in_unit[list(u["atoms"])] = True
    g = np.where(spec == "Zr", "Zr", np.where(spec == "C", "C", np.where(in_unit, "O2", "O in solid")))
    return g


def compare(fa, fb, g):
    """r.m.s. and largest length of the force difference, by atom group and overall."""
    diff = np.linalg.norm(np.array(fb) - np.array(fa), axis=1)
    row = {"rms_all": float(np.sqrt((diff ** 2).mean())), "max_all": float(diff.max())}
    for name in GROUPS:
        m = g == name
        row[f"rms_{name}"] = float(np.sqrt((diff[m] ** 2).mean())) if m.any() else np.nan
        row[f"max_{name}"] = float(diff[m].max()) if m.any() else np.nan
    return row, diff


def main():
    base = SD.load()
    new = json.loads((OUT / "results.json").read_text())
    rows, alld = [], {}
    for s, modes in new.items():
        d = base[s]
        b = d["nsp"]
        g = groups_of(d)
        n = len(g)
        fb = np.array(b["forces"])
        typical = float(np.sqrt((fb ** 2).sum(1).mean()))
        for mode, r in modes.items():
            if not r or not r["converged"]:
                if r:
                    print(f"{s} {mode}: not converged, left out")
                continue
            row, diff = compare(fb, r["forces"], g)
            alld.setdefault(mode, []).append((g, diff, np.linalg.norm(fb, axis=1)))
            rows.append(dict(structure=s, variant=mode, against="baseline", atoms=n, nkpts=r["nkpts"], encut=r["encut"],
                             F=r["free_energy"], F_base=b["free_energy"], dF=r["free_energy"] - b["free_energy"],
                             dF_per_atom_meV=1000 * (r["free_energy"] - b["free_energy"]) / n, typical_force=typical,
                             scf_steps=r["scf_steps"], elapsed_s=r["elapsed_s"], **row))
        if modes.get("k441") and modes.get("k221") and modes["k441"]["converged"] and modes["k221"]["converged"]:
            row, diff = compare(modes["k441"]["forces"], modes["k221"]["forces"], g)
            alld.setdefault("k221_vs_k441", []).append((g, diff, np.linalg.norm(fb, axis=1)))
            rows.append(dict(structure=s, variant="k221", against="k441", atoms=n, nkpts=modes["k221"]["nkpts"], encut=modes["k221"]["encut"],
                             F=modes["k221"]["free_energy"], F_base=modes["k441"]["free_energy"],
                             dF=modes["k221"]["free_energy"] - modes["k441"]["free_energy"],
                             dF_per_atom_meV=1000 * (modes["k221"]["free_energy"] - modes["k441"]["free_energy"]) / n, typical_force=typical, **row))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "Frames.csv", index=False)

    # pooled force differences by group
    pooled = {}
    for mode, items in alld.items():
        g = np.concatenate([x[0] for x in items])
        diff = np.concatenate([x[1] for x in items])
        f = np.concatenate([x[2] for x in items])
        pooled[mode] = {name: (float(np.sqrt((diff[g == name] ** 2).mean())), float(diff[g == name].max())) for name in GROUPS if (g == name).any()}
        pooled[mode]["all"] = (float(np.sqrt((diff ** 2).mean())), float(diff.max()))
        pooled[mode]["typical"] = float(np.sqrt((f ** 2).mean()))
        pooled[mode]["frames"] = len(items)

    pd.DataFrame([dict(variant=mode, group=k, rms=v[0], largest=v[1], frames=p["frames"], typical=p["typical"])
                  for mode, p in pooled.items() for k, v in p.items() if k in GROUPS + ["all"]]).to_csv(OUT / "ForcesByGroup.csv", index=False)

    # energy changes along the adsorption event (same composition)
    ev = sorted([s for s in new if s.startswith("E2_")], key=lambda s: int(s.rsplit("_f", 1)[1]))
    E = pd.DataFrame({"frame": [int(s.rsplit("_f", 1)[1]) for s in ev], "baseline": [base[s]["nsp"]["free_energy"] for s in ev]})
    for mode in ("k221", "e520", "k441"):
        E[mode] = [new[s][mode]["free_energy"] if new[s].get(mode) and new[s][mode]["converged"] else np.nan for s in ev]
    R = E.copy()
    for c in ("baseline", "k221", "e520", "k441"):
        R[c] = E[c] - E[c].iloc[0]
    R.to_csv(OUT / "EventEnergies.csv", index=False)
    temp = 1273.0
    kT = KB * temp

    # ---- tables
    L = ["# Convergence check: numbers", "",
         "Written by AnalyseConvergence.py. Baseline: static run with the MD settings (one special k-point, 400 eV). "
         "Each variant changes one setting. F is the free energy at the Fermi smearing of the MD. Forces in eV per A.", "",
         "## Force differences against the baseline, all frames pooled", "",
         "| Variant | Frames | Zr r.m.s. | C r.m.s. | O in solid r.m.s. | O2 r.m.s. | All r.m.s. | Largest | Typical force |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"k221": "1x2x2 mesh against baseline", "e520": "520 eV against baseline", "k441": "1x4x4 mesh against baseline",
             "k221_vs_k441": "1x2x2 against 1x4x4 mesh"}
    for mode in ("k221", "k441", "k221_vs_k441", "e520"):
        if mode not in pooled:
            continue
        p = pooled[mode]
        cell = lambda k: f"{p[k][0]:.3f}" if k in p else "n/a"          # noqa: E731
        L.append(f"| {names[mode]} | {p['frames']} | {cell('Zr')} | {cell('C')} | {cell('O in solid')} | {cell('O2')} | {p['all'][0]:.3f} | {p['all'][1]:.2f} | {p['typical']:.2f} |")
    L += ["", "## Per frame", "",
          "| Structure | Variant | Against | k-points | dF (eV) | dF per atom (meV) | Force r.m.s. | Force largest | SCF steps | Time (s) |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in T.itertuples():
        steps = "" if pd.isna(getattr(r, "scf_steps", np.nan)) else f"{r.scf_steps:.0f}"
        secs = "" if pd.isna(getattr(r, "elapsed_s", np.nan)) else f"{r.elapsed_s:.0f}"
        L.append(f"| {r.structure} | {r.variant} | {r.against} | {r.nkpts:.0f} | {r.dF:+.3f} | {r.dF_per_atom_meV:+.1f} | {r.rms_all:.3f} | {r.max_all:.2f} | {steps} | {secs} |")
    L += ["", "## Energy changes along the adsorption event (131 atoms in every frame)", "",
          "F(frame) minus F(frame 1), in eV. Error is the variant minus the baseline.", "",
          "| Frame | Baseline | 1x2x2 | error | 520 eV | error | 1x4x4 | error |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in R.itertuples():
        def pair(v):
            return ("", "") if np.isnan(v) else (f"{v:+.3f}", f"{v - r.baseline:+.3f}")
        a, b, c = pair(r.k221), pair(r.e520), pair(r.k441)
        L.append(f"| {r.frame} | {r.baseline:+.3f} | {a[0]} | {a[1]} | {b[0]} | {b[1]} | {c[0]} | {c[1]} |")
    L += [""]
    for mode in ("k221", "e520"):
        err = (R[mode] - R.baseline).iloc[1:].dropna()
        if len(err):
            L.append(f"- {names[mode]}: error of the energy changes mean absolute {err.abs().mean():.3f} eV, largest {err.abs().max():.3f} eV, "
                     f"over a span of {R.baseline.max() - R.baseline.min():.2f} eV. Thermal energy at {temp:.0f} K: {kT:.3f} eV.")
    shift = T[T.against == "baseline"].groupby("variant").dF_per_atom_meV.agg(["mean", "std", "min", "max"])
    L += ["", "## Shift of the free energy per atom (meV), variant minus baseline", "", "| Variant | Mean | S.d. between frames | Smallest | Largest |", "|---|---:|---:|---:|---:|"]
    for mode, r in shift.iterrows():
        L.append(f"| {mode} | {r['mean']:+.1f} | {r['std']:.1f} | {r['min']:+.1f} | {r['max']:+.1f} |")
    (OUT / "Tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))

    # ---- figure
    A.style()
    W = A.TEXT_WIDTH_MM / 25.4
    fig = plt.figure(figsize=(W, W * 0.52))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.1], height_ratios=[1.45, 1], hspace=0.10, wspace=0.38,
                          left=0.15, right=0.985, top=0.845, bottom=0.18)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 1], sharex=b)
    x = np.arange(len(GROUPS))
    shown = [m for m in ("k221", "k441", "e520") if m in pooled]
    width = 0.27
    for i, mode in enumerate(shown):
        p = pooled[mode]
        off = (i - (len(shown) - 1) / 2) * width
        a.bar(x + off, [p[k][0] if k in p else np.nan for k in GROUPS], width=width * 0.92, color=COLOUR[mode], edgecolor="0.2", lw=0.6, zorder=2, label=LABEL[mode])
        a.plot(x + off, [p[k][1] if k in p else np.nan for k in GROUPS], "v", ms=3.8, mfc="white", mec="0.2", mew=0.8, zorder=3)
    typical = pooled[shown[0]]["typical"]
    a.axhline(typical, color="0.55", lw=0.9, ls="--", zorder=1)
    a.text(len(GROUPS) - 0.55, typical * 0.86, "typical force", ha="right", va="top", fontsize=8, color="0.35")
    a.set_yscale("log")
    a.set_ylim(0.005, 6)
    a.set_xticks(x)
    a.set_xticklabels(["Zr", "C", "O in\nsolid", "O$_2$"])
    a.set_ylabel("Force difference from the\nsimulation settings (eV Å$^{-1}$)")
    # one legend for both panels: the colours mean the same in (a) and in the lower part of (b)
    from matplotlib.patches import Patch
    handles = [Line2D([], [], ls="none")] + [Patch(facecolor=COLOUR[m], edgecolor="0.2", lw=0.6) for m in shown]         + [Line2D([], [], marker="v", ls="none", mfc="white", mec="0.2", ms=4.5)]
    labels = ["One setting changed to"] + [LABEL[m] for m in shown] + ["largest"]
    leg = fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, fontsize=8,
                     handletextpad=0.4, columnspacing=1.1, handlelength=1.2, borderaxespad=0.2)
    leg.legend_handles[0].set_visible(False)

    b.plot(R.frame, R.baseline, "-o", color="0.25", ms=4.5, mfc="white", mew=1.0, lw=1.0, zorder=2, label="Simulation settings")
    b.legend(frameon=False, loc="lower left", handletextpad=0.4, handlelength=1.6, borderaxespad=0.4, fontsize=8)
    b.set_ylabel("Energy (eV)")
    b.tick_params(labelbottom=False)
    c.axhspan(-kT, kT, color="0.88", zorder=0)
    c.axhline(0, color="0.55", lw=0.8, zorder=1)
    for mode, marker in (("k221", "s"), ("k441", "D"), ("e520", "^")):
        ok = R[mode].notna() & (R.frame > 1)
        if ok.any():
            c.plot(R.frame[ok], (R[mode] - R.baseline)[ok], marker, color=COLOUR[mode], ms=4.2, mec="white", mew=0.4, ls="none", label=LABEL[mode], zorder=3)
    c.set_ylim(-0.16, 0.16)
    c.set_yticks([-0.1, 0, 0.1])
    c.set_ylabel("Change (eV)")
    # frames labelled as in the spin figure of the main text (same event)
    c.set_xlabel("Distance of the molecule to Zr (Å)")
    c.set_xticks(R.frame)
    c.set_xticklabels([f"{float(base[s]['index']['nearest_solid_atom_A']):.1f}" for s in ev])
    c.text(1.0, kT * 0.52, "thermal energy at 1273 K", fontsize=7.5, color="0.35", va="center", ha="left")
    for ax, t in ((a, "(a)"), (b, "(b)")):
        A.letter(ax, t)
    fig.align_ylabels([b, c])
    fig.savefig(OUT / "Fig_Convergence.png", dpi=600)
    fig.savefig(OUT / "Fig_Convergence.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
