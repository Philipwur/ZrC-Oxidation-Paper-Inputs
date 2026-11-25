"""Analysis of the spin-polarisation check.

Input: Results/results.json (energies, forces and moments collected from the static VASP
calculations) and Structures/ (atom groups and INDEX.csv, written by SelectSnapshots.py).

For every structure the non-spin-polarised calculation (nsp) is compared with the best
spin-polarised one: of the converged spin-polarised runs (first pass `sp`, second pass `sp2`)
the one with the lower free energy is used, because the spin-polarised SCF can stop at a
state that is not the ground state. The spin-polarised energy is therefore an upper bound and
the energy difference a lower bound.

An "O2 unit" is two O atoms that are each other's nearest O within 2.1 A. It is
  gas, isolated     further than 2.5 A from every Zr, no O atom of another unit within 2.5 A
  gas, in contact   further than 2.5 A from every Zr, an O atom of another unit within 2.5 A
  adsorbed          within 2.5 A of a Zr

Output in Results/:
  Tables.md                 every number quoted in the findings
  Structures.csv, O2Units.csv, GasPairs.csv
  Fig_SpinCheck.{pdf,png}   four panels
  Fig_SpinEvents.{pdf,png}  the two adsorption events
  Fig_SpinMethods.{pdf,png} the two-panel figure in the Methods of the paper
  Fig_SpinSupplementary.{pdf,png}  the four-panel figure in the Supplementary Information

Run:  python AnalyseSpinTest.py [results file name]
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import SpinData as SD

OUT = SD.RESULTS
TEXT_WIDTH_MM = 131.2            # sn-jnl text width, as for the figures of the paper
C_NSP, C_SP = "0.35", "#d95f02"
C_SET = {"S": "#1b9e77", "C": "#7570b3", "E1": "#d95f02", "E2": "#386cb0"}
SET_LABEL = {"S": "Snapshots", "C": "Chain snapshots", "E1": "Event, pristine surface", "E2": "Event, oxidised surface"}
AWAY, CONTACT = 2.5, 2.5
SET_NAME = {"S": "Slab", "C": "Chains", "E1": "Event A", "E2": "Event B"}


def best_sp(d):
    """The converged spin-polarised result with the lower free energy, and which pass it is."""
    cands = [(d[m]["free_energy"], m) for m in ("sp", "sp2") if d.get(m) and d[m]["converged"]]
    if not cands:
        cands = [(d[m]["free_energy"], m) for m in ("sp", "sp2") if d.get(m)]
        if not cands:
            return None, None, False
        e, m = min(cands)
        return d[m], m, False
    e, m = min(cands)
    return d[m], m, True


def set_of(name):
    return name[:2] if name.startswith("E") else name[0]


def build(data):
    rows, units, pairs = [], [], []
    for s, d in data.items():
        sp, which, ok = best_sp(d)
        if sp is None or not d.get("nsp"):
            continue
        d = dict(d, sp=sp)
        nsp = d["nsp"]
        g, spec = d["group"], d["species"]
        us = SD.o2_units(d)
        io = np.where(spec == "O")[0]
        in_unit = np.zeros(len(g), bool)
        for u in us:
            in_unit[list(u["atoms"])] = True
        m = np.array(sp["moment"]) if sp["moment"] else np.zeros(len(g))
        fn, fs = np.array(nsp["forces"]), np.array(sp["forces"])
        df = np.linalg.norm(fs - fn, axis=1)
        for u in us:
            a = list(u["atoms"])
            others = np.array([i for i in io if i not in a])
            u["other_O"] = float(SD.distances(d, a, others).min()) if len(others) else np.inf
            u["kind"] = "adsorbed" if u["d_zr"] <= AWAY else ("gas, in contact" if u["other_O"] < CONTACT else "gas, isolated")
        gas = [u for u in us if u["kind"] != "adsorbed"]
        row = dict(structure=s, set="O2" if s.startswith("O2_") else set_of(s), sp_used=which, sp_converged=ok,
                   nsp_converged=nsp["converged"], atoms=len(g),
                   dF=nsp["free_energy"] - sp["free_energy"], dE0=nsp["energy_sigma0"] - sp["energy_sigma0"],
                   dF_first_pass=nsp["free_energy"] - d["sp_first"]["free_energy"] if d.get("sp_first") else np.nan,
                   total_moment=sp["total_moment"], n_gas=len(gas), n_gas_isolated=sum(u["kind"] == "gas, isolated" for u in us),
                   n_gas_contact=sum(u["kind"] == "gas, in contact" for u in us), n_units=len(us),
                   sum_abs_moment_units=float(sum(abs(u["m_sp"]) for u in us)),
                   n_spin_units=int(sum(abs(u["m_sp"]) > 0.3 for u in us)),
                   max_moment_Zr=float(np.abs(m[spec == "Zr"]).max()) if (spec == "Zr").any() else np.nan,
                   max_moment_C=float(np.abs(m[spec == "C"]).max()) if (spec == "C").any() else np.nan,
                   max_moment_C_chain=float(np.abs(m[g == "C chain"]).max()) if (g == "C chain").any() else np.nan,
                   max_moment_O_single=float(np.abs(m[(spec == "O") & ~in_unit]).max()) if ((spec == "O") & ~in_unit).any() else np.nan,
                   F_rms=float(np.sqrt((fn ** 2).sum(1).mean())),
                   md_free_energy=float(d["index"]["md_free_energy_eV"]) if d["index"].get("md_free_energy_eV") else np.nan,
                   nsp_free_energy=nsp["free_energy"], sp_free_energy=sp["free_energy"])
        for lab, mask in (("Zr", spec == "Zr"), ("C", spec == "C"), ("O_single", (spec == "O") & ~in_unit), ("O_unit", in_unit)):
            row[f"dF_rms_{lab}"] = float(np.sqrt((df[mask] ** 2).mean())) if mask.any() else np.nan
            row[f"dF_max_{lab}"] = float(df[mask].max()) if mask.any() else np.nan
        for k in ("O_O_distance_A", "nearest_solid_atom_A", "time_from_binding_fs", "event_O_poscar_indices"):
            row[k] = d["index"].get(k, "")
        rows.append(row)
        for u in us:
            a, b = u["atoms"]
            units.append(dict(structure=s, set=row["set"], atom_a=a + 1, atom_b=b + 1, r=u["r"], d_zr=u["d_zr"], other_O=u["other_O"],
                              kind=u["kind"], q_nsp=u["q_nsp"], q_sp=u["q_sp"], moment=u["m_sp"], dF_atom_max=float(max(df[a], df[b]))))
        # force between two gas molecules, along the line joining their centres (positive pushes them apart)
        far = [u for u in us if u["d_zr"] > 4.0]
        for i in range(len(far)):
            for j in range(i + 1, len(far)):
                a, b = list(far[i]["atoms"]), list(far[j]["atoms"])
                close = float(SD.distances(d, a, b).min())
                if close > 3.6:
                    continue
                f0 = d["frac"][a[0]]

                def centre(idx):
                    x = d["frac"][idx] - f0
                    x -= np.round(x)
                    return x.mean(0) @ d["lattice"]
                v = centre(b) - centre(a)
                e = v / np.linalg.norm(v)
                pairs.append(dict(structure=s, contact_A=close, centre_distance_A=float(np.linalg.norm(v)),
                                  push_nsp=float((fn[b].sum(0) - fn[a].sum(0)) @ e / 2), push_sp=float((fs[b].sum(0) - fs[a].sum(0)) @ e / 2),
                                  moment_1=abs(far[i]["m_sp"]), moment_2=abs(far[j]["m_sp"]), sp_is_unpolarised=abs(sp["total_moment"] or 0) < 0.05))
    return pd.DataFrame(rows), pd.DataFrame(units), pd.DataFrame(pairs)


# ----------------------------------------------------------------------------- figures
def style():
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.labelsize": 10, "legend.fontsize": 8.5,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "mathtext.fontset": "stix", "axes.linewidth": 0.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "xtick.direction": "out", "ytick.direction": "out"})


def letter(ax, text):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=10, pad=4)


def figure_main(S, U):
    style()
    W = TEXT_WIDTH_MM / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.80))
    (a, b), (c, d) = axes
    iso = S[S.set == "O2"].copy()
    iso["r"] = [U[(U.structure == s)].r.iloc[0] for s in iso.structure]
    main = iso[~iso.structure.str.contains("sigma")].sort_values("r")
    full = float(U[U.structure == "O2_isolated"].moment.abs().iloc[0])
    slabs = S[S.set != "O2"]
    su = U[U.set != "O2"]

    # (a) the O-O potential of the lone molecule
    e0 = main.sp_free_energy.min()
    rr = np.linspace(main.r.min(), main.r.max(), 200)
    for col, colour, label, marker, fill in (("nsp_free_energy", C_NSP, "Without spin", "o", "white"), ("sp_free_energy", C_SP, "With spin", "s", C_SP)):
        y = main[col] - e0
        a.plot(rr, np.polyval(np.polyfit(main.r, y, 4), rr), color=colour, lw=1.1)
        a.plot(main.r, y, marker, ms=5, mfc=fill, mec=colour, mew=1.1, ls="none", label=label)
    ref = main[np.isclose(main.r, 1.23, atol=0.005)].iloc[0]
    y0n, y0s = ref.nsp_free_energy - e0, ref.sp_free_energy - e0
    a.annotate("", xy=(1.23, y0n), xytext=(1.23, y0s), arrowprops=dict(arrowstyle="<->", lw=0.8, color="0.2", shrinkA=3.5, shrinkB=3.5))
    a.text(1.246, y0s + 0.80 * (y0n - y0s), f"{y0n - y0s:.2f} eV", va="center", ha="left", fontsize=9)
    a.set_xlabel("O–O distance (Å)")
    a.set_ylabel("Free energy (eV)")
    a.legend(frameon=False, loc="upper left", handletextpad=0.3)

    # (b) moment of every O2 unit against its distance from the surface
    kinds = (("gas, isolated", dict(marker="o", s=17, facecolors=C_SP, edgecolors="0.2", linewidths=0.4), "Gas, isolated"),
             ("gas, in contact", dict(marker="o", s=17, facecolors="white", edgecolors=C_SP, linewidths=1.0), "Gas, touching another O$_2$"),
             ("adsorbed", dict(marker="s", s=15, facecolors="0.45", edgecolors="0.2", linewidths=0.4), "Adsorbed"))
    for kind, kw, lab in kinds:
        sub = su[su.kind == kind]
        b.scatter(sub.d_zr.clip(upper=15), sub.moment.abs(), zorder=3 if kind != "gas, in contact" else 2, label=lab, **kw)
    b.axhline(full, color="0.55", lw=0.9, ls="--", zorder=1)
    b.text(1.72, full + 0.05, "lone O$_2$", ha="left", va="bottom", fontsize=8.5, color="0.35")
    b.set_xscale("log")
    b.set_xlim(1.65, 16.5)
    b.set_xticks([2, 3, 5, 10, 15])
    b.set_xticklabels(["2", "3", "5", "10", "15"])
    b.minorticks_off()
    b.set_xlabel("Distance to nearest Zr (Å)")
    b.set_ylabel(r"Moment of O$_2$ unit ($\mu_\mathrm{B}$)")
    b.set_ylim(-0.09, full + 0.95)
    b.legend(frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2, fontsize=8)

    # (c) energy difference per structure against the summed moment of its O2 units
    xmax = max(slabs.sum_abs_moment_units.max(), 1.0)
    per_mu = float(iso[iso.structure == "O2_isolated"].dF.iloc[0]) / full
    c.plot([0, xmax * 1.03], [0, per_mu * xmax * 1.03], color="0.55", lw=0.9, ls="--", zorder=1)
    for key, colour in C_SET.items():
        sub = slabs[slabs.set == key]
        c.plot(sub.sum_abs_moment_units, sub.dF, "o", ms=4.5, mfc=colour, mec="white", mew=0.5, zorder=3)
    c.set_xlabel(r"Summed moment of O$_2$ units ($\mu_\mathrm{B}$)")
    c.set_ylabel(r"$\Delta F$ (eV)")
    c.set_xlim(-0.03 * xmax, 1.05 * xmax)
    c.set_ylim(-0.04 * per_mu * xmax, 1.08 * per_mu * xmax)
    c.text(1.03 * xmax, per_mu * 0.93 * xmax, "lone O$_2$", fontsize=8.5, color="0.35", ha="right", va="top")
    handles = [Line2D([], [], marker="o", ls="none", mfc=C_SET[k], mec="white", ms=5.5, label=SET_LABEL[k]) for k in C_SET]
    c.legend(handles=handles, frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2, fontsize=8)

    # (d) force differences by atom group
    labels = ["Zr", "C", "O in\nsolid", "O$_2$\nunits"]
    cols = ["Zr", "C", "O_single", "O_unit"]
    x = np.arange(len(cols))
    rms = [float(np.sqrt(np.nanmean(slabs[f"dF_rms_{k}"] ** 2))) for k in cols]
    mx = [float(np.nanmax(slabs[f"dF_max_{k}"])) for k in cols]
    d.bar(x, rms, width=0.55, color=["0.6", "0.6", "0.6", C_SP], edgecolor="0.2", lw=0.6, zorder=2)
    d.plot(x, mx, "v", ms=5, mfc="white", mec="0.2", mew=0.9, zorder=3)
    typical = float(slabs.F_rms.mean())
    d.axhline(typical, color="0.55", lw=0.9, ls="--", zorder=1)
    d.text(-0.42, typical * 1.15, "typical force on an atom", ha="left", va="bottom", fontsize=8.5, color="0.35")
    d.set_yscale("log")
    d.set_ylim(0.008, 60)
    d.set_xticks(x)
    d.set_xticklabels(labels)
    d.set_ylabel("Force difference (eV Å$^{-1}$)")
    d.legend(handles=[Line2D([], [], marker="s", ls="none", mfc="0.6", mec="0.2", ms=6, label="r.m.s."),
                      Line2D([], [], marker="v", ls="none", mfc="white", mec="0.2", ms=5, label="largest")],
             frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, ncol=2, columnspacing=1.0)
    for ax, t in zip((a, b, c, d), "abcd"):
        letter(ax, f"({t})")
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
    fig.savefig(OUT / "Fig_SpinCheck.png", dpi=600)
    fig.savefig(OUT / "Fig_SpinCheck.pdf")
    plt.close(fig)


def figure_events(S, U):
    style()
    W = TEXT_WIDTH_MM / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.62), sharex="col", gridspec_kw=dict(height_ratios=[1, 1], hspace=0.12, wspace=0.30))
    for col, key in enumerate(("E1", "E2")):
        sub = S[S.set == key].copy()
        sub["frame"] = sub.structure.str.extract(r"_f(\d+)$").astype(int)
        sub = sub.sort_values("frame")
        x = np.arange(len(sub))
        ids = [tuple(int(v) for v in str(p).split()) for p in sub.event_O_poscar_indices]
        mom = []
        for s, (a, b) in zip(sub.structure, ids):
            u = U[(U.structure == s) & (((U.atom_a == a) & (U.atom_b == b)) | ((U.atom_a == b) & (U.atom_b == a)))]
            mom.append(abs(float(u.moment.iloc[0])) if len(u) else 0.0)
        roo = sub.O_O_distance_A.astype(float).to_numpy()
        dzr = sub.nearest_solid_atom_A.astype(float).to_numpy()
        top, bot = axes[0, col], axes[1, col]
        top.plot(x, sub.dF, "-o", color=C_SET[key], ms=4.5, lw=1.1, mec="white", mew=0.5)
        top.set_ylabel(r"$\Delta F$ (eV)")
        top.set_ylim(min(0, sub.dF.min()) - 0.07 * sub.dF.max(), 1.20 * sub.dF.max())
        bot.plot(x, mom, "-s", color="0.25", ms=4, lw=1.0, mfc="0.25")
        bot.set_ylabel(r"Moment ($\mu_\mathrm{B}$)")
        bot.set_ylim(-0.12, 1.95)
        bot.set_xticks(x)
        bot.set_xticklabels([f"{a:.1f}\n{r:.2f}" for a, r in zip(dzr, roo)], fontsize=8)
        letter(top, f"({'ab'[col]}) {SET_LABEL[key].replace('Event, ', '').capitalize()}")
        if key == "E1":
            for xi, n, y in zip(x, sub.n_gas, sub.dF):
                top.annotate(f"{int(n)}", (xi, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8, color="0.3")
        top.set_xlim(-0.45, len(sub) - 0.55)
    fig.supxlabel("Distance to the surface (upper) and O–O distance (lower) of the followed molecule, in Å", fontsize=10, y=0.012)
    fig.subplots_adjust(left=0.085, right=0.99, top=0.93, bottom=0.20)
    fig.savefig(OUT / "Fig_SpinEvents.png", dpi=600)
    fig.savefig(OUT / "Fig_SpinEvents.pdf")
    plt.close(fig)


def figure_supplementary(S, U, P):
    """Four-panel figure of the Supplementary Information: (a) O-O potential of the isolated molecule,
    (b) energy difference per frame against the summed moment of its O2 molecules, (c) force
    differences by atom group, (d) force between two gas molecules."""
    style()
    W = TEXT_WIDTH_MM / 25.4
    fig, axes = plt.subplots(2, 2, figsize=(W, W * 0.80))
    (a, b), (c, d) = axes
    iso = S[S.set == "O2"].copy()
    iso["r"] = [U[U.structure == s].r.iloc[0] for s in iso.structure]
    main = iso[~iso.structure.str.contains("sigma")].sort_values("r")
    full = float(U[U.structure == "O2_isolated"].moment.abs().iloc[0])
    slabs = S[S.set != "O2"]

    # (a) O-O potential of the isolated molecule
    e0 = main.sp_free_energy.min()
    rr = np.linspace(main.r.min(), main.r.max(), 200)
    for col, colour, label, marker, fill in (("nsp_free_energy", C_NSP, "Without spin", "o", "white"),
                                             ("sp_free_energy", C_SP, "With spin", "s", C_SP)):
        y = main[col] - e0
        a.plot(rr, np.polyval(np.polyfit(main.r, y, 4), rr), color=colour, lw=1.1)
        a.plot(main.r, y, marker, ms=5, mfc=fill, mec=colour, mew=1.1, ls="none", label=label)
    a.set_xlabel("O–O distance (Å)")
    a.set_ylabel("Free energy (eV)")
    a.legend(frameon=False, loc="upper left", handletextpad=0.3)

    # (b) energy difference per frame against the summed moment of its O2 molecules
    xmax = slabs.sum_abs_moment_units.max()
    per_mu = float(iso[iso.structure == "O2_isolated"].dF.iloc[0]) / full
    b.plot([0, xmax * 1.03], [0, per_mu * xmax * 1.03], color="0.55", lw=0.9, ls="--", zorder=1)
    for key, colour in C_SET.items():
        sub = slabs[slabs.set == key]
        b.plot(sub.sum_abs_moment_units, sub.dF, "o", ms=4.5, mfc=colour, mec="white", mew=0.5, zorder=3)
    b.set_xlabel(r"Summed O$_2$ moment ($\mu_\mathrm{B}$)")
    b.set_ylabel(r"$\Delta F$ (eV)")
    b.set_xlim(-0.03 * xmax, 1.05 * xmax)
    b.set_ylim(-0.04 * per_mu * xmax, 1.08 * per_mu * xmax)
    handles = [Line2D([], [], marker="o", ls="none", mfc=C_SET[k], mec="white", ms=5.5, label=SET_NAME[k]) for k in C_SET]
    b.legend(handles=handles, frameon=False, loc="upper left", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2, fontsize=8)

    # (c) force differences by atom group
    cols = ["Zr", "C", "O_single", "O_unit"]
    x = np.arange(len(cols))
    rms = [float(np.sqrt(np.nanmean(slabs[f"dF_rms_{k}"] ** 2))) for k in cols]
    mx = [float(np.nanmax(slabs[f"dF_max_{k}"])) for k in cols]
    c.bar(x, rms, width=0.55, color=["0.6", "0.6", "0.6", C_SP], edgecolor="0.2", lw=0.6, zorder=2)
    c.plot(x, mx, "v", ms=5, mfc="white", mec="0.2", mew=0.9, zorder=3)
    typical = float(slabs.F_rms.mean())
    c.axhline(typical, color="0.55", lw=0.9, ls="--", zorder=1)
    c.set_yscale("log")
    c.set_ylim(0.008, 60)
    c.set_xticks(x)
    c.set_xticklabels(["Zr", "C", "O in\nsolid", "O$_2$"])
    c.set_ylabel("Force difference (eV Å$^{-1}$)")
    c.legend(handles=[Line2D([], [], marker="s", ls="none", mfc="0.6", mec="0.2", ms=6, label="r.m.s."),
                      Line2D([], [], marker="v", ls="none", mfc="white", mec="0.2", ms=5, label="largest")],
             frameon=False, loc="upper left", handletextpad=0.2, borderaxespad=0.2, ncol=2, columnspacing=1.0)

    # (d) force between two gas molecules
    p = P[~P.sp_is_unpolarised]
    d.axhline(0, color="0.55", lw=0.8, zorder=1)
    d.plot(p.contact_A, p.push_nsp, "o", ms=4.5, mfc="white", mec=C_NSP, mew=1.0, label="Without spin", zorder=3)
    d.plot(p.contact_A, p.push_sp, "s", ms=4.2, mfc=C_SP, mec="white", mew=0.4, label="With spin", zorder=3)
    d.set_xlabel("Closest O···O between two O$_2$ (Å)")
    d.set_ylabel("Force apart (eV Å$^{-1}$)")
    d.legend(frameon=False, loc="upper right", handletextpad=0.2, borderaxespad=0.2)
    for ax, t in zip((a, b, c, d), "abcd"):
        letter(ax, f"({t})")
    fig.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.0)
    fig.savefig(OUT / "Fig_SpinSupplementary.png", dpi=600)
    fig.savefig(OUT / "Fig_SpinSupplementary.pdf")
    plt.close(fig)
    return typical, rms, mx


# ----------------------------------------------------------------------------- tables
def tables(S, U, P):
    iso, slabs = S[S.set == "O2"], S[S.set != "O2"]
    su = U[U.set != "O2"]
    full = abs(float(U[U.structure == "O2_isolated"].moment.iloc[0]))
    L = ["# Spin-polarisation check: numbers", "",
         "Written by AnalyseSpinTest.py. F is the free energy that VASP reports with the Fermi smearing of the MD, the quantity whose "
         "gradient gives the MD forces. E0 is the energy extrapolated to zero smearing. dF = F(no spin) - F(spin). "
         "Moments are those inside the PAW spheres (LORBIT = 11), about 80% of the total for O2.", "",
         "## Lone O2 molecule", "", "| Structure | O-O (A) | dF (eV) | dE0 (eV) | Total moment | Moment in spheres |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in iso.iterrows():
        u = U[U.structure == r.structure].iloc[0]
        L.append(f"| {r.structure} | {u.r:.2f} | {r.dF:.3f} | {r.dE0:.3f} | {r.total_moment:.2f} | {abs(u.moment):.2f} |")
    L += ["", "## Slab structures", "",
          "| Structure | Spin run used | Gas O2 isolated | Gas O2 in contact | Units with a moment | Summed unit moment | dF (eV) | dE0 (eV) | Largest moment on Zr | on C | on single O |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in slabs.iterrows():
        L.append(f"| {r.structure} | {r.sp_used}{'' if r.sp_converged else ' (NOT CONVERGED)'} | {r.n_gas_isolated} | {r.n_gas_contact} | {r.n_spin_units} | "
                 f"{r.sum_abs_moment_units:.2f} | {r.dF:.3f} | {r.dE0:.3f} | {r.max_moment_Zr:.2f} | {r.max_moment_C:.2f} | {r.max_moment_O_single:.2f} |")
    one = slabs[(slabs.n_gas_isolated == 1) & (slabs.n_gas_contact == 0)]
    clean = one[[len(su[(su.structure == s) & (su.kind == "adsorbed") & (su.moment.abs() > 0.3)]) == 0 for s in one.structure]]
    L += ["", "## Summary numbers", "",
          f"- Slab structures: {len(slabs)}. All have a converged non-spin-polarised and a converged spin-polarised result: "
          f"{bool(slabs.sp_converged.all() and slabs.nsp_converged.all())}. Second pass used for {int((slabs.sp_used == 'sp2').sum())}, first pass for {int((slabs.sp_used == 'sp').sum())}.",
          f"- Largest moment on any Zr atom: {slabs.max_moment_Zr.max():.2f}. On any C atom: {slabs.max_moment_C.max():.2f}. "
          f"On any chain C atom: {slabs.max_moment_C_chain.max():.2f}. On any O atom outside an O2 unit: {slabs.max_moment_O_single.max():.2f}.",
          f"- Structures with exactly one gas O2, isolated, and no adsorbed unit with a moment ({len(clean)}): "
          + ", ".join(f"{r.structure.split('_')[0]} {r.dF:.3f}" for _, r in clean.iterrows())
          + f". Mean dF {clean.dF.mean():.3f} eV. Moment of that molecule {', '.join(f'{abs(float(su[(su.structure == s) & (su.kind == chr(103) + chr(97) + chr(115) + chr(44) + chr(32) + chr(105) + chr(115) + chr(111) + chr(108) + chr(97) + chr(116) + chr(101) + chr(100))].moment.iloc[0])):.2f}' for s in clean.structure)} (lone molecule {full:.2f})."]
    for kind in ("gas, isolated", "gas, in contact", "adsorbed"):
        k = su[su.kind == kind]
        L.append(f"- O2 units, {kind}: {len(k)}. Mean moment {k.moment.abs().mean():.2f}. Above 80% of the lone-molecule moment: "
                 f"{int((k.moment.abs() > 0.8 * full).sum())}. Between 0.3 and 80%: {int(((k.moment.abs() > 0.3) & (k.moment.abs() <= 0.8 * full)).sum())}. Below 0.3: {int((k.moment.abs() <= 0.3).sum())}.")
    ads = su[su.kind == "adsorbed"]
    L += [f"- Adsorbed units with O-O below 1.40 A: {int((ads.r < 1.40).sum())}, of which {int(((ads.r < 1.40) & (ads.moment.abs() > 0.3)).sum())} carry a moment above 0.3 "
          f"(mean {ads[(ads.r < 1.40) & (ads.moment.abs() > 0.3)].moment.abs().mean():.2f}). With O-O of 1.40 A or more: {int((ads.r >= 1.40).sum())}, of which "
          f"{int(((ads.r >= 1.40) & (ads.moment.abs() > 0.3)).sum())} carry a moment above 0.3.",
          "", "## Force differences between the two calculations (eV per A)", "",
          "| Atom group | r.m.s. over all slab structures | Largest | r.m.s. in structures where no O2 unit carries a moment |", "|---|---:|---:|---:|"]
    quiet = slabs[slabs.n_spin_units == 0]
    for lab, k in (("Zr", "Zr"), ("C", "C"), ("O outside O2 units", "O_single"), ("O in O2 units", "O_unit")):
        q = float(np.sqrt(np.nanmean(quiet[f"dF_rms_{k}"] ** 2))) if len(quiet) else np.nan
        L.append(f"| {lab} | {np.sqrt(np.nanmean(slabs[f'dF_rms_{k}'] ** 2)):.3f} | {np.nanmax(slabs[f'dF_max_{k}']):.2f} | {q:.4f} |")
    L += ["", f"Typical force on an atom in these frames (r.m.s., no spin): {slabs.F_rms.mean():.2f} eV per A. "
          f"Structures where no O2 unit carries a moment: {len(quiet)}.", "",
          "## Gas molecules in contact", "",
          "Force between two gas O2 molecules (both further than 4 A from every Zr) along the line joining their centres. Positive pushes them apart. "
          "Pairs from structures whose best spin-polarised result is unpolarised are left out, since there the two calculations are the same.", ""]
    p = P[~P.sp_is_unpolarised]
    close = p[p.contact_A < 2.3]
    L += [f"- Pairs with a closest O...O distance below 2.3 A: {len(close)}. Mean force apart: {close.push_nsp.mean():+.2f} eV/A without spin, "
          f"{close.push_sp.mean():+.2f} eV/A with spin. Attractive without spin in {int((close.push_nsp < 0).sum())}, with spin in {int((close.push_sp < 0).sum())}.",
          f"- Shortest contact in these structures: {P.contact_A.min():.2f} A.",
          "", "## Check against the MD", "",
          "Free energy from the non-spin-polarised static run minus the free energy the MD recorded for the same frame (frames written with energies only).", "",
          "| Structure | Static minus MD (eV) |", "|---|---:|"]
    for _, r in slabs[slabs.md_free_energy.notna()].iterrows():
        L.append(f"| {r.structure} | {r.nsp_free_energy - r.md_free_energy:+.3f} |")
    (OUT / "Tables.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return "\n".join(L)


def figure_methods():
    """Two-panel figure of the Methods subsection on spin polarisation, from the tables written above.

    (a) moment of every O2 unit in the 22 frames against its distance to the nearest Zr atom
    (b) free-energy difference and moment of the followed molecule along the adsorption event
        on the oxidised surface (1273 K cell 1)
    """
    C_GAS, C_ADS, C_DF = "#d95f02", "0.45", "#386cb0"
    S = pd.read_csv(OUT / "Structures.csv")
    U = pd.read_csv(OUT / "O2Units.csv")
    style()
    W = TEXT_WIDTH_MM / 25.4
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, W * 0.43), gridspec_kw=dict(width_ratios=[1, 1.12]))

    # (a) moment against distance from the surface, relative to the isolated molecule
    full = float(U[U.structure == "O2_isolated"].moment.abs().iloc[0])
    su = U[U.set != "O2"]
    kinds = (("gas, isolated", dict(marker="o", s=16, facecolors=C_GAS, edgecolors="0.2", linewidths=0.4), "Gas", 3),
             ("gas, in contact", dict(marker="o", s=16, facecolors="white", edgecolors=C_GAS, linewidths=1.0), "Gas, touching another O$_2$", 2),
             ("adsorbed", dict(marker="s", s=14, facecolors=C_ADS, edgecolors="0.2", linewidths=0.4), "Adsorbed", 3))
    for kind, kw, lab, z in kinds:
        sub = su[su.kind == kind]
        a.scatter(sub.d_zr.clip(upper=15), sub.moment.abs() / full, zorder=z, label=lab, **kw)
    a.axhline(1.0, color="0.55", lw=0.9, ls="--", zorder=1)
    a.set_xscale("log")
    a.set_xlim(1.65, 16.5)
    a.set_xticks([2, 3, 5, 10, 15])
    a.set_xticklabels(["2", "3", "5", "10", "15"])
    a.minorticks_off()
    a.set_ylim(-0.06, 1.62)
    a.set_yticks([0, 0.5, 1.0])
    a.set_xlabel("Distance to nearest Zr (Å)")
    a.set_ylabel("Moment relative to isolated O$_2$")
    a.legend(frameon=False, loc="upper right", handletextpad=0.1, borderaxespad=0.2, labelspacing=0.2, fontsize=8.5)

    # (b) the adsorption event on the oxidised surface
    sub = S[S.set == "E2"].copy()
    sub["frame"] = sub.structure.str.extract(r"_f(\d+)$").astype(int)
    sub = sub.sort_values("frame")
    x = np.arange(len(sub))
    mom = []
    for s, p in zip(sub.structure, sub.event_O_poscar_indices):
        i, j = (int(v) for v in str(p).split())
        u = U[(U.structure == s) & (((U.atom_a == i) & (U.atom_b == j)) | ((U.atom_a == j) & (U.atom_b == i)))]
        mom.append(abs(float(u.moment.iloc[0])) if len(u) else 0.0)
    dzr = sub.nearest_solid_atom_A.astype(float).to_numpy()
    roo = sub.O_O_distance_A.astype(float).to_numpy()
    tb = sub.time_from_binding_fs.astype(float).to_numpy()
    b.plot(x, sub.dF, "-o", color=C_DF, ms=4.5, lw=1.1, mec="white", mew=0.5, zorder=3, label=r"$\Delta F$")
    b.set_ylabel(r"$\Delta F$ (eV)", color=C_DF)
    b.tick_params(axis="y", colors=C_DF)
    b.set_ylim(-0.06, 1.62)
    b.set_yticks([0, 0.5, 1.0])
    b.set_xticks(x)
    b.set_xticklabels([f"{d:.1f}" for d in dzr])
    b.set_xlim(-0.45, len(sub) - 0.55)
    b.set_xlabel("Distance of the molecule to nearest Zr (Å)")
    t = b.twinx()
    t.plot(x, np.array(mom) / full, "--s", color="0.25", ms=3.8, lw=1.0, mfc="0.25", zorder=2, label="Moment")
    t.set_ylabel("Moment relative to isolated O$_2$")
    t.set_ylim(-0.06, 1.62)
    t.set_yticks([0, 0.5, 1.0])
    h = b.get_lines() + t.get_lines()
    b.legend(h, [l.get_label() for l in h], frameon=False, loc="upper right", handletextpad=0.4, borderaxespad=0.2, labelspacing=0.2, fontsize=8.5)
    print("time from binding (fs):", tb)
    print("event frames:", [(f"{d:.2f}", f"{r:.2f}", f"{e:.3f}", f"{m:.2f}") for d, r, e, m in zip(dzr, roo, sub.dF, mom)])
    for ax, lab in ((a, "(a)"), (b, "(b)")):
        letter(ax, lab)
    fig.tight_layout(pad=0.4, w_pad=1.4)
    fig.savefig(OUT / "Fig_SpinMethods.png", dpi=600)
    fig.savefig(OUT / "Fig_SpinMethods.pdf")
    plt.close(fig)


def main():
    data = SD.load(sys.argv[1] if len(sys.argv) > 1 else "results.json")
    S, U, P = build(data)
    S.to_csv(OUT / "Structures.csv", index=False)
    U.to_csv(OUT / "O2Units.csv", index=False)
    P.to_csv(OUT / "GasPairs.csv", index=False)
    print(tables(S, U, P))
    figure_main(S, U)
    figure_events(S, U)
    figure_methods()
    figure_supplementary(pd.read_csv(OUT / "Structures.csv"), pd.read_csv(OUT / "O2Units.csv"), pd.read_csv(OUT / "GasPairs.csv"))
    print("\nfigures written to", OUT)


if __name__ == "__main__":
    main()
