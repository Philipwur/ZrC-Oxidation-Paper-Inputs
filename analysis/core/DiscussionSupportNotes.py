"""
DiscussionSupportNotes.py
=======================================================================
Tier-2 SI-note calculations backing single Discussion sentences.
Three independent notes, one pass:

A) SCALE COMPOSITION vs Lin EDX. Final-window (126-140 ps) converted
   region = oxidised Zr (N_O >= 2 within 3.17 A, frame-averaged) +
   scale O + chain C (both from PhiTimeseries, window-averaged).
   Reports O/Zr and at% C per temperature for comparison with the
   experimental protective-scale window (O/Zr < 2, C 20-50 at%).
B) SYNTHETIC TGA. Net solid mass change per temperature,
   dm(t) = 16*NScaleO(t) - 12*NGasC(t), summed over the four cells.
   Linear net-rate fit on 20-140 ps per temperature, then an Arrhenius
   fit of the net rates = the apparent barrier a mass-based measurement
   would recover over this window.
C) RESERVOIR FILLING RATE. Chain and gas commitments per 35 ps quarter
   per temperature from the audited ledger. Tests whether the chain
   reservoir's filling rate declines late (self-limiting hint,
   Gasparrini limiting inner layer).

Inputs read-only: Refresh140/FrameCache, phiTimeseries/PhiTimeseries.csv,
Refresh140/Events/EventTableV2.csv.
Output: DiscussionSupportResults/ (Notes.json, Fig_SyntheticTGA.png/.pdf,
console tables). Seed-free (deterministic).

Also written:
  Fig_ConvolutionDemo.png/.pdf   net mass change against time and the mass
                                 flows against 1000/T (MassFlows.pdf in the paper)
  FluxNormalisedNetMass.json     net-mass activation energy with the rates
                                 divided by the mean oxygen flux
"""

import os
import json

import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.dirname(os.path.abspath(__file__))
CacheDir = os.path.join(ScriptDir, "Refresh140", "FrameCache")
PhiPath = os.path.join(ScriptDir, "phiTimeseries", "PhiTimeseries.csv")
EventPath = os.path.join(ScriptDir, "Refresh140", "Events", "EventTableV2.csv")
ResultsDir = os.path.join(ScriptDir, "DiscussionSupportResults")

Temperatures = [873, 973, 1073, 1273]
Sims = [1, 2, 3, 4]
ZrOCutoff = 3.17
MinO = 2
CodeZr, CodeO = 0, 1
WindowPs = (126.0, 140.0)
TgaFitPs = (20.0, 140.0)
MassO, MassC = 15.999, 12.011
Quarters = [(0, 35), (35, 70), (70, 105), (105, 140)]

TempColours = {873: "#1f77b4", 973: "#d62728", 1073: "#2ca02c", 1273: "#9467bd"}
plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10,
    "axes.labelsize": 10, "legend.fontsize": 8.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "stix",
})


def OxidisedZrFinalWindow(Temp, Sim):
    """Frame-averaged counts over 126-140 ps: (oxidised Zr, interface
    lattice C). Oxidised Zr = >= MinO oxygen contacts (3.17 A). Interface
    lattice C = carbon with >= 5 Zr neighbours (3.33 A, the Sec 2.3 lattice
    criterion) of which at least one is oxidised - the near-interface
    lattice carbon an EDX voxel over the converted region would include."""
    Z = np.load(os.path.join(CacheDir, f"{Temp}_{Sim}_grid50fs_140ps.npz"),
                allow_pickle=True)
    Times = Z["Times"] / 1000.0
    Starts = Z["FrameStarts"]
    Species, Coords, Lattices = Z["Species"], Z["Coords"], Z["Lattices"]
    CodeC = 2
    ZrCCutoff = 3.33
    ValsZr, ValsCInt = [], []
    for F in range(len(Times)):
        if not (WindowPs[0] <= Times[F] <= WindowPs[1]):
            continue
        Sp = Species[Starts[F]:Starts[F + 1]]
        Co = Coords[Starts[F]:Starts[F + 1]].astype(float)
        La = Lattices[F].astype(float)
        ZrIdx = np.where(Sp == CodeZr)[0]
        OIdx = np.where(Sp == CodeO)[0]
        CIdx = np.where(Sp == CodeC)[0]
        Frac = Co @ np.linalg.inv(La)
        if len(OIdx):
            Diff = Frac[ZrIdx][:, None, :] - Frac[OIdx][None, :, :]
            Diff -= np.round(Diff)
            Dist = np.sqrt(((Diff @ La) ** 2).sum(axis=2))
            ZrOx = (Dist < ZrOCutoff).sum(axis=1) >= MinO
        else:
            ZrOx = np.zeros(len(ZrIdx), bool)
        ValsZr.append(int(ZrOx.sum()))
        if len(CIdx):
            DiffC = Frac[CIdx][:, None, :] - Frac[ZrIdx][None, :, :]
            DiffC -= np.round(DiffC)
            DistC = np.sqrt(((DiffC @ La) ** 2).sum(axis=2))
            Contact = DistC < ZrCCutoff
            Lattice = Contact.sum(axis=1) >= 5
            TouchOx = (Contact & ZrOx[None, :]).any(axis=1)
            ValsCInt.append(int((Lattice & TouchOx).sum()))
        else:
            ValsCInt.append(0)
    return float(np.mean(ValsZr)), float(np.mean(ValsCInt))


def Main():
    os.makedirs(ResultsDir, exist_ok=True)
    Phi = pd.read_csv(PhiPath)
    Events = pd.read_csv(EventPath)
    Out = {}

    # ---- A) scale composition --------------------------------------------
    print("A) SCALE COMPOSITION, final window 126-140 ps")
    print(f"{'T':>6} {'Zr_ox':>7} {'O_scale':>8} {'C_chain':>8} {'C_intf':>7} "
          f"{'O/Zr':>6} {'at%C':>6} {'at%C_incl':>9}")
    CompRows = {}
    for T in Temperatures:
        ZrOx, OSc, CCh, CInt = 0.0, 0.0, 0.0, 0.0
        for S in Sims:
            Zr, Ci = OxidisedZrFinalWindow(T, S)
            ZrOx += Zr
            CInt += Ci
            Tr = Phi[(Phi.TrajectoryId == f"{T}_{S}")
                     & (Phi.Time_ps >= WindowPs[0]) & (Phi.Time_ps <= WindowPs[1])]
            OSc += Tr.NScaleO.mean()
            CCh += Tr.NChainC.mean()
        OZr = OSc / ZrOx
        AtC = 100.0 * CCh / (ZrOx + OSc + CCh)
        AtCIncl = 100.0 * (CCh + CInt) / (ZrOx + OSc + CCh + CInt)
        CompRows[T] = dict(ZrOx=ZrOx, OScale=OSc, CChain=CCh, CInterface=CInt,
                           OoverZr=OZr, AtPctC=AtC, AtPctCInclusive=AtCIncl)
        print(f"{T:>6} {ZrOx:>7.1f} {OSc:>8.1f} {CCh:>8.1f} {CInt:>7.1f} "
              f"{OZr:>6.2f} {AtC:>6.1f} {AtCIncl:>9.1f}")
    Out["ScaleComposition"] = CompRows

    # ---- B) synthetic TGA -------------------------------------------------
    print("\nB) SYNTHETIC TGA, dm(t) = 16*NScaleO - 12*NGasC (4 cells summed)")
    Fig, Ax = plt.subplots(figsize=(0.62 * 131.2 / 25.4, 62 / 25.4))
    Slopes = {}
    for T in Temperatures:
        # Trajectories carry slightly different time grids; interpolate each
        # onto a common grid before summing, then smooth (~2 ps moving
        # average) to remove transient network-fragmentation flicker in
        # NScaleO.
        Tt = np.arange(0.0, 140.0001, 0.1)
        SumO = np.zeros_like(Tt)
        SumC = np.zeros_like(Tt)
        for S in Sims:
            Tr = Phi[Phi.TrajectoryId == f"{T}_{S}"].sort_values("Time_ps")
            SumO += np.interp(Tt, Tr.Time_ps, Tr.NScaleO)
            SumC += np.interp(Tt, Tr.Time_ps, Tr.NGasC)
        Win = 21
        Kernel = np.ones(Win) / Win
        SumO = np.convolve(np.pad(SumO, Win // 2, mode="edge"), Kernel, "valid")
        SumC = np.convolve(np.pad(SumC, Win // 2, mode="edge"), Kernel, "valid")
        Dm = MassO * SumO - MassC * SumC
        Ax.plot(Tt, Dm / 1000.0, "-", lw=1.5,
                color=TempColours[T], label=f"{T} K")
        Sel = (Tt >= TgaFitPs[0]) & (Tt <= TgaFitPs[1])
        Res = linregress(Tt[Sel], Dm[Sel])
        Slopes[T] = dict(Slope_uPerPs=Res.slope, R2=Res.rvalue ** 2)
        print(f"  {T} K: net rate {Res.slope:.1f} u/ps (R2 {Res.rvalue**2:.3f})")
    X = np.array([1.0 / T for T in Temperatures])
    Y = np.array([np.log(Slopes[T]["Slope_uPerPs"]) for T in Temperatures])
    B, A = np.polyfit(X, Y, 1)
    R2 = 1 - np.var(Y - (B * X + A)) / np.var(Y)
    KB = 8.617333e-5
    EaEv = -B * KB
    print(f"  Arrhenius of net mass rate: Ea = {EaEv:.3f} eV "
          f"({EaEv*96.485:.1f} kJ/mol), R2 = {R2:.3f}")
    Out["SyntheticTGA"] = {"Slopes": Slopes,
                           "ApparentEa_eV": EaEv, "ArrheniusR2": R2}
    FluxNormalisedNetMass(Slopes, EaEv)
    Ax.set_xlabel("Time (ps)")
    Ax.set_ylabel(r"Net solid mass change (ku)")
    Ax.legend(frameon=False, loc="upper left", handlelength=1.4)
    Fig.tight_layout(pad=0.4)
    Fig.savefig(os.path.join(ResultsDir, "Fig_SyntheticTGA.png"), dpi=300)
    Fig.savefig(os.path.join(ResultsDir, "Fig_SyntheticTGA.pdf"))

    # ---- C) quarterly commitment counts -----------------------------------
    print("\nC) COMMITMENTS PER 35 ps QUARTER (chain | gas)")
    QuartRows = {}
    Head = "  ".join(f"{Q0}-{Q1}" for Q0, Q1 in Quarters)
    print(f"{'T':>6}  {Head}")
    for T in Temperatures:
        Ev = Events[Events.Temperature == T]
        Ch = [int(((Ev.Outcome == "Chain") & (Ev.Time_ps >= Q0)
                   & (Ev.Time_ps < Q1)).sum()) for Q0, Q1 in Quarters]
        Ga = [int(((Ev.Outcome == "Gas") & (Ev.Time_ps >= Q0)
                   & (Ev.Time_ps < Q1)).sum()) for Q0, Q1 in Quarters]
        QuartRows[T] = dict(Chain=Ch, Gas=Ga)
        print(f"{T:>6}  " + "  ".join(f"{C:>2}|{G:<2}" for C, G in zip(Ch, Ga)))
    Out["QuarterlyCommitments"] = QuartRows

    # ---- C2) piecewise cause-specific hazards -----------------------------
    # Exposure per quarter = summed time-at-risk of the 192 lattice C per
    # temperature (subjects leave the pool at commitment; unresolved carbons
    # contribute the full quarter). Hazard = events / exposure.
    print("\nC2) PIECEWISE CHAIN HAZARD (10^-3 ps^-1), exposure-corrected")
    Head = "  ".join(f"{Q0}-{Q1}" for Q0, Q1 in Quarters)
    print(f"{'T':>6}  {Head}")
    HazRows = {}
    NSubjects = 192
    for T in Temperatures:
        Ev = Events[Events.Temperature == T].sort_values("Time_ps")
        CommitTimes = Ev.Time_ps.to_numpy()
        ChainMask = (Ev.Outcome == "Chain").to_numpy()
        Haz = []
        for Q0, Q1 in Quarters:
            Exposure = 0.0
            # committed subjects: clip their in-quarter time
            for Tc in CommitTimes:
                Exposure += max(0.0, min(Tc, Q1) - Q0) if Tc > Q0 else 0.0
            # never-committed subjects: full quarter each
            Exposure += (NSubjects - len(CommitTimes)) * (Q1 - Q0)
            NCh = int((ChainMask & (CommitTimes >= Q0) & (CommitTimes < Q1)).sum())
            H = NCh / Exposure
            Se = np.sqrt(NCh) / Exposure if NCh else 0.0
            Haz.append((H, Se))
        HazRows[T] = [dict(Hazard=H, SE=Se) for H, Se in Haz]
        print(f"{T:>6}  " + "  ".join(f"{1e3*H:.1f}({1e3*Se:.1f})" for H, Se in Haz))
    Out["PiecewiseChainHazard"] = HazRows

    json.dump(Out, open(os.path.join(ResultsDir, "Notes.json"), "w"), indent=1)
    MakeMassFlowFigure()
    print(f"\nWritten: {ResultsDir}")


# -----------------------------------------------------------------------------
# Net mass rate normalised by the oxygen flux
# -----------------------------------------------------------------------------

def FluxNormalisedNetMass(Slopes, EaEv):
    """Net-mass activation energy after dividing each net mass rate by the mean
    oxygen flux of its temperature (free-molecule flux over 0-140 ps, from
    GasSupplyAudit/GasSupplyAudit.json). Quoted in the Methods of the paper."""
    Audit = json.load(open(os.path.join(ScriptDir, "GasSupplyAudit", "GasSupplyAudit.json")))
    Flux = np.array([Audit["PerTemperature"][str(T)]["WindowStats"]["0-140"]["free"]["J_TempMean"]
                     for T in Temperatures])
    Rates = np.array([Slopes[T]["Slope_uPerPs"] for T in Temperatures])
    X = np.array([1.0 / T for T in Temperatures])
    EaFlux = -np.polyfit(X, np.log(Rates / Flux), 1)[0] * 8.617333e-5
    Result = {"NetMassEa_eV": float(EaEv), "FluxNormalisedNetMassEa_eV": float(EaFlux),
              "MeanFlux_m2s": {str(T): float(J) for T, J in zip(Temperatures, Flux)}}
    json.dump(Result, open(os.path.join(ResultsDir, "FluxNormalisedNetMass.json"), "w"), indent=1)
    print(f"  Net mass rate divided by the mean O2 flux: Ea = {EaFlux:.3f} eV "
          f"(not divided: {EaEv:.3f} eV)")


# -----------------------------------------------------------------------------
# Figure of the net mass gain and the mass flows (MassFlows.pdf in the paper)
# -----------------------------------------------------------------------------

NotesPath = os.path.join(ScriptDir, "DiscussionSupportResults", "Notes.json")

# Panel (b) is deliberately achromatic: colour
# is reserved for temperature throughout the paper, so the mechanism series
# are separated by grey level + marker + line style instead.
OColour, GasColour, ChainColour = "0.30", "0.45", "0.60"



def MakeMassFlowFigure():
    Phi = pd.read_csv(PhiPath)
    Notes = json.load(open(NotesPath))  # noqa: F841 (kept for provenance)
    # CommitmentFits.json is intentionally no longer loaded: panel (b)
    # plots mass FLOWS from PhiTimeseries slopes, not commitment rates.

    Fig, Axes = plt.subplots(1, 2, figsize=(131.2 / 25.4, 70 / 25.4))

    # ---- (a) synthetic mass curves ----------------------------------------
    Ax = Axes[0]
    Tt = np.arange(0.0, 140.0001, 0.1)
    Win = 21
    Kernel = np.ones(Win) / Win
    for T in Temperatures:
        SumO = np.zeros_like(Tt)
        SumC = np.zeros_like(Tt)
        for S in Sims:
            Tr = Phi[Phi.TrajectoryId == f"{T}_{S}"].sort_values("Time_ps")
            SumO += np.interp(Tt, Tr.Time_ps, Tr.NScaleO)
            SumC += np.interp(Tt, Tr.Time_ps, Tr.NGasC)
        SumO = np.convolve(np.pad(SumO, Win // 2, mode="edge"), Kernel, "valid")
        SumC = np.convolve(np.pad(SumC, Win // 2, mode="edge"), Kernel, "valid")
        Dm = MassO * SumO - MassC * SumC
        Ax.plot(Tt, Dm, "-", lw=1.5, color=TempColours[T], label=f"{T} K")
    Ax.set_xlabel("Time (ps)")
    Ax.set_ylabel("Net mass change (amu)")
    Ax.set_yticks([0, 500, 1000, 1500, 2000, 2500])
    Ax.legend(frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), fontsize=7.5, handlelength=1.0,
              columnspacing=1.2, handletextpad=0.4)
    Ax.text(0.95, 0.05, "(a)", transform=Ax.transAxes, ha="right",
            fontweight="bold")

    # ---- (b) the mass flows, Arrhenius view --------------------------------
    # The balance sees exactly two flows: O uptake (+16 u) and gas C loss
    # (-12 u). The chain flow is plotted to show it enters the balance
    # nowhere. Flows = linear slopes on the smoothed, summed series, same
    # window as the net-rate fit (20-140 ps).
    Ax = Axes[1]
    InvT = np.array([1000.0 / T for T in Temperatures])
    Flows = {"O": [], "GasC": [], "ChainC": []}
    Sel = (Tt >= 20.0) & (Tt <= 140.0)
    for T in Temperatures:
        Series = {"O": np.zeros_like(Tt), "GasC": np.zeros_like(Tt),
                  "ChainC": np.zeros_like(Tt)}
        for S in Sims:
            Tr = Phi[Phi.TrajectoryId == f"{T}_{S}"].sort_values("Time_ps")
            Series["O"] += np.interp(Tt, Tr.Time_ps, Tr.NScaleO)
            Series["GasC"] += np.interp(Tt, Tr.Time_ps, Tr.NGasC)
            Series["ChainC"] += np.interp(Tt, Tr.Time_ps, Tr.NChainC)
        for Key, Mass in (("O", MassO), ("GasC", MassC), ("ChainC", MassC)):
            Sm = np.convolve(np.pad(Series[Key], Win // 2, mode="edge"),
                             Kernel, "valid")
            Flows[Key].append(Mass * np.polyfit(Tt[Sel], Sm[Sel], 1)[0])
    OFlow = np.array(Flows["O"])
    CFlow = np.array(Flows["GasC"])
    ChFlow = np.array(Flows["ChainC"])
    Net = OFlow - CFlow

    B, A = np.polyfit(1.0 / np.array(Temperatures, float), np.log(Net), 1)
    Xf = np.linspace(1.0 / 1273.0, 1.0 / 873.0, 50)
    Ax.plot(1000.0 * Xf, np.exp(A + B * Xf), "-", color="0.15", lw=1.4,
            zorder=2)
    Ax.plot(InvT, Net, "s", ms=6, color="0.15", label="Net mass rate",
            zorder=4)
    Ax.plot(InvT, OFlow, "D", ms=5, mfc="white", color=OColour,
            label="O uptake ($+16$ amu)", zorder=3)
    Ax.plot(InvT, OFlow, "--", lw=1.1, color=OColour, zorder=2)
    Ax.plot(InvT, ChFlow, "^", ms=5.5, mfc="white", color=ChainColour,
            label="C to chains ($\\pm 0$ amu)", zorder=3)
    Ax.plot(InvT, ChFlow, "-.", lw=1.1, color=ChainColour, zorder=2)
    Ax.plot(InvT, CFlow, "o", ms=5.5, mfc="white", color=GasColour,
            label="C to gas ($-12$ amu)", zorder=3)
    Ax.plot(InvT, CFlow, ":", lw=1.1, color=GasColour, zorder=2)
    Ax.set_yscale("log")
    Ax.set_ylim(1.6, 22)
    Ax.set_yticks([2, 5, 10, 20])
    Ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda V, _: f"{V:g}"))
    Ax.yaxis.set_minor_formatter(plt.NullFormatter())
    Ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    Ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    Ax.set_ylabel(r"Mass flow (amu ps$^{-1}$)")
    KB = 8.617333e-5
    Resid = np.log(Net) - (A + B / np.array(Temperatures, float))
    R2 = 1 - np.var(Resid) / np.var(np.log(Net))
    Ax.legend(frameon=False, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, 1.0), fontsize=7.5, handlelength=1.0,
              columnspacing=1.2, handletextpad=0.4)
    Ax.text(0.95, 0.05, "(b)", transform=Ax.transAxes, ha="right",
            fontweight="bold")

    Fig.tight_layout(pad=0.4)
    Fig.subplots_adjust(top=0.84)
    # Fit label along the fitted line, placed after layout so the angle
    # computed in display space is exact.
    Ax = Axes[1]
    XMid = 0.5 * (1.0 / 873.0 + 1.0 / 1273.0)
    P1 = Ax.transData.transform((1000.0 / 1273.0, np.exp(A + B / 1273.0)))
    P2 = Ax.transData.transform((1000.0 / 873.0, np.exp(A + B / 873.0)))
    Angle = np.degrees(np.arctan2(P2[1] - P1[1], P2[0] - P1[0]))
    Ax.text(1000.0 * XMid, np.exp(A + B * XMid) * 0.82,
            f"$E_\\mathrm{{a}} = {-B*KB:.2f}$ eV, $R^2 = {R2:.3f}$",
            fontsize=7.5, ha="center", va="top", rotation=Angle,
            rotation_mode="anchor", color="0.15")
    Fig.savefig(os.path.join(ResultsDir, "Fig_ConvolutionDemo.png"), dpi=300)
    Fig.savefig(os.path.join(ResultsDir, "Fig_ConvolutionDemo.pdf"))
    print("Figure written: Fig_ConvolutionDemo.png/.pdf")


if __name__ == "__main__":
    Main()
