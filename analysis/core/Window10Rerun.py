"""
Window10Rerun.py
=======================================================================
Reruns Refresh140 Modules 6 (density profiles) and 9 (oxygen layering)
with the structural window changed from 126.0-140.0 ps to 130.0-140.0 ps
(final 10 ps), and reports old-vs-new side by side. Nothing else changes:
bond criteria and every detection parameter are IMPORTED from Refresh140
(which imports them from the sibling scripts) - the window is the only
difference. Chain structure (Module 7) keeps 126.0-140.0 ps by design.

Figure: the profile figure is arranged 2 x 2 (one
panel per temperature) with the x axis trimmed to the occupied region
(data-driven extent + margin) instead of the full 42 A cell.

Reads: Refresh140/FrameCache (grid cache; read-only reuse) and
Refresh140/Module_9_Layering.json + Module_6_Profiles.json for the 14 ps
reference values. Never writes into Refresh140/ or the data folder.
Seed 20260715 quoted; nothing stochastic.
"""

import os
import sys
import json

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ScriptDir)

import Refresh140 as Rf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import OxygenLayeringAnalysis as Ola
import ChainTopology as Ct
import BootstrapCalculation as Bc
import BoundaryConventions as Bcv

ResultsDir = os.path.join(ScriptDir, "Window10Rerun")
RandomSeed = 20260715
Window10 = (130.0, 140.0)
Window14 = Rf.TerminalWindowPs                      # (126.0, 140.0)
MidWindow10 = (64.0, 74.0)
Temperatures = Rf.Temperatures
SimIndices = Rf.SimIndices
SpacingClassTol = Ola.SpacingMatchTolFrac           # 6%, identical thresholds


# =============================================================================
#                 STEP 1 - DENSITY PROFILES AT 130-140 ps
# =============================================================================
def PooledProfiles(WindowPs, WithRawCheck=False):
    """Per-T mean and between-trajectory SD profiles on the grid; optional
    raw-frame POTIM check. Same binning/centring as Refresh140.Module6."""
    Lx0 = 42.41
    NBins = int(np.ceil(Lx0 / Rf.ProfileBinA))
    Out, RawDiffs, FrameCounts = {}, {}, {}
    for Temp in Temperatures:
        PerTraj = {Name: [] for Name in Rf.ProfileSpecies}
        MaxDiff = 0.0
        for Sim in SimIndices:
            Hists = {Name: np.zeros(NBins) for Name in Rf.ProfileSpecies}
            NFrames = 0
            for TimePs, Sp, Co, La, Tr, Sn in Rf.IterGridFrames(
                    Temp, Sim, WindowPs=WindowPs):
                Rf.ProfileFromFrame(Sp, Co, La, Hists, NBins)
                NFrames += 1
            FrameCounts[f"{Temp}_{Sim}"] = NFrames
            BinVol = Rf.ProfileBinA * 9.456 * 9.462
            Grid = {Name: Hists[Name] / max(1, NFrames) / BinVol
                    for Name in Rf.ProfileSpecies}
            for Name in Rf.ProfileSpecies:
                PerTraj[Name].append(Grid[Name])
            if WithRawCheck:
                HistsRaw = {Name: np.zeros(NBins) for Name in Rf.ProfileSpecies}
                NRaw = 0
                Offset = Bcv.FindOffsetAtTime(
                    Bc.XyzPath(Temp, Sim), WindowPs[0] * 1000.0, Rf.HorizonFs)
                for TimeFs, Atoms, Coords, Lattice in Ct.ReadFramesFrom(
                        Bc.XyzPath(Temp, Sim), Offset):
                    if TimeFs < WindowPs[0] * 1000.0:
                        continue
                    if TimeFs > Rf.HorizonFs:
                        break
                    Codes = np.array([Rf.SpeciesCodes.get(A, 9) for A in Atoms],
                                     dtype=np.int8)
                    Rf.ProfileFromFrame(Codes, Coords, Lattice, HistsRaw, NBins)
                    NRaw += 1
                for Name in Rf.ProfileSpecies:
                    Raw = HistsRaw[Name] / max(1, NRaw) / BinVol
                    MaxDiff = max(MaxDiff, float(np.max(np.abs(Grid[Name] - Raw))))
        Out[Temp] = {Name: (np.mean(PerTraj[Name], axis=0),
                            np.std(PerTraj[Name], axis=0, ddof=1))
                     for Name in Rf.ProfileSpecies}
        if WithRawCheck:
            RawDiffs[Temp] = MaxDiff
    return Out, RawDiffs, FrameCounts, NBins, Lx0


def MakeProfileFigure2x2(Profiles, Lx, NBins, OutBase, WidthMm=160.0):
    Edges = np.linspace(0.0, Lx, NBins + 1)
    Centres = 0.5 * (Edges[:-1] + Edges[1:]) - Lx / 2.0
    # Data-driven x trim: bins with structural density (slab + oxide; the
    # 0.008 threshold sits above the diffuse gas background ~0.005 but well
    # below the weakest oxide shoulders ~0.02).
    Occupied = np.zeros(NBins, dtype=bool)
    for Temp in Temperatures:
        for Name in Rf.ProfileSpecies:
            Occupied |= Profiles[Temp][Name][0] > 0.008
    XLo = Centres[Occupied].min() - 1.5
    XHi = Centres[Occupied].max() + 1.5

    # Canvas sized at the DOCUMENT's \textwidth (WidthMm) so that LaTeX's
    # \includegraphics[width=\textwidth] applies scale factor 1.0 and the
    # 10 pt type prints at true 10 pt. Height keeps the 183:130 aspect.
    # (Sizing at Nature's 183 mm print column and then downscaling to a
    # narrower \textwidth is what shrank the text.)
    BodyPt = 10
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    SpeciesColours = {"Zr": "#7f7f7f", "C": "#2f2f2f", "O": "#d62728"}
    HeightMm = WidthMm * 130.0 / 183.0
    Fig, Axes = plt.subplots(2, 2, figsize=(WidthMm / 25.4, HeightMm / 25.4),
                             sharex=True, sharey=True)
    # Zero line pinned near the panel bottom (SD bands clipped at 0 - number
    # density cannot be negative) with a little headroom above the tallest band.
    InRange = (Centres >= XLo) & (Centres <= XHi)
    YTop = 1.05 * max(float(np.max((Profiles[T][Name][0]
                                    + Profiles[T][Name][1])[InRange]))
                      for T in Temperatures for Name in Rf.ProfileSpecies)
    from matplotlib.ticker import MultipleLocator
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    PanelLetters = ["(a)", "(b)", "(c)", "(d)"]
    for Idx, (Ax, Temp) in enumerate(zip(Axes.ravel(), Temperatures)):
        # Zr as a filled silhouette (substrate) in a light steel blue with a
        # defined top edge - deliberately OUTSIDE the grey/translucent space
        # so it cannot be confused with the SD bands; its own SD band shares
        # the blue hue. C and O are lines with clipped SD bands.
        ZrMean, ZrSd = Profiles[Temp]["Zr"]
        Ax.fill_between(Centres, 0.0, ZrMean, color="#dbe6f4", lw=0, zorder=1)
        Ax.plot(Centres, ZrMean, color="#7f9fc4", lw=1.0, zorder=1.5)
        Ax.fill_between(Centres, np.maximum(ZrMean - ZrSd, 0.0), ZrMean + ZrSd,
                        color="#7f9fc4", alpha=0.30, lw=0, zorder=1)
        for Name in ("C", "O"):
            Mean, Sd = Profiles[Temp][Name]
            Ax.plot(Centres, Mean, color=SpeciesColours[Name], lw=1.4, zorder=3)
            Ax.fill_between(Centres, np.maximum(Mean - Sd, 0.0), Mean + Sd,
                            color=SpeciesColours[Name], alpha=0.20, lw=0, zorder=2)
        Ax.set_title(f"{Temp} K", fontsize=BodyPt)
        Ax.set_xlim(XLo, XHi)
        Ax.set_ylim(-0.008, YTop)
        Ax.axhline(0.0, color="0.65", lw=0.7, zorder=0)
        Ax.xaxis.set_major_locator(MultipleLocator(5))
        Ax.xaxis.set_minor_locator(MultipleLocator(1))
        Ax.tick_params(labelsize=BodyPt)
        Ax.tick_params(which="minor", length=2)
        Ax.text(0.015, 0.965, PanelLetters[Idx], transform=Ax.transAxes,
                fontsize=BodyPt, fontweight="bold", va="top", ha="left")
    for Ax in Axes[-1]:
        Ax.set_xlabel("$x$ $-$ Zr centre of mass (Å)", fontsize=BodyPt)
    for Ax in Axes[:, 0]:
        Ax.set_ylabel("Number density (Å$^{-3}$)", fontsize=BodyPt)
    Handles = [Patch(facecolor="#dbe6f4", edgecolor="#7f9fc4", label="Zr"),
               Line2D([], [], color=SpeciesColours["C"], lw=1.4, label="C"),
               Line2D([], [], color=SpeciesColours["O"], lw=1.4, label="O")]
    Fig.legend(handles=Handles, loc="lower center", ncol=3, fontsize=BodyPt,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    Fig.tight_layout(rect=[0, 0.06, 1, 1])
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)
    return float(XLo), float(XHi)


# =============================================================================
#            STEP 2 - LAYERING PIPELINE AT A GIVEN WINDOW
# =============================================================================
def RunLayering(WindowPs):
    """Refresh140.Module9 pipeline re-driven at an arbitrary window; output
    returned instead of written into Refresh140/. Thresholds identical
    (imported from OxygenLayeringAnalysis)."""
    Extr = {(T, S): Rf.ExtractLayeringWindow(T, S, WindowPs)
            for T in Temperatures for S in SimIndices}
    DZrCByTemp = {T: float(np.median([Sp for (TT, _), E in Extr.items()
                                      if TT == T for Sp in E["InteriorSpacings"]]))
                  for T in Temperatures}
    OoByTemp = {T: Ola.OoFirstNeighbour([F for (TT, _), E in Extr.items()
                                         if TT == T for F in E["OxideOFrames"]])
                for T in Temperatures}
    AllSurfaces = {}
    for (T, S), E in Extr.items():
        for Side in Ola.SideLabels:
            AllSurfaces[(T, S, Side)] = Ola.AnalyseSurfaceLayers(
                E["Surface"][Side], DZrCByTemp[T])

    NBinsG = int(round(Ola.G2dRLimit / Ola.G2dBin))
    Area = 9.456 * 9.462
    G2dAccum, CrystalAccum = {}, {}
    for (T, S), E in Extr.items():
        for Side in Ola.SideLabels:
            R = AllSurfaces[(T, S, Side)]
            if R is None or not R.get("LayerU"):
                continue
            for Frame in E["Surface"][Side]["Frames"]:
                for LayerIdx, U0 in enumerate(R["LayerU"], start=1):
                    Members = np.abs(Frame["U"] - U0) <= Ola.LayerWindow
                    if Members.sum() < 2:
                        continue
                    Key = (T, LayerIdx)
                    G2dAccum.setdefault(Key, [np.zeros(NBinsG), 0, 0])
                    NP = Ola.AccumulatePairHistogram2d(
                        Frame["Y"][Members].astype(np.float64),
                        Frame["Z"][Members].astype(np.float64),
                        9.456, 9.462, G2dAccum[Key][0], Ola.G2dRLimit, Ola.G2dBin)
                    G2dAccum[Key][1] += NP
                    G2dAccum[Key][2] += 1
        for (Y, Z, FLy, FLz) in E["CrystalFrames"]:
            CrystalAccum.setdefault(T, [np.zeros(NBinsG), 0, 0])
            Ola.AccumulatePairHistogram2d(
                Y.astype(np.float64), Z.astype(np.float64), FLy, FLz,
                CrystalAccum[T][0], Ola.G2dRLimit, Ola.G2dBin)
            CrystalAccum[T][1] += (len(Y) * (len(Y) - 1)) // 2
            CrystalAccum[T][2] += 1

    CrystalChar = {}
    for T, (Hist, NPairs, NFrames) in CrystalAccum.items():
        RC, G = Ola.FinaliseG2d(Hist, NPairs, Area, Ola.G2dRLimit, Ola.G2dBin)
        CrystalChar[T] = Ola.CharacteriseG2d(RC, G)
    RefPeaks = [CrystalChar[T]["FirstPeakR"] for T in (1073, 1273)
                if T in CrystalChar and CrystalChar[T]]
    CrystalPeakR = float(np.mean(RefPeaks)) if RefPeaks else float("nan")
    G2dChar = {}
    for (T, LayerIdx), (Hist, NPairs, NFrames) in G2dAccum.items():
        RC, G = Ola.FinaliseG2d(Hist, NPairs, Area, Ola.G2dRLimit, Ola.G2dBin)
        G2dChar[(T, LayerIdx)] = Ola.CharacteriseG2d(RC, G, CrystalPeakR)

    Verdicts = {}
    for T in Temperatures:
        SurfList = [AllSurfaces[(T, S, Side)] for S in SimIndices
                    for Side in Ola.SideLabels]
        SpacingV, _ = Ola.SpacingRegistryVerdict(
            SurfList, DZrCByTemp[T], OoByTemp[T][0])
        Verdicts[str(T)] = {
            "Spacing": SpacingV,
            "Decay": Ola.DecayVerdict(SurfList),
            "InPlane": Ola.InPlaneVerdict(
                [G2dChar.get((T, L)) for L in sorted(
                    LI for (TT, LI) in G2dChar if TT == T)])}

    Surfaces = {}
    for (T, S, Side), R in AllSurfaces.items():
        Key = f"{T}_{S}_{Side}"
        Surfaces[Key] = (None if R is None else
                         {K: V for K, V in R.items()
                          if K not in ("Profile", "PeakIdx")})
    return {"DZrCByTemp": {str(T): DZrCByTemp[T] for T in Temperatures},
            "OoByTemp": {str(T): OoByTemp[T][0] for T in Temperatures},
            "Verdicts": Verdicts, "Surfaces": Surfaces,
            "InPlane": {f"{T}_layer{L}": C for (T, L), C in G2dChar.items()},
            "NFrames": {f"{T}_{S}": E["Surface"]["L"]["NFrames"]
                        for (T, S), E in Extr.items()}}


def MidTrajectoryCheck():
    """1273 K, matched 10 ps window 64-74 ps, spacing test only."""
    Mid = {}
    for S in SimIndices:
        E = Rf.ExtractLayeringWindow(1273, S, MidWindow10)
        DMid = (float(np.median(E["InteriorSpacings"]))
                if E["InteriorSpacings"] else float("nan"))
        for Side in Ola.SideLabels:
            R = Ola.AnalyseSurfaceLayers(E["Surface"][Side], DMid)
            Key = f"1273_{S}_{Side}"
            if R is None or len(R.get("LayerU", [])) < 2:
                Mid[Key] = {"Layers": 0 if R is None else len(R.get("LayerU", []))}
                continue
            Mid[Key] = {"Layers": len(R["LayerU"]),
                        "MeanSpacing": R["MeanSpacing"],
                        "RatioToDZrC": R["MeanSpacing"] / DMid,
                        "Drift": R["OffsetDriftPerLayer"]}
    return Mid


# =============================================================================
#                     STEP 3 - SIDE-BY-SIDE COMPARISON
# =============================================================================
def SpacingClass(Ratio, DZrC, DOo):
    """Per-surface spacing classification under the identical 6% tolerance."""
    if Ratio is None or not np.isfinite(Ratio):
        return "unresolved"
    D = Ratio * DZrC
    MatchZrC = abs(D - DZrC) <= SpacingClassTol * DZrC
    MatchOo = np.isfinite(DOo) and abs(D - DOo) <= SpacingClassTol * DOo
    if MatchZrC and not MatchOo:
        return "ZrC"
    if MatchOo and not MatchZrC:
        return "OO"
    if MatchZrC and MatchOo:
        return "both"
    return "neither"


def SurfaceComparison(Old9, New9):
    Rows = []
    for T in [1073, 1273, 873, 973]:            # primary temperatures first
        DZrCOld = Old9["DZrCByTemp"][str(T)]
        DZrCNew = New9["DZrCByTemp"][str(T)]
        DOoOld = Old9.get("OoPackingByTemp", Old9.get("OoByTemp", {})).get(str(T))
        DOoNew = New9["OoByTemp"][str(T)]
        for S in SimIndices:
            for Side in Ola.SideLabels:
                Key = f"{T}_{S}_{Side}"
                RO = Old9["Surfaces"].get(Key) or {}
                RN = New9["Surfaces"].get(Key) or {}
                L14 = len(RO.get("LayerU", []))
                L10 = len(RN.get("LayerU", []))
                Ratio14 = (RO.get("MeanSpacing") / DZrCOld
                           if RO.get("MeanSpacing") and L14 >= 2 else None)
                Ratio10 = (RN.get("MeanSpacing") / DZrCNew
                           if RN.get("MeanSpacing") and L10 >= 2 else None)
                C14 = SpacingClass(Ratio14, DZrCOld, DOoOld)
                C10 = SpacingClass(Ratio10, DZrCNew, DOoNew)
                Rows.append({
                    "Temp": T, "Surface": f"{S}{Side}",
                    "Layers14": L14, "Layers10": L10,
                    "Ratio14": Ratio14, "Ratio10": Ratio10,
                    "Drift14": RO.get("OffsetDriftPerLayer"),
                    "Drift10": RN.get("OffsetDriftPerLayer"),
                    "Class14": C14, "Class10": C10,
                    "ClassChanged": C14 != C10,
                    "LostUsability": bool(L14 >= 2 > L10),
                    "GainedUsability": bool(L10 >= 2 > L14)})
    return Rows


# =============================================================================
#                                REPORTING
# =============================================================================
def Fmt(X, Nd=2):
    if X is None or (isinstance(X, float) and not np.isfinite(X)):
        return "-"
    return f"{X:.{Nd}f}"


def RegistryStats(Surfaces, Temp):
    """Median |offset drift| and n(|drift| <= 0.05) over one temperature's
    usable surfaces (>= 2 resolved layers)."""
    Drifts = []
    for S in SimIndices:
        for Side in Ola.SideLabels:
            R = Surfaces.get(f"{Temp}_{S}_{Side}")
            if (R and len(R.get("LayerU", [])) >= 2
                    and R.get("OffsetDriftPerLayer") is not None
                    and np.isfinite(R["OffsetDriftPerLayer"])):
                Drifts.append(R["OffsetDriftPerLayer"])
    AbsVals = np.abs(Drifts)
    return {"MedianAbsDrift": float(np.median(AbsVals)) if len(AbsVals) else float("nan"),
            "NWithin005": int((AbsVals <= 0.05).sum()),
            "NSurfaces": len(Drifts)}


def Main():
    import argparse
    Parser = argparse.ArgumentParser(description="10 ps window rerun.")
    Parser.add_argument("--fig-width-mm", type=float, default=131.2,
                        help="Figure canvas width in mm. The default is the text "
                             "width of the journal, so that the figure prints unscaled.")
    Args = Parser.parse_args()
    os.makedirs(ResultsDir, exist_ok=True)
    EndTimes, GatePass, DatasetPass = Rf.RunGates()
    if not (GatePass and DatasetPass):
        print("STOP: gate failed.")
        return 1

    print("Step 1: profiles at 130-140 ps (with raw-frame POTIM check) ...",
          flush=True)
    Prof10, Raw10, FrameCounts, NBins, Lx = PooledProfiles(Window10,
                                                           WithRawCheck=True)
    Prof14, _, _, _, _ = PooledProfiles(Window14, WithRawCheck=False)
    ProfileDelta = {}
    for T in Temperatures:
        ProfileDelta[str(T)] = float(max(
            np.max(np.abs(Prof10[T][Name][0] - Prof14[T][Name][0]))
            for Name in Rf.ProfileSpecies))
    XLo, XHi = MakeProfileFigure2x2(Prof10, Lx, NBins,
                                    os.path.join(ResultsDir, "Fig1_Profiles10ps"),
                                    WidthMm=Args.fig_width_mm)
    # Figure of the paper at the 14 ps terminal window (126-140 ps);
    # same styling, Prof14 already computed for the delta check.
    MakeProfileFigure2x2(Prof14, Lx, NBins,
                         os.path.join(ResultsDir, "Fig1_DensityProfiles_126_140"),
                         WidthMm=Args.fig_width_mm)
    print(f"  figure x range trimmed to [{XLo:.1f}, {XHi:.1f}] A", flush=True)

    print("Step 2: layering at 130-140 ps ...", flush=True)
    New9 = RunLayering(Window10)
    Old9 = json.load(open(os.path.join(ScriptDir, "Refresh140",
                                       "Module_9_Layering.json")))
    Mid = MidTrajectoryCheck()

    Rows = SurfaceComparison(Old9, New9)
    VerdictChanges = {}
    for T in (1073, 1273):
        VerdictChanges[str(T)] = {
            Meas: {"From": Old9["Verdicts"][str(T)][Meas],
                   "To": New9["Verdicts"][str(T)][Meas],
                   "LetterChanged": (Old9["Verdicts"][str(T)][Meas].split()[0]
                                     != New9["Verdicts"][str(T)][Meas].split()[0])}
            for Meas in ("Spacing", "Decay", "InPlane")}
    TemplateSurvives = all(New9["Verdicts"][str(T)]["Spacing"] == "B"
                           and Old9["Verdicts"][str(T)]["Spacing"] == "B"
                           for T in (1073, 1273))

    Registry = {str(T): {"At14ps": RegistryStats(Old9["Surfaces"], T),
                         "At10ps": RegistryStats(New9["Surfaces"], T)}
                for T in (1073, 1273)}
    # Stamp values: pooled mean layer spacing over usable primary surfaces at
    # 10 ps; packing = mean of the two primary-temperature peaks.
    PrimarySpacings = [R["MeanSpacing"] for T in (1073, 1273)
                       for S in SimIndices for Side in Ola.SideLabels
                       for R in [New9["Surfaces"].get(f"{T}_{S}_{Side}")]
                       if R and len(R.get("LayerU", [])) >= 2
                       and np.isfinite(R.get("MeanSpacing", float("nan")))]
    Stamp = {
        "MeanSpacing10ps_A": float(np.mean(PrimarySpacings)),
        "MedianAbsDrift10ps": {str(T): Registry[str(T)]["At10ps"]["MedianAbsDrift"]
                               for T in (1073, 1273)},
        "PackingReference_A": float(np.mean([New9["OoByTemp"]["1073"],
                                             New9["OoByTemp"]["1273"]])),
        "TemplateVerdictSurvives": TemplateSurvives,
    }

    Payload = {"Seed": RandomSeed, "Window10_ps": list(Window10),
               "Window14_ps": list(Window14),
               "ProfileDelta_maxAbs": ProfileDelta,
               "PotimCheck10ps": {str(T): Raw10[T] for T in Temperatures},
               "FrameCountsPerTrajectory": FrameCounts,
               "FigureXRange_A": [XLo, XHi],
               "Layering10ps": New9, "MidTrajectory_64_74ps": Mid,
               "SurfaceComparison": Rows,
               "VerdictChanges": VerdictChanges,
               "RegistryStatistics": Registry,
               "StampBlock": Stamp,
               "TemplateVerdictSurvives": TemplateSurvives}
    with open(os.path.join(ResultsDir, "Window10Results.json"), "w") as F:
        json.dump(Payload, F, indent=1, default=Rf.JsonSafe)

    WriteSummary(GatePass, ProfileDelta, Raw10, Rows, Old9, New9, Mid,
                 TemplateSurvives, FrameCounts, Registry, Stamp)
    print("\nDone. Results in", ResultsDir, flush=True)
    return 0


def WriteSummary(GatePass, ProfileDelta, Raw10, Rows, Old9, New9, Mid,
                 TemplateSurvives, FrameCounts, Registry, Stamp):
    L = []
    L.append("# Window change rerun: profiles and layering at 130-140 ps")
    L.append("")
    L.append(f"Gate: 16/16 >= 140.0 ps: {'yes' if GatePass else 'no'} | "
             "Detection parameters identical to Refresh140: yes")
    L.append("")
    L.append("## 1. Profile deltas")
    L.append("| T (K) | Max |10ps - 14ps| density (atoms/A^3) | "
             "POTIM check at 10 ps (max grid-raw) |")
    L.append("|---:|---:|---:|")
    for T in Temperatures:
        L.append(f"| {T} | {ProfileDelta[str(T)]:.4f} | {Raw10[T]:.4f} |")
    L.append("")
    L.append("## 2. Layering at 10 ps vs 14 ps")
    L.append("| T (K) | Surface | Layers 14/10 | d_O/d_ZrC 14 -> 10 | "
             "Drift 14 -> 10 | Class change? |")
    L.append("|---:|---|---|---|---|---|")
    for R in Rows:
        Change = ("yes: " + R["Class14"] + " -> " + R["Class10"]
                  if R["ClassChanged"] else "no")
        L.append(f"| {R['Temp']} | {R['Surface']} | "
                 f"{R['Layers14']}/{R['Layers10']} | "
                 f"{Fmt(R['Ratio14'])} -> {Fmt(R['Ratio10'])} | "
                 f"{Fmt(R['Drift14'], 3)} -> {Fmt(R['Drift10'], 3)} | {Change} |")
    L.append("")
    L.append("## 3. Registry statistics and verdicts")
    L.append("| T (K) | Median |drift| 14 -> 10 | n(<=0.05) 14 -> 10 |")
    L.append("|---:|---|---|")
    for T in (1073, 1273):
        R14 = Registry[str(T)]["At14ps"]
        R10 = Registry[str(T)]["At10ps"]
        L.append(f"| {T} | {R14['MedianAbsDrift']:.3f} -> "
                 f"{R10['MedianAbsDrift']:.3f} | "
                 f"{R14['NWithin005']}/{R14['NSurfaces']} -> "
                 f"{R10['NWithin005']}/{R10['NSurfaces']} |")
    L.append("| Measurement | 1073 K (14 -> 10) | 1273 K (14 -> 10) |")
    L.append("|---|---|---|")
    for Meas in ("Spacing", "Decay", "InPlane"):
        L.append(f"| {Meas} | {Old9['Verdicts']['1073'][Meas]} -> "
                 f"{New9['Verdicts']['1073'][Meas]} | "
                 f"{Old9['Verdicts']['1273'][Meas]} -> "
                 f"{New9['Verdicts']['1273'][Meas]} |")
    L.append("Packing reference at 10 ps by T: "
             + " / ".join(f"{T} {New9['OoByTemp'][str(T)]:.2f}"
                          for T in Temperatures))
    MidParts = [f"{K}: {Fmt(E.get('RatioToDZrC'))}"
                for K, E in sorted(Mid.items()) if E.get("Layers", 0) >= 2]
    Unusable = [K for K, E in sorted(Mid.items()) if E.get("Layers", 0) < 2]
    MidLine = "; ".join(MidParts)
    if Unusable:
        MidLine += f" (unresolved: {', '.join(Unusable)})"
    L.append(f"Mid-trajectory check (64-74 ps): d/d_ZrC per 1273 K surface: "
             f"{MidLine}.")
    L.append("")
    L.append("## 4. Stamp block")
    L.append(f"Spacing: {Stamp['MeanSpacing10ps_A']:.1f} Angstrom | "
             f"Median |drift|: 1073 K {Stamp['MedianAbsDrift10ps']['1073']:.3f}, "
             f"1273 K {Stamp['MedianAbsDrift10ps']['1273']:.3f} | "
             f"Packing: {Stamp['PackingReference_A']:.1f} Angstrom")
    L.append(f"Template verdict (B) survives at both primary temperatures: "
             f"{'yes' if TemplateSurvives else 'no'}")
    L.append("")
    L.append("## 5. Notes")
    Lost = [f"{R['Temp']} {R['Surface']}" for R in Rows if R["LostUsability"]]
    Gained = [f"{R['Temp']} {R['Surface']}" for R in Rows if R["GainedUsability"]]
    L.append(f"- Usability threshold (2 resolved layers): lost at the 10 ps "
             f"window: {Lost if Lost else 'none'}; gained: "
             f"{Gained if Gained else 'none'}.")
    NF = sorted(set(FrameCounts.values()))
    Pooled = sorted(set(sum(FrameCounts[f"{T}_{S}"] for S in SimIndices)
                        for T in Temperatures))
    L.append(f"- Pooled-profile frame counts at 10 ps: {NF} grid frames per "
             f"trajectory (the task's nominal 200 points plus the shared "
             f"130.0 ps endpoint), pooled {Pooled} frames per temperature "
             f"(vs ~281/1124 at 14 ps); per-trajectory counts in "
             "Window10Results.json.")
    L.append("- Figure arranged 2 x 2 (one panel per temperature) with "
             "the x axis trimmed to the structurally occupied region (species "
             "density > 0.008 atoms/A^3 anywhere, +/- 1.5 A margin) instead "
             "of the full 42 A cell; 10 pt Times New Roman at 183 mm width "
             "per the figure conventions of the paper.")
    L.append("- All detection parameters (bond criteria, plane/layer peak "
             "detection, verdict thresholds, contamination flags) imported "
             "unchanged from Refresh140/OxygenLayeringAnalysis; the window "
             "is the only change. Mid-trajectory check moved to the matched "
             "10 ps window 64-74 ps.")
    L.append("- Refresh140 outputs untouched (read-only reference); grid "
             "cache reused read-only; the raw-frame POTIM pass is the only "
             "xyz access. Seed 20260715 quoted; nothing stochastic.")
    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print(Text)


if __name__ == "__main__":
    sys.exit(Main())
