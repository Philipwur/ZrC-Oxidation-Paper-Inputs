"""
ExemplarProfilesFigure.py
=======================================================================
SI Section S1 figure (2x2 layout): layer
detection shown explicitly on the two layer-richest surfaces.
Columns: 1073_1_L and 1273_3_R. Top row: pooled oxygen density over
the primary window (126-140 ps) with the detected layers marked.
Bottom row: zirconium density on the same axis and grid, in its own
panels so the carbide front peak is shown in full. Both rows carry
the carbide plane grid continued across the front.

Profiles are recomputed from the Refresh140 frame cache with the
IDENTICAL pipeline (Refresh140.ExtractLayeringWindow +
OxygenLayeringAnalysis.AnalyseSurfaceLayers, thresholds imported), and
the stored per-temperature d_ZrC of Module_9_Layering.json is passed
in unchanged. A sanity gate asserts that the recomputed layer
positions and mean spacing reproduce the stored values. The Zr curves
use the same 0.10 A bins and 0.15 A Gaussian smoothing as the O
profiles (constants imported).

Style: conventions of the paper (131.2 mm width, 10 pt Times New Roman,
stix mathtext, main-text temperature colours).
Output: RegistryFigureResults/Fig_ExemplarProfiles.png/.pdf
The SI carries the PDF as Supplementary/Figures/ExemplarProfiles.pdf.
"""

import os
import sys
import json

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ScriptDir)

import Refresh140 as Rf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import OxygenLayeringAnalysis as Ola

ModulePath = os.path.join(ScriptDir, "Refresh140", "Module_9_Layering.json")
ResultsDir = os.path.join(ScriptDir, "RegistryFigureResults")

Exemplars = [(1073, 1, "L"), (1273, 3, "R")]
TempColours = {1073: "#2ca02c", 1273: "#9467bd"}
WindowPs = (126.0, 140.0)

plt.rcParams.update({
    "font.family": "Times New Roman", "font.size": 10,
    "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "stix",
})


def ZrProfile(T, S, Side):
    """Zr density on the layer-analysis outward-distance axis, identical
    geometry, binning, normalization, and smoothing conventions."""
    NBins = len(Ola.UCentres())
    Hist = np.zeros(NBins)
    NFr = 0
    for TimePs, Sp, Co, La, Tr, Sn in Rf.IterGridFrames(
            T, S, WindowPs=WindowPs):
        Geo = Ola.AnalyseFrameGeometry(Sp, Co, La)
        if Geo is None:
            continue
        XA = Geo["XA"]
        U = (Geo["FrontL"] - XA) if Side == "L" else (XA - Geo["FrontR"])
        IsZr = ~Geo["IsO"] & ~Geo["IsC"]
        UCap = min(Ola.UHistMax, Geo["HalfGas"])
        Sel = IsZr & (U > Ola.UHistMin) & (U < UCap)
        Hist += np.histogram(U[Sel], bins=NBins,
                             range=(Ola.UHistMin, Ola.UHistMax))[0]
        NFr += 1
    Sigma = Ola.LayerSmoothSigmaAng / Ola.ProfileBin
    return gaussian_filter1d(Hist / max(NFr, 1), Sigma)


def DrawGrid(Ax, DZrC, Centres):
    for K in range(1, 8):
        U = K * DZrC
        if U > Centres[-1]:
            break
        Ax.axvline(U, color="0.6", lw=0.6, ls="--", zorder=1)
    Ax.axvline(0.0, color="0.15", lw=1.0, zorder=2)


def Main():
    os.makedirs(ResultsDir, exist_ok=True)
    Module = json.load(open(ModulePath))
    DZrCByTemp = {int(K): V for K, V in Module["DZrCByTemp"].items()}
    Stored = Module["Surfaces"]

    Fig, Axes = plt.subplots(2, 2, figsize=(131.2 / 25.4, 78 / 25.4),
                             sharex=True)
    Centres = Ola.UCentres()
    for ColIdx, (T, S, Side) in enumerate(Exemplars):
        Tag = f"{T}_{S}_{Side}"
        Col = TempColours[T]
        DZrC = DZrCByTemp[T]

        E = Rf.ExtractLayeringWindow(T, S, WindowPs)
        R = Ola.AnalyseSurfaceLayers(E["Surface"][Side], DZrC)
        Ref = Stored[Tag]
        assert np.allclose(R["LayerU"], Ref["LayerU"], atol=1e-3), Tag
        assert abs(R["MeanSpacing"] - Ref["MeanSpacing"]) < 1e-3, Tag
        print(f"{Tag}: {len(R['LayerU'])} layers reproduce stored values")

        AxO = Axes[0, ColIdx]
        DrawGrid(AxO, DZrC, Centres)
        AxO.plot(Centres, R["Profile"], "-", lw=1.2, color=Col, zorder=3)
        Prof = np.asarray(R["Profile"])
        for U0 in R["LayerU"]:
            H = Prof[int(np.argmin(np.abs(Centres - U0)))]
            AxO.plot(U0, H, "v", ms=4.5, color=Col, mec="0.15", mew=0.5,
                     zorder=4)
        AxO.set_title(f"({'ab'[ColIdx]}) {Tag}", fontsize=10,
                      fontweight="bold", pad=4)
        AxO.set_ylim(0.0, 0.72)

        AxZ = Axes[1, ColIdx]
        DrawGrid(AxZ, DZrC, Centres)
        AxZ.plot(Centres, ZrProfile(T, S, Side), "-", lw=1.2,
                 color="0.45", zorder=3)
        AxZ.set_ylim(0.0, 1.45)
        AxZ.text(0.94, 0.90, f"({'cd'[ColIdx]})", transform=AxZ.transAxes,
                 ha="right", va="top", fontsize=10, fontweight="bold")

    Axes[0, 0].set_xlim(-0.8, 15.0)
    Axes[0, 0].set_ylabel("O count density")
    Axes[1, 0].set_ylabel("Zr count density")
    Fig.supxlabel("Outward distance from carbide front (Å)", fontsize=10)
    Fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.13,
                        wspace=0.18, hspace=0.12)

    OutBase = os.path.join(ResultsDir, "Fig_ExemplarProfiles")
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    print("saved", OutBase + ".png/.pdf")


if __name__ == "__main__":
    Main()
