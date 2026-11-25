"""
BootstrapFigureRemake.py
=======================================================================
Remakes the bootstrap carbon-retention figure (Fig. 2 of the paper) in the
FRACTION form phi_C = chain/(chain+gas), styled identically to the
density-profile figure: 10 pt Times New Roman on a 160 mm canvas, STIX
mathtext, no gridlines, frameless legend at the bottom centre. Dashed
reference line at phi_C = 0.5 (equal cumulative extent of the two carbon
pathways).

The 10,000 trajectory-stratified resamples are regenerated from
Refresh140/Events/EventTableV2.csv with the imported machinery
(BootstrapCalculation.RunBootstrap, seed 20260715) - deterministic, and
the fraction form is the exact monotone transform r/(1+r) of the stored
ratio-form results, verified against Module_2_Partitioning.json
(P(phi<0.5) == stored P_below_1) at run time.

Output: phiTimeseries/Fig_PhiC_Bootstrap.png/.pdf + values echoed to
BootstrapFigureValues.json in the same folder.
"""

import os
import sys
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ScriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ScriptDir)

import Refresh140 as Rf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import BootstrapCalculation as Bc

ResultsDir = os.path.join(ScriptDir, "phiTimeseries")
RandomSeed = 20260715
Temperatures = Rf.Temperatures
TemperatureColours = Rf.TemperatureColours


def Main():
    os.makedirs(ResultsDir, exist_ok=True)
    Table = pd.read_csv(os.path.join(ScriptDir, "Refresh140", "Events",
                                     "EventTableV2.csv"))
    EventFrame = Table.rename(columns={"Time_ps": "TimeStep"}).copy()
    EventFrame["TimeStep"] = EventFrame["TimeStep"] * 1000.0

    RatioByTemp, NanCount = Bc.RunBootstrap(EventFrame, 10000, RandomSeed)
    M1 = json.load(open(os.path.join(ScriptDir, "Refresh140",
                                     "Module_1_Events.json")))
    M2 = json.load(open(os.path.join(ScriptDir, "Refresh140",
                                     "Module_2_Partitioning.json")))

    Values = {}
    for Temp in Temperatures:
        Ratio = RatioByTemp[Temp]
        Frac = Ratio[~np.isnan(Ratio)]
        Frac = Frac / (1.0 + Frac)
        Ev = M1["Totals"][str(Temp)]
        Gas = Ev["CO"] + Ev["CO2"]
        Observed = Ev["Retained"] / (Ev["Retained"] + Gas)
        PBelow = float(np.mean(Frac < 0.5))
        StoredP = M2["DominanceProbability"][str(Temp)]["P_below_1"]
        assert abs(PBelow - StoredP) < 1e-9, (Temp, PBelow, StoredP)
        Values[str(Temp)] = {
            "Observed": Observed,
            "P2_5": float(np.percentile(Frac, 2.5)),
            "P97_5": float(np.percentile(Frac, 97.5)),
            "P_below_05": PBelow,
            "NanRounds": int(NanCount[Temp]),
            "FracSamples": Frac,
        }
        print(f"  {Temp} K: observed {Observed:.3f}, 95% "
              f"[{Values[str(Temp)]['P2_5']:.3f}, "
              f"{Values[str(Temp)]['P97_5']:.3f}], "
              f"P(<0.5) {PBelow:.3f} (== stored {StoredP:.4f})", flush=True)

    MakeFigure(Values, os.path.join(ResultsDir, "Fig_PhiC_Bootstrap"))
    with open(os.path.join(ResultsDir, "BootstrapFigureValues.json"), "w") as F:
        json.dump({T: {K: V for K, V in E.items() if K != "FracSamples"}
                   for T, E in Values.items()}, F, indent=1)
    print("Done. Figure in", ResultsDir, flush=True)
    return 0


def MakeFigure(Values, OutBase):
    BodyPt = 10
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    # Canvas at the sn-jnl \textwidth (131.2 mm) so 10 pt fonts render true.
    Fig, Ax = plt.subplots(figsize=(131.2 / 25.4, 82 / 25.4))
    Positions = list(range(1, len(Temperatures) + 1))

    Violins = Ax.violinplot([Values[str(T)]["FracSamples"] for T in Temperatures],
                            positions=Positions, widths=0.7, showextrema=False)
    for Body, Temp in zip(Violins["bodies"], Temperatures):
        Body.set_facecolor(TemperatureColours[Temp])
        Body.set_alpha(0.45)
        Body.set_edgecolor("none")
    for Pos, Temp in zip(Positions, Temperatures):
        E = Values[str(Temp)]
        Ax.plot([Pos, Pos], [E["P2_5"], E["P97_5"]], color="black", lw=1.4,
                solid_capstyle="butt", zorder=4)
        Ax.plot([Pos], [E["Observed"]], marker="D",
                color=TemperatureColours[Temp], markeredgecolor="black",
                markersize=8, zorder=5)
    Ax.axhline(0.5, color="grey", ls="--", lw=1.1, zorder=1)

    Ax.set_xticks(Positions)
    Ax.set_xticklabels([f"{T} K" for T in Temperatures], fontsize=BodyPt)
    Ax.set_ylabel("$\\phi_C$", fontsize=BodyPt)
    Ax.set_ylim(0, 1.0)
    Ax.tick_params(labelsize=BodyPt)

    Handles = [
        Patch(facecolor="0.6", alpha=0.45, label="Bootstrap distribution"),
        Line2D([], [], color="black", lw=1.4, label="95\\% interval"),
        Line2D([], [], marker="D", color="0.4", markeredgecolor="black",
               lw=0, markersize=8, label="Observed"),
        Line2D([], [], color="grey", ls="--", lw=1.1, label="Equal extent"),
    ]
    Fig.legend(handles=Handles,
               labels=["Bootstrap distribution", "95% interval", "Observed",
                       "Equal extent ($\\phi_C = 0.5$)"],
               loc="lower center", ncol=2, fontsize=BodyPt, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    Fig.tight_layout(rect=[0, 0.16, 1, 1])
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


if __name__ == "__main__":
    sys.exit(Main())
