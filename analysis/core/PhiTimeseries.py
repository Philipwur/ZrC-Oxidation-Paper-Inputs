"""
PhiTimeseries.py
=======================================================================
Per-trajectory time series of the retention ratios of the revised
equations of the paper (eq. phi_c / phi_o):

    PhiC(t) = nChainC(t) / (nChainC(t) + nGasC(t))
    PhiO(t) = nScaleO(t) / (nScaleO(t) + nGasO(t))

Definitions (operational, consistent with the Refresh140 pipeline):
  nChainC(t)  confirmed-Chain carbons at grid time t (0.2 ps time-based
              persistence filter, pending transitions not flushed) - from
              the shared-pass state snapshots in Refresh140/FrameCache.
  nGasC(t)    confirmed-Gas carbons at t (cumulative by construction: a
              gasified carbon stays in the Gas state), excluding identities
              marked ArtifactRemoval in SubsequentChainFateResults/AuditMatches.csv.
  nScaleO(t)  oxygen bound into the slab network at t (bonded component
              containing Zr, intact O2 excluded from the network first)
              - the same operational 'scale O' as Refresh140 Module 2.
  nGasO(t)    cumulative oxygen removed in gas products up to t (CO -> 1 O,
              CO2 -> 2 O), timestamps from the RateAnalysis CSVs (closed
              interval, t <= 140.0 ps). Free O2 in the void is supply, not
              product, and is NOT counted as gas-phase oxygen.

Both ratios are NaN while their denominator is zero (before the first
mobilisation / first gas product). Window 0-140.0 ps on the 50 fs grid;
seed 20260715 quoted (nothing stochastic).

Outputs (phiTimeseries/): PhiTimeseries.csv (long format, per grid point),
Fig_PhiC_Timeseries.png/.pdf, Fig_PhiO_Timeseries.png/.pdf, SUMMARY.md.
The figure of the Supplementary Information (both ratios per temperature)
is written to PhiTimeseriesSIFigureResults/PhiTimeseries.png/.pdf. With
--si-figures-only it is redrawn from the stored PhiTimeseries.csv.
Horizon cross-checks against Module_1/Module_2 JSONs are printed and
recorded in the summary.
"""

import argparse
import os
import sys
import json

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ScriptDir)

import Refresh140 as Rf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import BootstrapCalculation as Bc

ResultsDir = os.path.join(ScriptDir, "phiTimeseries")
RandomSeed = 20260715
Temperatures = Rf.Temperatures
SimIndices = Rf.SimIndices
TemperatureColours = Rf.TemperatureColours
# Trajectory colours (Okabe-Ito, colourblind-safe), deliberately distinct from
# the temperature palette of the paper; temperature is carried by the panel title.
TrajectoryColours = {1: "#0072B2", 2: "#E69F00", 3: "#009E73", 4: "#CC79A7"}


def ArtifactIDs(Temp, Sim):
    Audit = pd.read_csv(os.path.join(
        ScriptDir, "SubsequentChainFateResults", "AuditMatches.csv"))
    Rows = Audit[(Audit["Trajectory"] == f"{Temp}_{Sim}")
                 & (Audit["Kind"] == "ArtifactRemoval")]
    IDs = Rows["AtomID"].to_numpy(dtype=int)
    if len(np.unique(IDs)) != len(IDs) or np.any((IDs < 0) | (IDs >= Bc.TotalStartC)):
        raise ValueError(f"Invalid artifact identities for {Temp}_{Sim}")
    return IDs


def ConfirmedGasCounts(States, ExcludedIDs):
    """Count confirmed Gas states along the identity axis, excluding artifacts."""
    IsGas = States == Rf.StateCodes["Gas"]
    IsGas[..., ExcludedIDs] = False
    return IsGas.sum(axis=-1)


def RefreshGasCountsFromCache():
    """Rebuild gas-carbon counts from state snapshots, idempotently.

    Other stored counts are unchanged. This avoids recalculating oxygen bonding
    when only the gas-removal audit has changed.
    """
    Frame = pd.read_csv(os.path.join(ResultsDir, "PhiTimeseries.csv"))
    for Temp in Temperatures:
        for Sim in SimIndices:
            Mask = Frame["TrajectoryId"] == f"{Temp}_{Sim}"
            with np.load(Rf.GridCachePath(Temp, Sim)) as Data:
                Times = Data["Times"] / 1000.0
                if not np.allclose(Frame.loc[Mask, "Time_ps"], Times, rtol=0, atol=1e-9):
                    raise ValueError(f"Cached time grid mismatch for {Temp}_{Sim}")
                Frame.loc[Mask, "NGasC"] = ConfirmedGasCounts(
                    Data["StateSnapshots"], ArtifactIDs(Temp, Sim))
    Denominator = Frame["NChainC"] + Frame["NGasC"]
    Frame["PhiC"] = Frame["NChainC"] / Denominator.where(Denominator > 0)
    return Frame


def ProcessTrajectory(TaskArgs):
    Temp, Sim = TaskArgs
    ExcludedIDs = ArtifactIDs(Temp, Sim)
    GasEvents, _ = Bc.LoadGasEvents(Bc.CsvPath(Temp, Sim))
    GasOTimes = np.sort(np.concatenate([
        np.full(1 if E["Species"] == "CO" else 2, E["TimeStep"] / 1000.0)
        for E in GasEvents if E["TimeStep"] <= Rf.HorizonFs] or [np.empty(0)]))

    Rows = []
    for TimePs, Sp, Co, La, Tr, Sn in Rf.IterGridFrames(Temp, Sim):
        NChain = int((Sn == Rf.StateCodes["Chain"]).sum())
        NGasC = int(ConfirmedGasCounts(Sn, ExcludedIDs))
        NScaleO = Rf.ScaleOxygenCount(Sp, Co, La)
        NGasO = int(np.searchsorted(GasOTimes, TimePs, side="right"))
        Rows.append({
            "Temperature": Temp, "TrajectoryId": f"{Temp}_{Sim}",
            "Time_ps": TimePs,
            "NChainC": NChain, "NGasC": NGasC,
            "NScaleO": NScaleO, "NGasO": NGasO,
            "PhiC": NChain / (NChain + NGasC) if NChain + NGasC else float("nan"),
            "PhiO": NScaleO / (NScaleO + NGasO) if NScaleO + NGasO else float("nan"),
        })
    return Rows


def MakeTimeseriesFigure(Frame, Column, YLabel, OutBase):
    BodyPt = 10
    matplotlib.rcParams["font.family"] = "Times New Roman"
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    Fig, Axes = plt.subplots(2, 2, figsize=(160 / 25.4, 110 / 25.4),
                             sharex=True, sharey=True)
    for Ax, Temp in zip(Axes.ravel(), Temperatures):
        for Sim in SimIndices:
            Sub = Frame[Frame["TrajectoryId"] == f"{Temp}_{Sim}"]
            Ax.plot(Sub["Time_ps"], Sub[Column], "-",
                    color=TrajectoryColours[Sim], lw=1.0, alpha=0.9)
        Ax.set_title(f"{Temp} K", fontsize=BodyPt)
        Ax.set_xlim(0, Rf.HorizonPs)
        Ax.set_ylim(0, 1.02)
        Ax.tick_params(labelsize=BodyPt)
    for Ax in Axes[-1]:
        Ax.set_xlabel("Time (ps)", fontsize=BodyPt)
    for Ax in Axes[:, 0]:
        Ax.set_ylabel(YLabel, fontsize=BodyPt)
    Handles = [plt.Line2D([], [], color=TrajectoryColours[S], lw=1.4)
               for S in SimIndices]
    Fig.legend(Handles, [f"Trajectory {S}" for S in SimIndices],
               loc="lower center", ncol=4, fontsize=BodyPt, frameon=False,
               bbox_to_anchor=(0.5, 0.0))
    Fig.tight_layout(rect=[0, 0.07, 1, 1])
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


def Main():
    Parser = argparse.ArgumentParser(description=__doc__)
    Parser.add_argument("--refresh-gas-counts", action="store_true",
                        help="Rebuild gas-carbon counts from the existing state cache only")
    Parser.add_argument("--si-figures-only", action="store_true",
                        help="Only redraw the Supplementary figures from the stored PhiTimeseries.csv")
    Args = Parser.parse_args()
    os.makedirs(ResultsDir, exist_ok=True)
    if Args.si_figures_only:
        MakeSIFigures()
        return 0
    Frame = RefreshGasCountsFromCache() if Args.refresh_gas_counts else BuildTimeseries()
    SaveOutputs(Frame)
    MakeSIFigures()
    return 0


def BuildTimeseries():
    Tasks = [(T, S) for T in Temperatures for S in SimIndices]
    print("Building phi time series on the 50 fs grid (16 trajectories) ...",
          flush=True)
    from concurrent.futures import ProcessPoolExecutor, as_completed
    AllRows = []
    with ProcessPoolExecutor(max_workers=max(1, min(6, (os.cpu_count() or 2) - 2))) as Pool:
        Futures = {Pool.submit(ProcessTrajectory, T): T for T in Tasks}
        for Fut in as_completed(Futures):
            Rows = Fut.result()
            AllRows.extend(Rows)
            Last = Rows[-1]
            print(f"  {Last['TrajectoryId']}: PhiC(140) = {Last['PhiC']:.3f}, "
                  f"PhiO(140) = {Last['PhiO']:.3f} "
                  f"(chain {Last['NChainC']}, gasC {Last['NGasC']}, "
                  f"scaleO {Last['NScaleO']}, gasO {Last['NGasO']})", flush=True)

    Frame = pd.DataFrame(AllRows).sort_values(["Temperature", "TrajectoryId",
                                               "Time_ps"])
    return Frame


def SaveOutputs(Frame):
    # Horizon cross-checks against the Refresh140 module JSONs.
    M1 = json.load(open(os.path.join(ScriptDir, "Refresh140", "Module_1_Events.json")))
    M2 = json.load(open(os.path.join(ScriptDir, "Refresh140",
                                     "Module_2_Partitioning.json")))
    Final = Frame.sort_values("Time_ps").groupby("TrajectoryId").tail(1)
    Checks = []
    for Temp in Temperatures:
        End = Final[Final["Temperature"] == Temp]
        Retained = int(End["NChainC"].sum())
        GasC = int(End["NGasC"].sum())
        ScaleO = int(End["NScaleO"].sum())
        GasO = int(End["NGasO"].sum())
        Ev = M1["Totals"][str(Temp)]
        Checks.append({
            "Temperature": Temp,
            "NGasC_140_vs_Module1_CO_plus_CO2":
                [GasC, Ev["CO"] + Ev["CO2"], GasC == Ev["CO"] + Ev["CO2"]],
            "NChainC_140_vs_Module1_Retained":
                [Retained, Ev["Retained"], Retained == Ev["Retained"]],
            "NScaleO_140_vs_Module2_ScaleO":
                [ScaleO, M2["Table2"][str(Temp)]["ScaleO_140ps"],
                 ScaleO == M2["Table2"][str(Temp)]["ScaleO_140ps"]],
            "NGasO_140_vs_Module1_CO_plus_2CO2":
                [GasO, Ev["CO"] + 2 * Ev["CO2"],
                 GasO == Ev["CO"] + 2 * Ev["CO2"]],
        })
        print(f"  check {Temp} K: gasC {GasC} vs {Ev['CO'] + Ev['CO2']}, "
              f"retained {Retained} vs {Ev['Retained']}, "
              f"scaleO {ScaleO} vs {M2['Table2'][str(Temp)]['ScaleO_140ps']}, "
              f"gasO {GasO} vs {Ev['CO'] + 2 * Ev['CO2']}", flush=True)

    if not all(Value[-1] for Check in Checks for Key, Value in Check.items()
               if Key != "Temperature"):
        raise ValueError("Horizon counts do not match the audited reference")
    Frame.to_csv(os.path.join(ResultsDir, "PhiTimeseries.csv"), index=False)
    MakeTimeseriesFigure(Frame, "PhiC", "$\\phi_C$",
                         os.path.join(ResultsDir, "Fig_PhiC_Timeseries"))
    MakeTimeseriesFigure(Frame, "PhiO", "$\\phi_O$",
                         os.path.join(ResultsDir, "Fig_PhiO_Timeseries"))
    WriteSummary(Frame, Checks)
    print("\nDone. Results in", ResultsDir, flush=True)
    return 0


def WriteSummary(Frame, Checks):
    L = []
    L.append("# Phi retention-ratio time series (per trajectory, 0-140 ps)")
    L.append("")
    L.append("Source: Refresh140/FrameCache grid (50 fs, confirmed states) + "
             "RateAnalysis CSVs (gas-product timestamps). Seed 20260715; "
             "nothing stochastic. Definitions per the paper "
             "equations: PhiC = nChainC/(nChainC+nGasC), "
             "PhiO = nScaleO/(nScaleO+nGasO).")
    L.append("")
    L.append("Horizon values (140.0 ps), mean +/- SD over 4 trajectories:")
    L.append("| T (K) | PhiC(140) | PhiO(140) |")
    L.append("|---:|---|---|")
    Final = Frame.sort_values("Time_ps").groupby("TrajectoryId").tail(1)
    for Temp in Temperatures:
        End = Final[Final["Temperature"] == Temp]
        L.append(f"| {Temp} | {End['PhiC'].mean():.3f} +/- {End['PhiC'].std(ddof=1):.3f} "
                 f"| {End['PhiO'].mean():.3f} +/- {End['PhiO'].std(ddof=1):.3f} |")
    L.append("")
    L.append("Horizon cross-checks vs Refresh140 module JSONs (value, reference, match):")
    for C in Checks:
        L.append(f"- {C['Temperature']} K: gas C "
                 f"{C['NGasC_140_vs_Module1_CO_plus_CO2']}, retained "
                 f"{C['NChainC_140_vs_Module1_Retained']}, scale O "
                 f"{C['NScaleO_140_vs_Module2_ScaleO']}, gas O "
                 f"{C['NGasO_140_vs_Module1_CO_plus_2CO2']}")
    L.append("")
    L.append("Notes:")
    L.append("- nGasC counts confirmed-Gas carbons, excluding identities marked "
             "ArtifactRemoval in SubsequentChainFateResults/AuditMatches.csv. "
             "It is cumulative and can lag the audited gas census by the "
             "0.2 ps persistence delay; horizon agreement is enforced above.")
    L.append("- nGasO counts only PRODUCT oxygen (CO -> 1, CO2 -> 2, "
             "RateAnalysis timestamps, closed at 140.0 ps); free O2 in the "
             "void is supply, not product, and is excluded. Substitute "
             "definitions would shift PhiO globally.")
    L.append("- Ratios are NaN before the first event in their denominator "
             "(plotted as gaps at early times).")
    L.append("- These are the FRACTION-form definitions (bounded [0,1]) of the "
             "revised equations; they differ from the earlier Table 2 forms "
             "(PhiC = chain/gas ratio, PhiO = scaleO/(CO+2CO2)), which are "
             "recoverable from the stored counts in PhiTimeseries.csv.")
    L.append("- Figures: 2 x 2 panels (one per temperature), one line per "
             "trajectory (line style = run), 10 pt Times New Roman, 160 mm.")
    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print(Text)


# -----------------------------------------------------------------------------
# Figures for the Supplementary Information, drawn from PhiTimeseries.csv
# -----------------------------------------------------------------------------

SICsvPath = os.path.join(ScriptDir, "phiTimeseries", "PhiTimeseries.csv")
SIResultsDir = os.path.join(ScriptDir, "PhiTimeseriesSIFigureResults")

SIHorizonPs = 140.0

SIStyle = {
    "font.family": "Times New Roman", "font.size": 10,
    "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "stix",
}


def MakeSIFigure(Frame, Column, YLabel, LegendLoc, OutBase):
    Fig, Axes = plt.subplots(2, 2, figsize=(160 / 25.4, 110 / 25.4),
                             sharex=True, sharey=True)
    for PanelIdx, (Ax, Temp) in enumerate(zip(Axes.ravel(), Temperatures)):
        for Sim in SimIndices:
            Sub = Frame[Frame["TrajectoryId"] == f"{Temp}_{Sim}"]
            Ax.plot(Sub["Time_ps"], Sub[Column], "-",
                    color=TrajectoryColours[Sim], lw=1.0, alpha=0.9)
        Ax.set_title(f"({'abcd'[PanelIdx]}) {Temp} K", fontsize=10,
                     fontweight="bold", pad=4)
        Ax.set_xlim(0, SIHorizonPs)
        Ax.set_ylim(0, 1.02)
    for Ax in Axes[-1]:
        Ax.set_xlabel("Time (ps)")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel(YLabel)
    Handles = [plt.Line2D([], [], color=TrajectoryColours[S], lw=1.4)
               for S in SimIndices]
    Axes[0, 0].legend(Handles, [f"Trajectory {S}" for S in SimIndices],
                      loc=LegendLoc, ncol=2, frameon=False,
                      handlelength=1.6, columnspacing=1.0)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


def MakeSICombinedFigure(Frame, OutBase):
    """Both ratios in one 2x2 grid: PhiC in trajectory colours, PhiO in
    thin grey (per-trajectory identity not used by the text for PhiO)."""
    GreyO = "#909090"
    Fig, Axes = plt.subplots(2, 2, figsize=(160 / 25.4, 110 / 25.4),
                             sharex=True, sharey=True)
    for PanelIdx, (Ax, Temp) in enumerate(zip(Axes.ravel(), Temperatures)):
        for Sim in SimIndices:
            Sub = Frame[Frame["TrajectoryId"] == f"{Temp}_{Sim}"]
            Ax.plot(Sub["Time_ps"], Sub["PhiO"], "-", color=GreyO,
                    lw=0.7, alpha=0.8)
            Ax.plot(Sub["Time_ps"], Sub["PhiC"], "-",
                    color=TrajectoryColours[Sim], lw=1.0, alpha=0.9)
        Ax.set_title(f"({'abcd'[PanelIdx]}) {Temp} K", fontsize=10,
                     fontweight="bold", pad=4)
        Ax.set_xlim(0, SIHorizonPs)
        Ax.set_ylim(0, 1.02)
    for Ax in Axes[-1]:
        Ax.set_xlabel("Time (ps)")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel(r"$\phi$")
    Handles = ([plt.Line2D([], [], color=TrajectoryColours[S], lw=1.4)
                for S in SimIndices]
               + [plt.Line2D([], [], color=GreyO, lw=1.4)])
    Labels = [rf"$\phi_C$, trajectory {S}" for S in SimIndices] \
        + [r"$\phi_O$, all trajectories"]
    Fig.legend(Handles, Labels, loc="lower center", ncol=5, frameon=False,
               handlelength=1.4, columnspacing=0.8, fontsize=8,
               bbox_to_anchor=(0.5, 0.0))
    Fig.tight_layout(rect=[0, 0.05, 1, 1])
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


def MakeSIFigures():
    with plt.rc_context():
        matplotlib.rc_file_defaults()
        plt.rcParams.update(SIStyle)
        os.makedirs(SIResultsDir, exist_ok=True)
        Frame = pd.read_csv(SICsvPath)
        MakeSIFigure(Frame, "PhiC", r"$\phi_C$", "upper right",
                   os.path.join(SIResultsDir, "PhiCTimeseries"))
        MakeSIFigure(Frame, "PhiO", r"$\phi_O$", "lower right",
                   os.path.join(SIResultsDir, "PhiOTimeseries"))
        MakeSICombinedFigure(Frame, os.path.join(SIResultsDir, "PhiTimeseries"))
        print("Wrote figures to", SIResultsDir)


if __name__ == "__main__":
    sys.exit(Main())
