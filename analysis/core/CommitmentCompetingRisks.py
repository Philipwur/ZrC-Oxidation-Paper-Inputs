"""
CommitmentCompetingRisks.py
=======================================================================
Fate-commitment competing-risks analysis of carbon liberation,
the COMPLETE-EVENTS variant: no dependence on traced onsets.

Model: two first-order channels drain the lattice-carbon reservoir.
    lattice C --(lambda_g)--> gas (commitment = desorption time)
    lattice C --(lambda_c)--> chains (commitment = confirmed chain entry)
Events from Refresh140/Events/EventTableV2.csv (audited, post-correction):
gas rows at desorption timestamps, chain rows at last-chain-entry
timestamps. Risk set per temperature: 192 lattice carbons. A carbon leaves
the risk set at its first commitment. Artifact-deleted carbons censored at
deletion; never-committed carbons censored at 140.0 ps. Ever-chained
carbons that ended re-lattice-bound (3-4 per T) are not in the event table
and enter as uncommitted (disclosed).

CHECKS this run performs (decision gates for restructuring Sec 2.3):
 1. Cause-specific time dependence: Weibull shape per cause per T.
    Pass = the 1273 K front-loading reproduces in the gas channel.
 2. Trajectory-stratified bootstrap (10,000, seed 20260715) over the
    channel rates: P(gas peak at 973 K), P(chain rise), P(total monotonic),
    plus rho_gas(1273) bootstrap (2,000 Weibull refits).

Outputs: CompetingRisksResults/ (fits JSON, bootstrap JSON, console).
"""

import os
import json
import argparse

import numpy as np
import pandas as pd
from scipy.stats import chi2

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
from SurvivalAnalysis import FitExponentialRate, FitWeibull, LikelihoodRatioTest

ScriptDir = os.path.dirname(os.path.abspath(__file__))
ResultsDir = os.path.join(ScriptDir, "CompetingRisksResults")
EventTablePath = os.path.join(ScriptDir, "Refresh140", "Events", "EventTableV2.csv")

Temperatures = [873, 973, 1073, 1273]
CarbonsPerCell = 48
HorizonPs = 140.0
ArtifactCensor = {"973_1": [31.672, 33.944], "1273_3": [3.504, 5.440, 16.704]}
Seed = 20260715
NBootRates = 10000
NBootRho = 2000
KB_EV = 8.617333e-5


def BuildSubjects(Df, Temp):
    """Per-trajectory subject arrays: (Durations, Cause) with Cause in
    {0: censored, 1: gas, 2: chain}."""
    ByTraj = {}
    for Sim in (1, 2, 3, 4):
        Traj = f"{Temp}_{Sim}"
        Sub = Df[Df["TrajectoryId"] == Traj]
        Gas = Sub[Sub["Outcome"] == "Gas"]["Time_ps"].to_numpy()
        Chain = Sub[Sub["Outcome"] == "Chain"]["Time_ps"].to_numpy()
        Art = np.array(ArtifactCensor.get(Traj, []))
        NCommitted = len(Gas) + len(Chain) + len(Art)
        NUncommitted = CarbonsPerCell - NCommitted
        Durations = np.concatenate([Gas, Chain, Art,
                                    np.full(NUncommitted, HorizonPs)])
        Cause = np.concatenate([np.full(len(Gas), 1), np.full(len(Chain), 2),
                                np.zeros(len(Art) + NUncommitted)])
        ByTraj[Traj] = (Durations, Cause)
    return ByTraj


def CauseFits(Durations, Cause):
    """Exponential + Weibull per cause (competing event treated as censoring
    at its commitment time, the standard cause-specific construction)."""
    Out = {}
    for CauseId, Name in ((1, "gas"), (2, "chain")):
        Events = (Cause == CauseId).astype(float)
        Exp = FitExponentialRate(Durations, Events)
        Wei = FitWeibull(Durations, Events)
        _, LrtP = LikelihoodRatioTest(Durations, Events, Wei)
        Out[Name] = dict(Exponential=Exp, Weibull=Wei, LRT_p=LrtP)
    EventsAny = (Cause > 0).astype(float)
    ExpT = FitExponentialRate(Durations, EventsAny)
    WeiT = FitWeibull(Durations, EventsAny)
    _, LrtPT = LikelihoodRatioTest(Durations, EventsAny, WeiT)
    Out["total"] = dict(Exponential=ExpT, Weibull=WeiT, LRT_p=LrtPT)
    return Out


def ArrheniusOf(Lams):
    InvT = 1.0 / np.array(Temperatures, float)
    Slope, Intercept = np.polyfit(InvT, np.log(Lams), 1)
    Pred = np.exp(Intercept + Slope * InvT)
    R2 = 1 - (np.sum((np.log(Lams) - np.log(Pred)) ** 2)
              / np.sum((np.log(Lams) - np.mean(np.log(Lams))) ** 2))
    return dict(Ea_eV=float(-Slope * KB_EV), R2=float(R2),
                Slope=float(Slope), Intercept=float(Intercept))


def MakeFigure(OutBase):
    """Main-text fig:Survival (publication layout).

    (a) Commitment rates vs 1000/T: committed total (filled squares,
        legend "Committed lambda" - the fitted
        events are commitments, not geometric liberation onsets) on
        its Arrhenius fit, gas and chain channels (open circles/triangles).
        Legend lower left (y axis extended downward to make room); tag top
        right. Error bars = exact 95 % CIs (chi-square) - stated in caption.
    (b) Cumulative committed fraction for gas, chains and their total at
        1273 K. Outcome legend at upper right, without a background;
        shade the initial 0-40 ps interval. These are cumulative
        counts divided by the starting inventory, not time-local hazards.
    Reads CommitmentFits.json + EventTableV2.csv. sn-jnl textwidth
    131.2 mm, fonts at true size.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter

    with open(os.path.join(ResultsDir, "CommitmentFits.json")) as Fh:
        Fits = json.load(Fh)
    Ev = pd.read_csv(EventTablePath)

    plt.rcParams.update({
        "font.family": "Times New Roman", "font.size": 10,
        "axes.labelsize": 10, "legend.fontsize": 8.5,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
    })
    CGas, CChain, CTot = "#d95f02", "#1b9e77", "0.15"

    Fig, Axes = plt.subplots(1, 2, figsize=(131.2 / 25.4, 62 / 25.4))

    # ---- (a) rates ---------------------------------------------------------
    Ax = Axes[0]
    TGrid = np.linspace(850, 1300, 200)
    A = Fits["Arrhenius"]["total"]
    Ax.plot(1000 / TGrid, np.exp(A["Intercept"] + A["Slope"] / TGrid) * 1e3,
            color="0.55", lw=1.1, ls="-", zorder=1)
    Series = {"total": ("s", CTot, CTot, 0.0),
              "gas": ("o", CGas, "white", -0.007),
              "chain": ("^", CChain, "white", 0.007)}
    for Name, (M, C, Fc, Off) in Series.items():
        for T in Temperatures:
            E = Fits["Fits"][str(T)][Name]["Exponential"]
            V = E["Lambda"] * 1e3
            Ax.errorbar(1000 / T + Off, V,
                        yerr=[[V - E["CI_Lower"] * 1e3],
                              [E["CI_Upper"] * 1e3 - V]],
                        fmt=M, ms=5.5, color=C, mfc=Fc, capsize=2.5,
                        lw=1.3, zorder=3)
    Ax.errorbar([], [], fmt="s", ms=5.5, color=CTot,
                label="Total")
    Ax.errorbar([], [], fmt="o", ms=5.5, color=CGas, mfc="white",
                label="Gas")
    Ax.errorbar([], [], fmt="^", ms=5.5, color=CChain, mfc="white",
                label="Chains")
    Ax.set_yscale("log")
    Ax.yaxis.set_major_locator(FixedLocator([1, 2, 3, 5, 7]))
    Ax.yaxis.set_minor_locator(NullLocator())
    Ax.yaxis.set_major_formatter(ScalarFormatter())
    Ax.set_ylim(0.5, 8.2)
    Ax.set_xlabel(r"$1000/T$ (K$^{-1}$)")
    Ax.set_ylabel(r"$\lambda$ ($10^{-3}$ ps$^{-1}$)")
    Ax.legend(frameon=False, loc="lower left", handlelength=1.2)
    Ax.text(0.97, 0.97, "(a)", transform=Ax.transAxes, va="top", ha="right",
            fontweight="bold")

    # ---- (b) cumulative committed fraction ---------------------------------
    Ax = Axes[1]
    Ax.axvspan(0, 40, facecolor="0.93", edgecolor="none", zorder=0)
    Ax.axvline(40, color="0.65", lw=0.7, ls=":", zorder=1)
    S = Ev[Ev["Temperature"] == 1273]
    GasTimes = np.sort(S.loc[S["Outcome"] == "Gas", "Time_ps"].to_numpy())
    ChainTimes = np.sort(S.loc[S["Outcome"] == "Chain", "Time_ps"].to_numpy())
    TotalTimes = np.sort(np.concatenate([GasTimes, ChainTimes]))
    Grid = np.unique(np.r_[0, TotalTimes, HorizonPs])
    assert np.array_equal(np.searchsorted(TotalTimes, Grid, side="right"),
                          np.searchsorted(GasTimes, Grid, side="right") +
                          np.searchsorted(ChainTimes, Grid, side="right"))
    for Times, C, Width in ((GasTimes, CGas, 1.4),
                            (ChainTimes, CChain, 1.4),
                            (TotalTimes, CTot, 1.8)):
        Tt = np.concatenate([[0], Times, [HorizonPs]])
        Ff = np.concatenate([[0], np.arange(1, len(Times) + 1),
                             [len(Times)]]) / 192.0
        Ax.step(Tt, Ff, where="post", color=C, lw=Width)
    Ax.set_xlim(0, HorizonPs)
    Ax.set_ylim(0, 0.75)
    Ax.set_xticks([0, 40, 80, 120, 140])
    Ax.set_yticks([0, 0.2, 0.4, 0.6])
    Ax.set_xlabel("Time (ps)")
    Ax.set_ylabel("C fraction committed")
    Ax.legend(handles=[
        Line2D([], [], color=CTot, lw=1.8, label="Total"),
        Line2D([], [], color=CGas, lw=1.4, label="Gas"),
        Line2D([], [], color=CChain, lw=1.4, label="Chains")],
        frameon=False, loc="upper right", handlelength=1.3,
        borderaxespad=0.4, borderpad=0.2, labelspacing=0.25)
    Ax.text(20, 0.30, "Initial\ntransient", ha="center", va="center",
            fontsize=8, color="0.4")
    Ax.text(0.97, 0.03, "(b)", transform=Ax.transAxes, va="bottom",
            ha="right", fontweight="bold")

    Fig.tight_layout(pad=0.4)
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)
    print(f"Figure written: {OutBase}.png/.pdf")


def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--figure-only", action="store_true",
                        help="Regenerate the figure from stored fits.")
    Args = Parser.parse_args()
    if Args.figure_only:
        MakeFigure(os.path.join(ResultsDir, "Fig_CommitmentRates"))
        return

    os.makedirs(ResultsDir, exist_ok=True)
    Df = pd.read_csv(EventTablePath)
    Rng = np.random.default_rng(Seed)

    Subjects = {T: BuildSubjects(Df, T) for T in Temperatures}

    # ---------------- observed fits ----------------
    Fits = {}
    print(f"{'T':>5} {'ch':>6} {'n':>4} {'lam(e-3/ps)':>16} "
          f"{'rho':>20} {'LRT p':>9}")
    for T in Temperatures:
        AllD = np.concatenate([Subjects[T][K][0] for K in Subjects[T]])
        AllC = np.concatenate([Subjects[T][K][1] for K in Subjects[T]])
        Fits[T] = CauseFits(AllD, AllC)
        for Name in ("total", "gas", "chain"):
            F = Fits[T][Name]
            E, W = F["Exponential"], F["Weibull"]
            print(f"{T:>5} {Name:>6} {E['Events']:>4} "
                  f"{E['Lambda']*1e3:>6.2f} [{E['CI_Lower']*1e3:.2f},{E['CI_Upper']*1e3:.2f}] "
                  f"{W['Rho']:>6.3f} [{W['Rho_CI_Lower']:.3f},{W['Rho_CI_Upper']:.3f}] "
                  f"{F['LRT_p']:>9.3g}")

    Arrhenius = {Name: ArrheniusOf([Fits[T][Name]["Exponential"]["Lambda"]
                                    for T in Temperatures])
                 for Name in ("total", "gas", "chain")}
    for Name, A in Arrhenius.items():
        print(f"Arrhenius {Name}: Ea = {A['Ea_eV']:.3f} eV, R2 = {A['R2']:.3f}")

    # ---------------- bootstrap of channel rates ----------------
    print(f"\nBootstrap rates ({NBootRates} two-level resamples)...")
    TrajKeys = {T: list(Subjects[T].keys()) for T in Temperatures}
    BootLam = {T: {"gas": np.empty(NBootRates), "chain": np.empty(NBootRates),
                   "total": np.empty(NBootRates)} for T in Temperatures}
    for B in range(NBootRates):
        for T in Temperatures:
            DrawTraj = Rng.choice(4, size=4, replace=True)
            DList, CList = [], []
            for Ti in DrawTraj:
                D, C = Subjects[T][TrajKeys[T][Ti]]
                Idx = Rng.integers(0, len(D), size=len(D))
                DList.append(D[Idx]); CList.append(C[Idx])
            Dd = np.concatenate(DList); Cc = np.concatenate(CList)
            Texp = Dd.sum()
            BootLam[T]["gas"][B] = (Cc == 1).sum() / Texp
            BootLam[T]["chain"][B] = (Cc == 2).sum() / Texp
            BootLam[T]["total"][B] = (Cc > 0).sum() / Texp

    def P(Cond):
        return float(np.mean(Cond))

    Claims = dict(
        GasPeak973_vs873=P(BootLam[973]["gas"] > BootLam[873]["gas"]),
        GasPeak973_vs1273=P(BootLam[973]["gas"] > BootLam[1273]["gas"]),
        Gas973_vs1073=P(BootLam[973]["gas"] > BootLam[1073]["gas"]),
        ChainDip973_vs873=P(BootLam[973]["chain"] < BootLam[873]["chain"]),
        ChainDip973_vs1073=P(BootLam[973]["chain"] < BootLam[1073]["chain"]),
        ChainRise973_to_1273=P(BootLam[1273]["chain"] > BootLam[973]["chain"]),
        TotalMonotonic=P((BootLam[973]["total"] > BootLam[873]["total"])
                         & (BootLam[1073]["total"] > BootLam[973]["total"])
                         & (BootLam[1273]["total"] > BootLam[1073]["total"])),
        GasOvertakesChain973=P(BootLam[973]["gas"] > BootLam[973]["chain"]),
        ChainOverGas1273=P(BootLam[1273]["chain"] > BootLam[1273]["gas"]),
    )
    print("Bootstrap claim probabilities:")
    for K, V in Claims.items():
        print(f"  {K}: {V:.3f}")

    # ---------------- bootstrap of rho_gas(1273) ----------------
    print(f"\nBootstrap rho_gas(1273 K) ({NBootRho} Weibull refits)...")
    RhoBoot = []
    Fails = 0
    for B in range(NBootRho):
        DrawTraj = Rng.choice(4, size=4, replace=True)
        DList, CList = [], []
        for Ti in DrawTraj:
            D, C = Subjects[1273][TrajKeys[1273][Ti]]
            Idx = Rng.integers(0, len(D), size=len(D))
            DList.append(D[Idx]); CList.append(C[Idx])
        Dd = np.concatenate(DList); Ev = (np.concatenate(CList) == 1).astype(float)
        if Ev.sum() < 3:
            Fails += 1
            continue
        try:
            W = FitWeibull(Dd, Ev)
            RhoBoot.append(W["Rho"])
        except Exception:
            Fails += 1
    RhoBoot = np.array(RhoBoot)
    RhoSummary = dict(
        N=len(RhoBoot), Fails=Fails,
        CI=[float(np.percentile(RhoBoot, 2.5)), float(np.percentile(RhoBoot, 97.5))],
        P_below_1=float(np.mean(RhoBoot < 1.0)),
    )
    print(f"  rho_gas(1273) bootstrap: CI [{RhoSummary['CI'][0]:.3f}, "
          f"{RhoSummary['CI'][1]:.3f}], P(rho<1) = {RhoSummary['P_below_1']:.3f}, "
          f"fails {Fails}")

    Out = dict(Fits=Fits, Arrhenius=Arrhenius, Claims=Claims,
               RhoGas1273Bootstrap=RhoSummary,
               BootLamCI={str(T): {K: [float(np.percentile(BootLam[T][K] * 1e3, 2.5)),
                                       float(np.percentile(BootLam[T][K] * 1e3, 97.5))]
                                   for K in ("gas", "chain", "total")}
                          for T in Temperatures})
    with open(os.path.join(ResultsDir, "CommitmentFits.json"), "w") as Fh:
        json.dump(Out, Fh, indent=1, default=float)
    print(f"Written: {ResultsDir}")
    MakeFigure(os.path.join(ResultsDir, "Fig_CommitmentRates"))


if __name__ == "__main__":
    Main()
