"""
EaBootstrap.py
=======================================================================
Bootstrap confidence intervals for the two Arrhenius activation
energies quoted in the paper:

  1. Commitment Ea (Sec 2.3): trajectory-stratified two-level bootstrap
     mirroring CommitmentCompetingRisks.py exactly (4 trajectories per
     temperature drawn with replacement, then the 48 subjects of each
     drawn trajectory resampled with replacement; artifact atoms
     censored at deletion). Per resample, lambda_total per temperature,
     then the Arrhenius refit of ArrheniusOf (unweighted OLS of
     ln lambda vs 1/T).
  2. Net-mass Ea (Sec 2.4): same outer draws (paired), net mass rate
     per resample. Because interpolation to the common 0.1 ps grid,
     the 2 ps moving average, cell summing, and the 20-140 ps slope
     fit are all linear in the data, the resampled net rate equals the
     sum of precomputed per-trajectory rate contributions, reproducing
     the DiscussionSupportNotes.py pipeline exactly.
  3. Paired difference: P(Ea_mass > Ea_commitment) from the shared
     outer draws.

Sanity gates before bootstrapping: point lambdas must reproduce
CommitmentFits.json and point net rates must reproduce Notes.json.

Seed 20260715, 10000 resamples.
Output: EaBootstrapResults/ (Results.json, console summary).
"""

import os
import json

import numpy as np
import pandas as pd

ScriptDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
EventTablePath = os.path.join(ScriptDir, "Refresh140", "Events", "EventTableV2.csv")
PhiPath = os.path.join(ScriptDir, "phiTimeseries", "PhiTimeseries.csv")
FitsPath = os.path.join(ScriptDir, "CompetingRisksResults", "CommitmentFits.json")
NotesPath = os.path.join(ScriptDir, "DiscussionSupportResults", "Notes.json")
ResultsDir = os.path.join(ScriptDir, "EaBootstrapResults")

Temperatures = [873, 973, 1073, 1273]
Sims = [1, 2, 3, 4]
CarbonsPerCell = 48
HorizonPs = 140.0
ArtifactCensor = {"973_1": [31.672, 33.944], "1273_3": [3.504, 5.440, 16.704]}
Seed = 20260715
NBoot = 10000
KB_EV = 8.617333e-5
MassO, MassC = 15.999, 12.011
TgaFitPs = (20.0, 140.0)
SmoothWin = 21


def BuildSubjects(Df):
    """(Durations, Cause) per trajectory, Cause in {0 censored, 1 gas, 2 chain}."""
    ByTraj = {}
    for T in Temperatures:
        for S in Sims:
            Traj = f"{T}_{S}"
            Sub = Df[Df["TrajectoryId"] == Traj]
            Gas = Sub[Sub["Outcome"] == "Gas"]["Time_ps"].to_numpy()
            Chain = Sub[Sub["Outcome"] == "Chain"]["Time_ps"].to_numpy()
            Art = np.array(ArtifactCensor.get(Traj, []))
            NUnc = CarbonsPerCell - len(Gas) - len(Chain) - len(Art)
            Durations = np.concatenate([Gas, Chain, Art, np.full(NUnc, HorizonPs)])
            Cause = np.concatenate([np.ones(len(Gas)), np.full(len(Chain), 2),
                                    np.zeros(len(Art) + NUnc)])
            ByTraj[Traj] = (Durations, Cause)
    return ByTraj


def PerTrajectoryMassRate(Phi):
    """Slope contribution of each trajectory to the summed, smoothed net
    mass signal over 20-140 ps (linear pipeline, so contributions add)."""
    Tt = np.arange(0.0, 140.0001, 0.1)
    Sel = (Tt >= TgaFitPs[0]) & (Tt <= TgaFitPs[1])
    Kernel = np.ones(SmoothWin) / SmoothWin
    Rates = {}
    for T in Temperatures:
        for S in Sims:
            Tr = Phi[Phi.TrajectoryId == f"{T}_{S}"].sort_values("Time_ps")
            O = np.interp(Tt, Tr.Time_ps, Tr.NScaleO)
            C = np.interp(Tt, Tr.Time_ps, Tr.NGasC)
            Dm = MassO * O - MassC * C
            Sm = np.convolve(np.pad(Dm, SmoothWin // 2, mode="edge"),
                             Kernel, "valid")
            Rates[f"{T}_{S}"] = float(np.polyfit(Tt[Sel], Sm[Sel], 1)[0])
    return Rates


def EaOf(Vals):
    InvT = 1.0 / np.array(Temperatures, float)
    Slope, _ = np.polyfit(InvT, np.log(Vals), 1)
    return -Slope * KB_EV


def Main():
    os.makedirs(ResultsDir, exist_ok=True)
    Subjects = BuildSubjects(pd.read_csv(EventTablePath))
    MassRates = PerTrajectoryMassRate(pd.read_csv(PhiPath))

    # ---- sanity gates --------------------------------------------------
    PointLam, PointNet = [], []
    for T in Temperatures:
        D = np.concatenate([Subjects[f"{T}_{S}"][0] for S in Sims])
        C = np.concatenate([Subjects[f"{T}_{S}"][1] for S in Sims])
        PointLam.append((C > 0).sum() / D.sum())
        PointNet.append(sum(MassRates[f"{T}_{S}"] for S in Sims))
    Ref = json.load(open(FitsPath))
    RefEa = Ref["Arrhenius"]["total"]["Ea_eV"]
    Notes = json.load(open(NotesPath))
    RefNet = [Notes["SyntheticTGA"]["Slopes"][str(T)]["Slope_uPerPs"]
              for T in Temperatures]
    EaC0, EaM0 = EaOf(PointLam), EaOf(PointNet)
    print(f"gate commitment Ea: {EaC0:.4f} eV (stored {RefEa:.4f})")
    print(f"gate net rates: {[f'{V:.2f}' for V in PointNet]} "
          f"(stored {[f'{V:.2f}' for V in RefNet]}), Ea {EaM0:.4f} eV")
    assert abs(EaC0 - RefEa) < 1e-6, "commitment reproduction failed"
    assert all(abs(A - B) < 0.05 for A, B in zip(PointNet, RefNet)), \
        "net-rate reproduction failed"

    # ---- paired bootstrap ----------------------------------------------
    Rng = np.random.default_rng(Seed)
    EaCommit = np.empty(NBoot)
    EaMass = np.empty(NBoot)
    for B in range(NBoot):
        LamB, NetB = [], []
        for T in Temperatures:
            Draw = Rng.choice(4, size=4, replace=True)
            DList, CList, Net = [], [], 0.0
            for Ti in Draw:
                Traj = f"{T}_{Sims[Ti]}"
                D, C = Subjects[Traj]
                Idx = Rng.integers(0, len(D), size=len(D))
                DList.append(D[Idx])
                CList.append(C[Idx])
                Net += MassRates[Traj]
            Dd = np.concatenate(DList)
            Cc = np.concatenate(CList)
            LamB.append(max((Cc > 0).sum(), 1) / Dd.sum())
            NetB.append(Net)
        EaCommit[B] = EaOf(LamB)
        EaMass[B] = EaOf(NetB)

    def Stats(A):
        return dict(Mean=float(A.mean()), SD=float(A.std(ddof=1)),
                    CI95=[float(np.percentile(A, 2.5)),
                          float(np.percentile(A, 97.5))])

    Diff = EaMass - EaCommit
    Out = dict(PointEaCommit_eV=float(EaC0), PointEaMass_eV=float(EaM0),
               BootEaCommit=Stats(EaCommit), BootEaMass=Stats(EaMass),
               BootDiff=Stats(Diff),
               P_MassGreaterCommit=float(np.mean(Diff > 0)),
               NBoot=NBoot, Seed=Seed,
               Method="two-level (trajectory + subject) for commitment, "
                      "trajectory-level for mass rates, shared outer draws")
    json.dump(Out, open(os.path.join(ResultsDir, "Results.json"), "w"),
              indent=1)

    print(f"\ncommitment Ea = {EaC0:.3f} eV, bootstrap 95% CI "
          f"[{Out['BootEaCommit']['CI95'][0]:.3f}, "
          f"{Out['BootEaCommit']['CI95'][1]:.3f}], SD {Out['BootEaCommit']['SD']:.3f}")
    print(f"net-mass Ea = {EaM0:.3f} eV, bootstrap 95% CI "
          f"[{Out['BootEaMass']['CI95'][0]:.3f}, "
          f"{Out['BootEaMass']['CI95'][1]:.3f}], SD {Out['BootEaMass']['SD']:.3f}")
    print(f"difference (mass - commitment) 95% CI "
          f"[{Out['BootDiff']['CI95'][0]:.3f}, {Out['BootDiff']['CI95'][1]:.3f}], "
          f"P(mass > commitment) = {Out['P_MassGreaterCommit']:.4f}")
    print(f"Written: {ResultsDir}")


if __name__ == "__main__":
    Main()
