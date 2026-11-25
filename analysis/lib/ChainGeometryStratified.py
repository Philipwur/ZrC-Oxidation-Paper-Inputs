"""
ChainGeometryStratified.py
=======================================================================
Length-stratified and thermally controlled re-analysis of interfacial carbon
chain geometry. Separates three confounded effects in the ChainTopology result:
    (a) chain length  - branching/rings need length >= 4, so a length-mix
                         difference alone can produce the group contrast;
    (b) thermal vibration - instantaneous angles are bounded at 180 deg, so a
                         hotter identical structure reads as more bent;
    (c) genuine structural difference.

Decisive output: a collapse test of geometry vs chain length. If at fixed length
the four temperature series lie on a common curve, geometry is a function of size
(single condensation route, different progress); if the Above group is
systematically more bent/branched at fixed length, temperature acts on structure.

Descriptive only: NO p-values or significance claims. Strata with < 20 angle
samples or < 5 distinct chains are printed but flagged as too thin to quote.

Reuse
-----
Branch statistics come from the cached ChainTopologyResults/ChainInventory.csv
(per-frame-chain Length + NBranchPoints). Angles-tagged-by-length and persistent
triplets are not cached, so the final-10% window is re-streamed with the SAME
imported detection stack, seed 20260715, 50 fs subsampling, and 0.2 ps burn-in.

Time-averaged angles (thermal control)
--------------------------------------
For each persistent triplet (a center carbon bonded to the same two carbons over a
contiguous run of >= 2 ps at the 50 fs sampling), the angle is measured on the
time-averaged geometry. Averaging is done on the two minimum-image C-C bond
vectors (center->n1, center->n2) per frame, then the angle between the means:
this is algebraically the angle of the unwrapped per-atom mean positions
(mean(n1 - c) = mean n1_unwrapped - mean c_unwrapped) and NEVER averages wrapped
coordinates across a periodic boundary.

Usage
-----
    python ChainGeometryStratified.py            # all 16 trajectories, 6 workers
    python ChainGeometryStratified.py --only 1273_3
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
sys.path.insert(0, ScriptDir)

# Reuse the streaming + detection stack from the previous scripts (import, not copy).
from ChainTopology import (
    ScanFrameOffsets, ReadFramesFrom, ComputeBonds, ConnectedComponents,
    MinImageVector, XyzPath, GroupOf, Temperatures, SimIndices,
    WindowStartPct, SubsampleFs, BurnInFs, GroupBelow, GroupAbove,
)
from BootstrapCalculation import (
    CartesianToFractional, MapAtomIdentities, DetermineInstantaneousStates,
    TotalStartC, StatePersistenceTimePs,
)

ResultsDir = os.path.join(ScriptDir, "ChainGeometryResults")
InventoryCsv = os.path.join(ScriptDir, "ChainTopologyResults", "ChainInventory.csv")

RandomSeed        = 20260715
PersistRunMinPs   = 2.0          # minimum contiguous run for a persistent triplet
ThinAngleSamples  = 20           # < this many angle samples -> too thin to quote
ThinChainCount    = 5            # < this many distinct chains -> too thin to quote
AngleStrataL      = [3, 4]       # lengths populated in both groups (Step 1a)

TemperatureStyle = {
    873:  {'color': '#1f77b4', 'marker': 'o', 'group': GroupBelow},
    973:  {'color': '#17becf', 'marker': 's', 'group': GroupBelow},
    1073: {'color': '#ff7f0e', 'marker': '^', 'group': GroupAbove},
    1273: {'color': '#d62728', 'marker': 'D', 'group': GroupAbove},
}


def AngleBetween(V1, V2):
    Cos = np.dot(V1, V2) / (np.linalg.norm(V1) * np.linalg.norm(V2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(Cos, -1.0, 1.0))))


# =============================================================================
#                       PER-FRAME GEOMETRY (angles + triplets)
# =============================================================================
def AnalyseFrameForGeometry(Atoms, FracCoords, Lattice, ConfirmedStates, TrackedIDs, TimePs):
    """
    Returns (AngleRecords, CurrentTriplets) for one subsampled frame.
      AngleRecords    : list of (TimePs, Angle, Length) for degree-2 chain carbons.
      CurrentTriplets : {(CenterAtomID, frozenset{N1AtomID, N2AtomID}): (V1, V2, Angle, Length)}
                        over all chain carbons with >= 2 chain-C neighbours (all pairs).
    """
    Bond = ComputeBonds(Atoms, FracCoords, Lattice)
    CarbonFullIdx = np.where(Atoms == 'C')[0]
    IsCarbon = (Atoms == 'C')

    ChainFull = []
    FullToAtomID = {}
    for i in range(len(TrackedIDs)):
        AtomID = int(TrackedIDs[i])
        if ConfirmedStates.get(AtomID) == 'Chain':
            Full = int(CarbonFullIdx[i])
            ChainFull.append(Full)
            FullToAtomID[Full] = AtomID
    ChainSet = set(ChainFull)

    Adjacency = {F: [] for F in ChainFull}
    for A in ChainFull:
        for B in np.where(Bond[A] & IsCarbon)[0]:
            if int(B) in ChainSet:
                Adjacency[A].append(int(B))

    LengthOf = {}
    for Comp in ConnectedComponents(ChainFull, Adjacency):
        for F in Comp:
            LengthOf[F] = len(Comp)

    AngleRecords = []
    CurrentTriplets = {}
    for A in ChainFull:
        Neigh = Adjacency[A]
        if len(Neigh) < 2:
            continue
        FracA = FracCoords[A]
        for J in range(len(Neigh)):
            for K in range(J + 1, len(Neigh)):
                N1, N2 = Neigh[J], Neigh[K]
                V1 = MinImageVector(FracA, FracCoords[N1], Lattice)
                V2 = MinImageVector(FracA, FracCoords[N2], Lattice)
                Ang = AngleBetween(V1, V2)
                Key = (FullToAtomID[A], frozenset((FullToAtomID[N1], FullToAtomID[N2])))
                CurrentTriplets[Key] = (V1, V2, Ang, LengthOf[A])
        if len(Neigh) == 2:     # Step-1 angle definition: exactly 2 C neighbours
            V1 = MinImageVector(FracA, FracCoords[Neigh[0]], Lattice)
            V2 = MinImageVector(FracA, FracCoords[Neigh[1]], Lattice)
            AngleRecords.append((TimePs, AngleBetween(V1, V2), LengthOf[A]))

    return AngleRecords, CurrentTriplets


def FinalizeTriplet(State, Temp, Group):
    """Emit a persistent-triplet record if the run reached PersistRunMinPs."""
    RunPs = (State['LastTime'] - State['StartTime']) / 1000.0
    if RunPs < PersistRunMinPs:
        return None
    MeanV1 = State['SumV1'] / State['Count']
    MeanV2 = State['SumV2'] / State['Count']
    Inst = np.array(State['Inst'])
    return {
        'Temperature': Temp, 'Group': Group,
        'RunLengthPs': RunPs, 'RunFrames': State['Count'], 'StartL': State['StartL'],
        'TimeAveragedAngle': AngleBetween(MeanV1, MeanV2),
        'InstMean': float(Inst.mean()), 'InstSD': float(Inst.std(ddof=1)) if Inst.size > 1 else 0.0,
    }


# =============================================================================
#                            PER-TRAJECTORY WORKER
# =============================================================================
def ProcessTrajectory(TaskArgs):
    Temp, Sim = TaskArgs
    Path = XyzPath(Temp, Sim)
    TrajId = f"{Temp}_{Sim}"
    Group = GroupOf(Temp)

    NFrames, Offsets = ScanFrameOffsets(Path)
    IdxStart = int(NFrames * WindowStartPct)
    StartOffset = Offsets[IdxStart]

    TrackedIDs = np.arange(TotalStartC)
    ConfirmedStates = {ID: 'Lattice' for ID in range(TotalStartC)}
    PendingStates = {}
    PrevFracC = None

    WindowStartTime = None
    LastRecordedTime = None
    NAnalysed = 0

    AngleRecords = []
    ActiveTriplets = {}
    FinalTriplets = []

    for TimeFs, Atoms, Coords, Lattice in ReadFramesFrom(Path, StartOffset):
        if WindowStartTime is None:
            WindowStartTime = TimeFs
        FracCoords = CartesianToFractional(Coords, Lattice)
        CurrFracC = FracCoords[Atoms == 'C']
        if PrevFracC is not None:
            TrackedIDs = MapAtomIdentities(PrevFracC, CurrFracC, TrackedIDs, Lattice)
        else:
            if len(CurrFracC) < len(TrackedIDs):
                TrackedIDs = TrackedIDs[:len(CurrFracC)]
        PrevFracC = CurrFracC

        RawStates, _ = DetermineInstantaneousStates(FracCoords, Atoms, Lattice)
        CurrentFrameStates = {int(TrackedIDs[i]): RawStates[i] for i in range(len(RawStates))}

        Tp = TimeFs / 1000.0
        for AtomID in range(TotalStartC):
            RawState = CurrentFrameStates.get(AtomID, 'Gas')
            if RawState == ConfirmedStates[AtomID]:
                PendingStates.pop(AtomID, None)
                continue
            if AtomID in PendingStates and PendingStates[AtomID]['State'] == RawState:
                if Tp - PendingStates[AtomID]['StartTime'] >= StatePersistenceTimePs:
                    ConfirmedStates[AtomID] = RawState
                    del PendingStates[AtomID]
            else:
                PendingStates[AtomID] = {'State': RawState, 'StartTime': Tp}

        Stable = (TimeFs - WindowStartTime) >= BurnInFs
        Due = LastRecordedTime is None or (TimeFs - LastRecordedTime) >= SubsampleFs
        if not (Stable and Due):
            continue
        LastRecordedTime = TimeFs
        NAnalysed += 1

        FrameAngles, CurrentTriplets = AnalyseFrameForGeometry(
            Atoms, FracCoords, Lattice, ConfirmedStates, TrackedIDs, Tp)
        AngleRecords.extend(FrameAngles)

        # Persistent-triplet run tracking across consecutive analysed frames.
        for Key, (V1, V2, Ang, L) in CurrentTriplets.items():
            if Key in ActiveTriplets:
                St = ActiveTriplets[Key]
                St['SumV1'] += V1; St['SumV2'] += V2
                St['Inst'].append(Ang); St['LastTime'] = TimeFs; St['Count'] += 1
            else:
                ActiveTriplets[Key] = {'StartTime': TimeFs, 'LastTime': TimeFs,
                                       'SumV1': V1.copy(), 'SumV2': V2.copy(),
                                       'Inst': [Ang], 'Count': 1, 'StartL': L}
        for Key in list(ActiveTriplets.keys()):
            if Key not in CurrentTriplets:
                Rec = FinalizeTriplet(ActiveTriplets.pop(Key), Temp, Group)
                if Rec:
                    FinalTriplets.append(Rec)

    for St in ActiveTriplets.values():
        Rec = FinalizeTriplet(St, Temp, Group)
        if Rec:
            FinalTriplets.append(Rec)

    return {'TrajId': TrajId, 'Temp': Temp, 'Group': Group,
            'NAnalysedFrames': NAnalysed,
            'AngleRecords': [(T, A, L) for (T, A, L) in AngleRecords],
            'Triplets': FinalTriplets}


# =============================================================================
#                          BRANCHING (reused from cache)
# =============================================================================
def LoadBranchStats():
    """Branch-by-length from the cached ChainInventory.csv (reuse; no re-stream)."""
    Inv = pd.read_csv(InventoryCsv)
    Inv['Branched'] = (Inv['NBranchPoints'] >= 1).astype(int)
    return Inv


# =============================================================================
#                               HELPERS
# =============================================================================
def MedianIqr(Values):
    Arr = np.asarray(Values, dtype=float)
    if Arr.size == 0:
        return float('nan'), float('nan'), float('nan')
    return float(np.median(Arr)), float(np.percentile(Arr, 25)), float(np.percentile(Arr, 75))


# =============================================================================
#                                 FIGURES
# =============================================================================
def MakeCollapseAngleFigure(AngleDf, OutPng, OutPdf):
    Fig, Ax = plt.subplots(figsize=(7.6, 5.2))
    Lengths = sorted(L for L in AngleDf['L'].unique() if L >= 3)
    Offsets = {873: -0.18, 973: -0.06, 1073: 0.06, 1273: 0.18}
    for Temp in Temperatures:
        Style = TemperatureStyle[Temp]
        Xs, Meds, LoErr, HiErr = [], [], [], []
        for L in Lengths:
            Vals = AngleDf[(AngleDf['Temperature'] == Temp) & (AngleDf['L'] == L)]['Angle'].values
            if len(Vals) < 3:
                continue
            Med, Q1, Q3 = MedianIqr(Vals)
            Xs.append(L + Offsets[Temp]); Meds.append(Med)
            LoErr.append(Med - Q1); HiErr.append(Q3 - Med)
        Ax.errorbar(Xs, Meds, yerr=[LoErr, HiErr], fmt=Style['marker'], color=Style['color'],
                    markersize=9, capsize=4, lw=1.6, label=f"{Temp} K ({Style['group'][:5]})")
    Ax.axhline(120, color='0.6', ls='--', lw=1); Ax.axhline(180, color='0.6', ls=':', lw=1)
    Ax.set_xlabel("Chain length L (atoms)")
    Ax.set_ylabel("Median C-C-C angle (deg), IQR bars")
    Ax.set_title("Collapse test: chain angle vs length, by temperature")
    Ax.set_xticks(Lengths)
    Ax.legend(title="Temperature", fontsize=9)
    Ax.grid(alpha=0.25)
    Fig.tight_layout(); Fig.savefig(OutPng, dpi=300); Fig.savefig(OutPdf); plt.close(Fig)


def MakeCollapseBranchFigure(Inv, OutPng, OutPdf):
    Fig, Ax = plt.subplots(figsize=(7.6, 5.2))
    Lengths = sorted(L for L in Inv['Length'].unique() if L >= 2)
    Offsets = {873: -0.18, 973: -0.06, 1073: 0.06, 1273: 0.18}
    for Temp in Temperatures:
        Style = TemperatureStyle[Temp]
        Xs, Ps = [], []
        for L in Lengths:
            Sub = Inv[(Inv['Temperature'] == Temp) & (Inv['Length'] == L)]
            if len(Sub) < 3:
                continue
            Xs.append(L + Offsets[Temp]); Ps.append(Sub['Branched'].mean())
        Ax.plot(Xs, Ps, Style['marker'], color=Style['color'], markersize=9,
                label=f"{Temp} K ({Style['group'][:5]})")
    Ax.set_xlabel("Chain length L (atoms)")
    Ax.set_ylabel("P(chain contains a branch point)")
    Ax.set_title("Collapse test: branch probability vs length, by temperature")
    Ax.set_xticks(Lengths)
    Ax.legend(title="Temperature", fontsize=9)
    Ax.grid(alpha=0.25)
    Fig.tight_layout(); Fig.savefig(OutPng, dpi=300); Fig.savefig(OutPdf); plt.close(Fig)


def MakeInstVsAvgFigure(AngleDf, TripletDf, OutPng, OutPdf):
    Fig, Axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    Bins = np.linspace(60, 180, 41)
    for Ax, GName, Colour in [(Axes[0], GroupBelow, '#1f77b4'), (Axes[1], GroupAbove, '#d62728')]:
        Inst = AngleDf[AngleDf['Group'] == GName]['Angle'].values
        Avg = TripletDf[TripletDf['Group'] == GName]['TimeAveragedAngle'].values
        Ax.hist(Inst, bins=Bins, density=True, alpha=0.5, color=Colour,
                label=f"Instantaneous (n={len(Inst)})")
        Ax.hist(Avg, bins=Bins, density=True, histtype='step', lw=2.2, color='black',
                label=f"Time-averaged (n={len(Avg)})")
        Ax.axvline(120, color='0.6', ls='--', lw=1); Ax.axvline(180, color='0.6', ls=':', lw=1)
        Ax.set_title(GName); Ax.set_xlabel("C-C-C angle (deg)"); Ax.set_xlim(60, 180)
        Ax.legend(fontsize=8)
    Axes[0].set_ylabel("Probability density")
    Fig.suptitle("Instantaneous vs time-averaged chain angles (thermal control)")
    Fig.tight_layout(); Fig.savefig(OutPng, dpi=300); Fig.savefig(OutPdf); plt.close(Fig)


# =============================================================================
#                                 DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Length-stratified / thermally controlled chain geometry.")
    Parser.add_argument('--jobs', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Parser.add_argument('--only', type=str, default=None)
    Args = Parser.parse_args()

    np.random.seed(RandomSeed)
    os.makedirs(ResultsDir, exist_ok=True)

    Tasks = ([tuple(int(x) for x in Args.only.split('_'))] if Args.only
             else [(T, S) for T in Temperatures for S in SimIndices])

    print(f"Re-streaming {len(Tasks)} trajectories for angles + triplets (jobs={Args.jobs}) ...", flush=True)
    Results = []
    if Args.jobs > 1 and len(Tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=Args.jobs) as Pool:
            Futures = {Pool.submit(ProcessTrajectory, T): T for T in Tasks}
            for Fut in as_completed(Futures):
                R = Fut.result(); Results.append(R)
                print(f"  {R['TrajId']}: {R['NAnalysedFrames']} frames, {len(R['AngleRecords'])} angles, "
                      f"{len(R['Triplets'])} persistent triplets", flush=True)
    else:
        for T in Tasks:
            R = ProcessTrajectory(T); Results.append(R)
            print(f"  {R['TrajId']}: {R['NAnalysedFrames']} frames, {len(R['AngleRecords'])} angles, "
                  f"{len(R['Triplets'])} persistent triplets", flush=True)

    # ---- Assemble angle + triplet frames ----
    AngleRows = []
    for R in Results:
        for (T, A, L) in R['AngleRecords']:
            AngleRows.append({'Temperature': R['Temp'], 'Group': R['Group'], 'L': L,
                              'Time_ps': round(T, 3), 'Angle': A})
    AngleDf = pd.DataFrame(AngleRows)

    TripletRows = [dict(Rec) for R in Results for Rec in R['Triplets']]
    TripletDf = pd.DataFrame(TripletRows)

    Inv = LoadBranchStats()

    # ---- StratifiedGeometry.csv (angle + branch observations) ----
    StratRows = []
    for _, Row in AngleDf.iterrows():
        StratRows.append({'Type': 'angle', 'Temperature': Row['Temperature'], 'Group': Row['Group'],
                          'L': int(Row['L']), 'Time_ps': Row['Time_ps'], 'Value': Row['Angle']})
    for _, Row in Inv.iterrows():
        StratRows.append({'Type': 'branch', 'Temperature': int(Row['Temperature']), 'Group': Row['Group'],
                          'L': int(Row['Length']), 'Time_ps': Row['Time_ps'], 'Value': int(Row['Branched'])})
    pd.DataFrame(StratRows).to_csv(os.path.join(ResultsDir, "StratifiedGeometry.csv"), index=False)
    TripletDf.to_csv(os.path.join(ResultsDir, "TimeAveragedAngles.csv"), index=False)

    # ---- Figures ----
    MakeCollapseAngleFigure(AngleDf, os.path.join(ResultsDir, "CollapseAngle.png"),
                            os.path.join(ResultsDir, "CollapseAngle.pdf"))
    MakeCollapseBranchFigure(Inv, os.path.join(ResultsDir, "CollapseBranch.png"),
                             os.path.join(ResultsDir, "CollapseBranch.pdf"))
    MakeInstVsAvgFigure(AngleDf, TripletDf, os.path.join(ResultsDir, "InstVsTimeAveraged.png"),
                        os.path.join(ResultsDir, "InstVsTimeAveraged.pdf"))

    Stats = BuildStatistics(AngleDf, TripletDf, Inv, Results)
    with open(os.path.join(ResultsDir, "GeometryResults.json"), 'w') as F:
        json.dump(Stats, F, indent=2)

    WriteSummary(Stats, Results)
    print("\nDone. Results in", ResultsDir, flush=True)


def BuildStatistics(AngleDf, TripletDf, Inv, Results):
    def GroupName(G):
        return "Below" if G == GroupBelow else "Above"

    # Step 1a: fixed-length angle comparison.
    FixedLength = {}
    for L in AngleStrataL:
        for G in (GroupBelow, GroupAbove):
            Vals = AngleDf[(AngleDf['Group'] == G) & (AngleDf['L'] == L)]['Angle'].values
            Med, _, _ = MedianIqr(Vals)
            FixedLength[f"L{L}_{GroupName(G)}"] = {
                'AngleSamples': int(len(Vals)),
                'MedianAngle': Med,
                'FractionAbove150': float(np.mean(Vals > 150)) if len(Vals) else float('nan'),
                'Thin': bool(len(Vals) < ThinAngleSamples),
            }

    # Step 1b / 2: branch probability by length, per group and per temperature.
    BranchByLength = {}
    for L in sorted(Inv['Length'].unique()):
        Entry = {'L': int(L)}
        for G in (GroupBelow, GroupAbove):
            Sub = Inv[(Inv['Group'] == G) & (Inv['Length'] == L)]
            Entry[f"{GroupName(G)}_chains"] = int(len(Sub))
            Entry[f"{GroupName(G)}_Pbranched"] = float(Sub['Branched'].mean()) if len(Sub) else float('nan')
            Entry[f"{GroupName(G)}_thin"] = bool(len(Sub) < ThinChainCount)
        BranchByLength[int(L)] = Entry

    BranchByTempLength = {}
    for Temp in Temperatures:
        for L in sorted(Inv['Length'].unique()):
            Sub = Inv[(Inv['Temperature'] == Temp) & (Inv['Length'] == L)]
            if len(Sub):
                BranchByTempLength[f"{Temp}_L{int(L)}"] = {
                    'Chains': int(len(Sub)), 'Pbranched': float(Sub['Branched'].mean())}

    AngleByTempLength = {}
    for Temp in Temperatures:
        for L in sorted(AngleDf['L'].unique()):
            Vals = AngleDf[(AngleDf['Temperature'] == Temp) & (AngleDf['L'] == L)]['Angle'].values
            if len(Vals):
                Med, Q1, Q3 = MedianIqr(Vals)
                AngleByTempLength[f"{Temp}_L{int(L)}"] = {
                    'Samples': int(len(Vals)), 'Median': Med, 'Q1': Q1, 'Q3': Q3}

    # Step 3: thermal control.
    Thermal = {}
    for G in (GroupBelow, GroupAbove):
        Inst = AngleDf[AngleDf['Group'] == G]['Angle'].values
        Avg = TripletDf[TripletDf['Group'] == G]['TimeAveragedAngle'].values if len(TripletDf) else np.array([])
        Sds = TripletDf[TripletDf['Group'] == G]['InstSD'].values if len(TripletDf) else np.array([])
        Thermal[GroupName(G)] = {
            'MedianInstantaneous': float(np.median(Inst)) if len(Inst) else float('nan'),
            'MedianTimeAveraged': float(np.median(Avg)) if len(Avg) else float('nan'),
            'MeanPerTripletSD': float(np.mean(Sds)) if len(Sds) else float('nan'),
            'NTriplets': int(len(Avg)),
        }
    ThermalPerTemp = {}
    for Temp in Temperatures:
        Sds = TripletDf[TripletDf['Temperature'] == Temp]['InstSD'].values if len(TripletDf) else np.array([])
        ThermalPerTemp[str(Temp)] = {'MeanPerTripletSD': float(np.mean(Sds)) if len(Sds) else float('nan'),
                                     'NTriplets': int(len(Sds))}
    TimeAvgFixedL = {}
    for L in AngleStrataL:
        for G in (GroupBelow, GroupAbove):
            Vals = TripletDf[(TripletDf['Group'] == G) & (TripletDf['StartL'] == L)]['TimeAveragedAngle'].values \
                if len(TripletDf) else np.array([])
            TimeAvgFixedL[f"L{L}_{GroupName(G)}"] = {
                'NTriplets': int(len(Vals)),
                'MedianTimeAveraged': float(np.median(Vals)) if len(Vals) else float('nan'),
                'Thin': bool(len(Vals) < ThinChainCount),
            }

    Verdict = ComputeVerdict(FixedLength, BranchByLength)

    return {
        'Seed': RandomSeed, 'SubsampleInterval_fs': SubsampleFs, 'PersistRunMin_ps': PersistRunMinPs,
        'FramesAnalysed': {'below': sum(R['NAnalysedFrames'] for R in Results if R['Group'] == GroupBelow),
                           'above': sum(R['NAnalysedFrames'] for R in Results if R['Group'] == GroupAbove)},
        'FixedLengthAngles': FixedLength,
        'BranchByLength': BranchByLength,
        'BranchByTempLength': BranchByTempLength,
        'AngleByTempLength': AngleByTempLength,
        'Thermal': Thermal,
        'ThermalPerTemperature': ThermalPerTemp,
        'TimeAveragedFixedLength': TimeAvgFixedL,
        'Verdict': Verdict,
    }


def ComputeVerdict(FixedLength, BranchByLength):
    """Apply the interpretation rule (Step 2) numerically; return phrase + evidence."""
    # Angle: compare Below vs Above median at fixed L (3, 4).
    AngleDiffs = {}
    for L in AngleStrataL:
        Bel = FixedLength[f"L{L}_Below"]['MedianAngle']
        Abv = FixedLength[f"L{L}_Above"]['MedianAngle']
        AngleDiffs[L] = Abv - Bel      # negative => Above more bent
    Valid = [d for d in AngleDiffs.values() if not np.isnan(d)]
    if Valid and all(d <= -3 for d in Valid):
        AngleVerdict = "Above systematically more bent"
    elif Valid and all(abs(d) < 3 for d in Valid):
        AngleVerdict = "common curve"
    else:
        AngleVerdict = "mixed"

    # Branch: compare P(branched) Below vs Above at fixed L (>= 4).
    BranchDiffs = {}
    for L, E in BranchByLength.items():
        if L >= 4 and E['Below_chains'] >= ThinChainCount and E['Above_chains'] >= ThinChainCount:
            BranchDiffs[L] = E['Above_Pbranched'] - E['Below_Pbranched']
    BValid = list(BranchDiffs.values())
    if BValid and all(d >= 0.05 for d in BValid):
        BranchVerdict = "Above systematically more branched"
    elif BValid and all(abs(d) < 0.05 for d in BValid):
        BranchVerdict = "common curve"
    elif not BValid:
        BranchVerdict = "insufficient fixed-L data"
    else:
        BranchVerdict = "mixed"

    return {'AngleVerdict': AngleVerdict, 'AngleDiffs': {str(k): v for k, v in AngleDiffs.items()},
            'BranchVerdict': BranchVerdict, 'BranchDiffs': {str(k): v for k, v in BranchDiffs.items()}}


def Fmt(X, Nd=1):
    return "nan" if (X is None or (isinstance(X, float) and np.isnan(X))) else f"{X:.{Nd}f}"


def WriteSummary(Stats, Results):
    FL = Stats['FixedLengthAngles']
    Th = Stats['Thermal']
    V = Stats['Verdict']

    L = []
    L.append("# Length-stratified chain geometry and thermal control")
    L.append("")
    L.append(f"Source: ChainInventory.csv reused for branching + final-10% window re-streamed for angles/triplets; "
             f"seed {RandomSeed}, subsampling {int(SubsampleFs)} fs")
    L.append(f"Frames analysed: below {Stats['FramesAnalysed']['below']} / above {Stats['FramesAnalysed']['above']}")
    L.append("")
    L.append("## 1. Fixed-length comparison (instantaneous)")
    L.append("")
    L.append("| L | Group | Angle samples | Median angle (deg) | Fraction > 150 deg | Thin? |")
    L.append("|--:|---|---:|---:|---:|---|")
    for Lval in AngleStrataL:
        for GName in ("Below", "Above"):
            E = FL[f"L{Lval}_{GName}"]
            L.append(f"| {Lval} | {GName} | {E['AngleSamples']} | {Fmt(E['MedianAngle'])} | "
                     f"{Fmt(E['FractionAbove150'], 3)} | {'yes' if E['Thin'] else 'no'} |")
    L.append("")
    L.append("## 2. Branch probability by length")
    L.append("")
    L.append("| L | Below: chains obs. | Below: P(branched) | Above: chains obs. | Above: P(branched) |")
    L.append("|--:|---:|---:|---:|---:|")
    for Lval, E in sorted(Stats['BranchByLength'].items()):
        BThin = " (thin)" if E['Below_thin'] else ""
        AThin = " (thin)" if E['Above_thin'] else ""
        L.append(f"| {Lval} | {E['Below_chains']}{BThin} | {Fmt(E['Below_Pbranched'], 3)} | "
                 f"{E['Above_chains']}{AThin} | {Fmt(E['Above_Pbranched'], 3)} |")
    L.append("")
    L.append("## 3. Collapse test verdict (per interpretation rule)")
    L.append("")
    Ad = V['AngleDiffs']
    L.append(f"Angle vs L: [{V['AngleVerdict']}] - Above-minus-Below median at fixed L: "
             f"L3 {Fmt(Ad.get('3'))} deg, L4 {Fmt(Ad.get('4'))} deg "
             f"(negative = Above more bent).")
    Bd = V['BranchDiffs']
    BdStr = ", ".join(f"L{k} {Fmt(v, 3)}" for k, v in sorted(Bd.items())) or "no fixed-L stratum with >=5 chains in both groups"
    L.append(f"Branch vs L: [{V['BranchVerdict']}] - Above-minus-Below P(branched) at fixed L: {BdStr}.")
    L.append("")
    L.append("## 4. Thermal control")
    L.append("")
    L.append("| Group | Median instantaneous angle | Median time-averaged angle | Mean per-triplet SD (deg) |")
    L.append("|---|---:|---:|---:|")
    for GName in ("Below", "Above"):
        E = Th[GName]
        L.append(f"| {GName} | {Fmt(E['MedianInstantaneous'])} | {Fmt(E['MedianTimeAveraged'])} | "
                 f"{Fmt(E['MeanPerTripletSD'])} |")
    L.append("")
    PT = Stats['ThermalPerTemperature']
    L.append(f"Per-temperature thermal bending amplitude (mean per-triplet SD, deg): "
             f"873: {Fmt(PT['873']['MeanPerTripletSD'])} / 973: {Fmt(PT['973']['MeanPerTripletSD'])} / "
             f"1073: {Fmt(PT['1073']['MeanPerTripletSD'])} / 1273: {Fmt(PT['1273']['MeanPerTripletSD'])}")
    TA = Stats['TimeAveragedFixedLength']
    TaParts = []
    for Lval in AngleStrataL:
        for GName in ("Below", "Above"):
            E = TA[f"L{Lval}_{GName}"]
            Flag = " [thin]" if E['Thin'] else ""
            TaParts.append(f"L{Lval} {GName}: {Fmt(E['MedianTimeAveraged'])} (n={E['NTriplets']}){Flag}")
    L.append("Fixed-L time-averaged medians (deg): " + "; ".join(TaParts))
    L.append("")
    L.append("## 5. Notes")
    for N in BuildNotes(Stats, Results):
        L.append(f"- {N}")

    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), 'w') as F:
        F.write(Text + "\n")
    print(Text)


def BuildNotes(Stats, Results):
    Notes = []
    Notes.append(f"Persistent triplet = a centre carbon bonded to the same two carbons across a contiguous "
                 f"run of >= {PersistRunMinPs:.0f} ps at the {int(SubsampleFs)} fs sampling (continuity checked "
                 f"on consecutive analysed frames); {sum(len(R['Triplets']) for R in Results)} triplets total.")
    Notes.append("Time-averaged angle: the two minimum-image C-C bond vectors (centre->n1, centre->n2) are "
                 "averaged over the run, then the angle is taken between the means. This equals the angle of "
                 "the unwrapped per-atom mean positions (mean(n1-c) = mean n1_unwrapped - mean c_unwrapped) and "
                 "never averages wrapped coordinates across a PBC boundary - the required unwrapping is implicit "
                 "in using per-frame minimum-image bond vectors.")
    Notes.append("Branching reused from ChainTopologyResults/ChainInventory.csv (per-frame-chain Length + "
                 "NBranchPoints); angles and triplets re-streamed with the imported detection stack, same seed, "
                 "window, subsampling - so both sit on the identical analysed-frame set.")
    Notes.append(f"Thin strata (< {ThinAngleSamples} angle samples or < {ThinChainCount} distinct chains) are "
                 f"printed but flagged; do not quote them.")
    Notes.append("Step-1 angles use centres with exactly 2 chain-C neighbours; persistent triplets additionally "
                 "include branch-point angle pairs (all C-C-C bonded triples), so the triplet population is a "
                 "slight superset - branch points are rare so the medians remain comparable.")
    Notes.append("Per-triplet StartL (length used for fixed-L time-averaged strata) is the chain length at the "
                 "first frame of the run; length can drift over a >=2 ps run.")
    return Notes


if __name__ == "__main__":
    Main()
