"""
Refresh140.py
=======================================================================
Single authoritative refresh of every Results-section number, table and
figure on the FINAL ZrC dataset at the uniform 140.0 ps horizon (closed
interval). Outputs under Refresh140/ supersede all previous results
folders and are the source of the numbers in the paper.

Architecture
------------
One orchestrator; sibling scripts are imported and parameterised (module
globals overridden at call time), never copied. New logic that has no
sibling (entropy, profiles, temporal) lives here. Everything downstream
of Module 1 runs from:
  * Refresh140/FrameCache/  - one npz per trajectory: every 50 fs grid
    frame over 0-140.0 ps (species, coords, lattice, time) PLUS the
    carbon tracker IDs and confirmed-state snapshot at each grid frame;
  * Refresh140/Events/EventTableV2.csv and the fate/turnover records
    produced by the same single full-resolution pass.

The full-resolution pass is the only expensive xyz read: it runs the
established chain-state machine (BootstrapCalculation stack: bond
criterion r < (Ri+Rj)*1.5 with RC 0.77 / RO 0.66 / RZr 1.45 A, minimum
image, 0.2 ps TIME-BASED persistence filter evaluated on every raw MD
frame) from t = 0 to the last frame <= 140.0 ps.

Conventions (BoundaryConventions/Conventions.json)
-----------------------------------------------------------------------
* Horizon 140.0 ps closed; censoring 140.0 ps for all subjects.
* Horizon state snapshots use confirmed states only - pending
  transitions are NOT flushed (differs from the superseded pipeline,
  which flushed at trajectory end; the boundary census showed zero
  events near the horizon, so this cannot move any published number by
  more than the persistence filter's own latency).
* Structural window: the fixed 14 ps terminal window 126.0-140.0 ps.
* Grid frames are the FIRST raw frame at/after each 50 fs grid time
  (within <= 1.8 fs of nominal; raw spacing 0.7-1.8 fs - see the
  gas-audit POTIM audit). Endpoint verification: every trajectory has a
  frame within 1.0 fs of 140.0 ps.
* Seed 20260715 for all resampling.
* FluxConvention (Module 4 primary): CLI parameter. The dispatch left
  the placeholder '<TO BE FILLED AFTER AUDIT>' unfilled; the default
  'free_full' (J_free, 0-140 ps - the audit's Table 1 candidate) is
  used and FLAGGED in MASTER_SUMMARY Notes. All three conventions are
  computed regardless.

Stale-cache rule: the dataset was replaced during the project;
nothing cached in the previous results folders is reused. Gas-supply
numbers come from GasSupplyAudit/GasSupplyAudit.json, which was built
from the final dataset (its gate output is re-verified here).

Usage
-----
    python Refresh140.py                          # all modules + master
    python Refresh140.py --modules 1,2,3          # subset (cache reused)
    python Refresh140.py --flux-convention far_40_140
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

ScriptDir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ScriptDir)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))  # shared modules
import BootstrapCalculation as Bc
import SurvivalAnalysis as Sa
import ChainTopology as Ct
import ChainGeometryStratified as Cg
import OxygenLayeringAnalysis as Ola
import GasSupplyAudit as Gsa

RootDir = os.path.join(ScriptDir, "Refresh140")
CacheDir = os.path.join(RootDir, "FrameCache")
DataDir = Bc.DataDir

RandomSeed = 20260715
Temperatures = [873, 973, 1073, 1273]
SimIndices = [1, 2, 3, 4]
HorizonPs = 140.0
HorizonFs = HorizonPs * 1000.0
GridFs = 50.0
NGrid = int(HorizonFs / GridFs) + 1              # 2801
TerminalWindowPs = (126.0, 140.0)                # fixed 14 ps terminal window
GroupBelow, GroupAbove = "Below (873+973)", "Above (1073+1273)"
ThinChains, ThinEvents = 5, 20

TemperatureColours = {873: "#1f77b4", 973: "#d62728", 1073: "#2ca02c", 1273: "#9467bd"}
StateCodes = {"Lattice": 0, "Displaced": 1, "Chain": 2, "Gas": 3}
StateNames = {V: K for K, V in StateCodes.items()}
SpeciesCodes = {"Zr": 0, "O": 1, "C": 2, "N": 3}

AuditJsonPath = os.path.join(ScriptDir, "GasSupplyAudit", "GasSupplyAudit.json")

# Superseded-dataset reference totals (reported alongside, never reconciled).
OldCO = {873: 21, 973: 46, 1073: 35, 1273: 36}
OldCO2 = {873: 10, 973: 4, 1073: 2, 1273: 3}
OldRetained = {873: 35, 973: 31, 1073: 48, 1273: 66}
OldCumulative = {873: 39, 973: 35, 1073: 52, 1273: 69}

FluxConventions = ("free_full", "free_40_140", "far_40_140")


def SubDir(Name):
    Path = os.path.join(RootDir, Name)
    os.makedirs(Path, exist_ok=True)
    return Path


def SaveModuleJson(Name, Payload):
    with open(os.path.join(RootDir, f"Module_{Name}.json"), "w") as F:
        json.dump(Payload, F, indent=1, default=JsonSafe)


def LoadModuleJson(Name):
    with open(os.path.join(RootDir, f"Module_{Name}.json")) as F:
        return json.load(F)


def JsonSafe(X):
    if isinstance(X, (np.floating, np.integer)):
        return float(X)
    if isinstance(X, np.ndarray):
        return X.tolist()
    if isinstance(X, (np.bool_,)):
        return bool(X)
    return str(X)


# =============================================================================
#                                   GATES
# =============================================================================
def RunGates():
    EndTimes = {f"{T}_{S}": Gsa.ReadFinalTimePs(Gsa.XyzPath(T, S))
                for T in Temperatures for S in SimIndices}
    GatePass = all(V >= HorizonPs for V in EndTimes.values())
    InBand = sum(1 for V in EndTimes.values() if 140.0 <= V <= 147.5)
    MostTight = sum(1 for V in EndTimes.values() if 140.0 <= V <= 140.5)
    DatasetPass = GatePass and InBand == 16 and MostTight >= 8
    return EndTimes, GatePass, DatasetPass


# =============================================================================
#          SHARED FULL-RESOLUTION PASS (grid cache + state machine)
# =============================================================================
def GridCachePath(Temp, Sim):
    return os.path.join(CacheDir, f"{Temp}_{Sim}_grid50fs_140ps.npz")


def StatePath(Temp, Sim):
    return os.path.join(CacheDir, f"{Temp}_{Sim}_states_140ps.json")


def AppendGridSnapshot(FrameData, Confirmed, EverChain, GridTimes, GridSpecies,
                       GridCoords, GridLattices, GridStarts, GridTracked,
                       GridTrackedStarts, GridStateSnapshots, GridCumulativeEver):
    TimeFs, Atoms, Coords, Lattice, TrackedIDs = FrameData
    GridTimes.append(TimeFs)
    Codes = np.array([SpeciesCodes.get(A, 9) for A in Atoms], dtype=np.int8)
    GridSpecies.append(Codes)
    GridCoords.append(Coords.astype(np.float32))
    GridLattices.append(np.asarray(Lattice, dtype=np.float32))
    GridStarts.append(GridStarts[-1] + len(Codes))
    GridTracked.append(np.asarray(TrackedIDs, dtype=np.int16))
    GridTrackedStarts.append(GridTrackedStarts[-1] + len(TrackedIDs))
    Snapshot = np.array([StateCodes[Confirmed[ID]]
                         for ID in range(Bc.TotalStartC)], dtype=np.int8)
    GridStateSnapshots.append(Snapshot)
    GridCumulativeEver.append(len(EverChain))


def SharedPassWorker(TaskArgs):
    """Full-resolution stream 0 -> last frame <= 140.0 ps for one trajectory.
    Runs the established chain-state machine on every raw frame and writes
    the 50 fs grid cache + per-carbon state history."""
    Temp, Sim = TaskArgs
    Path = Bc.XyzPath(Temp, Sim)

    TrackedIDs = np.arange(Bc.TotalStartC)
    Confirmed = {ID: "Lattice" for ID in range(Bc.TotalStartC)}
    Pending = {}
    PrevFracC = None

    FirstGasPs = {}
    FirstChainPs = {}
    LastChainEntryPs = {}
    EverChain = set()
    Episodes = []            # closed chain episodes {AtomID, Start_ps, End_ps, ExitTo}
    OpenEpisode = {}         # AtomID -> Start_ps

    GridTimes, GridStarts = [], [0]
    GridSpecies, GridCoords, GridLattices = [], [], []
    GridTracked, GridTrackedStarts = [], [0]
    GridStateSnapshots = []
    GridCumulativeEver = []
    NextTargetIdx = 0
    LastFrame = None         # (TimeFs, chainCount) at last frame <= horizon

    for TimeFs, Atoms, Coords, Lattice in Bc.ReadTrajectoryGenerator(Path):
        if TimeFs > HorizonFs:
            break
        Tp = TimeFs / 1000.0
        FracCoords = Bc.CartesianToFractional(Coords, Lattice)
        CurrFracC = FracCoords[Atoms == "C"]
        if PrevFracC is not None:
            TrackedIDs = Bc.MapAtomIdentities(PrevFracC, CurrFracC, TrackedIDs, Lattice)
        else:
            if len(CurrFracC) < len(TrackedIDs):
                TrackedIDs = TrackedIDs[:len(CurrFracC)]
        PrevFracC = CurrFracC

        RawStates, _ = Bc.DetermineInstantaneousStates(FracCoords, Atoms, Lattice)
        FrameStates = {int(TrackedIDs[i]): RawStates[i] for i in range(len(RawStates))}
        for AtomID in range(Bc.TotalStartC):
            RawState = FrameStates.get(AtomID, "Gas")
            if RawState == Confirmed[AtomID]:
                Pending.pop(AtomID, None)
                continue
            if AtomID in Pending and Pending[AtomID]["State"] == RawState:
                if Tp - Pending[AtomID]["StartTime"] >= Bc.StatePersistenceTimePs:
                    Previous = Confirmed[AtomID]
                    Confirmed[AtomID] = RawState
                    del Pending[AtomID]
                    if RawState == "Chain":
                        EverChain.add(AtomID)
                        FirstChainPs.setdefault(AtomID, Tp)
                        LastChainEntryPs[AtomID] = Tp
                        OpenEpisode[AtomID] = Tp
                    elif Previous == "Chain":
                        ExitTo = "Gas" if RawState == "Gas" else "Solid"
                        Episodes.append({"AtomID": AtomID,
                                         "Start_ps": OpenEpisode.pop(AtomID, Tp),
                                         "End_ps": Tp, "ExitTo": ExitTo})
                    if RawState == "Gas":
                        FirstGasPs.setdefault(AtomID, Tp)
            else:
                Pending[AtomID] = {"State": RawState, "StartTime": Tp}

        # Grid snapshot: FIRST raw frame at/after each 50 fs grid time.
        FrameData = (TimeFs, Atoms, Coords, Lattice, TrackedIDs)
        while NextTargetIdx < NGrid and TimeFs >= NextTargetIdx * GridFs:
            AppendGridSnapshot(FrameData, Confirmed, EverChain, GridTimes,
                               GridSpecies, GridCoords, GridLattices, GridStarts,
                               GridTracked, GridTrackedStarts,
                               GridStateSnapshots, GridCumulativeEver)
            NextTargetIdx += 1

        LastFrame = (TimeFs, {ID: S for ID, S in Confirmed.items()})
        LastFrameData = FrameData

    # Final grid point (140.0 ps): when no raw frame sits exactly at 140000 fs,
    # snapshot the LAST frame <= horizon so the grid ends at the declared
    # last-frame-<=-140.0 endpoint (gap <= 1.0 fs per BoundaryConventions).
    while NextTargetIdx < NGrid:
        AppendGridSnapshot(LastFrameData, Confirmed, EverChain, GridTimes,
                           GridSpecies, GridCoords, GridLattices, GridStarts,
                           GridTracked, GridTrackedStarts,
                           GridStateSnapshots, GridCumulativeEver)
        NextTargetIdx += 1

    HorizonStates = LastFrame[1]
    RetainedIDs = sorted(ID for ID, S in HorizonStates.items() if S == "Chain")
    os.makedirs(CacheDir, exist_ok=True)
    np.savez_compressed(
        GridCachePath(Temp, Sim),
        Times=np.array(GridTimes), FrameStarts=np.array(GridStarts, dtype=np.int64),
        Species=np.concatenate(GridSpecies), Coords=np.concatenate(GridCoords),
        Lattices=np.stack(GridLattices),
        Tracked=np.concatenate(GridTracked),
        TrackedStarts=np.array(GridTrackedStarts, dtype=np.int64),
        StateSnapshots=np.stack(GridStateSnapshots),
        CumulativeEver=np.array(GridCumulativeEver, dtype=np.int32),
    )
    StateRecord = {
        "Temp": Temp, "Sim": Sim,
        "HorizonFrameFs": LastFrame[0],
        "RetainedIDs": RetainedIDs,
        "RetainedCount": len(RetainedIDs),
        "CumulativeEverChain": len(EverChain),
        "FirstGasPs": {str(K): V for K, V in FirstGasPs.items()},
        "FirstChainPs": {str(K): V for K, V in FirstChainPs.items()},
        "LastChainEntryPs": {str(K): V for K, V in LastChainEntryPs.items()},
        "ClosedEpisodes": Episodes,
        "OpenEpisodes": {str(K): V for K, V in OpenEpisode.items()},
        "HorizonStates": {str(K): V for K, V in HorizonStates.items()},
        "NGridFrames": len(GridTimes),
    }
    with open(StatePath(Temp, Sim), "w") as F:
        json.dump(StateRecord, F)
    return Temp, Sim, len(GridTimes), len(RetainedIDs), len(EverChain)


def EnsureSharedPass(Jobs, Force=False):
    Missing = [(T, S) for T in Temperatures for S in SimIndices
               if Force or not (os.path.exists(GridCachePath(T, S))
                                and os.path.exists(StatePath(T, S)))]
    if not Missing:
        print("Shared pass: all 16 grid caches present.", flush=True)
        return
    print(f"Shared pass: streaming {len(Missing)} trajectories at full "
          f"resolution (jobs={Jobs}) ...", flush=True)
    if Jobs > 1 and len(Missing) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=Jobs) as Pool:
            Futures = {Pool.submit(SharedPassWorker, T): T for T in Missing}
            for Fut in as_completed(Futures):
                Temp, Sim, NG, NR, NE = Fut.result()
                print(f"  {Temp}_{Sim}: {NG} grid frames, retained {NR}, "
                      f"ever-chained {NE}", flush=True)
    else:
        for T in Missing:
            Temp, Sim, NG, NR, NE = SharedPassWorker(T)
            print(f"  {Temp}_{Sim}: {NG} grid frames, retained {NR}, "
                  f"ever-chained {NE}", flush=True)


def LoadStateRecord(Temp, Sim):
    with open(StatePath(Temp, Sim)) as F:
        return json.load(F)


def IterGridFrames(Temp, Sim, WindowPs=None):
    """Yield (TimePs, SpeciesCodes, Coords, Lattice, TrackedIDs, StateSnapshot)
    for grid frames, optionally restricted to a [lo, hi] ps window."""
    Data = np.load(GridCachePath(Temp, Sim))
    Starts, TStarts = Data["FrameStarts"], Data["TrackedStarts"]
    for I in range(len(Data["Times"])):
        TimePs = float(Data["Times"][I]) / 1000.0
        if WindowPs and not (WindowPs[0] <= TimePs <= WindowPs[1] + 1e-9):
            continue
        Lo, Hi = Starts[I], Starts[I + 1]
        yield (TimePs, Data["Species"][Lo:Hi],
               Data["Coords"][Lo:Hi].astype(np.float64),
               Data["Lattices"][I].astype(np.float64),
               Data["Tracked"][TStarts[I]:TStarts[I + 1]].astype(np.int64),
               Data["StateSnapshots"][I])


# =============================================================================
#                        MODULE 1 - EVENT TABLE V2
# =============================================================================
def Module1(Jobs):
    OutDir = SubDir("Events")
    EnsureSharedPass(Jobs)

    Rows, OtherByTraj = [], {}
    Totals = {T: {"CO": 0, "CO2": 0, "Retained": 0, "Cumulative": 0}
              for T in Temperatures}
    for Temp in Temperatures:
        for Sim in SimIndices:
            TrajId = f"{Temp}_{Sim}"
            GasEvents, Other = Bc.LoadGasEvents(Bc.CsvPath(Temp, Sim))
            if Other:
                OtherByTraj[TrajId] = [str(O) for O in Other]
            for E in GasEvents:
                if E["TimeStep"] <= HorizonFs:            # closed interval
                    Rows.append({"Temperature": Temp, "TrajectoryId": TrajId,
                                 "Outcome": "Gas", "Species": E["Species"],
                                 "Time_ps": E["TimeStep"] / 1000.0})
                    Totals[Temp][E["Species"]] += 1
            State = LoadStateRecord(Temp, Sim)
            for ID in State["RetainedIDs"]:
                Rows.append({"Temperature": Temp, "TrajectoryId": TrajId,
                             "Outcome": "Chain", "Species": "Chain",
                             "Time_ps": State["LastChainEntryPs"][str(ID)]})
            Totals[Temp]["Retained"] += State["RetainedCount"]
            Totals[Temp]["Cumulative"] += State["CumulativeEverChain"]

    Table = pd.DataFrame(Rows, columns=["Temperature", "TrajectoryId", "Outcome",
                                        "Species", "Time_ps"])
    Table.to_csv(os.path.join(OutDir, "EventTableV2.csv"), index=False)

    print("\nModule 1 totals (new vs superseded):")
    for Temp in Temperatures:
        N = Totals[Temp]
        print(f"  {Temp} K: CO {N['CO']} (was {OldCO[Temp]}), "
              f"CO2 {N['CO2']} (was {OldCO2[Temp]}), "
              f"retained {N['Retained']} (was {OldRetained[Temp]}), "
              f"cumulative {N['Cumulative']} (was {OldCumulative[Temp]})", flush=True)

    Payload = {"Totals": {str(T): Totals[T] for T in Temperatures},
               "Old": {"CO": OldCO, "CO2": OldCO2, "Retained": OldRetained,
                       "Cumulative": OldCumulative},
               "OtherSpeciesRemovals": OtherByTraj,
               "NEvents": len(Table),
               "HorizonRule": "closed: Time_ps <= 140.0 included",
               "SnapshotRule": "confirmed states, pending NOT flushed at horizon"}
    SaveModuleJson("1_Events", Payload)
    return Payload


# =============================================================================
#                 SHARED STRUCTURAL HELPERS (grid-frame analyses)
# =============================================================================
def LastGridFrame(Temp, Sim):
    for Frame in IterGridFrames(Temp, Sim, WindowPs=(HorizonPs - 0.06, HorizonPs)):
        Last = Frame
    return Last


def ScaleOxygenCount(SpeciesC, Coords, Lattice):
    """O atoms bound into the slab network (bonded component containing Zr;
    intact O2 molecules excluded from the network first) - the 'scale O' of
    phi_O and the net-mass observable. Reuses the gas-audit machinery."""
    Names = np.array([list(SpeciesCodes)[C] if C < 4 else "X" for C in SpeciesC],
                     dtype=object)
    Inv = np.linalg.inv(Lattice)
    Frac = Coords @ Inv
    OIdx = np.where(Names == "O")[0]
    DistOO = Gsa.MinImageDistanceMatrix(Frac[OIdx], Frac[OIdx], Lattice)
    Pairs = Gsa.MatchO2Pairs(DistOO)
    PairedO = {int(OIdx[I]) for P in Pairs for I in P}
    NetIdx = np.array([K for K in range(len(Names)) if K not in PairedO])
    FracNet = Frac[NetIdx]
    DistNet = Gsa.MinImageDistanceMatrix(FracNet, FracNet, Lattice)
    Radii = np.array([Gsa.CovalentRadii.get(Names[K], 0.8) for K in NetIdx])
    Adjacency = DistNet < (Radii[:, None] + Radii[None, :]) * Gsa.RadiusTol
    np.fill_diagonal(Adjacency, False)
    IsZr = Names[NetIdx] == "Zr"
    ScaleO = 0
    for Component in Gsa.ConnectedComponentsBool(Adjacency):
        if IsZr[Component].any():
            ScaleO += sum(1 for K in Component if Names[NetIdx[K]] == "O")
    return ScaleO


def ChainAdjacencyAtFrame(SpeciesC, Coords, Lattice, Tracked, Snapshot,
                          CutoffCC=None):
    """(ChainFullIdx, Adjacency dict) of confirmed-chain carbons at one grid
    frame, C-C bonds under the given cutoff (default: established 2.31 A)."""
    Cutoff = CutoffCC if CutoffCC is not None else Bc.CutoffCC
    Inv = np.linalg.inv(Lattice)
    Frac = Coords @ Inv
    CFull = np.where(SpeciesC == SpeciesCodes["C"])[0]
    ChainFull = [int(CFull[I]) for I in range(len(Tracked))
                 if Snapshot[Tracked[I]] == StateCodes["Chain"]]
    if not ChainFull:
        return [], {}
    FracChain = Frac[ChainFull]
    Dist = Gsa.MinImageDistanceMatrix(FracChain, FracChain, Lattice)
    Adjacency = {F: [] for F in ChainFull}
    for A in range(len(ChainFull)):
        for B in range(A + 1, len(ChainFull)):
            if Dist[A, B] < Cutoff:
                Adjacency[ChainFull[A]].append(ChainFull[B])
                Adjacency[ChainFull[B]].append(ChainFull[A])
    return ChainFull, Adjacency


def ChainComponentSizes(ChainFull, Adjacency):
    return [len(C) for C in Ct.ConnectedComponents(ChainFull, Adjacency)]


# =============================================================================
#              MODULE 2 - PARTITIONING AND BOOTSTRAP  [Table 2, Fig 2]
# =============================================================================
def Module2(Jobs):
    OutDir = SubDir("Partitioning")
    Events = LoadModuleJson("1_Events")
    Table = pd.read_csv(os.path.join(RootDir, "Events", "EventTableV2.csv"))
    EventFrame = Table.rename(columns={"Time_ps": "TimeStep"}).copy()
    EventFrame["TimeStep"] = EventFrame["TimeStep"] * 1000.0

    Summary = {T: {"Chain": Events["Totals"][str(T)]["Retained"],
                   "CO": Events["Totals"][str(T)]["CO"],
                   "CO2": Events["Totals"][str(T)]["CO2"]} for T in Temperatures}

    # -- Table 2 scalars ------------------------------------------------------
    PerTemp = {}
    ScaleOByTraj = {}
    for Temp in Temperatures:
        ScaleO, ChainSizesAll = 0, []
        for Sim in SimIndices:
            TimePs, Sp, Co, La, Tr, Sn = LastGridFrame(Temp, Sim)
            SO = ScaleOxygenCount(Sp, Co, La)
            ScaleOByTraj[f"{Temp}_{Sim}"] = SO
            ScaleO += SO
            ChainFull, Adj = ChainAdjacencyAtFrame(Sp, Co, La, Tr, Sn)
            ChainSizesAll.extend(ChainComponentSizes(ChainFull, Adj))
        N = Summary[Temp]
        Gas = N["CO"] + N["CO2"]
        PerTemp[str(Temp)] = {
            "PhiC": N["Chain"] / Gas if Gas else float("nan"),
            "PhiO": ScaleO / (N["CO"] + 2 * N["CO2"]) if (N["CO"] + 2 * N["CO2"]) else float("nan"),
            "ScaleO_140ps": ScaleO,
            "CoOverCo2": N["CO"] / N["CO2"] if N["CO2"] else float("inf"),
            "ChainLengthMean_140ps": float(np.mean(ChainSizesAll)) if ChainSizesAll else 0.0,
            "ChainLengthMax_140ps": int(max(ChainSizesAll)) if ChainSizesAll else 0,
            "NChains_140ps": len(ChainSizesAll),
        }

    # -- Bootstrap (imported machinery, same seed/rounds) ----------------------
    PhiCByTemp, NanCount = Bc.RunBootstrap(EventFrame, 10000, RandomSeed)
    Headline = Bc.ComputeHeadlineStatistics(PhiCByTemp, NanCount, EventFrame, Summary)

    A = {T: PhiCByTemp[T] for T in Temperatures}
    ValidAll = ~(np.isnan(A[873]) | np.isnan(A[973]) | np.isnan(A[1073])
                 | np.isnan(A[1273]))
    Ordering = {
        "Monotonic_973_1073_1273": float(np.mean(
            (A[973][ValidAll] < A[1073][ValidAll])
            & (A[1073][ValidAll] < A[1273][ValidAll]))),
        "VShape_873_973_1073_1273": float(np.mean(
            (A[873][ValidAll] > A[973][ValidAll])
            & (A[973][ValidAll] < A[1073][ValidAll])
            & (A[1073][ValidAll] < A[1273][ValidAll]))),
        "Reference_1073_1273": float(np.mean(
            A[1073][ValidAll] < A[1273][ValidAll])),
        "ValidRounds": int(ValidAll.sum()),
    }
    DominanceP = {str(T): {"P_below_1": float(np.mean(
        A[T][~np.isnan(A[T])] < 1.0))} for T in Temperatures}

    MakeFigure2(PhiCByTemp, Headline, DominanceP,
                os.path.join(OutDir, "Fig2_PhiC"))

    Payload = {"Table2": PerTemp, "ScaleOByTrajectory": ScaleOByTraj,
               "Headline": Headline, "Ordering": Ordering,
               "DominanceProbability": DominanceP,
               "ZeroDenominator": {str(T): int(NanCount[T]) for T in Temperatures},
               "NResamples": 10000, "Seed": RandomSeed}
    SaveModuleJson("2_Partitioning", Payload)
    return Payload


def MakeFigure2(PhiCByTemp, Headline, DominanceP, OutBase):
    Fig, Ax = plt.subplots(figsize=(7.4, 5.2))
    Positions = list(range(1, len(Temperatures) + 1))
    Data = [PhiCByTemp[T][~np.isnan(PhiCByTemp[T])] for T in Temperatures]
    Violins = Ax.violinplot(Data, positions=Positions, widths=0.7,
                            showextrema=False)
    for Body, Temp in zip(Violins["bodies"], Temperatures):
        Body.set_facecolor(TemperatureColours[Temp])
        Body.set_alpha(0.45)
    for Pos, Temp in zip(Positions, Temperatures):
        E = Headline["PerTemperature"][str(Temp)]
        Ax.plot([Pos, Pos], [E["P2_5"], E["P97_5"]], color="black", lw=1.6)
        Ax.plot([Pos], [E["Observed"]], marker="D", color=TemperatureColours[Temp],
                markeredgecolor="black", markersize=9, zorder=5)
        Top = max(np.percentile(PhiCByTemp[Temp][~np.isnan(PhiCByTemp[Temp])], 99.5),
                  E["P97_5"])
        Ax.text(Pos, Top + 0.12,
                f"$P(\\phi_C<1)$ = {DominanceP[str(Temp)]['P_below_1']:.2f}",
                ha="center", fontsize=9)
    Ax.axhline(1.0, color="grey", ls="--", lw=1.2)
    Ax.text(0.62, 1.03, "$\\phi_C = 1$: retention = gasification", fontsize=9,
            color="grey")
    Ax.set_xticks(Positions)
    Ax.set_xticklabels([f"{T} K" for T in Temperatures])
    Ax.set_ylabel("Carbon retention ratio $\\phi_C$")
    Ax.set_ylim(bottom=0)
    Ax.grid(axis="y", alpha=0.25)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#                 MODULE 3 - SURVIVAL KINETICS  [Table 3, Fig 3]
# =============================================================================
def Module3(Jobs):
    OutDir = SubDir("Survival")
    Sa.ResultsDir = OutDir                    # parameterise, don't copy
    Table = pd.read_csv(os.path.join(RootDir, "Events", "EventTableV2.csv"))
    Events = Table.rename(columns={"Time_ps": "TimeStep"}).copy()
    Events["TimeStep"] = Events["TimeStep"] * 1000.0
    EndTimes = {f"{T}_{S}": HorizonPs for T in Temperatures for S in SimIndices}

    Dataset = Sa.BuildSurvivalDataset(Events, EndTimes)
    Fits, Cv = {}, {}
    for Temp in Temperatures:
        Fits[Temp] = Sa.FitTemperature(Dataset, Temp)
        Cv[Temp] = Sa.InterEventCv(Dataset, Temp)
        print(f"  {Temp} K: Lambda {Fits[Temp]['Exponential']['Lambda']:.5f}, "
              f"rho {Fits[Temp]['Weibull']['Rho']:.3f} "
              f"(LRT p {Fits[Temp]['LRT_p']:.4f})", flush=True)
    RatesByTemp = {T: Fits[T]["Exponential"] for T in Temperatures}
    Arr = Sa.ArrheniusExtrapolation(RatesByTemp)

    MakeFigure3(Fits, Arr, os.path.join(OutDir, "Fig3_Kinetics"))
    Payload = {"Fits": {str(T): Fits[T] for T in Temperatures},
               "InterEventCV": {str(T): Cv[T] for T in Temperatures},
               "ArrheniusReplacement": Arr,
               "Censoring_ps": HorizonPs,
               "RiskSet": "mobilised carbons: gas events at event time; "
                          "chained-at-140.0 censored at 140.0"}
    SaveModuleJson("3_Survival", Payload)
    return Payload


def MakeFigure3(Fits, Arr, OutBase):
    Fig, (AxA, AxB) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    InvT = [1.0 / T for T in Temperatures]
    for Temp in Temperatures:
        E = Fits[Temp]["Exponential"]
        AxA.errorbar([1.0 / Temp], [E["Lambda"]],
                     yerr=[[E["Lambda"] - E["CI_Lower"]], [E["CI_Upper"] - E["Lambda"]]],
                     fmt="o", color=TemperatureColours[Temp], capsize=4, markersize=7)
    Xs = np.linspace(min(InvT) * 0.98, max(InvT) * 1.02, 100)
    AxA.plot(Xs, np.exp(Arr["Intercept"] + Arr["Slope"] * Xs), "--", color="grey",
             lw=1.2, label="873-973 K Arrhenius line")
    AxA.set_yscale("log")
    AxA.set_xlabel("$1/T$ (K$^{-1}$)")
    AxA.set_ylabel("Desorption rate $\\lambda$ (ps$^{-1}$)")
    AxA.legend(fontsize=8)
    AxA.text(0.03, 0.94, "(a)", transform=AxA.transAxes, fontweight="bold")
    for Temp in Temperatures:
        W = Fits[Temp]["Weibull"]
        AxB.errorbar([Temp], [W["Rho"]],
                     yerr=[[W["Rho"] - W["Rho_CI_Lower"]],
                           [W["Rho_CI_Upper"] - W["Rho"]]],
                     fmt="o", color=TemperatureColours[Temp], capsize=4, markersize=7)
    AxB.axhline(1.0, color="grey", ls="--", lw=1.2)
    AxB.text(1290, 1.01, "constant rate", fontsize=8, color="grey", ha="right")
    P1273 = Fits[1273]["LRT_p"]
    AxB.annotate(f"$p$ = {P1273:.4f}", xy=(1273, Fits[1273]["Weibull"]["Rho"]),
                 xytext=(1210, Fits[1273]["Weibull"]["Rho"] - 0.06), fontsize=9,
                 ha="right", color=TemperatureColours[1273])
    AxB.set_xlabel("Temperature (K)")
    AxB.set_ylabel("Hazard shape $\\rho$")
    AxB.set_xticks(Temperatures)
    AxB.text(0.03, 0.94, "(b)", transform=AxB.transAxes, fontweight="bold")
    for Ax in (AxA, AxB):
        Ax.grid(alpha=0.25)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#             MODULE 4 - ENSEMBLE ARRHENIUS FITS  [Table 4 + SI]
# =============================================================================
KbEv = 8.617333262e-5


def FluxLookup(Audit, Convention):
    Window, Label = {"free_full": ("0-140", "free"),
                     "free_40_140": ("40-140", "free"),
                     "far_40_140": ("40-140", "far")}[Convention]
    TempMean = {T: Audit["PerTemperature"][str(T)]["WindowStats"][Window][Label]["J_TempMean"]
                for T in Temperatures}
    PerTraj = {f"{T}_{S}": Audit["PerTrajectory"][f"{T}_{S}"]["WindowStats"][Window][Label]["J_Mean"]
               for T in Temperatures for S in SimIndices}
    return TempMean, PerTraj


def ArrheniusFit(TempsK, Ks):
    """ln k = ln A - Ea/(kB T): returns Ea (eV), sigma_Ea, R2, n."""
    X = 1.0 / np.asarray(TempsK, dtype=float)
    Y = np.log(np.asarray(Ks, dtype=float))
    N = len(X)
    Slope, Intercept = np.polyfit(X, Y, 1)
    Pred = Slope * X + Intercept
    SsRes = float(np.sum((Y - Pred) ** 2))
    SsTot = float(np.sum((Y - Y.mean()) ** 2))
    R2 = 1.0 - SsRes / SsTot if SsTot > 0 else float("nan")
    if N > 2:
        SigmaSlope = np.sqrt(SsRes / (N - 2) / np.sum((X - X.mean()) ** 2))
    else:
        SigmaSlope = float("nan")
    return {"Ea_eV": -Slope * KbEv, "SigmaEa_eV": SigmaSlope * KbEv,
            "R2": R2, "N": N, "Slope": float(Slope), "Intercept": float(Intercept)}


def Module4(Jobs, FluxConvention):
    OutDir = SubDir("ArrheniusFits")
    with open(AuditJsonPath) as F:
        Audit = json.load(F)
    Part = LoadModuleJson("2_Partitioning")
    Events = LoadModuleJson("1_Events")

    THorizonS = HorizonPs * 1e-12
    Observables = {}
    for Temp in Temperatures:
        for Sim in SimIndices:
            TrajId = f"{Temp}_{Sim}"
            State = LoadStateRecord(Temp, Sim)
            Sub = pd.read_csv(os.path.join(RootDir, "Events", "EventTableV2.csv"))
            Sub = Sub[(Sub["TrajectoryId"] == TrajId) & (Sub["Outcome"] == "Gas")]
            NCo = int((Sub["Species"] == "CO").sum())
            NCo2 = int((Sub["Species"] == "CO2").sum())
            ScaleO = Part["ScaleOByTrajectory"][TrajId]
            Observables[TrajId] = {
                "CumulativeChain": State["CumulativeEverChain"],
                "NetMassAmu": ScaleO * 15.999 - (NCo + NCo2) * 12.011,
                "Desorption": NCo + NCo2,
            }

    Results = {}
    for Convention in FluxConventions:
        JTemp, JTraj = FluxLookup(Audit, Convention)
        Results[Convention] = {}
        for Obs in ("CumulativeChain", "NetMassAmu", "Desorption"):
            KTraj = {Id: (Observables[Id][Obs] / THorizonS) / JTraj[Id]
                     for Id in Observables}
            KTempMean = {T: float(np.mean([KTraj[f"{T}_{S}"] for S in SimIndices]))
                         for T in Temperatures}
            Entry = {
                "MeanFit": ArrheniusFit(Temperatures, [KTempMean[T] for T in Temperatures]),
                "TrajFit": ArrheniusFit(
                    [int(Id.split("_")[0]) for Id in KTraj],
                    [KTraj[Id] for Id in KTraj]),
                "KTempMean": KTempMean,
            }
            if Obs == "NetMassAmu":
                HighIds = [f"{T}_{S}" for T in (1073, 1273) for S in SimIndices]
                Entry["RegimeFit_1073_1273_Traj"] = ArrheniusFit(
                    [int(Id.split("_")[0]) for Id in HighIds],
                    [KTraj[Id] for Id in HighIds])
            Results[Convention][Obs] = Entry

    Comparison = {}
    for Obs in ("CumulativeChain", "NetMassAmu"):
        Eas = {C: Results[C][Obs]["MeanFit"]["Ea_eV"] for C in FluxConventions}
        Sigma = Results[FluxConvention][Obs]["MeanFit"]["SigmaEa_eV"]
        Spread = max(Eas.values()) - min(Eas.values())
        Comparison[Obs] = {"Ea_eV": Eas, "PrimarySigma": Sigma,
                           "SpreadExceedsSigma": bool(Spread > Sigma)}

    Payload = {"FluxConventionPrimary": FluxConvention,
               "PlaceholderUnfilled": True,
               "Results": Results, "Comparison": Comparison,
               "Observables": Observables,
               "Units": "k = (N / 140 ps) / J  [m^2]; Ea from ln k vs 1/T"}
    SaveModuleJson("4_ArrheniusFits", Payload)
    with open(os.path.join(OutDir, "ArrheniusFits.json"), "w") as F:
        json.dump(Payload, F, indent=1, default=JsonSafe)
    for Obs, C in Comparison.items():
        print(f"  {Obs}: Ea(free_full) {C['Ea_eV']['free_full']:.3f} eV / "
              f"Ea(free_40_140) {C['Ea_eV']['free_40_140']:.3f} / "
              f"Ea(far_40_140) {C['Ea_eV']['far_40_140']:.3f} "
              f"(sigma {C['PrimarySigma']:.3f})", flush=True)
    return Payload


# =============================================================================
#              MODULE 5 - CONFIGURATIONAL ENTROPY  [Figure 4]
# =============================================================================
ZrOCutoff = 3.17
OxidisedZrMinO = 2


def Module5(Jobs):
    OutDir = SubDir("Entropy")
    PerTraj = {}
    for Temp in Temperatures:
        for Sim in SimIndices:
            Counts = {}
            for TimePs, Sp, Co, La, Tr, Sn in IterGridFrames(Temp, Sim):
                Inv = np.linalg.inv(La)
                Frac = Co @ Inv
                ZrIdx = np.where(Sp == SpeciesCodes["Zr"])[0]
                OIdx = np.where(Sp == SpeciesCodes["O"])[0]
                if len(ZrIdx) == 0 or len(OIdx) == 0:
                    continue
                Dist = Gsa.MinImageDistanceMatrix(Frac[ZrIdx], Frac[OIdx], La)
                NO = (Dist < ZrOCutoff).sum(axis=1)
                for N in NO[NO >= OxidisedZrMinO]:
                    Counts[int(N)] = Counts.get(int(N), 0) + 1
            Total = sum(Counts.values())
            Probs = np.array([C / Total for C in Counts.values()])
            H = float(-(Probs * np.log(Probs)).sum()) if Total else float("nan")
            PerTraj[f"{Temp}_{Sim}"] = {"H_nats": H, "Distribution": Counts,
                                        "NObservations": Total}
            print(f"  {Temp}_{Sim}: H = {H:.4f} nats ({Total} Zr-frame obs)",
                  flush=True)
    PerTemp = {}
    for Temp in Temperatures:
        Hs = [PerTraj[f"{Temp}_{S}"]["H_nats"] for S in SimIndices]
        PerTemp[str(Temp)] = {"Mean": float(np.mean(Hs)), "SD": float(np.std(Hs, ddof=1)),
                              "Min": float(min(Hs)), "Max": float(max(Hs))}
    MakeFigure4(PerTraj, PerTemp, os.path.join(OutDir, "Fig4_Entropy"))
    Payload = {"PerTrajectory": PerTraj, "PerTemperature": PerTemp,
               "Definition": f"Zr-O coordination (N_O >= {OxidisedZrMinO}, cutoff "
                             f"{ZrOCutoff} A) pooled over the 50 fs grid 0-140 ps; "
                             "Shannon H in nats"}
    SaveModuleJson("5_Entropy", Payload)
    return Payload


def MakeFigure4(PerTraj, PerTemp, OutBase):
    Fig, Ax = plt.subplots(figsize=(7.0, 5.0))
    for Temp in Temperatures:
        Colour = TemperatureColours[Temp]
        for Sim in SimIndices:
            Ax.plot([Temp], [PerTraj[f"{Temp}_{Sim}"]["H_nats"]], "o",
                    color=Colour, alpha=0.35, markersize=6)
        E = PerTemp[str(Temp)]
        Ax.errorbar([Temp], [E["Mean"]], yerr=[[E["SD"]], [E["SD"]]], fmt="D",
                    color=Colour, markeredgecolor="black", capsize=5,
                    markersize=9, zorder=5)
    Ax.set_xticks(Temperatures)
    Ax.set_xlabel("Temperature (K)")
    Ax.set_ylabel("Zr-O coordination Shannon entropy $H$ (nats)")
    Ax.grid(alpha=0.25)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#                MODULE 6 - DENSITY PROFILES  [Figure 1]
# =============================================================================
ProfileBinA = 0.25
ProfileSpecies = ["Zr", "C", "O"]


def CircularZrCentre(FracX):
    Theta = 2.0 * np.pi * FracX
    return float(np.arctan2(np.sin(Theta).mean(), np.cos(Theta).mean())
                 / (2.0 * np.pi)) % 1.0


def ProfileFromFrame(Sp, Co, La, Hists, NBins):
    """Accumulate species histograms of Zr-COM-centred x (fractional shift)."""
    Inv = np.linalg.inv(La)
    Frac = (Co @ Inv) % 1.0
    Lx = float(np.linalg.norm(La[0]))
    Centre = CircularZrCentre(Frac[Sp == SpeciesCodes["Zr"], 0])
    X = ((Frac[:, 0] - Centre + 0.5) % 1.0) * Lx
    for Name in ProfileSpecies:
        Sel = Sp == SpeciesCodes[Name]
        Hists[Name] += np.histogram(X[Sel], bins=NBins, range=(0.0, Lx))[0]
    return Lx


def Module6(Jobs):
    OutDir = SubDir("Profiles")
    import BoundaryConventions as Bcv

    Lx0 = 42.41
    NBins = int(np.ceil(Lx0 / ProfileBinA))
    GridProfiles, RawDiffs = {}, {}
    for Temp in Temperatures:
        PerTrajMeans = {Name: [] for Name in ProfileSpecies}
        MaxDiff = 0.0
        for Sim in SimIndices:
            HistsGrid = {Name: np.zeros(NBins) for Name in ProfileSpecies}
            NGridFrames = 0
            for TimePs, Sp, Co, La, Tr, Sn in IterGridFrames(
                    Temp, Sim, WindowPs=TerminalWindowPs):
                Lx = ProfileFromFrame(Sp, Co, La, HistsGrid, NBins)
                NGridFrames += 1
            # Raw-frame variant (POTIM bias check): every raw frame 126-140 ps.
            HistsRaw = {Name: np.zeros(NBins) for Name in ProfileSpecies}
            NRawFrames = 0
            Offset = Bcv.FindOffsetAtTime(
                Bc.XyzPath(Temp, Sim), TerminalWindowPs[0] * 1000.0, HorizonFs)
            for TimeFs, Atoms, Coords, Lattice in Ct.ReadFramesFrom(
                    Bc.XyzPath(Temp, Sim), Offset):
                if TimeFs < TerminalWindowPs[0] * 1000.0:
                    continue
                if TimeFs > HorizonFs:
                    break
                Codes = np.array([SpeciesCodes.get(A, 9) for A in Atoms], dtype=np.int8)
                ProfileFromFrame(Codes, Coords, Lattice, HistsRaw, NBins)
                NRawFrames += 1
            BinVol = ProfileBinA * 9.456 * 9.462
            for Name in ProfileSpecies:
                Grid = HistsGrid[Name] / max(1, NGridFrames) / BinVol
                Raw = HistsRaw[Name] / max(1, NRawFrames) / BinVol
                PerTrajMeans[Name].append(Grid)
                MaxDiff = max(MaxDiff, float(np.max(np.abs(Grid - Raw))))
        GridProfiles[Temp] = {Name: (np.mean(PerTrajMeans[Name], axis=0),
                                     np.std(PerTrajMeans[Name], axis=0, ddof=1))
                              for Name in ProfileSpecies}
        RawDiffs[Temp] = MaxDiff
        print(f"  {Temp} K: POTIM bias check max |grid - raw| = "
              f"{MaxDiff:.4f} atoms/A^3", flush=True)

    MakeFigure1(GridProfiles, Lx0, NBins, os.path.join(OutDir, "Fig1_Profiles"))
    Payload = {"PotimBiasMaxDiff_atomsPerA3": {str(T): RawDiffs[T] for T in Temperatures},
               "Bin_A": ProfileBinA,
               "Window": "126.0-140.0 ps (fixed 14 ps terminal window), 50 fs grid",
               "Centring": "Zr centre of mass (circular mean), x axis"}
    SaveModuleJson("6_Profiles", Payload)
    return Payload


def MakeFigure1(GridProfiles, Lx, NBins, OutBase):
    Edges = np.linspace(0.0, Lx, NBins + 1)
    Centres = 0.5 * (Edges[:-1] + Edges[1:]) - Lx / 2.0
    SpeciesColours = {"Zr": "#7f7f7f", "C": "#2f2f2f", "O": "#d62728"}
    Fig, Axes = plt.subplots(4, 1, figsize=(7.5, 10.5), sharex=True, sharey=True)
    for Ax, Temp in zip(Axes, Temperatures):
        for Name in ProfileSpecies:
            Mean, Sd = GridProfiles[Temp][Name]
            Ax.plot(Centres, Mean, color=SpeciesColours[Name], lw=1.4, label=Name)
            Ax.fill_between(Centres, Mean - Sd, Mean + Sd,
                            color=SpeciesColours[Name], alpha=0.20, lw=0)
        Ax.set_ylabel(f"{Temp} K\nnumber density (A$^{{-3}}$)", fontsize=9)
        Ax.grid(alpha=0.2)
    Axes[0].legend(fontsize=9, ncol=3)
    Axes[-1].set_xlabel("x - Zr centre of mass (A)")
    Fig.suptitle("Species-resolved density, fixed 14 ps terminal window "
                 "(126-140 ps), $\\pm$1 SD over trajectories", fontsize=10)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#           MODULE 7 - CHAIN STRUCTURE  [Figure 5 + text numbers]
# =============================================================================
def GroupNameOf(Temp):
    return GroupBelow if Temp in (873, 973) else GroupAbove


def SpeciesNamesArray(Sp):
    Lookup = np.array(["Zr", "O", "C", "N"] + ["X"] * 6, dtype=object)
    return Lookup[np.clip(Sp, 0, 9)]


def Module7(Jobs):
    OutDir = SubDir("Chains")
    GroupAccum = {G: {F: {"Termini": [], "ChainObs": 0, "BridgingObs": 0}
                      for F in (1.5, 1.3)} for G in (GroupBelow, GroupAbove)}
    RingSizes = {G: [] for G in (GroupBelow, GroupAbove)}
    BranchObs = {}          # (Temp, L) -> [chains, branched]
    LengthDist140 = {G: [] for G in (GroupBelow, GroupAbove)}
    DistinctChains = {G: set() for G in (GroupBelow, GroupAbove)}
    AngleRecords = {G: [] for G in (GroupBelow, GroupAbove)}   # (angle, L)
    TripletsByGroup = {G: [] for G in (GroupBelow, GroupAbove)}

    OrigTol = Ct.RadiusTol
    for Temp in Temperatures:
        Group = GroupNameOf(Temp)
        for Sim in SimIndices:
            ActiveTriplets = {}
            for TimePs, Sp, Co, La, Tr, Sn in IterGridFrames(
                    Temp, Sim, WindowPs=TerminalWindowPs):
                Atoms = SpeciesNamesArray(Sp)
                Frac = Co @ np.linalg.inv(La)
                Confirmed = {int(ID): StateNames[int(Sn[ID])]
                             for ID in range(Bc.TotalStartC)}
                for Factor in (1.5, 1.3):
                    Ct.RadiusTol = Factor
                    Topo = Ct.AnalyseFrameTopology(Atoms, Frac, La, Confirmed, Tr)
                    Acc = GroupAccum[Group][Factor]
                    Acc["Termini"].extend(Topo["Termini"])
                    Acc["ChainObs"] += Topo["NChains"]
                    Acc["BridgingObs"] += sum(
                        1 for C in Topo["ChainRecords"] if C["Bridging"])
                    if Factor == 1.5:
                        RingSizes[Group].extend(Topo["RingSizes"])
                        for Rec in Topo["ChainRecords"]:
                            Key = (Temp, Rec["Length"])
                            BranchObs.setdefault(Key, [0, 0])
                            BranchObs[Key][0] += 1
                            BranchObs[Key][1] += int(Rec["NBranchPoints"] >= 1)
                        # Distinct chains: member tracked-ID sets over the window.
                        CFull = np.where(Atoms == "C")[0]
                        FullToId = {int(CFull[I]): int(Tr[I]) for I in range(len(Tr))}
                        for Comp in Topo["Components"]:
                            DistinctChains[Group].add(
                                frozenset(FullToId[A] for A in Comp))
                Ct.RadiusTol = OrigTol
                # Instantaneous angles tagged by length + persistent triplets.
                FrameAngles, CurrentTriplets = Cg.AnalyseFrameForGeometry(
                    Atoms, Frac, La, Confirmed, Tr, TimePs)
                AngleRecords[Group].extend([(A, L) for (_, A, L) in FrameAngles])
                for Key, (V1, V2, Ang, L) in CurrentTriplets.items():
                    if Key in ActiveTriplets:
                        St = ActiveTriplets[Key]
                        St["SumV1"] += V1
                        St["SumV2"] += V2
                        St["Inst"].append(Ang)
                        St["LastTime"] = TimePs * 1000.0
                        St["Count"] += 1
                    else:
                        ActiveTriplets[Key] = {
                            "StartTime": TimePs * 1000.0, "LastTime": TimePs * 1000.0,
                            "SumV1": V1.copy(), "SumV2": V2.copy(),
                            "Inst": [Ang], "Count": 1, "StartL": L}
                for Key in [K for K in ActiveTriplets if K not in CurrentTriplets]:
                    Rec = Cg.FinalizeTriplet(ActiveTriplets.pop(Key), Temp, Group)
                    if Rec:
                        TripletsByGroup[Group].append(Rec)
            for St in ActiveTriplets.values():
                Rec = Cg.FinalizeTriplet(St, Temp, Group)
                if Rec:
                    TripletsByGroup[Group].append(Rec)
            # Length distribution at the horizon frame.
            TimePs, Sp, Co, La, Tr, Sn = LastGridFrame(Temp, Sim)
            ChainFull, Adj = ChainAdjacencyAtFrame(Sp, Co, La, Tr, Sn)
            LengthDist140[Group].extend(ChainComponentSizes(ChainFull, Adj))
    Ct.RadiusTol = OrigTol

    CrossTab = {}
    for Group in (GroupBelow, GroupAbove):
        CrossTab[Group] = {}
        for Factor in (1.5, 1.3):
            Acc = GroupAccum[Group][Factor]
            NT = max(1, len(Acc["Termini"]))
            CrossTab[Group][str(Factor)] = {
                Cat: sum(1 for T in Acc["Termini"] if T[Cat]) / NT
                for Cat in ("CarbideAnchored", "OxideAnchored",
                            "OxygenBonded", "Dangling")}
            CrossTab[Group][str(Factor)]["BridgingChainFraction"] = (
                Acc["BridgingObs"] / Acc["ChainObs"] if Acc["ChainObs"] else float("nan"))
            CrossTab[Group][str(Factor)]["NTermini"] = len(Acc["Termini"])
            CrossTab[Group][str(Factor)]["NChainObs"] = Acc["ChainObs"]

    BranchByTempLength = {}
    for (Temp, L), (NChainsObs, NBranched) in sorted(BranchObs.items()):
        BranchByTempLength[f"{Temp}_L{L}"] = {
            "Chains": NChainsObs, "PBranched": NBranched / NChainsObs,
            "Thin": bool(NChainsObs < ThinChains)}

    Angles = {}
    for Group in (GroupBelow, GroupAbove):
        Angles[Group] = {}
        for L in (3, 4):
            Vals = np.array([A for (A, AL) in AngleRecords[Group] if AL == L])
            TAvg = np.array([R["TimeAveragedAngle"] for R in TripletsByGroup[Group]
                             if R["StartL"] == L])
            Angles[Group][f"L{L}"] = {
                "MedianInstantaneous": float(np.median(Vals)) if len(Vals) else float("nan"),
                "FractionAbove150": float(np.mean(Vals > 150)) if len(Vals) else float("nan"),
                "NInstantaneous": int(len(Vals)),
                "MedianTimeAveraged": float(np.median(TAvg)) if len(TAvg) else float("nan"),
                "NTriplets": int(len(TAvg)),
                "Thin": bool(len(Vals) < ThinEvents)}

    Rings = {G: {"N3Cycles": sum(1 for R in RingSizes[G] if R == 3),
                 "NLargerCycles": sum(1 for R in RingSizes[G] if R > 3),
                 "Note": "3-cycles reported separately as cutoff artefacts"}
             for G in (GroupBelow, GroupAbove)}

    MakeFigure5(BranchByTempLength, os.path.join(OutDir, "Fig5_BranchProbability"))
    Payload = {"CrossTab": CrossTab, "Rings": Rings,
               "BranchByTempLength": BranchByTempLength,
               "LengthDistribution140": {G: sorted(LengthDist140[G])
                                         for G in LengthDist140},
               "DistinctChainsInWindow": {G: len(DistinctChains[G])
                                          for G in DistinctChains},
               "Angles": Angles,
               "Window": "126-140 ps, 50 fs grid"}
    SaveModuleJson("7_Chains", Payload)
    return Payload


def MakeFigure5(BranchByTempLength, OutBase):
    Fig, Ax = plt.subplots(figsize=(7.4, 5.0))
    Markers = {873: "o", 973: "s", 1073: "^", 1273: "D"}
    Offsets = {873: -0.18, 973: -0.06, 1073: 0.06, 1273: 0.18}
    for Temp in Temperatures:
        Xs, Ys = [], []
        for Key, E in sorted(BranchByTempLength.items()):
            T, L = Key.split("_L")
            if int(T) == Temp and E["Chains"] >= 3:
                Xs.append(int(L) + Offsets[Temp])
                Ys.append(E["PBranched"])
        Ax.plot(Xs, Ys, Markers[Temp], color=TemperatureColours[Temp],
                markersize=9, label=f"{Temp} K")
    Ax.set_xlabel("Chain length $L$ (atoms)")
    Ax.set_ylabel("P(chain contains a branch point)")
    Ax.legend(title="Temperature", fontsize=9)
    Ax.grid(alpha=0.25)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#                 MODULE 8 - TEMPORAL  [new subsection]
# =============================================================================
def Module8(Jobs):
    OutDir = SubDir("Temporal")

    # ---- a. fate curves --------------------------------------------------------
    Fates = {T: {"Gas": [], "Chain": []} for T in Temperatures}
    ChainThenGas = {T: 0 for T in Temperatures}
    for Temp in Temperatures:
        for Sim in SimIndices:
            State = LoadStateRecord(Temp, Sim)
            Gas = {int(K): V for K, V in State["FirstGasPs"].items()}
            Chain = {int(K): V for K, V in State["FirstChainPs"].items()}
            for ID in set(Gas) | set(Chain):
                TG = Gas.get(ID, float("inf"))
                TC = Chain.get(ID, float("inf"))
                if TC < TG:
                    Fates[Temp]["Chain"].append(TC)
                    if np.isfinite(TG):
                        ChainThenGas[Temp] += 1
                else:
                    Fates[Temp]["Gas"].append(TG)
    T50 = {}
    for Temp in Temperatures:
        T50[str(Temp)] = {}
        for Kind in ("Gas", "Chain"):
            Times = np.sort(Fates[Temp][Kind])
            T50[str(Temp)][f"t50_{Kind.lower()}"] = (
                float(Times[int(np.ceil(len(Times) / 2)) - 1]) if len(Times)
                else float("nan"))
            T50[str(Temp)][f"N_{Kind.lower()}"] = int(len(Times))
    MakeFateFigure(Fates, os.path.join(OutDir, "FateCurves"))

    # ---- b. developmental clock -------------------------------------------------
    ClockWindows = {"EARLY": (40.0, 90.0), "LATE": (90.0, 140.0)}
    WindowStats = {G: {W: {"Lengths": [], "Branched": 0, "ChainObs": 0,
                           "Distinct": set()}
                       for W in ClockWindows} for G in (GroupBelow, GroupAbove)}
    ClockSamples = []
    SampleTimes = np.arange(40.0, HorizonPs + 1e-9, 10.0)
    for Temp in Temperatures:
        Group = GroupNameOf(Temp)
        for Sim in SimIndices:
            Data = np.load(GridCachePath(Temp, Sim))
            CumEver = Data["CumulativeEver"]
            Times = Data["Times"] / 1000.0
            for TimePs, Sp, Co, La, Tr, Sn in IterGridFrames(
                    Temp, Sim, WindowPs=(40.0, HorizonPs)):
                OnSample = any(abs(TimePs - S) <= 0.026 for S in SampleTimes)
                InWindows = [W for W, (Lo, Hi) in ClockWindows.items()
                             if Lo <= TimePs <= Hi]
                if not OnSample and not InWindows:
                    continue
                ChainFull, Adj = ChainAdjacencyAtFrame(Sp, Co, La, Tr, Sn)
                Comps = Ct.ConnectedComponents(ChainFull, Adj)
                Sizes = [len(C) for C in Comps]
                NBranched = sum(1 for C in Comps if any(
                    len(Adj[A]) >= 3 for A in C))
                CFull = np.where(SpeciesNamesArray(Sp) == "C")[0]
                FullToId = {int(CFull[I]): int(Tr[I]) for I in range(len(Tr))}
                for W in InWindows:
                    WS = WindowStats[Group][W]
                    WS["Lengths"].extend(Sizes)
                    WS["Branched"] += NBranched
                    WS["ChainObs"] += len(Comps)
                    for C in Comps:
                        WS["Distinct"].add(frozenset(FullToId[A] for A in C))
                if OnSample and Sizes:
                    Idx = int(np.argmin(np.abs(Times - TimePs)))
                    ClockSamples.append({
                        "Temp": Temp, "Time_ps": TimePs,
                        "CumulativeChainCarbons": int(CumEver[Idx]),
                        "MeanChainLength": float(np.mean(Sizes)),
                        "MaxChainLength": int(max(Sizes)),
                        "BranchFraction": NBranched / len(Comps)})

    ClockTable = {}
    for Group in (GroupBelow, GroupAbove):
        ClockTable[Group] = {}
        for W in ClockWindows:
            WS = WindowStats[Group][W]
            ClockTable[Group][W] = {
                "DistinctChains": len(WS["Distinct"]),
                "MeanLength": float(np.mean(WS["Lengths"])) if WS["Lengths"] else float("nan"),
                "MaxLength": int(max(WS["Lengths"])) if WS["Lengths"] else 0,
                "BranchFraction": WS["Branched"] / WS["ChainObs"] if WS["ChainObs"] else float("nan"),
                "Thin": bool(len(WS["Distinct"]) < ThinChains)}

    CollapseVerdict = AssessCollapse(ClockSamples)
    MakeCollapseFigure(ClockSamples, os.path.join(OutDir, "ProgressCollapse"))

    EarlyAbove = WindowStats[GroupAbove]["EARLY"]["Lengths"]
    LateBelow = WindowStats[GroupBelow]["LATE"]["Lengths"]
    LengthComparison = {
        "EarlyAbove": {"N": len(EarlyAbove),
                       "Mean": float(np.mean(EarlyAbove)) if EarlyAbove else float("nan"),
                       "Hist": {str(L): int(np.sum(np.array(EarlyAbove) == L))
                                for L in sorted(set(EarlyAbove))}},
        "LateBelow": {"N": len(LateBelow),
                      "Mean": float(np.mean(LateBelow)) if LateBelow else float("nan"),
                      "Hist": {str(L): int(np.sum(np.array(LateBelow) == L))
                               for L in sorted(set(LateBelow))}}}

    # ---- c. turnover -------------------------------------------------------------
    Turnover = {}
    for Temp in Temperatures:
        Ever, Retained, ToGas, ToSolid = 0, 0, 0, 0
        ExitDurations, CensoredDurations = [], []
        for Sim in SimIndices:
            State = LoadStateRecord(Temp, Sim)
            RetainedIDs = set(State["RetainedIDs"])
            LastExit = {}
            for E in State["ClosedEpisodes"]:
                LastExit[E["AtomID"]] = E["ExitTo"]
            EverIDs = set(int(K) for K in State["FirstChainPs"])
            Ever += len(EverIDs)
            Retained += len(RetainedIDs)
            for ID in EverIDs - RetainedIDs:
                if LastExit.get(ID) == "Gas":
                    ToGas += 1
                else:
                    ToSolid += 1
            ExitDurations.extend([E["End_ps"] - E["Start_ps"]
                                  for E in State["ClosedEpisodes"]])
            for ID in RetainedIDs:
                Start = State["OpenEpisodes"].get(
                    str(ID), State["LastChainEntryPs"][str(ID)])
                CensoredDurations.append(HorizonPs - Start)
        Turnover[str(Temp)] = {
            "EverChained": Ever, "RetainedAt140": Retained,
            "ExitedToGas": ToGas, "ExitedToSolid": ToSolid,
            "RetentionFraction": Retained / Ever if Ever else float("nan"),
            "ChainThenGasFirstFateChain": ChainThenGas[Temp],
            "ExitResidenceMedian_ps": float(np.median(ExitDurations)) if ExitDurations else float("nan"),
            "ExitResidenceIqr_ps": ([float(np.percentile(ExitDurations, 25)),
                                     float(np.percentile(ExitDurations, 75))]
                                    if ExitDurations else [float("nan")] * 2),
            "CensoredResidenceMedian_ps": float(np.median(CensoredDurations)) if CensoredDurations else float("nan"),
            "NClosedEpisodes": len(ExitDurations)}

    Payload = {"T50": T50, "ClockTable": ClockTable,
               "CollapseVerdict": CollapseVerdict,
               "LengthComparison": LengthComparison,
               "Turnover": Turnover,
               "FateRule": "first confirmed state (state machine times); "
                           "chain-then-gas carbons keep Chain as first fate and "
                           "are tallied under turnover"}
    SaveModuleJson("8_Temporal", Payload)
    return Payload


def AssessCollapse(ClockSamples):
    """Common curve vs temperature offsets: quadratic pooled fit of
    MeanChainLength vs CumulativeChainCarbons; per-temperature mean residual
    compared against the pooled residual SD."""
    X = np.array([S["CumulativeChainCarbons"] for S in ClockSamples], dtype=float)
    Y = np.array([S["MeanChainLength"] for S in ClockSamples])
    Temps = np.array([S["Temp"] for S in ClockSamples])
    if len(X) < 8:
        return {"Verdict": "insufficient samples", "N": int(len(X))}
    Coef = np.polyfit(X, Y, 2)
    Resid = Y - np.polyval(Coef, X)
    PooledSd = float(Resid.std(ddof=1))
    PerTemp = {str(T): float(Resid[Temps == T].mean()) for T in Temperatures}
    Exceed = [T for T in Temperatures if abs(PerTemp[str(T)]) > PooledSd]
    if not Exceed:
        Verdict = "common curve"
    elif len(Exceed) >= 3:
        Verdict = "temperature offsets"
    else:
        Verdict = f"mixed (offsets at {Exceed})"
    return {"Verdict": Verdict, "PerTempMeanResidual": PerTemp,
            "PooledResidualSD": PooledSd, "N": int(len(X)),
            "Rule": "offset if |per-T mean residual| > pooled residual SD "
                    "(quadratic pooled fit, MeanChainLength vs progress)"}


def MakeFateFigure(Fates, OutBase):
    Fig, Axes = plt.subplots(2, 2, figsize=(10.0, 7.5), sharex=True, sharey=True)
    for Ax, Temp in zip(Axes.ravel(), Temperatures):
        for Kind, Style in (("Gas", "-"), ("Chain", "--")):
            Times = np.sort(Fates[Temp][Kind])
            if len(Times) == 0:
                continue
            Frac = np.arange(1, len(Times) + 1) / len(Times)
            Ax.step(np.concatenate([[0], Times, [HorizonPs]]),
                    np.concatenate([[0], Frac, [1.0]]), Style,
                    where="post", color=TemperatureColours[Temp], lw=1.8,
                    label=f"{Kind.lower()} (n={len(Times)})")
        Ax.set_title(f"{Temp} K", fontsize=10)
        Ax.legend(fontsize=8)
        Ax.grid(alpha=0.25)
    for Ax in Axes[-1]:
        Ax.set_xlabel("Time (ps)")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel("Cumulative fraction of final count")
    Fig.suptitle("First-fate cumulative incidence: desorbed (solid) vs "
                 "chain-retained (dashed)", fontsize=11)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


def MakeCollapseFigure(ClockSamples, OutBase):
    Fig, Axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    Markers = {873: "o", 973: "s", 1073: "^", 1273: "D"}
    for Ax, Field, Label in ((Axes[0], "MeanChainLength", "Mean chain length"),
                             (Axes[1], "BranchFraction", "Branch fraction")):
        for Temp in Temperatures:
            Xs = [S["CumulativeChainCarbons"] for S in ClockSamples
                  if S["Temp"] == Temp]
            Ys = [S[Field] for S in ClockSamples if S["Temp"] == Temp]
            Ax.plot(Xs, Ys, Markers[Temp], color=TemperatureColours[Temp],
                    alpha=0.75, markersize=6, label=f"{Temp} K")
        Ax.set_xlabel("Cumulative chain carbons (progress variable)")
        Ax.set_ylabel(Label)
        Ax.grid(alpha=0.25)
    Axes[0].legend(fontsize=8)
    Fig.suptitle("Progress-variable collapse, 10 ps sampling from 40 ps",
                 fontsize=11)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#              MODULE 9 - OXYGEN LAYERING  [3 sentences + SI]
# =============================================================================
def ExtractLayeringWindow(Temp, Sim, WindowPs):
    """Re-drive of OxygenLayeringAnalysis.ExtractTrajectory on the shared grid
    cache for an arbitrary window (functions imported from Ola; only the frame
    source differs). Returns the same record structure."""
    NBins = int(round((Ola.UHistMax - Ola.UHistMin) / Ola.ProfileBin))
    Surface = {Side: {"UHist": np.zeros(NBins), "Frames": [], "NFrames": 0}
               for Side in Ola.SideLabels}
    InteriorSpacings, CrystalFrames, OxideOFrames = [], [], []
    NoFront = 0
    for TimePs, Sp, Co, La, Tr, Sn in IterGridFrames(Temp, Sim, WindowPs=WindowPs):
        Geo = Ola.AnalyseFrameGeometry(Sp, Co, La)
        if Geo is None:
            NoFront += 1
            continue
        InteriorSpacings.extend(Geo["InteriorSpacings"])
        XA, IsO = Geo["XA"], Geo["IsO"]
        Y, Z = Co[:, 1], Co[:, 2]
        OxideMask = np.zeros(len(XA), dtype=bool)
        for Side in Ola.SideLabels:
            U = (Geo["FrontL"] - XA) if Side == "L" else (XA - Geo["FrontR"])
            UCap = min(Ola.UHistMax, Geo["HalfGas"])
            Sel = IsO & (U > Ola.UHistMin) & (U < UCap)
            Rec = Surface[Side]
            Rec["UHist"] += np.histogram(U[Sel], bins=NBins,
                                         range=(Ola.UHistMin, Ola.UHistMax))[0]
            Rec["Frames"].append({"U": U[Sel].astype(np.float32),
                                  "Y": Y[Sel].astype(np.float32),
                                  "Z": Z[Sel].astype(np.float32)})
            Rec["NFrames"] += 1
            OxideMask |= Sel & (U > Ola.OoOxideUMin) & (U < Ola.OoOxideUMax)
        if OxideMask.sum() >= 2:
            OxideOFrames.append((Geo["Frac"][OxideMask], Geo["Lattice"]))
        if len(Geo["InteriorPlanes"]) > 0:
            Centre = 0.5 * (Geo["FrontL"] + Geo["FrontR"])
            PlaneX = Geo["InteriorPlanes"][
                int(np.argmin(np.abs(Geo["InteriorPlanes"] - Centre)))]
            CSel = Geo["IsC"] & (np.abs(XA - PlaneX) <= Ola.LayerWindow)
            if CSel.sum() >= 2:
                CrystalFrames.append((Y[CSel].copy(), Z[CSel].copy(),
                                      Geo["Ly"], Geo["Lz"]))
    return {"Surface": Surface, "InteriorSpacings": InteriorSpacings,
            "CrystalFrames": CrystalFrames, "OxideOFrames": OxideOFrames,
            "NoFrontFrames": NoFront}


def Module9(Jobs):
    OutDir = SubDir("Layering")
    Extr = {(T, S): ExtractLayeringWindow(T, S, TerminalWindowPs)
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

    # In-plane g2D pooled per (temperature, layer index)
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
            NP = Ola.AccumulatePairHistogram2d(
                Y.astype(np.float64), Z.astype(np.float64), FLy, FLz,
                CrystalAccum[T][0], Ola.G2dRLimit, Ola.G2dBin)
            CrystalAccum[T][1] += NP
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
        SpacingV, Evidence = Ola.SpacingRegistryVerdict(
            SurfList, DZrCByTemp[T], OoByTemp[T][0])
        Verdicts[str(T)] = {
            "Spacing": SpacingV, "SpacingEvidence": Evidence,
            "Decay": Ola.DecayVerdict(SurfList),
            "InPlane": Ola.InPlaneVerdict(
                [G2dChar.get((T, L)) for L in sorted(
                    LI for (TT, LI) in G2dChar if TT == T)])}

    # Mid-trajectory spacing/registry check: 1273 K, 60-74 ps, spacing only.
    MidCheck = {}
    for S in SimIndices:
        E = ExtractLayeringWindow(1273, S, (60.0, 74.0))
        DMid = float(np.median(E["InteriorSpacings"])) if E["InteriorSpacings"] else float("nan")
        for Side in Ola.SideLabels:
            R = Ola.AnalyseSurfaceLayers(E["Surface"][Side], DMid)
            Key = f"1273_{S}_{Side}"
            if R is None or len(R.get("LayerU", [])) < 2:
                MidCheck[Key] = {"Layers": 0 if R is None else len(R.get("LayerU", []))}
                continue
            MidCheck[Key] = {"Layers": len(R["LayerU"]),
                             "MeanSpacing": R["MeanSpacing"],
                             "RatioToDZrC": R["MeanSpacing"] / DMid,
                             "Drift": R["OffsetDriftPerLayer"]}

    SurfaceTable = {}
    for (T, S, Side), R in AllSurfaces.items():
        Key = f"{T}_{S}_{Side}"
        if R is None:
            SurfaceTable[Key] = None
            continue
        SurfaceTable[Key] = {K: V for K, V in R.items()
                             if K not in ("Profile", "PeakIdx")}

    Payload = {"DZrCByTemp": {str(T): DZrCByTemp[T] for T in Temperatures},
               "OoPackingByTemp": {str(T): OoByTemp[T][0] for T in Temperatures},
               "Verdicts": {T: {K: V for K, V in D.items()
                                if K != "SpacingEvidence"}
                            for T, D in Verdicts.items()},
               "Surfaces": SurfaceTable,
               "InPlane": {f"{T}_layer{L}": C for (T, L), C in G2dChar.items()},
               "CrystalReference": {str(T): C for T, C in CrystalChar.items()},
               "MidTrajectoryCheck_1273_60_74ps": MidCheck,
               "Window": "126-140 ps, 50 fs grid; thresholds identical to "
                         "OxygenLayeringAnalysis (imported)"}
    SaveModuleJson("9_Layering", Payload)
    return Payload


# =============================================================================
#             MODULE 10 - REGIME SUMMARY  [Figure 6]
# =============================================================================
def Module10(Jobs):
    OutDir = SubDir("RegimeSummary")
    Part = LoadModuleJson("2_Partitioning")
    Surv = LoadModuleJson("3_Survival")
    Entr = LoadModuleJson("5_Entropy")

    MmToInch = 1.0 / 25.4
    matplotlib.rcParams["svg.hashsalt"] = "Fig6_Refresh140"
    Fig = plt.figure(figsize=(180 * MmToInch, 230 * MmToInch))
    Grid = Fig.add_gridspec(5, 1, height_ratios=[1.0, 1.0, 1.0, 1.0, 0.42],
                            hspace=0.14, left=0.105, right=0.975, top=0.975,
                            bottom=0.055)
    AxA = Fig.add_subplot(Grid[0])
    AxB = Fig.add_subplot(Grid[1], sharex=AxA)
    AxC = Fig.add_subplot(Grid[2], sharex=AxA)
    AxD = Fig.add_subplot(Grid[3], sharex=AxA)
    AxBand = Fig.add_subplot(Grid[4], sharex=AxA)

    for Ax in (AxA, AxB, AxC, AxD):
        Ax.axvspan(923, 1023, color="#d62728", alpha=0.12, zorder=0, lw=0)
        Ax.grid(axis="y", color="#dddddd", lw=0.6)
        Ax.set_axisbelow(True)
        Ax.set_xlim(823, 1323)
        Ax.set_xticks(Temperatures)
        Ax.tick_params(axis="x", labelbottom=False, length=2)
        Ax.tick_params(axis="both", labelsize=8)

    def Bars(Ax, Points):
        for T in Temperatures:
            V, Lo, Hi = Points[T]
            Ax.errorbar([T], [V], yerr=[[V - Lo], [Hi - V]], fmt="o",
                        color=TemperatureColours[T], capsize=3, markersize=5,
                        elinewidth=1.3, capthick=1.3, zorder=3)

    # (a) phi_C
    PhiPoints = {T: (Part["Headline"]["PerTemperature"][str(T)]["Observed"],
                     Part["Headline"]["PerTemperature"][str(T)]["P2_5"],
                     Part["Headline"]["PerTemperature"][str(T)]["P97_5"])
                 for T in Temperatures}
    Bars(AxA, PhiPoints)
    AxA.axhline(1.0, color="grey", ls="--", lw=1.0)
    AxA.text(1170, 1.03, "retention = gasification", fontsize=7.5, color="grey",
             ha="center", va="bottom")
    for T in Temperatures:
        P = Part["DominanceProbability"][str(T)]["P_below_1"]
        AxA.text(T, PhiPoints[T][2] + 0.13, f"$P(<1)$ = {P:.2f}", fontsize=7.5,
                 ha="center", color=TemperatureColours[T])
    AxA.set_ylim(0, 4.1)
    AxA.set_ylabel("Carbon retention ratio $\\phi_C$", fontsize=9)
    AxA.text(973, 3.98, "gasification window (bracketed)", fontsize=7.5,
             color="#a03030", ha="center", va="top", style="italic")
    AxA.text(0.012, 0.955, "(a)", transform=AxA.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # (b) Lambda, log y, with stored-extrapolation line
    import math
    LambdaPoints = {T: (Surv["Fits"][str(T)]["Exponential"]["Lambda"],
                        Surv["Fits"][str(T)]["Exponential"]["CI_Lower"],
                        Surv["Fits"][str(T)]["Exponential"]["CI_Upper"])
                    for T in Temperatures}
    Arr = Surv["ArrheniusReplacement"]
    LineT = np.linspace(823, 1323, 200)
    AxB.plot(LineT, np.exp(Arr["Intercept"] + Arr["Slope"] / LineT), "--",
             color="grey", lw=1.0)
    AxB.text(1318, math.exp(Arr["Intercept"] + Arr["Slope"] / 1318) * 0.70,
             "873-973 K Arrhenius line", fontsize=7.5, color="grey",
             ha="right", va="top")
    Bars(AxB, LambdaPoints)
    AxB.set_yscale("log")
    AxB.set_ylabel("Desorption rate $\\lambda$ (ps$^{-1}$)", fontsize=9)
    AxB.text(0.012, 0.955, "(b)", transform=AxB.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # (c) rho
    RhoPoints = {T: (Surv["Fits"][str(T)]["Weibull"]["Rho"],
                     Surv["Fits"][str(T)]["Weibull"]["Rho_CI_Lower"],
                     Surv["Fits"][str(T)]["Weibull"]["Rho_CI_Upper"])
                 for T in Temperatures}
    Bars(AxC, RhoPoints)
    AxC.axhline(1.0, color="grey", ls="--", lw=1.0)
    AxC.text(1318, 1.005, "constant rate", fontsize=7.5, color="grey", ha="right")
    P1273 = Surv["Fits"]["1273"]["LRT_p"]
    AxC.annotate(f"$p$ = {P1273:.4f}", xy=(1273, RhoPoints[1273][0]),
                 xytext=(1250, RhoPoints[1273][0] - 0.015), fontsize=7.5,
                 ha="right", va="top", color=TemperatureColours[1273])
    AxC.set_ylabel("Hazard shape $\\rho$", fontsize=9)
    AxC.text(0.012, 0.955, "(c)", transform=AxC.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # (d) entropy (mean +/- SD from Module 5)
    EntropyPoints = {T: (Entr["PerTemperature"][str(T)]["Mean"],
                         Entr["PerTemperature"][str(T)]["Mean"]
                         - Entr["PerTemperature"][str(T)]["SD"],
                         Entr["PerTemperature"][str(T)]["Mean"]
                         + Entr["PerTemperature"][str(T)]["SD"])
                     for T in Temperatures}
    Bars(AxD, EntropyPoints)
    AxD.set_ylabel("Oxide disorder\n(Shannon entropy, nats)", fontsize=9)
    AxD.text(0.012, 0.955, "(d)", transform=AxD.transAxes, fontsize=10,
             fontweight="bold", va="top")

    # Stage band
    AxBand.set_ylim(0, 1)
    AxBand.set_yticks([])
    AxBand.set_xlabel("Temperature (K)", fontsize=9)
    AxBand.tick_params(axis="x", labelbottom=True, labelsize=8)
    for Spine in ("top", "right", "left"):
        AxBand.spines[Spine].set_visible(False)
    for Lo, Hi, Colour in [(823, 1000, "#e3ecf5"), (1000, 1200, "#f7f0dd"),
                           (1200, 1323, "#f6e2e2")]:
        AxBand.axvspan(Lo, Hi, ymin=0.30, ymax=1.0, color=Colour, zorder=0, lw=0)
    for Boundary in (1000, 1200):
        AxBand.plot([Boundary, Boundary], [0.30, 1.0], ls=":", color="#888888",
                    lw=1.0, clip_on=False)
    for Centre, Label in [(911, "protective: compact,\ncarbon-retaining scale"),
                          (1100, "linear kinetics\nfrom outset"),
                          (1261, "scale failure\nregime")]:
        AxBand.text(Centre, 0.65, Label, fontsize=7.5, ha="center", va="center")
    AxBand.text(1320, 0.10, "macroscopic stages after Lin et al.; alignment "
                "qualitative (simulation flux ~2.5 orders higher)",
                fontsize=6.5, style="italic", ha="right", color="#555555")

    for Ext in ("png", "pdf", "svg"):
        Fig.savefig(os.path.join(OutDir, f"Fig6_RegimeSummary.{Ext}"), dpi=300)
    plt.close(Fig)
    Payload = {"Sources": ["Module_2_Partitioning", "Module_3_Survival",
                           "Module_5_Entropy"],
               "PanelD_Spread": "mean +/- between-trajectory SD (Module 5)"}
    SaveModuleJson("10_RegimeSummary", Payload)
    return Payload


# =============================================================================
#                       MASTER SUMMARY + RESULTS NUMBERS
# =============================================================================
OldEntropy = {873: (1.615, None), 973: (1.700, 0.04),
              1073: (1.670, None), 1273: (1.695, 0.017)}   # values in the paper


def Get(Mapping, *Keys, Default="-"):
    Node = Mapping
    for K in Keys:
        if not isinstance(Node, dict) or K not in Node:
            return Default
        Node = Node[K]
    return Node


def LoadOldJson(*PathParts):
    Path = os.path.join(ScriptDir, *PathParts)
    if not os.path.exists(Path):
        return {}
    with open(Path) as F:
        return json.load(F)


def FmtNum(X, Nd=3):
    if isinstance(X, str):
        return X
    if X is None or (isinstance(X, float) and not np.isfinite(X)):
        return "-"
    return f"{X:.{Nd}f}"


def BuildMaster(FluxConvention):
    Gates = LoadOldJson("Refresh140", "Gates.json")
    Audit = LoadOldJson("GasSupplyAudit", "GasSupplyAudit.json")
    OldBoot = LoadOldJson("BootstrapResults", "BootstrapResults.json")
    OldSurv = LoadOldJson("SurvivalResults", "SurvivalResults.json")
    OldTopo = LoadOldJson("ChainTopologyResults", "TopologyResults.json")
    OldGeom = LoadOldJson("ChainGeometryResults", "GeometryResults.json")
    OldLayer = LoadOldJson("OxygenLayeringResults", "OxygenLayeringResults.json")
    M = {N: LoadModuleJson(N) for N in
         ("1_Events", "2_Partitioning", "3_Survival", "4_ArrheniusFits",
          "5_Entropy", "6_Profiles", "7_Chains", "8_Temporal", "9_Layering")}

    Numbers = {}     # flat map: key -> {"Old": ..., "New": ...}

    def Add(Key, Old, New):
        Numbers[Key] = {"Old": Old, "New": New}

    # ---- collect the numbers quoted in the paper -----------------------------------
    for T in Temperatures:
        Ts = str(T)
        W = Audit["PerTemperature"][Ts]["WindowStats"]["0-140"]["free"]
        Add(f"Table1_P_bar_{T}", "-", round(W["P_bar_TempMean"], 1))
        Add(f"Table1_J_1e27_{T}", "-", round(W["J_TempMean"] / 1e27, 1))
        E = M["2_Partitioning"]["Headline"]["PerTemperature"][Ts]
        Add(f"PhiC_Observed_{T}",
            Get(OldBoot, "PerTemperature", Ts, "Observed"), E["Observed"])
        Add(f"PhiC_CI_{T}",
            [Get(OldBoot, "PerTemperature", Ts, "P2_5"),
             Get(OldBoot, "PerTemperature", Ts, "P97_5")],
            [E["P2_5"], E["P97_5"]])
        Add(f"P_PhiC_below1_{T}",
            Get(OldBoot, "FollowUpStatistics", "DominanceProbability", Ts, "P_below_1"),
            M["2_Partitioning"]["DominanceProbability"][Ts]["P_below_1"])
        F = M["3_Survival"]["Fits"][Ts]
        Add(f"Lambda_perPs_{T}",
            Get(OldSurv, "Fits", Ts, "Exponential", "Lambda"),
            F["Exponential"]["Lambda"])
        Add(f"Rho_{T}", Get(OldSurv, "Fits", Ts, "Weibull", "Rho"),
            F["Weibull"]["Rho"])
        Add(f"LRT_p_{T}", Get(OldSurv, "Fits", Ts, "LRT_p"), F["LRT_p"])
        Add(f"InterEventCV_{T}", Get(OldSurv, "InterEventCV", Ts, "CV"),
            M["3_Survival"]["InterEventCV"][Ts]["CV"])
        Add(f"Entropy_nats_{T}", OldEntropy[T][0],
            M["5_Entropy"]["PerTemperature"][Ts]["Mean"])
        Ev = M["1_Events"]["Totals"][Ts]
        Add(f"CO_{T}", OldCO[T], Ev["CO"])
        Add(f"CO2_{T}", OldCO2[T], Ev["CO2"])
        Add(f"ChainRetained_{T}", OldRetained[T], Ev["Retained"])
        Add(f"ChainCumulative_{T}", OldCumulative[T], Ev["Cumulative"])
        T50 = M["8_Temporal"]["T50"][Ts]
        Add(f"t50_gas_ps_{T}", "-", T50["t50_gas"])
        Add(f"t50_chain_ps_{T}", "-", T50["t50_chain"])
        Add(f"RetentionFraction_{T}", "-",
            M["8_Temporal"]["Turnover"][Ts]["RetentionFraction"])
    Add("Ordering_Primary_973min",
        Get(OldBoot, "PrimaryFraction_973_below_873_and_1073"),
        M["2_Partitioning"]["Headline"]["PrimaryFraction_973_below_873_and_1073"])
    Add("Ordering_973GlobalMin", Get(OldBoot, "Fraction_973_global_minimum"),
        M["2_Partitioning"]["Headline"]["Fraction_973_global_minimum"])
    for K in ("Monotonic_973_1073_1273", "VShape_873_973_1073_1273",
              "Reference_1073_1273"):
        Add(f"Ordering_{K}",
            Get(OldBoot, "FollowUpStatistics", "OrderingProbability", K),
            M["2_Partitioning"]["Ordering"][K])
    for Obs in ("CumulativeChain", "NetMassAmu"):
        R = M["4_ArrheniusFits"]["Results"][FluxConvention][Obs]["MeanFit"]
        Add(f"Ea_eV_{Obs}_{FluxConvention}", "-",
            [round(R["Ea_eV"], 3), round(R["SigmaEa_eV"], 3), round(R["R2"], 3)])
    for G, GOldKey in ((GroupBelow, "Below"), (GroupAbove, "Above")):
        for Factor in ("1.5", "1.3"):
            Add(f"BridgingFraction_f{Factor}_{GOldKey}",
                Get(OldTopo, GOldKey, "BridgingChainFraction")
                if Factor == "1.5" else "-",
                M["7_Chains"]["CrossTab"][G][Factor]["BridgingChainFraction"])
        for L in (3, 4):
            Add(f"AngleMedian_L{L}_{GOldKey}",
                Get(OldGeom, "FixedLengthAngles", f"L{L}_{GOldKey}", "MedianAngle"),
                M["7_Chains"]["Angles"][G][f"L{L}"]["MedianInstantaneous"])
    for T in Temperatures:
        Key = f"{T}_L4"
        Add(f"BranchP_L4_{T}",
            Get(OldGeom, "BranchByTempLength", Key, "Pbranched"),
            Get(M["7_Chains"]["BranchByTempLength"], Key, "PBranched"))
    Add("LayeringVerdict_Spacing_1073",
        Get(OldLayer, "Verdicts", "1073", "Spacing"),
        M["9_Layering"]["Verdicts"]["1073"]["Spacing"])
    Add("LayeringVerdict_Spacing_1273",
        Get(OldLayer, "Verdicts", "1273", "Spacing"),
        M["9_Layering"]["Verdicts"]["1273"]["Spacing"])

    with open(os.path.join(RootDir, "ResultsNumbers.json"), "w") as F:
        json.dump(Numbers, F, indent=1, default=JsonSafe)

    WriteMasterSummary(Gates, Audit, M, Numbers, FluxConvention)


def WriteMasterSummary(Gates, Audit, M, Numbers, FluxConvention):
    Ppt = Audit["Meta"]["PotimAudit"]["FramesPerPsByTemp"]
    L = []
    L.append("# Results refresh at the 140.0 ps horizon (ZrC final dataset)")
    L.append("")
    L.append(f"Gate: 16/16 >= 140.0 ps: {'yes' if Gates['GatePass'] else 'no'} | "
             f"Dataset check (140-147 ps spread): "
             f"{'pass' if Gates['DatasetPass'] else 'fail'}")
    L.append("Seed 20260715 | Persistence filter time-based: yes")
    L.append(f"FluxConvention primary: {FluxConvention} | POTIM frames/ps by T: "
             f"{Ppt['873']:.0f} / {Ppt['973']:.0f} / {Ppt['1073']:.0f} / "
             f"{Ppt['1273']:.0f}")
    L.append("")
    L.append("## Changed numbers (quoted in the paper)")
    L.append("| Quantity | Old | New |")
    L.append("|---|---|---|")
    for Key, V in Numbers.items():
        OldS = FmtNum(V["Old"]) if not isinstance(V["Old"], list) else \
            "[" + ", ".join(FmtNum(X) for X in V["Old"]) + "]"
        NewS = FmtNum(V["New"]) if not isinstance(V["New"], list) else \
            "[" + ", ".join(FmtNum(X) for X in V["New"]) + "]"
        L.append(f"| {Key} | {OldS} | {NewS} |")
    L.append("")
    L.append("## Tables of the paper")
    L.append("")
    Window, Label = {"free_full": ("0-140", "free"),
                     "free_40_140": ("40-140", "free"),
                     "far_40_140": ("40-140", "far")}[FluxConvention]
    L.append(f"Table 1 (gas supply; J_{Label}, {Window} ps - "
             f"per FluxConvention {FluxConvention}):")
    L.append("| T (K) | P +/- SD (bar) | J +/- SD (10^27 m^-2 s^-1) |")
    L.append("|---:|---:|---:|")
    for T in Temperatures:
        W = Audit["PerTemperature"][str(T)]["WindowStats"][Window][Label]
        L.append(f"| {T} | {W['P_bar_TempMean']:.1f} +/- "
                 f"{W.get('P_bar_TrajSD', float('nan')):.1f} | "
                 f"{W['J_TempMean'] / 1e27:.1f} +/- {W['J_TrajSD'] / 1e27:.1f} |")
    L.append("")
    L.append("Table 2 (carbon partitioning at 140.0 ps):")
    L.append("| T (K) | phi_C [2.5-97.5%] | P(phi_C<1) | phi_O | CO/CO2 | "
             "Chain length mean / max |")
    L.append("|---:|---|---:|---:|---:|---|")
    for T in Temperatures:
        Ts = str(T)
        E = M["2_Partitioning"]["Headline"]["PerTemperature"][Ts]
        T2 = M["2_Partitioning"]["Table2"][Ts]
        P1 = M["2_Partitioning"]["DominanceProbability"][Ts]["P_below_1"]
        L.append(f"| {T} | {E['Observed']:.3f} [{E['P2_5']:.3f}, {E['P97_5']:.3f}] "
                 f"| {P1:.2f} | {T2['PhiO']:.2f} | {T2['CoOverCo2']:.1f} | "
                 f"{T2['ChainLengthMean_140ps']:.2f} / {T2['ChainLengthMax_140ps']} |")
    L.append("")
    L.append("Table 3 (desorption kinetics, censoring 140.0 ps):")
    L.append("| T (K) | lambda (ps^-1) [95% CI] | rho [95% CI] | LRT p | CV | "
             "LOO rho range |")
    L.append("|---:|---|---|---:|---:|---|")
    for T in Temperatures:
        F = M["3_Survival"]["Fits"][str(T)]
        E, W = F["Exponential"], F["Weibull"]
        Loo = F["LeaveOneOut_Rho_range"]
        L.append(f"| {T} | {E['Lambda']:.5f} [{E['CI_Lower']:.5f}, "
                 f"{E['CI_Upper']:.5f}] | {W['Rho']:.3f} [{W['Rho_CI_Lower']:.3f}, "
                 f"{W['Rho_CI_Upper']:.3f}] | {F['LRT_p']:.4f} | "
                 f"{M['3_Survival']['InterEventCV'][str(T)]['CV']:.2f} | "
                 f"[{Loo[0]:.3f}, {Loo[1]:.3f}] |")
    L.append("")
    L.append(f"Table 4 (flux-normalised ensemble Arrhenius, primary = "
             f"{FluxConvention}):")
    L.append("| Observable | Ea (eV, mean fit n=4) | R^2 | Ea (eV, trajectory "
             "fit n=16) | R^2 |")
    L.append("|---|---|---:|---|---:|")
    for Obs in ("CumulativeChain", "NetMassAmu"):
        R = M["4_ArrheniusFits"]["Results"][FluxConvention][Obs]
        Mf, Tf = R["MeanFit"], R["TrajFit"]
        L.append(f"| {Obs} | {Mf['Ea_eV']:.3f} +/- {Mf['SigmaEa_eV']:.3f} | "
                 f"{Mf['R2']:.3f} | {Tf['Ea_eV']:.3f} +/- {Tf['SigmaEa_eV']:.3f} | "
                 f"{Tf['R2']:.3f} |")
    Reg = M["4_ArrheniusFits"]["Results"][FluxConvention]["NetMassAmu"].get(
        "RegimeFit_1073_1273_Traj", {})
    Des = M["4_ArrheniusFits"]["Results"][FluxConvention]["Desorption"]["MeanFit"]
    L.append(f"SI: net-mass regime fit 1073-1273 K (trajectory level): Ea "
             f"{FmtNum(Reg.get('Ea_eV'))} +/- {FmtNum(Reg.get('SigmaEa_eV'))} eV, "
             f"R^2 {FmtNum(Reg.get('R2'))}. Diagnostic full-range desorption "
             f"mean fit R^2 = {Des['R2']:.3f}.")
    L.append("")
    L.append("## New-material summaries")
    L.append("")
    L.append("Temporal - t50 (ps) and turnover:")
    L.append("| T (K) | t50_gas | t50_chain | Ever chained | Retained | "
             "To gas | To solid | Retention |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for T in Temperatures:
        Ts = str(T)
        T50 = M["8_Temporal"]["T50"][Ts]
        Tu = M["8_Temporal"]["Turnover"][Ts]
        L.append(f"| {T} | {T50['t50_gas']:.1f} | {T50['t50_chain']:.1f} | "
                 f"{Tu['EverChained']} | {Tu['RetainedAt140']} | "
                 f"{Tu['ExitedToGas']} | {Tu['ExitedToSolid']} | "
                 f"{Tu['RetentionFraction']:.2f} |")
    CV = M["8_Temporal"]["CollapseVerdict"]
    L.append(f"Progress collapse verdict: {CV['Verdict']} (rule: {CV.get('Rule', '-')}; "
             f"n = {CV.get('N', '-')}).")
    L.append("")
    L.append("Layering verdicts (126-140 ps window):")
    L.append("| Measurement | 1073 K | 1273 K | (873/973 K controls) |")
    L.append("|---|---|---|---|")
    V9 = M["9_Layering"]["Verdicts"]
    for Meas in ("Spacing", "Decay", "InPlane"):
        L.append(f"| {Meas} | {V9['1073'][Meas]} | {V9['1273'][Meas]} | "
                 f"{V9['873'][Meas]} / {V9['973'][Meas]} |")
    Mid = M["9_Layering"]["MidTrajectoryCheck_1273_60_74ps"]
    MidParts = [f"{K}: d/d_ZrC {FmtNum(E.get('RatioToDZrC'), 2)}"
                for K, E in sorted(Mid.items()) if E.get("Layers", 0) >= 2]
    L.append(f"Mid-trajectory check (1273 K, 60-74 ps, spacing only): "
             + ("; ".join(MidParts) if MidParts else "no surface with >= 2 layers")
             + ".")
    L.append("")
    Potim = M["6_Profiles"]["PotimBiasMaxDiff_atomsPerA3"]
    L.append("POTIM bias check (max |grid - raw| density, atoms/A^3): "
             + " / ".join(f"{T} K {Potim[str(T)]:.4f}" for T in Temperatures) + ".")
    L.append("")
    L.append("Module 4 three-convention Ea comparison (eV, mean fits):")
    L.append("| Observable | free_full | free_40_140 | far_40_140 | sigma (primary) |")
    L.append("|---|---:|---:|---:|---:|")
    for Obs in ("CumulativeChain", "NetMassAmu"):
        C = M["4_ArrheniusFits"]["Comparison"][Obs]
        L.append(f"| {Obs} | {C['Ea_eV']['free_full']:.3f} | "
                 f"{C['Ea_eV']['free_40_140']:.3f} | {C['Ea_eV']['far_40_140']:.3f} | "
                 f"{C['PrimarySigma']:.3f} |")
    L.append("")
    L.append("## Notes")
    for N in MasterNotes(M, FluxConvention):
        L.append(f"- {N}")
    Text = "\n".join(L)
    with open(os.path.join(RootDir, "MASTER_SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print("\nMASTER_SUMMARY.md and ResultsNumbers.json written.", flush=True)


def MasterNotes(M, FluxConvention):
    Notes = []
    Notes.append(
        f"FluxConvention: the dispatch placeholder was never filled; the "
        f"default '{FluxConvention}' (the audit's Table 1 candidate) is used "
        "as primary. All three conventions are computed; the three-convention "
        "table shows the spread "
        + ("exceeds" if any(M["4_ArrheniusFits"]["Comparison"][O]["SpreadExceedsSigma"]
                            for O in ("CumulativeChain", "NetMassAmu")) else "stays within")
        + " the fitted sigma per observable.")
    Notes.append(
        "Persistence filter: time-based (0.2 ps vs physical frame times), "
        "evaluated on every raw MD frame during the shared pass - not on the "
        "50 fs grid (the grid only samples structure). Horizon snapshots use "
        "confirmed states with pending transitions NOT flushed "
        "(BoundaryConventions convention). The superseded pipeline flushed "
        "pendings at trajectory end; together with the dataset update this "
        "moves 1273 K retained 66 -> 69 and cumulative 69 -> 73 - a shift in "
        "degree, not kind (phi_C ordering and the 973 K minimum survive).")
    Notes.append(
        "Old-column availability: Table 1, Ea, t50, retention and phi_O have "
        "no machine-readable superseded values ('-'); event totals use the "
        "task-supplied reference; bootstrap/survival/layering olds come from "
        "the superseded results JSONs (read-only).")
    Notes.append(
        "Zero-denominator bootstrap rounds: "
        + ", ".join(f"{T} K {M['2_Partitioning']['ZeroDenominator'][str(T)]}"
                    for T in Temperatures)
        + " of 10,000 (NaN rounds excluded pairwise per the superseded rule).")
    Notes.append(
        "Thin strata: branch-by-length rows with < 5 chain observations and "
        "angle strata with < 20 samples are flagged in Module_7_Chains.json; "
        "clock windows flag < 5 distinct chains (Module_8_Temporal.json).")
    Notes.append(
        "Layering contamination flags (molecular O2 in outermost 'layers', "
        "front-adjacent u~1.1 A peaks) carry forward from the "
        "OxygenLayeringAnalysis thresholds - flagged rows are listed in "
        "Module_9_Layering.json InPlane entries with O2Contamination=true.")
    Notes.append(
        "lifelines was missing on this machine and installed (0.30.0) so the "
        "Weibull fits use the identical library path as the superseded run.")
    Notes.append(
        "All previous results folders remain untouched read-only reference; "
        "their frame caches were NOT reused (stale after the in-place dataset "
        "replacement). Gas-supply numbers imported from "
        "GasSupplyAudit/GasSupplyAudit.json (built on the final dataset).")
    return Notes


# =============================================================================
#                                  DRIVER
# =============================================================================
ModuleRegistry = {}


def RegisterModules(FluxConvention):
    ModuleRegistry.update({
        1: ("Events", Module1),
        2: ("Partitioning", Module2),
        3: ("Survival", Module3),
        4: ("ArrheniusFits", lambda Jobs: Module4(Jobs, FluxConvention)),
        5: ("Entropy", Module5),
        6: ("Profiles", Module6),
        7: ("Chains", Module7),
        8: ("Temporal", Module8),
        9: ("Layering", Module9),
        10: ("RegimeSummary", Module10),
    })


def Main():
    Parser = argparse.ArgumentParser(description="Authoritative 140 ps results refresh.")
    Parser.add_argument("--jobs", type=int,
                        default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Parser.add_argument("--modules", type=str, default="all")
    Parser.add_argument("--flux-convention", type=str, default="free_full",
                        choices=list(FluxConventions))
    Parser.add_argument("--force-cache", action="store_true")
    Args = Parser.parse_args()

    os.makedirs(RootDir, exist_ok=True)
    RegisterModules(Args.flux_convention)

    EndTimes, GatePass, DatasetPass = RunGates()
    print("Gate:", "PASS" if GatePass else "FAIL",
          "| Dataset check:", "PASS" if DatasetPass else "FAIL")
    for K in sorted(EndTimes):
        print(f"  {K}: {EndTimes[K]:.3f} ps")
    with open(os.path.join(RootDir, "Gates.json"), "w") as F:
        json.dump({"EndTimes_ps": EndTimes, "GatePass": GatePass,
                   "DatasetPass": DatasetPass,
                   "FluxConvention": Args.flux_convention}, F, indent=1)
    if not (GatePass and DatasetPass):
        print("STOP: gates failed.")
        return 1

    if Args.force_cache:
        EnsureSharedPass(Args.jobs, Force=True)

    if Args.modules == "master":
        BuildMaster(Args.flux_convention)
        return 0
    Wanted = (sorted(ModuleRegistry) if Args.modules == "all"
              else [int(X) for X in Args.modules.split(",")])
    for M in Wanted:
        if M not in ModuleRegistry:
            print(f"Module {M} not implemented yet; skipping.")
            continue
        Name, Fn = ModuleRegistry[M]
        print(f"\n=== Module {M} ({Name}) ===", flush=True)
        Fn(Args.jobs)
    if all(os.path.exists(os.path.join(RootDir, f"Module_{N}.json")) for N in
           ("1_Events", "2_Partitioning", "3_Survival", "4_ArrheniusFits",
            "5_Entropy", "6_Profiles", "7_Chains", "8_Temporal", "9_Layering")):
        BuildMaster(Args.flux_convention)
    else:
        print("Master summary deferred: not all module JSONs present yet.")
    return 0


if __name__ == "__main__":
    sys.exit(Main())
