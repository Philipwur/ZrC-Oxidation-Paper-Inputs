"""
GasSupplyAudit.py
=======================================================================
Gas-phase supply audit of the FINAL ZrC dataset (140.0 ps uniform horizon).

Quantifies O2 partial pressure and Hertz-Knudsen impingement flux under three
O2 population definitions, measures non-dissociative O2 recycling
(bounce-back), and tests whether "flux held comparable across temperatures"
survives. No carbon/chain analysis here.

Gates (both enforced before any analysis; the script STOPS and reports if
either fails):
  1. all 16 trajectories reach >= 140.0 ps;
  2. the dataset is the UPDATED one: per-trajectory total times spread
     140-147 ps with most in 140-140.5 ps (the OLD dataset had several
     trajectories well below 140 ps).

Conventions
-----------
* Uniform horizon 140.0 ps; all times physical (fs -> ps).
* Evaluation on a uniform 50 fs grid, nearest raw frame per grid point.
* Streaming: xyz files are read frame-by-frame; never loaded whole.
* Nothing stochastic; seed 20260715 quoted for consistency only.
* Bond criteria as established: O-O bonded if r < (2*0.66)*1.5 = 1.98 A;
  Zr-O surface-association cutoff 3.17 A; slab-network bonds use
  (Ri+Rj)*1.5 with radii Zr 1.45, O 0.66, C 0.76, N 0.71; minimum image.
* The data folder is READ-ONLY. All outputs live in GasSupplyAudit/.
  The legacy partial-pressure product (--with-legacy-pp) is therefore
  written to GasSupplyAudit/partial_pressure/, NOT into the data folder.

Three O2 populations (per grid frame)
-------------------------------------
  NTotal : all intact O2 (unique greedy O-O pairing at 1.98 A) anywhere.
  NFree  : O2 with no O atom within 3.17 A of any Zr (legacy protocol
           definition behind the published flux).
  NFar   : O2 with BOTH atoms > 4.0 A from every slab atom, slab = every
           atom in a bonded connected component containing Zr (paired-O2
           atoms are excluded from the network before the component search
           so a hovering molecule cannot count as its own slab contact).

P = N kB T / Vgas ;  J = (N / Vgas) * sqrt(kB T / (2 pi mO2)).
Vgas per the protocol void-fraction method (imported from
postprocessing/CalculatePartialPressure._FrameVolumeZrDistance when
importable, else an identical local reimplementation - recorded in outputs).

GasSupplyAudit.json key map (stable; consumed by the full refresh)
------------------------------------------------------------------
  Meta                : Seed, Grid_fs, Horizon_ps, GatePass, DatasetCheckPass,
                        VgasMethod, BondCutoffs_A, WindowDefs_ps, Area_m2,
                        TrajectoryEndTimes_ps, PotimAudit (per trajectory:
                        MeanSpacing_fs / MinSpacing_fs / MaxSpacing_fs /
                        FramesPerPs; per temperature: FramesPerPs)
  PerTrajectory[id]   : Series (Times_ps, NTotal, NFree, NFar, Vgas_m3,
                        Pfree_bar, Pfar_bar, Jfree_m2s, Jfar_m2s),
                        WindowStats[window][def] = {P_bar_Mean, P_bar_SD,
                        J_Mean, J_SD}, BounceEvents (list of
                        {Start_ps, End_ps, Residence_ps}), Uptakes_ps (list of
                        {Time_ps, EverFree, Origin}; EverFree=False marks
                        surface-born transient O-O pairs, not arriving gas),
                        AddedMolecules, AddedDesorbedAtHorizon,
                        EdgeCases {Exchanges, GasPhaseDissociations,
                        VanishedIntact, Reassociated}
  PerTemperature[T]   : WindowStats[window][def] = {J_TempMean, J_TrajSD,
                        P_bar_TempMean, P_bar_TrajSD},
                        Inflation[window] = Jfree/Jfar,
                        BounceCensus[window] = {Bounces, MedianResidence_ps,
                        IqrResidence_ps, Impingements, BouncePerImpingement,
                        Uptakes, UptakesGasOrigin, UptakesSurfaceBorn},
                        Sticking[window] = {Uptakes (gas-origin only),
                        UptakesAllInclSurfaceBorn, Impingements, Probability,
                        PoissonSigma}
  Comparability       : [window][def] = {TempMeans, MaxOverMin, CV,
                        Within15Pct}
"""

import os
import re
import sys
import json
import argparse

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ScriptDir = os.path.dirname(os.path.abspath(__file__))
DataDir = os.environ.get("ZRC_DATA_DIR", os.path.join(ScriptDir, "..", "data"))
ResultsDir = os.path.join(ScriptDir, "GasSupplyAudit")

RandomSeed = 20260715           # quoted for consistency; nothing stochastic
Temperatures = [873, 973, 1073, 1273]
SimIndices = [1, 2, 3, 4]

HorizonPs = 140.0
GridFs = 50.0
GridTargetsFs = np.arange(0.0, HorizonPs * 1000.0 + 0.5 * GridFs, GridFs)

CovalentRadii = {"Zr": 1.45, "O": 0.66, "C": 0.76, "N": 0.71}
RadiusTol = 1.5
OoCutoff = 2.0 * CovalentRadii["O"] * RadiusTol          # 1.98 A
ZrOCutoff = 3.17                                          # A, as established
FarCutoff = 4.0                                           # A from any slab atom
IdentityMatchMax = 1.5                                    # A, O tracking between grid frames

Kb = 1.380649e-23                                         # J / K
MO2 = 32.0 * 1.66054e-27                                  # kg

Windows = {"0-140": (0.0, 140.0), "0-40": (0.0, 40.0),
           "40-140": (40.0, 140.0), "126-140": (126.0, 140.0)}
WindowOrder = ["0-140", "0-40", "40-140", "126-140"]
BounceWindows = ["0-40", "40-140"]

TemperatureColours = {873: "#1f77b4", 973: "#d62728", 1073: "#2ca02c", 1273: "#9467bd"}

TimeRegex = re.compile(r"Time\s*=\s*([\d.]+)")
LatticeRegex = re.compile(r'Lattice="([^"]+)"')


# =============================================================================
#                    Vgas: import the framework function if possible
# =============================================================================
def _LocalFrameVolumeZrDistance(CellMat, InvCell, Elements, CartArray):
    """Identical reimplementation of the protocol void-fraction volume:
    largest contiguous gap (incl. wrap) between Zr atoms along x."""
    CellVolA3 = abs(np.dot(CellMat[0], np.cross(CellMat[1], CellMat[2])))
    ZrMask = np.array([E == "Zr" for E in Elements])
    if np.any(ZrMask):
        FracX = np.sort((CartArray[ZrMask] @ InvCell)[:, 0] % 1.0)
        Diffs = np.diff(FracX)
        WrapGap = 1.0 - FracX[-1] + FracX[0]
        GasFraction = max(WrapGap, np.max(Diffs)) if Diffs.size > 0 else WrapGap
    else:
        GasFraction = 1.0
    return CellVolA3 * GasFraction * 1e-30, CellVolA3 * 1e-30


def ResolveVgasFunction():
    """Try the framework implementation first; fall back to the local copy."""
    try:
        sys.path.insert(0, os.environ.get("SGUSCHI_POSTPROCESSING", os.path.join(ScriptDir, "..", "postprocessing")))
        from CalculatePartialPressure import _FrameVolumeZrDistance
        return _FrameVolumeZrDistance, "imported (CalculatePartialPressure._FrameVolumeZrDistance)"
    except Exception as Err:
        return _LocalFrameVolumeZrDistance, f"reimplemented (import failed: {type(Err).__name__})"


# =============================================================================
#                          STREAMING + GATES
# =============================================================================
def XyzPath(Temp, Sim):
    return os.path.join(DataDir, f"{Temp}_{Sim}.xyz")


def ReadFinalTimePs(Path):
    """Final frame time from the file tail; never loads the file."""
    Size = os.path.getsize(Path)
    with open(Path, "rb") as Fh:
        Fh.seek(max(0, Size - 65536))
        Tail = Fh.read().decode("utf-8", errors="replace")
    Matches = TimeRegex.findall(Tail)
    return float(Matches[-1]) / 1000.0 if Matches else float("nan")


def IterRawFrames(Path):
    """Stream raw frames: (TimeFs, Lattice, AtomLines). Atom lines are kept
    unparsed (strings) so only grid-selected frames pay float parsing."""
    with open(Path, "r") as Fh:
        while True:
            Header = Fh.readline()
            if not Header:
                return
            Stripped = Header.strip()
            if not Stripped:
                continue
            try:
                NAtoms = int(Stripped)
            except ValueError:
                return
            Comment = Fh.readline()
            AtomLines = [Fh.readline() for _ in range(NAtoms)]
            if not AtomLines or len(AtomLines[-1].split()) < 4:
                return                      # truncated final frame: drop it
            TimeMatch = TimeRegex.search(Comment)
            LatticeMatch = LatticeRegex.search(Comment)
            if not TimeMatch or not LatticeMatch:
                continue
            Lattice = np.array([float(X) for X in LatticeMatch.group(1).split()]
                               ).reshape(3, 3)
            yield float(TimeMatch.group(1)), Lattice, AtomLines


def ParseAtomLines(AtomLines):
    Species = np.empty(len(AtomLines), dtype=object)
    Coords = np.empty((len(AtomLines), 3), dtype=float)
    for I, Line in enumerate(AtomLines):
        Parts = Line.split()
        Species[I] = Parts[0]
        Coords[I, 0] = float(Parts[1])
        Coords[I, 1] = float(Parts[2])
        Coords[I, 2] = float(Parts[3])
    return Species, Coords


# =============================================================================
#                       PER-FRAME GEOMETRY AND POPULATIONS
# =============================================================================
def MinImageDistanceMatrix(FracA, FracB, Lattice):
    Diff = FracA[:, None, :] - FracB[None, :, :]
    Diff -= np.round(Diff)
    return np.linalg.norm(Diff @ Lattice, axis=-1)


def MatchO2Pairs(DistOO):
    """Greedy shortest-first unique O-O pairing below OoCutoff (the
    established protocol rule). Returns list of (i, j) index pairs into the
    O-atom subarray."""
    N = DistOO.shape[0]
    TriI, TriJ = np.triu_indices(N, k=1)
    Mask = DistOO[TriI, TriJ] < OoCutoff
    CandI, CandJ, CandD = TriI[Mask], TriJ[Mask], DistOO[TriI, TriJ][Mask]
    Order = np.argsort(CandD)
    Used = np.zeros(N, dtype=bool)
    Pairs = []
    for K in Order:
        I, J = int(CandI[K]), int(CandJ[K])
        if not Used[I] and not Used[J]:
            Pairs.append((I, J))
            Used[I] = Used[J] = True
    return Pairs


def ConnectedComponentsBool(Adjacency):
    N = Adjacency.shape[0]
    Visited = np.zeros(N, dtype=bool)
    Components = []
    for Start in range(N):
        if Visited[Start]:
            continue
        Queue = [Start]
        Current = []
        while Queue:
            Node = Queue.pop()
            if Visited[Node]:
                continue
            Visited[Node] = True
            Current.append(Node)
            Queue.extend(np.where(Adjacency[Node])[0].tolist())
        Components.append(Current)
    return Components


def AnalyseGridFrame(Species, Coords, Lattice, VgasFunction):
    """All per-frame quantities. Returns a dict."""
    Inv = np.linalg.inv(Lattice)
    Frac = Coords @ Inv

    OIdx = np.where(Species == "O")[0]
    ZrIdx = np.where(Species == "Zr")[0]
    FracO = Frac[OIdx]

    DistOO = MinImageDistanceMatrix(FracO, FracO, Lattice)
    Pairs = MatchO2Pairs(DistOO)

    DistOZr = (MinImageDistanceMatrix(FracO, Frac[ZrIdx], Lattice)
               if len(ZrIdx) else np.full((len(OIdx), 0), np.inf))
    MinToZr = DistOZr.min(axis=1) if DistOZr.size else np.full(len(OIdx), np.inf)

    # Slab network: all atoms except paired-O2 oxygens; bonds (Ri+Rj)*1.5.
    PairedO = set()
    for I, J in Pairs:
        PairedO.add(int(OIdx[I]))
        PairedO.add(int(OIdx[J]))
    NetIdx = np.array([K for K in range(len(Species)) if K not in PairedO])
    SlabIdx = np.array([], dtype=int)
    if len(NetIdx):
        FracNet = Frac[NetIdx]
        DistNet = MinImageDistanceMatrix(FracNet, FracNet, Lattice)
        Radii = np.array([CovalentRadii.get(Species[K], 0.8) for K in NetIdx])
        Cutoff = (Radii[:, None] + Radii[None, :]) * RadiusTol
        Adjacency = DistNet < Cutoff
        np.fill_diagonal(Adjacency, False)
        IsZrNet = Species[NetIdx] == "Zr"
        Slab = []
        for Component in ConnectedComponentsBool(Adjacency):
            if IsZrNet[Component].any():
                Slab.extend(Component)
        SlabIdx = NetIdx[np.array(Slab, dtype=int)] if Slab else np.array([], dtype=int)

    MinToSlab = (MinImageDistanceMatrix(FracO, Frac[SlabIdx], Lattice).min(axis=1)
                 if len(SlabIdx) else np.full(len(OIdx), np.inf))

    Molecules = []
    for I, J in Pairs:
        Assoc = bool(MinToZr[I] < ZrOCutoff or MinToZr[J] < ZrOCutoff)
        Far = bool(MinToSlab[I] > FarCutoff and MinToSlab[J] > FarCutoff)
        Molecules.append({"OSubIdx": (I, J), "Assoc": Assoc, "Far": Far})

    VgasM3, _CellM3 = VgasFunction(Lattice, Inv, list(Species), Coords)
    AreaM2 = float(np.linalg.norm(np.cross(Lattice[1], Lattice[2]))) * 1e-20

    return {"Molecules": Molecules, "OIdx": OIdx, "FracO": FracO,
            "Vgas_m3": VgasM3, "Area_m2": AreaM2,
            "NTotal": len(Molecules),
            "NFree": sum(not M["Assoc"] for M in Molecules),
            "NFar": sum(M["Far"] for M in Molecules)}


# =============================================================================
#                       O-ATOM IDENTITY TRACKING
# =============================================================================
def MapOxygenIds(PrevFracO, PrevIds, CurrFracO, Lattice, NextId):
    """Map previous O tracker IDs onto current O atoms. Same-count frames
    with small same-index displacements keep index order (protocol preserves
    atom order; additions append after the O block). Otherwise greedy
    nearest-neighbour minimum-image matching with a 1.5 A cap. Returns
    (CurrIds, NextId, VanishedIds, NewIds)."""
    if PrevFracO is None:
        Ids = np.arange(len(CurrFracO)) + NextId
        return Ids, NextId + len(CurrFracO), [], list(Ids)

    if len(PrevFracO) == len(CurrFracO):
        Diff = CurrFracO - PrevFracO
        Diff -= np.round(Diff)
        Disp = np.linalg.norm(Diff @ Lattice, axis=1)
        if Disp.max() < IdentityMatchMax:
            return PrevIds.copy(), NextId, [], []

    Dist = MinImageDistanceMatrix(PrevFracO, CurrFracO, Lattice)
    Order = np.dstack(np.unravel_index(np.argsort(Dist, axis=None), Dist.shape))[0]
    UsedPrev = np.zeros(len(PrevFracO), dtype=bool)
    UsedCurr = np.zeros(len(CurrFracO), dtype=bool)
    CurrIds = np.full(len(CurrFracO), -1, dtype=np.int64)
    for I, J in Order:
        if Dist[I, J] > IdentityMatchMax:
            break
        if UsedPrev[I] or UsedCurr[J]:
            continue
        UsedPrev[I] = UsedCurr[J] = True
        CurrIds[J] = PrevIds[I]
    NewIds = []
    for J in range(len(CurrFracO)):
        if CurrIds[J] < 0:
            CurrIds[J] = NextId
            NewIds.append(int(NextId))
            NextId += 1
    Vanished = [int(PrevIds[I]) for I in range(len(PrevIds)) if not UsedPrev[I]]
    return CurrIds, NextId, Vanished, NewIds


# =============================================================================
#                          PER-TRAJECTORY WORKER
# =============================================================================
def ProcessTrajectory(TaskArgs):
    Temp, Sim = TaskArgs
    VgasFunction, VgasMethod = ResolveVgasFunction()
    Path = XyzPath(Temp, Sim)

    Times, NTotalS, NFreeS, NFarS, VgasS = [], [], [], [], []
    AreaSamples = []
    SpacingSum, SpacingN = 0.0, 0
    SpacingMin, SpacingMax = np.inf, -np.inf
    PrevRaw = None                # (TimeFs, Lattice, AtomLines)
    PrevTimeFs = None
    TargetIdx = 0

    PrevFracO, PrevIds = None, None
    NextId = 0
    Active = {}                   # frozenset({idA,idB}) -> molecule state
    BounceEvents, UptakesPs = [], []
    EdgeCases = {"Exchanges": 0, "GasPhaseDissociations": 0,
                 "VanishedIntact": 0, "Reassociated": 0}
    AddedMoleculeKeys = set()
    FirstFrameDone = False

    def ProcessChosen(TimeFs, Lattice, AtomLines):
        nonlocal PrevFracO, PrevIds, NextId, FirstFrameDone
        Species, Coords = ParseAtomLines(AtomLines)
        Frame = AnalyseGridFrame(Species, Coords, Lattice, VgasFunction)
        TimePs = TimeFs / 1000.0
        Times.append(TimePs)
        NTotalS.append(Frame["NTotal"])
        NFreeS.append(Frame["NFree"])
        NFarS.append(Frame["NFar"])
        VgasS.append(Frame["Vgas_m3"])
        AreaSamples.append(Frame["Area_m2"])

        CurrIds, NextId, Vanished, NewIds = MapOxygenIds(
            PrevFracO, PrevIds, Frame["FracO"], Lattice, NextId)
        NewIdSet = set(NewIds)
        VanishedSet = set(Vanished)

        CurrentPairs = {}
        for M in Frame["Molecules"]:
            I, J = M["OSubIdx"]
            Key = frozenset((int(CurrIds[I]), int(CurrIds[J])))
            CurrentPairs[Key] = M["Assoc"]

        # --- update / open molecules -------------------------------------
        for Key, Assoc in CurrentPairs.items():
            if Key in Active:
                St = Active[Key]
                if St["State"] == "free" and Assoc:
                    St["State"] = "assoc"
                    St["AssocStartPs"] = TimePs
                elif St["State"] == "assoc" and not Assoc:
                    if St["EverFree"]:
                        BounceEvents.append({
                            "Start_ps": St["AssocStartPs"], "End_ps": TimePs,
                            "Residence_ps": TimePs - St["AssocStartPs"]})
                    St["State"] = "free"
                    St["AssocStartPs"] = None
                if not Assoc:
                    St["EverFree"] = True
                St["LastSeenPs"] = TimePs
            else:
                BothNew = all(A in NewIdSet for A in Key)
                if not FirstFrameDone:
                    Origin = "initial"
                elif BothNew:
                    Origin = "added"
                    AddedMoleculeKeys.add(Key)
                else:
                    Origin = "reassociated"
                    EdgeCases["Reassociated"] += 1
                Active[Key] = {"Origin": Origin, "State": "assoc" if Assoc else "free",
                               "AssocStartPs": TimePs if Assoc else None,
                               "EverFree": not Assoc, "FirstSeenPs": TimePs,
                               "LastSeenPs": TimePs}

        # --- close molecules that disappeared -----------------------------
        for Key in [K for K in Active if K not in CurrentPairs]:
            St = Active.pop(Key)
            if any(A in VanishedSet for A in Key):
                EdgeCases["VanishedIntact"] += 1
                continue
            if St["State"] == "assoc":
                UptakesPs.append({"Time_ps": TimePs, "EverFree": St["EverFree"],
                                  "Origin": St["Origin"]})
            else:
                InNewPair = any(A in P for A in Key for P in CurrentPairs)
                if InNewPair:
                    EdgeCases["Exchanges"] += 1
                else:
                    EdgeCases["GasPhaseDissociations"] += 1

        PrevFracO, PrevIds = Frame["FracO"], CurrIds
        FirstFrameDone = True

    for TimeFs, Lattice, AtomLines in IterRawFrames(Path):
        if PrevTimeFs is not None:
            Dt = TimeFs - PrevTimeFs
            if Dt > 0:
                SpacingSum += Dt
                SpacingN += 1
                SpacingMin = min(SpacingMin, Dt)
                SpacingMax = max(SpacingMax, Dt)
        PrevTimeFs = TimeFs

        while TargetIdx < len(GridTargetsFs) and TimeFs >= GridTargetsFs[TargetIdx]:
            Target = GridTargetsFs[TargetIdx]
            if PrevRaw is not None and abs(PrevRaw[0] - Target) < abs(TimeFs - Target):
                ProcessChosen(*PrevRaw)
            else:
                ProcessChosen(TimeFs, Lattice, AtomLines)
            TargetIdx += 1
        if TargetIdx >= len(GridTargetsFs):
            break
        PrevRaw = (TimeFs, Lattice, AtomLines)

    # Horizon survivors: protocol-added molecules, intact, ending free.
    AddedDesorbed = sum(1 for Key, St in Active.items()
                        if St["Origin"] == "added" and St["State"] == "free")

    Temperature = float(Temp)
    Series = {
        "Times_ps": Times, "NTotal": NTotalS, "NFree": NFreeS, "NFar": NFarS,
        "Vgas_m3": VgasS,
    }
    for Label, NSeries in [("free", NFreeS), ("far", NFarS)]:
        NArr = np.array(NSeries, dtype=float)
        VArr = np.array(VgasS, dtype=float)
        PPa = NArr * Kb * Temperature / VArr
        JFlux = (NArr / VArr) * np.sqrt(Kb * Temperature / (2.0 * np.pi * MO2))
        Series[f"P{Label}_bar"] = (PPa * 1e-5).tolist()
        Series[f"J{Label}_m2s"] = JFlux.tolist()

    return {
        "Temp": Temp, "Sim": Sim, "VgasMethod": VgasMethod,
        "Series": Series, "Area_m2": float(np.mean(AreaSamples)),
        "Potim": {"MeanSpacing_fs": SpacingSum / max(1, SpacingN),
                  "MinSpacing_fs": float(SpacingMin), "MaxSpacing_fs": float(SpacingMax),
                  "FramesPerPs": 1000.0 * SpacingN / SpacingSum if SpacingSum else float("nan")},
        "BounceEvents": BounceEvents, "Uptakes_ps": UptakesPs,
        "AddedMolecules": len(AddedMoleculeKeys),
        "AddedDesorbedAtHorizon": int(AddedDesorbed),
        "EdgeCases": EdgeCases,
        "NGridFrames": len(Times),
    }


# =============================================================================
#                       AGGREGATION (Steps 2-4)
# =============================================================================
def WindowMask(Times, Window):
    Lo, Hi = Windows[Window]
    T = np.asarray(Times)
    return (T >= Lo) & (T <= Hi)


def TrajectoryWindowStats(Result):
    Out = {}
    Times = Result["Series"]["Times_ps"]
    for Window in WindowOrder:
        Mask = WindowMask(Times, Window)
        Out[Window] = {}
        for Label in ("free", "far"):
            P = np.array(Result["Series"][f"P{Label}_bar"])[Mask]
            J = np.array(Result["Series"][f"J{Label}_m2s"])[Mask]
            Out[Window][Label] = {
                "P_bar_Mean": float(P.mean()), "P_bar_SD": float(P.std(ddof=1)),
                "J_Mean": float(J.mean()), "J_SD": float(J.std(ddof=1)),
            }
    return Out


def Aggregate(Results):
    PerTrajStats = {f"{R['Temp']}_{R['Sim']}": TrajectoryWindowStats(R) for R in Results}

    PerTemperature = {}
    for Temp in Temperatures:
        TempResults = [R for R in Results if R["Temp"] == Temp]
        Entry = {"WindowStats": {}, "Inflation": {}, "BounceCensus": {},
                 "Sticking": {}}
        AreaM2 = float(np.mean([R["Area_m2"] for R in TempResults]))
        for Window in WindowOrder:
            Entry["WindowStats"][Window] = {}
            for Label in ("free", "far"):
                Js = [PerTrajStats[f"{Temp}_{R['Sim']}"][Window][Label]["J_Mean"]
                      for R in TempResults]
                Ps = [PerTrajStats[f"{Temp}_{R['Sim']}"][Window][Label]["P_bar_Mean"]
                      for R in TempResults]
                Entry["WindowStats"][Window][Label] = {
                    "J_TempMean": float(np.mean(Js)),
                    "J_TrajSD": float(np.std(Js, ddof=1)),
                    "P_bar_TempMean": float(np.mean(Ps)),
                    "P_bar_TrajSD": float(np.std(Ps, ddof=1)),
                }
            Free = Entry["WindowStats"][Window]["free"]["J_TempMean"]
            Far = Entry["WindowStats"][Window]["far"]["J_TempMean"]
            Entry["Inflation"][Window] = Free / Far if Far > 0 else float("inf")

        for Window in BounceWindows:
            Lo, Hi = Windows[Window]
            DurationS = (Hi - Lo) * 1e-12
            Bounces = [E for R in TempResults for E in R["BounceEvents"]
                       if Lo <= E["End_ps"] <= Hi]
            Uptakes = [U for R in TempResults for U in R["Uptakes_ps"]
                       if Lo <= U["Time_ps"] <= Hi]
            GasOriginUptakes = [U for U in Uptakes if U["EverFree"]]
            JFreeMean = Entry["WindowStats"][Window]["free"]["J_TempMean"]
            # Impingements summed over the 4 trajectories (2 surfaces each).
            Impingements = JFreeMean * 2.0 * AreaM2 * DurationS * len(TempResults)
            Residences = np.array([E["Residence_ps"] for E in Bounces])
            Entry["BounceCensus"][Window] = {
                "Bounces": len(Bounces),
                "MedianResidence_ps": float(np.median(Residences)) if len(Residences) else float("nan"),
                "IqrResidence_ps": ([float(np.percentile(Residences, 25)),
                                     float(np.percentile(Residences, 75))]
                                    if len(Residences) else [float("nan")] * 2),
                "Impingements": float(Impingements),
                "BouncePerImpingement": (len(Bounces) / Impingements
                                         if Impingements > 0 else float("nan")),
                "Uptakes": len(Uptakes),
                "UptakesGasOrigin": len(GasOriginUptakes),
                "UptakesSurfaceBorn": len(Uptakes) - len(GasOriginUptakes),
            }
            # Sticking counts GAS-ORIGIN uptakes only (molecules that were free
            # at least once); surface-born transient O-O pairs are not arriving
            # flux and would push the "probability" above 1.
            Entry["Sticking"][Window] = {
                "Uptakes": len(GasOriginUptakes),
                "UptakesAllInclSurfaceBorn": len(Uptakes),
                "Impingements": float(Impingements),
                "Probability": (len(GasOriginUptakes) / Impingements
                                if Impingements > 0 else float("nan")),
                "PoissonSigma": (np.sqrt(len(GasOriginUptakes)) / Impingements
                                 if Impingements > 0 else float("nan")),
            }
        Entry["Area_m2"] = AreaM2
        PerTemperature[str(Temp)] = Entry

    Comparability = {}
    for Window in WindowOrder:
        Comparability[Window] = {}
        for Label in ("free", "far"):
            Means = {str(T): PerTemperature[str(T)]["WindowStats"][Window][Label]["J_TempMean"]
                     for T in Temperatures}
            Vals = np.array(list(Means.values()))
            Ratio = float(Vals.max() / Vals.min()) if Vals.min() > 0 else float("inf")
            Cv = float(Vals.std(ddof=1) / Vals.mean()) if Vals.mean() > 0 else float("nan")
            Comparability[Window][Label] = {
                "TempMeans": Means, "MaxOverMin": Ratio, "CV": Cv,
                "Within15Pct": bool(Ratio <= 1.15),
            }
    return PerTrajStats, PerTemperature, Comparability


# =============================================================================
#                                 FIGURES
# =============================================================================
def RunningMean(Values):
    Arr = np.asarray(Values, dtype=float)
    return np.cumsum(Arr) / np.arange(1, len(Arr) + 1)


def MakeRunningMeanFigure(Results, OutBase):
    Fig, Axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True, sharey=True)
    for Ax, Temp in zip(Axes.ravel(), Temperatures):
        Colour = TemperatureColours[Temp]
        MeanCurves = {"free": [], "far": []}
        for R in [R for R in Results if R["Temp"] == Temp]:
            T = np.array(R["Series"]["Times_ps"])
            for Label, Style in [("free", "-"), ("far", "--")]:
                Curve = RunningMean(R["Series"][f"J{Label}_m2s"])
                Ax.plot(T, Curve, Style, color=Colour, alpha=0.25, lw=0.8)
                MeanCurves[Label].append(Curve)
        for Label, Style in [("free", "-"), ("far", "--")]:
            Common = min(len(C) for C in MeanCurves[Label])
            Mean = np.mean([C[:Common] for C in MeanCurves[Label]], axis=0)
            Ax.plot(T[:Common], Mean, Style, color=Colour, lw=2.2,
                    label=f"J_{Label} (mean)")
        Ax.set_title(f"{Temp} K", fontsize=10)
        Ax.set_yscale("log")
        Ax.legend(fontsize=8)
        Ax.grid(alpha=0.25)
    for Ax in Axes[-1]:
        Ax.set_xlabel("Time (ps)")
    for Ax in Axes[:, 0]:
        Ax.set_ylabel("Running-mean J (m$^{-2}$ s$^{-1}$)")
    Fig.suptitle("Running-mean impingement flux: J_free (solid) vs J_far (dashed)",
                 fontsize=11)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


def MakeStickingFigure(PerTemperature, OutBase):
    Fig, Ax = plt.subplots(figsize=(7.0, 5.0))
    Offsets = {"0-40": -8.0, "40-140": 8.0}
    Markers = {"0-40": "o", "40-140": "s"}
    for Window in BounceWindows:
        Xs, Ys, Errs = [], [], []
        for Temp in Temperatures:
            S = PerTemperature[str(Temp)]["Sticking"][Window]
            Xs.append(Temp + Offsets[Window])
            Ys.append(S["Probability"])
            Errs.append(S["PoissonSigma"])
        Ax.errorbar(Xs, Ys, yerr=Errs, fmt=Markers[Window], capsize=4,
                    markersize=8, lw=1.4, label=f"{Window} ps")
    Ax.set_xticks(Temperatures)
    Ax.set_xlabel("Temperature (K)")
    Ax.set_ylabel("Effective dissociative sticking probability")
    Ax.legend(title="Window")
    Ax.grid(alpha=0.25)
    Fig.tight_layout()
    Fig.savefig(OutBase + ".png", dpi=300)
    Fig.savefig(OutBase + ".pdf")
    plt.close(Fig)


# =============================================================================
#                      LEGACY PARTIAL-PRESSURE PRODUCT
# =============================================================================
def RunLegacyPartialPressure(Jobs, Stride=50):
    """Create the standard partial-pressure product with the EXISTING
    postprocessing functions (ParseTrajectory/_ProcessSingleFile), written to
    the dataset's own partial_pressure/ folder - the location the stock
    ProcessSystemFolder naturally generates."""
    from pathlib import Path as PathLib
    sys.path.insert(0, os.environ.get("SGUSCHI_POSTPROCESSING", os.path.join(ScriptDir, "..", "postprocessing")))
    import pandas as pd
    from CalculatePartialPressure import RunTask, _ProcessSingleFile, PlotSummaryFigures

    OutDir = PathLib(DataDir) / "partial_pressure"
    OutDir.mkdir(parents=True, exist_ok=True)
    ColorMap = {873: "#1f77b4", 973: "#2ca02c", 1073: "#ff7f0e", 1273: "#d62728"}
    Tasks = [RunTask(XyzPath=PathLib(XyzPath(T, S)), OutputDir=OutDir,
                     Temperature=T, RunNumber=S, Stride=Stride,
                     Color=ColorMap[T], Label=f"{T}K Run {S}")
             for T in Temperatures for S in SimIndices]

    Traces, Rows = [], []
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=Jobs) as Pool:
        Futures = {Pool.submit(_ProcessSingleFile, Task): Task for Task in Tasks}
        for Fut in as_completed(Futures):
            Result, Messages = Fut.result()
            for Msg in Messages:
                print(Msg, flush=True)
            if Result is not None:
                Trace, Row = Result
                Traces.append(Trace)
                Rows.append(Row)
    if Traces:
        PlotSummaryFigures(Traces, "ZrC", OutDir)
    if Rows:
        pd.DataFrame(Rows).sort_values(["Temperature (K)", "Run Number"]).to_csv(
            OutDir / "Summary.csv", index=False)
    print(f"Legacy partial-pressure product written to {OutDir} (stride {Stride}).",
          flush=True)


# =============================================================================
#                                REPORTING
# =============================================================================
def Fmt(X, Nd=2):
    if X is None or (isinstance(X, float) and not np.isfinite(X)):
        return "-"
    return f"{X:.{Nd}f}"


def RoundList(Values, Sig=5):
    return [float(f"{V:.{Sig}g}") for V in Values]


def WriteOutputs(Results, PerTrajStats, PerTemperature, Comparability,
                 EndTimes, GatePass, DatasetPass, VgasMethod):
    PotimByTraj = {f"{R['Temp']}_{R['Sim']}": R["Potim"] for R in Results}
    PotimByTemp = {str(T): float(np.mean(
        [R["Potim"]["FramesPerPs"] for R in Results if R["Temp"] == T]))
        for T in Temperatures}

    PerTrajectory = {}
    for R in Results:
        TrajId = f"{R['Temp']}_{R['Sim']}"
        Series = dict(R["Series"])
        Series["Vgas_m3"] = RoundList(Series["Vgas_m3"])
        for Label in ("free", "far"):
            Series[f"P{Label}_bar"] = RoundList(Series[f"P{Label}_bar"])
            Series[f"J{Label}_m2s"] = RoundList(Series[f"J{Label}_m2s"])
        PerTrajectory[TrajId] = {
            "Series": Series,
            "WindowStats": PerTrajStats[TrajId],
            "BounceEvents": R["BounceEvents"],
            "Uptakes_ps": R["Uptakes_ps"],
            "AddedMolecules": R["AddedMolecules"],
            "AddedDesorbedAtHorizon": R["AddedDesorbedAtHorizon"],
            "EdgeCases": R["EdgeCases"],
            "NGridFrames": R["NGridFrames"],
        }

    Payload = {
        "Meta": {
            "Seed": RandomSeed, "Grid_fs": GridFs, "Horizon_ps": HorizonPs,
            "GatePass": GatePass, "DatasetCheckPass": DatasetPass,
            "VgasMethod": VgasMethod,
            "BondCutoffs_A": {"O-O": OoCutoff, "Zr-O_association": ZrOCutoff,
                              "SlabNetwork": "(Ri+Rj)*1.5, radii Zr 1.45 / O 0.66 / C 0.76 / N 0.71",
                              "Far": FarCutoff},
            "WindowDefs_ps": {K: list(V) for K, V in Windows.items()},
            "Area_m2": {f"{R['Temp']}_{R['Sim']}": R["Area_m2"] for R in Results},
            "TrajectoryEndTimes_ps": EndTimes,
            "PotimAudit": {"PerTrajectory": PotimByTraj, "FramesPerPsByTemp": PotimByTemp},
        },
        "PerTrajectory": PerTrajectory,
        "PerTemperature": PerTemperature,
        "Comparability": Comparability,
    }
    with open(os.path.join(ResultsDir, "GasSupplyAudit.json"), "w") as F:
        json.dump(Payload, F, indent=1)
    WriteSummary(Payload)
    return Payload


def WriteSummary(Payload):
    Meta = Payload["Meta"]
    PerT = Payload["PerTemperature"]
    Comp = Payload["Comparability"]
    Ppt = Meta["PotimAudit"]["FramesPerPsByTemp"]

    L = []
    L.append("# Gas-supply audit (ZrC final dataset, 140.0 ps horizon)")
    L.append("")
    L.append(f"Gate: 16/16 >= 140.0 ps: {'yes' if Meta['GatePass'] else 'no'} | "
             f"Dataset check (lengths 140-147 ps): "
             f"{'pass' if Meta['DatasetCheckPass'] else 'fail'}")
    L.append(f"Grid: 50 fs | Vgas method: {Meta['VgasMethod']}")
    L.append(f"POTIM audit: frames per ps by T: "
             f"873 {Ppt['873']:.1f} / 973 {Ppt['973']:.1f} / "
             f"1073 {Ppt['1073']:.1f} / 1273 {Ppt['1273']:.1f}")
    L.append("")
    L.append("## 1. Table 1 candidates (J_free, 0-140 ps)")
    L.append("| T (K) | P +/- SD (bar) | J +/- SD (10^27 m^-2 s^-1) |")
    L.append("|---:|---:|---:|")
    for Temp in Temperatures:
        W = PerT[str(Temp)]["WindowStats"]["0-140"]["free"]
        L.append(f"| {Temp} | {W['P_bar_TempMean']:.1f} +/- {W['P_bar_TrajSD']:.1f} | "
                 f"{W['J_TempMean'] / 1e27:.1f} +/- {W['J_TrajSD'] / 1e27:.1f} |")
    L.append("")
    L.append("## 2. Comparability")
    L.append("| Window | Definition | max/min | CV | Within 15%? |")
    L.append("|---|---|---:|---:|---|")
    for Window in WindowOrder:
        for Label in ("free", "far"):
            C = Comp[Window][Label]
            L.append(f"| {Window} | J_{Label} | {C['MaxOverMin']:.2f} | "
                     f"{C['CV']:.3f} | {'yes' if C['Within15Pct'] else 'no'} |")
    L.append("")
    Inflation = " / ".join(
        f"{T} {PerT[str(T)]['Inflation']['40-140']:.2f}" for T in Temperatures)
    L.append(f"Inflation J_free/J_far by T (40-140 ps): {Inflation}")
    L.append("")
    L.append("## 3. Bounce census (40-140 ps)")
    L.append("| T (K) | Bounces | Median residence (ps) | Bounces per impingement | "
             "Dissociative uptakes |")
    L.append("|---:|---:|---:|---:|---:|")
    for Temp in Temperatures:
        B = PerT[str(Temp)]["BounceCensus"]["40-140"]
        L.append(f"| {Temp} | {B['Bounces']} | {Fmt(B['MedianResidence_ps'])} | "
                 f"{Fmt(B['BouncePerImpingement'], 3)} | {B['Uptakes']} |")
    L.append("")
    L.append("## 4. Sticking probability")
    L.append("| T (K) | 0-40 ps | 40-140 ps |")
    L.append("|---:|---:|---:|")
    for Temp in Temperatures:
        S0 = PerT[str(Temp)]["Sticking"]["0-40"]
        S1 = PerT[str(Temp)]["Sticking"]["40-140"]
        L.append(f"| {Temp} | {S0['Probability']:.3f} +/- {S0['PoissonSigma']:.3f} | "
                 f"{S1['Probability']:.3f} +/- {S1['PoissonSigma']:.3f} |")
    L.append("")
    L.append("## 5. Notes")
    for N in BuildNotes(Payload):
        L.append(f"- {N}")
    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print(Text)


def BuildNotes(Payload):
    PerTraj = Payload["PerTrajectory"]
    Notes = []
    Edge = {K: sum(PerTraj[T]["EdgeCases"][K] for T in PerTraj)
            for K in ("Exchanges", "GasPhaseDissociations", "VanishedIntact",
                      "Reassociated")}
    Added = sum(PerTraj[T]["AddedMolecules"] for T in PerTraj)
    AddedDesorbed = sum(PerTraj[T]["AddedDesorbedAtHorizon"] for T in PerTraj)
    Notes.append(
        "Identity tracking: O atoms matched between consecutive grid frames "
        "(index order kept when counts match and same-index displacement "
        f"< {IdentityMatchMax} A; greedy minimum-image nearest-neighbour "
        "matching otherwise). Molecule identity = the unordered pair of O "
        "tracker IDs; a within-molecule swap of the two atoms preserves "
        "identity. Edge cases at the 50 fs grid across all 16 trajectories: "
        f"{Edge['Exchanges']} O-exchange events (an O re-pairs with a "
        f"different O; both identities end), {Edge['GasPhaseDissociations']} "
        f"apparent gas-phase pair losses, {Edge['VanishedIntact']} intact "
        f"molecules whose atom vanished (protocol removals), "
        f"{Edge['Reassociated']} re-associations of previously seen O.")
    Notes.append(
        f"Protocol additions: {Added} molecules first seen as two brand-new O "
        f"atoms after frame 0; {AddedDesorbed} of them end at the horizon "
        "intact and free ('desorbed-after-addition'). Molecules already "
        "surface-associated when first seen count no bounce until they have "
        "been free once (EverFree flag).")
    Notes.append(
        "Bounce residences are quantised by the 50 fs grid (minimum resolvable "
        "residence = one grid step); no hysteresis is applied at the 3.17 A "
        "association boundary, so single-frame flickers count as 0.05 ps "
        "bounces - the median/IQR reported are raw per protocol.")
    Notes.append(
        "Sticking probability counts GAS-ORIGIN dissociative uptakes only "
        "(molecules that were free at least once before dissociating while "
        "surface-associated). The raw uptake census also contains "
        "surface-born transient O-O pairs (lattice/adsorbed O that pair "
        "briefly and re-break, the 'Reassociated' origin); counting those as "
        "sticking pushed the apparent probability above 1 and they are not "
        "arriving flux - the split is in the JSON (UptakesGasOrigin vs "
        "UptakesSurfaceBorn) and the Section 3 uptake column is the total.")
    Notes.append(
        "Table 1 SDs (both P and J) are between-trajectory SDs (ddof=1, n=4) "
        "of the per-trajectory window means - the replicate spread relevant "
        "for a temperature-mean error bar. Per-frame time-series SDs (much "
        "larger; Poisson-dominated) are in the JSON as P_bar_SD / J_SD per "
        "trajectory.")
    Notes.append(
        "Comparability rule: 'within 15%' means max/min of the four "
        "temperature-mean fluxes <= 1.15. CV is SD/mean (ddof=1) over the "
        "four temperature means.")
    Notes.append(
        "Impingement denominators use the window's temperature-mean J_free, "
        "2 surfaces x mean in-plane area x window duration x 4 trajectories.")
    Notes.append(
        "N_far excludes molecules within 4.0 A of any slab atom; the slab is "
        "every bonded connected component containing Zr, computed after "
        "removing paired-O2 atoms from the bond network so hovering molecules "
        "cannot shield themselves.")
    Notes.append(
        "Vgas: protocol void-fraction (largest Zr-gap along x, incl. wrap); "
        f"method = {Payload['Meta']['VgasMethod']}.")
    Notes.append(
        "The legacy partial-pressure product (existing "
        "CalculatePartialPressure functions, stride 50) lives in the "
        "dataset's own partial_pressure/ folder, the location the stock pipeline naturally generates. "
        "Its per-frame CSVs use the legacy hardcoded C radius 0.77 A (audit "
        "notes in src/postprocessing apply).")
    Notes.append(
        "Nothing stochastic; seed 20260715 quoted for consistency. Earlier "
        "caches (e.g. OxygenLayeringResults/FrameCache) were "
        "built from the OLD dataset and are stale for the updated "
        "trajectories; this audit reads only the xyz files directly.")
    return Notes


# =============================================================================
#                                  DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Gas-phase supply audit (ZrC final dataset).")
    Parser.add_argument("--jobs", type=int,
                        default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Parser.add_argument("--only", type=str, default=None)
    Parser.add_argument("--with-legacy-pp", action="store_true",
                        help="Also build the legacy partial-pressure product "
                             "(under GasSupplyAudit/partial_pressure/).")
    Parser.add_argument("--legacy-pp-only", action="store_true")
    Parser.add_argument("--flux-figure-only", action="store_true",
                        help="Only redraw the flux figure from the stored GasSupplyAudit.json.")
    Args = Parser.parse_args()

    os.makedirs(ResultsDir, exist_ok=True)
    if Args.flux_figure_only:
        MakeFluxControlFigure()
        return 0

    # ---- Gates -------------------------------------------------------------------
    EndTimes = {f"{T}_{S}": ReadFinalTimePs(XyzPath(T, S))
                for T in Temperatures for S in SimIndices}
    Short = {K: V for K, V in EndTimes.items() if not (V >= HorizonPs)}
    GatePass = len(Short) == 0
    InBand = sum(1 for V in EndTimes.values() if 140.0 <= V <= 147.5)
    DatasetPass = GatePass and InBand == 16
    print("Per-trajectory total simulated time (ps):")
    for K in sorted(EndTimes):
        print(f"  {K}: {EndTimes[K]:.3f}")
    if not GatePass or not DatasetPass:
        print(f"\nSTOP: gate failed. Trajectories short of {HorizonPs} ps: {Short}. "
              f"Lengths in the 140-147 ps band: {InBand}/16.")
        with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
            F.write("# Gas-supply audit (ZrC final dataset, 140.0 ps horizon)\n\n"
                    f"Gate: 16/16 >= 140.0 ps: {'yes' if GatePass else 'no'} | "
                    f"Dataset check (lengths 140-147 ps): "
                    f"{'pass' if DatasetPass else 'fail'}\n\n"
                    f"STOPPED before analysis. End times (ps): {EndTimes}\n")
        return 1

    if Args.legacy_pp_only:
        RunLegacyPartialPressure(Args.jobs)
        return 0

    Tasks = ([tuple(int(x) for x in Args.only.split("_"))] if Args.only
             else [(T, S) for T in Temperatures for S in SimIndices])

    print(f"\nStreaming {len(Tasks)} trajectories on the 50 fs grid "
          f"(jobs={Args.jobs}) ...", flush=True)
    Results = []
    if Args.jobs > 1 and len(Tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=Args.jobs) as Pool:
            Futures = {Pool.submit(ProcessTrajectory, T): T for T in Tasks}
            for Fut in as_completed(Futures):
                R = Fut.result()
                Results.append(R)
                print(f"  {R['Temp']}_{R['Sim']}: {R['NGridFrames']} grid frames, "
                      f"{len(R['BounceEvents'])} bounces, "
                      f"{len(R['Uptakes_ps'])} uptakes", flush=True)
    else:
        for T in Tasks:
            R = ProcessTrajectory(T)
            Results.append(R)
            print(f"  {R['Temp']}_{R['Sim']}: {R['NGridFrames']} grid frames, "
                  f"{len(R['BounceEvents'])} bounces, "
                  f"{len(R['Uptakes_ps'])} uptakes", flush=True)

    if Args.only:
        print(json.dumps({"WindowStats": TrajectoryWindowStats(Results[0]),
                          "Potim": Results[0]["Potim"]}, indent=2))
        return 0

    Results.sort(key=lambda R: (R["Temp"], R["Sim"]))
    PerTrajStats, PerTemperature, Comparability = Aggregate(Results)

    MakeRunningMeanFigure(Results, os.path.join(ResultsDir, "FateOfFlux_RunningMeanJ"))
    MakeStickingFigure(PerTemperature, os.path.join(ResultsDir, "StickingProbability"))

    VgasMethod = Results[0]["VgasMethod"]
    WriteOutputs(Results, PerTrajStats, PerTemperature, Comparability,
                 EndTimes, GatePass, DatasetPass, VgasMethod)

    MakeFluxControlFigure()
    if Args.with_legacy_pp:
        RunLegacyPartialPressure(Args.jobs)
    print("\nDone. Results in", ResultsDir, flush=True)
    return 0


# -----------------------------------------------------------------------------
# Flux over time (figure of the Supplementary Information), drawn from GasSupplyAudit.json
# -----------------------------------------------------------------------------

FluxJsonPath = os.path.join(ScriptDir, "GasSupplyAudit", "GasSupplyAudit.json")
FluxResultsDir = os.path.join(ScriptDir, "FluxControlFigureResults")

FluxTempColours = {873: "#0072B2", 973: "#E69F00", 1073: "#009E73",
               1273: "#CC79A7"}
FluxWindowPs = 10.0
FluxSettleInPs = 40.0

FluxStyle = {
    "font.family": "Times New Roman", "font.size": 10,
    "axes.labelsize": 10, "legend.fontsize": 8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "mathtext.fontset": "stix",
}


def RollingMean(Values, NWindow):
    Kernel = np.ones(NWindow) / NWindow
    return np.convolve(Values, Kernel, mode="valid")


def MakeFluxControlFigure():
    with plt.rc_context():
        matplotlib.rc_file_defaults()
        plt.rcParams.update(FluxStyle)
        os.makedirs(FluxResultsDir, exist_ok=True)
        Audit = json.load(open(FluxJsonPath))["PerTrajectory"]

        Fig, Ax = plt.subplots(figsize=(160 / 25.4, 70 / 25.4))
        for Temp in Temperatures:
            Smoothed = []
            for Sim in SimIndices:
                Series = Audit[f"{Temp}_{Sim}"]["Series"]
                Times = np.asarray(Series["Times_ps"], dtype=float)
                J = np.asarray(Series["Jfree_m2s"], dtype=float) / 1e27
                Step = np.median(np.diff(Times))
                NWindow = max(1, int(round(FluxWindowPs / Step)))
                JSmooth = RollingMean(J, NWindow)
                TSmooth = RollingMean(Times, NWindow)
                Ax.plot(TSmooth, JSmooth, "-", color=FluxTempColours[Temp],
                        lw=0.6, alpha=0.35)
                Smoothed.append((TSmooth, JSmooth))
            NCommon = min(len(T) for T, _ in Smoothed)
            TMean = Smoothed[0][0][:NCommon]
            JMean = np.mean([J[:NCommon] for _, J in Smoothed], axis=0)
            Ax.plot(TMean, JMean, "-", color=FluxTempColours[Temp], lw=1.6,
                    label=f"{Temp} K")

        Decomposition = {}
        for Temp in Temperatures:
            Early, Late = {}, {}
            for Key in ("NFree", "Vgas_m3", "Jfree_m2s"):
                EVals, LVals = [], []
                for Sim in SimIndices:
                    Series = Audit[f"{Temp}_{Sim}"]["Series"]
                    Times = np.asarray(Series["Times_ps"], dtype=float)
                    Vals = np.asarray(Series[Key], dtype=float)
                    EVals.append(Vals[(Times >= 40) & (Times < 90)].mean())
                    LVals.append(Vals[(Times >= 90) & (Times <= 140)].mean())
                Early[Key], Late[Key] = float(np.mean(EVals)), float(np.mean(LVals))
            Decomposition[Temp] = {
                "Early40to90": Early, "Late90to140": Late,
                "RelChange": {K: Late[K] / Early[K] - 1.0 for K in Early},
            }
        with open(os.path.join(FluxResultsDir, "LateDriftDecomposition.json"),
                  "w") as Handle:
            json.dump(Decomposition, Handle, indent=1)
        for Temp, D in Decomposition.items():
            R = D["RelChange"]
            print(f"{Temp} K: NFree {R['NFree']:+.0%}  Vgas {R['Vgas_m3']:+.0%}"
                  f"  Jfree {R['Jfree_m2s']:+.0%}")

        Ax.axvline(FluxSettleInPs, color="0.4", lw=0.8, ls="--")
        Ax.set_yscale("log")
        Ax.set_xlim(0, 140)
        Ax.set_xlabel("Time (ps)")
        Ax.set_ylabel(r"$J_\mathrm{free}$ ($10^{27}$ m$^{-2}$ s$^{-1}$)")
        Ax.legend(frameon=False, ncol=2, loc="upper right")
        Fig.tight_layout()
        Fig.savefig(os.path.join(FluxResultsDir, "FluxControl.png"), dpi=300)
        Fig.savefig(os.path.join(FluxResultsDir, "FluxControl.pdf"))
        plt.close(Fig)
        print("Wrote figures to", FluxResultsDir)


if __name__ == "__main__":
    sys.exit(Main())
