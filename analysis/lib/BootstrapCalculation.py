"""
BootstrapCalculation.py
=======================================================================
Trajectory-stratified bootstrap validation of the carbon retention ratio
minimum at 973 K for the ZrC oxidation study.

Statistic
---------
    PhiC = N_chain / N_gasified            (paper's phi_c = N_chain / N_gasified)

where, per trajectory,
    * N_gasified = number of carbons that left as CO or CO2
                   (taken from the RateAnalysis CSVs, "Gas Removed" column;
                    per-event with timestamps).
    * N_chain    = number of carbons in a confirmed interfacial "Chain" state
                   AT THE FINAL FRAME (net carbon retention), reconstructed from
                   the xyz trajectories using the paper's exact chain-detection
                   stack: instantaneous C-C / C-Zr coordination + 0.2 ps state-
                   persistence smoothing.

                   NOTE (verified against the paper's own
                   legacy_arrhenius_outputs/*/Final_Trajectory_Values.csv):
                   Table 3 "chain atoms" (35/31/48/66) is the FINAL-frame
                   confirmed Chain count, NOT the cumulative count of carbons
                   that ever touched a chain (that gives 39/35/52/69). The
                   paper's PhiC uses the final-frame count, so that is what is
                   used here. A chain event is therefore one carbon retained in
                   a chain at trajectory end, timestamped at its last entry into
                   the Chain state.

The chain-detection functions (geometry, state determination, persistence
smoothing, identity mapping) are ported verbatim from the paper's own
analysis code (pipeline/ZrC_Paper_Plots.py) so the reconstructed chain
totals must reproduce the published Table 3 values exactly. That equality
is asserted as a hard guardrail in the sanity-check step.

HARD CONSTRAINTS honoured here:
    * The xyz files are NEVER loaded whole. ReadTrajectoryGenerator streams
      one frame at a time (atom-count header -> that many atom lines ->
      process -> discard).
    * The CSVs are preferred over the xyz files: gas events come entirely
      from the CSVs; the xyz files are touched only for chain detection,
      which is not recorded in the CSVs.
    * All identifiers are PascalCase.
    * The RNG seed is fixed and reported.

The expensive xyz pass is cached to BootstrapResults/EventTable.csv. Re-runs
reuse the cache unless --rebuild-events is given.

Usage
-----
    python BootstrapCalculation.py                 # full pipeline (uses cache if present)
    python BootstrapCalculation.py --rebuild-events
    python BootstrapCalculation.py --nresamples 10000
    python BootstrapCalculation.py --smoke 3000    # cap frames/xyz (plumbing test only)
"""

import os
import re
import ast
import json
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#                                CONFIGURATION
# =============================================================================
ScriptDir  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
ResultsDir = os.path.join(ScriptDir, "BootstrapResults")
DataDir    = os.environ.get("ZRC_DATA_DIR", os.path.join(ScriptDir, "..", "data"))

Temperatures = [873, 973, 1073, 1273]
SimIndices   = [1, 2, 3, 4]

# --- Reproducibility ---------------------------------------------------------
RandomSeed   = 20260715      # fixed seed, reported in every output
NResamples   = 10000         # bootstrap rounds per temperature (overridable via CLI)

# --- Chain-detection constants (ported verbatim from ZrC_Paper_Plots.py) -----
TotalStartC          = 48
CovalentRadii        = {'C': 0.77, 'Zr': 1.45, 'O': 0.66}
RadiusTol            = 1.5
CutoffCC             = (CovalentRadii['C']  + CovalentRadii['C'])  * RadiusTol   # 2.31 A
CutoffZrC            = (CovalentRadii['Zr'] + CovalentRadii['C'])  * RadiusTol   # 3.33 A
VolatileZrThreshold  = 4
StatePersistenceTimePs = 0.2

# --- XYZ comment-line parsing regexes (ported verbatim) ----------------------
TimeCommentRegex = re.compile(r'(?:Time|t)\s*[:=]\s*([\d\.]+)', re.IGNORECASE)
FirstFloatRegex  = re.compile(r'([\d\.]+)')
LatticeRegex     = re.compile(r'Lattice="([^"]+)"')

# --- Published Table 3 totals (for the sanity check) -------------------------
PaperCO    = {873: 21, 973: 46, 1073: 35, 1273: 36}
PaperCO2   = {873: 10, 973: 4,  1073: 2,  1273: 3}
PaperChain = {873: 35, 973: 31, 1073: 48, 1273: 66}

# Per-trajectory final-frame Chain counts, read from the paper's own
# legacy_arrhenius_outputs/Plots_*/Final_Trajectory_Values.csv (identical across
# the three smoothing configs). Summing each temperature reproduces PaperChain.
# Used as a per-trajectory guardrail: the bootstrap depends on the per-trajectory
# split, so matching only the per-temperature sum would be insufficient.
PaperChainByTraj = {
    '873_1': 14, '873_2': 4,  '873_3': 11, '873_4': 6,
    '973_1': 10, '973_2': 6,  '973_3': 8,  '973_4': 7,
    '1073_1': 16, '1073_2': 7, '1073_3': 16, '1073_4': 9,
    '1273_1': 12, '1273_2': 13, '1273_3': 26, '1273_4': 15,
}


# =============================================================================
#           GEOMETRY + CHAIN DETECTION  (ported verbatim from the paper)
# =============================================================================
def CartesianToFractional(CartCoords, Lattice):
    """Cartesian -> fractional in [0, 1). Lattice rows are the cell vectors."""
    return np.mod(CartCoords @ np.linalg.inv(Lattice), 1.0)


def MinimumImageDistances(FracCoords, Lattice):
    """Full pairwise minimum-image Cartesian distance matrix (triclinic-safe)."""
    FracDiff = FracCoords[:, np.newaxis, :] - FracCoords[np.newaxis, :, :]
    FracDiff -= np.round(FracDiff)
    CartDiff = FracDiff @ Lattice
    return np.sqrt(np.sum(CartDiff ** 2, axis=2))


def MinimumImageCrossDistances(FracA, FracB, Lattice):
    """Pairwise minimum-image Cartesian distances between two atom sets -> (M, N)."""
    FracDiff = FracA[:, np.newaxis, :] - FracB[np.newaxis, :, :]
    FracDiff -= np.round(FracDiff)
    CartDiff = FracDiff @ Lattice
    return np.sqrt(np.sum(CartDiff ** 2, axis=2))


def ParseLatticeFromComment(CommentLine):
    """Parse the 3x3 lattice (rows = cell vectors) from an extended-xyz comment."""
    Match = LatticeRegex.search(CommentLine)
    if Match:
        Nums = np.fromstring(Match.group(1), sep=' ')
        if len(Nums) == 9:
            return np.array(Nums).reshape(3, 3)
    return np.diag([42.4, 9.47, 9.46])


def ReadTrajectoryGenerator(FilePath, MaxFrames=None):
    """
    Stream an xyz trajectory frame-by-frame. NEVER loads the file whole:
    reads the atom-count header, then exactly that many atom lines, yields
    the frame, and discards it before reading the next.

    Yields (TimeFs, Atoms, Coords, Lattice) per frame.
    """
    FrameIndex = 0
    with open(FilePath, 'r') as F:
        while True:
            if MaxFrames is not None and FrameIndex >= MaxFrames:
                break
            Line = F.readline()
            if not Line:
                break
            try:
                NumAtoms = int(Line.strip())
            except ValueError:
                break

            CommentLine = F.readline()
            Atoms  = np.empty(NumAtoms, dtype=object)
            Coords = np.empty((NumAtoms, 3), dtype=float)
            IncompleteFrame = False
            for i in range(NumAtoms):
                AtomLine = F.readline()
                if not AtomLine:
                    IncompleteFrame = True
                    break
                Parts = AtomLine.split()
                if len(Parts) < 4:
                    IncompleteFrame = True
                    break
                Atoms[i] = Parts[0]
                Coords[i, 0] = float(Parts[1])
                Coords[i, 1] = float(Parts[2])
                Coords[i, 2] = float(Parts[3])
            if IncompleteFrame:
                break

            TimeVal = 0.0
            Match = TimeCommentRegex.search(CommentLine)
            if Match:
                TimeVal = float(Match.group(1))
            else:
                MatchAny = FirstFloatRegex.search(CommentLine)
                if MatchAny:
                    TimeVal = float(MatchAny.group(1))

            Lattice = ParseLatticeFromComment(CommentLine)
            FrameIndex += 1
            yield TimeVal, Atoms, Coords, Lattice


def MapAtomIdentities(PrevFracC, CurrFracC, PrevIDs, Lattice, MaxDisplacement=2.0):
    """
    Map current-frame carbons to persistent IDs by minimum-image nearest-
    neighbour assignment (Hungarian). Departed carbons (gas) leave their ID
    unqueried, so they are later detected as 'Gas' via dict .get() defaults.

    Ported verbatim; the >MaxDisplacement warning print is dropped because it
    is purely cosmetic (does not affect the returned mapping) and would flood
    over tens of thousands of frames.
    """
    if len(PrevFracC) == 0 or len(CurrFracC) == 0:
        return np.array([])

    NCurr     = len(CurrFracC)
    CrossDist = MinimumImageCrossDistances(CurrFracC, PrevFracC, Lattice)

    RowInds, ColInds = linear_sum_assignment(CrossDist)
    SortOrder = np.argsort(RowInds)
    RowInds = RowInds[SortOrder]
    ColInds = ColInds[SortOrder]
    if not np.array_equal(RowInds, np.arange(NCurr)):
        raise ValueError(
            "Carbon identity mapping failed to assign every current carbon exactly once."
        )
    return PrevIDs[ColInds]


def DetermineInstantaneousStates(FracCoords, Atoms, Lattice):
    """
    Raw instantaneous state of every carbon (current-frame carbon order):
        'Chain'     : >=1 carbon neighbour within CutoffCC
        'Displaced' : else, few Zr neighbours (<= VolatileZrThreshold within CutoffZrC)
        'Lattice'   : otherwise
    Ported verbatim.
    """
    MaskC  = Atoms == 'C'
    MaskZr = Atoms == 'Zr'

    NumCarbons = int(np.sum(MaskC))
    States = np.full(NumCarbons, 'Lattice', dtype=object)
    if NumCarbons == 0:
        return States, MaskC

    FracC  = FracCoords[MaskC]
    FracZr = FracCoords[MaskZr]

    if NumCarbons > 1:
        CCDists = MinimumImageDistances(FracC, Lattice)
        np.fill_diagonal(CCDists, np.inf)
        CNeighbourCounts = np.sum(CCDists < CutoffCC, axis=1)
    else:
        CNeighbourCounts = np.zeros(NumCarbons, dtype=int)

    ZrNeighbourCounts = np.zeros(NumCarbons, dtype=int)
    if len(FracZr) > 0:
        CZrDists = MinimumImageCrossDistances(FracC, FracZr, Lattice)
        ZrNeighbourCounts = np.sum(CZrDists < CutoffZrC, axis=1)

    for i in range(NumCarbons):
        if CNeighbourCounts[i] >= 1:
            States[i] = 'Chain'
        elif ZrNeighbourCounts[i] <= VolatileZrThreshold:
            States[i] = 'Displaced'
        else:
            States[i] = 'Lattice'
    return States, MaskC


def ExtractChainEvents(FilePath, MaxFrames=None, ProgressEvery=20000, Label=""):
    """
    Stream one xyz trajectory and return the chain-retention events: one event
    per carbon whose FINAL confirmed state (after the 0.2 ps persistence filter
    and end-of-trajectory flush) is 'Chain'. Each event is timestamped at the
    carbon's LAST entry into the Chain state (start of the final Chain episode).

    len(events) == the paper's FinalStats["Chain"] for this trajectory, i.e. the
    Table 3 "chain atoms" contribution. (See module docstring: this final-frame
    count is what the paper's PhiC uses, not the cumulative unique-chain count.)

    Returns (ChainEvents, NFrames) where ChainEvents is a list of dicts.
    """
    TrackedIDs      = np.arange(TotalStartC)
    ConfirmedStates = {ID: 'Lattice' for ID in range(TotalStartC)}
    PendingStates   = {}
    ChainEntryTimeFs = {}          # AtomID -> fs of most recent entry into Chain
    PrevFracC = None
    NFrames = 0

    for TimeFs, Atoms, Coords, Lattice in ReadTrajectoryGenerator(FilePath, MaxFrames):
        CurrentTimePs = TimeFs / 1000.0
        FracCoords = CartesianToFractional(Coords, Lattice)

        # 1. persistent-identity mapping of the carbons
        CurrFracC = FracCoords[Atoms == 'C']
        if PrevFracC is not None:
            TrackedIDs = MapAtomIdentities(PrevFracC, CurrFracC, TrackedIDs, Lattice)
        else:
            if len(CurrFracC) < len(TrackedIDs):
                TrackedIDs = TrackedIDs[:len(CurrFracC)]
        PrevFracC = CurrFracC

        # 2. raw instantaneous states
        RawStatesList, _ = DetermineInstantaneousStates(FracCoords, Atoms, Lattice)
        CurrentFrameStates = {TrackedIDs[i]: State for i, State in enumerate(RawStatesList)}

        # 3. persistence filter (carbons missing this frame -> raw 'Gas')
        for AtomID in range(TotalStartC):
            RawState = CurrentFrameStates.get(AtomID, 'Gas')
            CurrentConfirmed = ConfirmedStates[AtomID]

            if RawState == CurrentConfirmed:
                PendingStates.pop(AtomID, None)
                continue

            if AtomID in PendingStates:
                Pending = PendingStates[AtomID]
                if Pending['State'] == RawState:
                    Duration = CurrentTimePs - Pending['StartTime']
                    if Duration >= StatePersistenceTimePs:
                        ConfirmedStates[AtomID] = RawState
                        del PendingStates[AtomID]
                        if RawState == 'Chain':
                            # Record time of THIS entry (overwrites earlier ones,
                            # so the final value is the last entry into a chain).
                            ChainEntryTimeFs[AtomID] = TimeFs
                else:
                    PendingStates[AtomID] = {'State': RawState, 'StartTime': CurrentTimePs}
            else:
                PendingStates[AtomID] = {'State': RawState, 'StartTime': CurrentTimePs}

        NFrames += 1
        if ProgressEvery and NFrames % ProgressEvery == 0:
            CurrentChain = sum(1 for S in ConfirmedStates.values() if S == 'Chain')
            print(f"      [{Label}] {NFrames} frames, confirmed chain carbons now: "
                  f"{CurrentChain}", flush=True)

    # 4. end-of-trajectory flush of still-pending transitions (matches the paper:
    #    a pending state is retroactively confirmed from its StartTime).
    for AtomID, Pending in PendingStates.items():
        ConfirmedStates[AtomID] = Pending['State']
        if Pending['State'] == 'Chain':
            ChainEntryTimeFs[AtomID] = Pending['StartTime'] * 1000.0

    # A chain event is one carbon whose FINAL confirmed state is 'Chain'.
    ChainEvents = [{'AtomID': int(AtomID), 'TimeStep': float(ChainEntryTimeFs[AtomID])}
                   for AtomID in range(TotalStartC)
                   if ConfirmedStates[AtomID] == 'Chain']
    return ChainEvents, NFrames


# =============================================================================
#                     GAS EVENTS FROM THE RATEANALYSIS CSVs
# =============================================================================
def ClassifyRemovedMolecule(MoleculeTuple):
    """Map a removed-gas tuple to 'CO', 'CO2', or 'Other'."""
    Sorted = tuple(sorted(MoleculeTuple))
    if Sorted == ('C', 'O'):
        return 'CO'
    if Sorted == ('C', 'O', 'O'):
        return 'CO2'
    return 'Other'


def LoadGasEvents(CsvPath):
    """
    Parse gas-removal events from one RateAnalysis CSV. The "Gas Removed"
    column holds a python-literal list of element tuples removed at that
    row's "Time (fs)". Each tuple is one gas event.

    Returns (GasEvents, OtherSpecies) where GasEvents is a list of dicts with
    keys Species ('CO'/'CO2') and TimeStep (fs), and OtherSpecies collects any
    unexpected (non CO/CO2) removals for reporting.
    """
    Frame = pd.read_csv(CsvPath)
    GasEvents = []
    OtherSpecies = []
    for _, RowData in Frame.iterrows():
        Raw = str(RowData['Gas Removed']).strip()
        if not Raw or Raw == '[]':
            continue
        Removed = ast.literal_eval(Raw)
        for Molecule in Removed:
            Species = ClassifyRemovedMolecule(Molecule)
            if Species == 'Other':
                OtherSpecies.append(tuple(sorted(Molecule)))
                continue
            GasEvents.append({'Species': Species, 'TimeStep': float(RowData['Time (fs)'])})
    return GasEvents, OtherSpecies


# =============================================================================
#                            EVENT-TABLE ASSEMBLY
# =============================================================================
def XyzPath(Temp, Sim):
    return os.path.join(DataDir, f"{Temp}_{Sim}.xyz")


def CsvPath(Temp, Sim):
    return os.path.join(DataDir, f"RateAnalysis_{Temp}_{Sim}.csv")


def ChainWorker(TaskArgs):
    """Process-pool worker: extract chain events for one trajectory."""
    Temp, Sim, MaxFrames = TaskArgs
    Events, NFrames = ExtractChainEvents(XyzPath(Temp, Sim), MaxFrames=MaxFrames,
                                         ProgressEvery=0, Label=f"{Temp}_{Sim}")
    return Temp, Sim, Events, NFrames


def BuildEventTable(MaxFrames=None, Jobs=1):
    """
    Build the resolved-carbon event table across all 16 trajectories.
    One row per event:
        Temperature, TrajectoryId, Outcome ('Gas'|'Chain'),
        Species ('CO'|'CO2'|'Chain'), TimeStep (fs).

    Gas events come from the CSVs (cheap, serial); chain events from streaming
    the xyz files (expensive, optionally across Jobs worker processes since each
    trajectory is independent and CPU-bound).

    Returns (EventFrame, OtherReport).
    """
    # --- Gas events from the CSVs (serial, cheap) ---
    Rows = []
    OtherReport = {}
    for Temp in Temperatures:
        for Sim in SimIndices:
            TrajId = f"{Temp}_{Sim}"
            GasEvents, OtherSpecies = LoadGasEvents(CsvPath(Temp, Sim))
            for Event in GasEvents:
                Rows.append({'Temperature': Temp, 'TrajectoryId': TrajId,
                             'Outcome': 'Gas', 'Species': Event['Species'],
                             'TimeStep': Event['TimeStep']})
            if OtherSpecies:
                OtherReport[TrajId] = OtherSpecies

    # --- Chain events from the xyz files (parallel or serial) ---
    Tasks = [(Temp, Sim, MaxFrames) for Temp in Temperatures for Sim in SimIndices]
    ChainResults = {}
    if Jobs and Jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        print(f"    Streaming {len(Tasks)} xyz files across {Jobs} worker processes ...",
              flush=True)
        with ProcessPoolExecutor(max_workers=Jobs) as Pool:
            Futures = {Pool.submit(ChainWorker, T): (T[0], T[1]) for T in Tasks}
            for Fut in as_completed(Futures):
                Temp, Sim, Events, NFrames = Fut.result()
                ChainResults[(Temp, Sim)] = Events
                print(f"      done {Temp}_{Sim}: {NFrames} frames, "
                      f"{len(Events)} chain carbons.", flush=True)
    else:
        for Temp, Sim, MF in Tasks:
            print(f"    Streaming xyz {Temp}_{Sim} ...", flush=True)
            Events, NFrames = ExtractChainEvents(XyzPath(Temp, Sim), MaxFrames=MF,
                                                 Label=f"{Temp}_{Sim}")
            ChainResults[(Temp, Sim)] = Events
            print(f"      {Temp}_{Sim}: {NFrames} frames, "
                  f"{len(Events)} chain carbons.", flush=True)

    for Temp in Temperatures:
        for Sim in SimIndices:
            TrajId = f"{Temp}_{Sim}"
            for Event in ChainResults[(Temp, Sim)]:
                Rows.append({'Temperature': Temp, 'TrajectoryId': TrajId,
                             'Outcome': 'Chain', 'Species': 'Chain',
                             'TimeStep': Event['TimeStep']})

    EventFrame = pd.DataFrame(Rows,
                              columns=['Temperature', 'TrajectoryId', 'Outcome',
                                       'Species', 'TimeStep'])
    return EventFrame, OtherReport


# =============================================================================
#                    STEP 2 - SANITY CHECK vs PUBLISHED TOTALS
# =============================================================================
def SanityCheck(EventFrame, OtherReport, Strict=True):
    """Aggregate per-temperature CO/CO2/Chain totals and compare to Table 3."""
    Lines = []
    Lines.append("STEP 2 - SANITY CHECK vs PUBLISHED TABLE 3")
    Lines.append("=" * 64)
    Header = f"{'T (K)':>6} | {'CO':>12} | {'CO2':>10} | {'Chain':>12}"
    Lines.append(Header)
    Lines.append(f"{'':>6} | {'mine  paper':>12} | {'mine paper':>10} | {'mine  paper':>12}")
    Lines.append("-" * len(Header))

    Discrepancies = []
    Summary = {}
    for Temp in Temperatures:
        Sub = EventFrame[EventFrame['Temperature'] == Temp]
        CO    = int(((Sub['Outcome'] == 'Gas') & (Sub['Species'] == 'CO')).sum())
        CO2   = int(((Sub['Outcome'] == 'Gas') & (Sub['Species'] == 'CO2')).sum())
        Chain = int((Sub['Outcome'] == 'Chain').sum())
        Summary[Temp] = {'CO': CO, 'CO2': CO2, 'Chain': Chain,
                         'CO_paper': PaperCO[Temp], 'CO2_paper': PaperCO2[Temp],
                         'Chain_paper': PaperChain[Temp]}
        Lines.append(f"{Temp:>6} | {CO:>5} {PaperCO[Temp]:>6} | "
                     f"{CO2:>4} {PaperCO2[Temp]:>5} | {Chain:>5} {PaperChain[Temp]:>6}")
        if CO != PaperCO[Temp]:
            Discrepancies.append(f"CO {Temp} K: mine={CO} paper={PaperCO[Temp]} (d={CO-PaperCO[Temp]})")
        if CO2 != PaperCO2[Temp]:
            Discrepancies.append(f"CO2 {Temp} K: mine={CO2} paper={PaperCO2[Temp]} (d={CO2-PaperCO2[Temp]})")
        if Chain != PaperChain[Temp]:
            Discrepancies.append(f"Chain {Temp} K: mine={Chain} paper={PaperChain[Temp]} (d={Chain-PaperChain[Temp]})")

    # Per-trajectory chain check (stronger than the per-temperature sum; the
    # bootstrap resamples within trajectories, so the split must match too).
    ChainByTrajMismatch = []
    Lines.append("")
    Lines.append("Per-trajectory final-frame Chain count (mine vs paper):")
    for Temp in Temperatures:
        for Sim in SimIndices:
            TrajId = f"{Temp}_{Sim}"
            Mine = int(((EventFrame['TrajectoryId'] == TrajId) &
                        (EventFrame['Outcome'] == 'Chain')).sum())
            Expected = PaperChainByTraj[TrajId]
            Flag = "" if Mine == Expected else "  <-- MISMATCH"
            Lines.append(f"    {TrajId:>7}: {Mine:>3} vs {Expected:>3}{Flag}")
            if Mine != Expected:
                ChainByTrajMismatch.append(f"{TrajId}: mine={Mine} paper={Expected}")

    Lines.append("")
    if OtherReport:
        Lines.append(f"Unexpected (non CO/CO2) removed gases: {OtherReport}")
    if Discrepancies:
        Lines.append("Discrepancies:")
        for D in Discrepancies:
            Lines.append(f"  - {D}")
    else:
        Lines.append("No discrepancies: all totals reproduce Table 3 exactly.")

    Report = "\n".join(Lines)
    print(Report, flush=True)

    # Hard guardrail: the CHAIN reconstruction must match the paper exactly,
    # since it is the numerator under validation. Gas CO/CO2 come from the CSVs
    # (already validated) and a small documented xyz "overhang" can leave CO a
    # few short; that gas gap is reported but not fatal.
    ChainMismatch = [D for D in Discrepancies if D.startswith('Chain')]
    if Strict and (ChainMismatch or ChainByTrajMismatch):
        raise SystemExit(
            "STOP: reconstructed chain counts do not match the paper exactly.\n"
            + ("Per-temperature:\n  " + "\n  ".join(ChainMismatch) + "\n" if ChainMismatch else "")
            + ("Per-trajectory:\n  " + "\n  ".join(ChainByTrajMismatch) + "\n" if ChainByTrajMismatch else "")
            + "The chain-detection port has diverged from the paper. "
              "Fix before bootstrapping."
        )
    return Summary


# =============================================================================
#                         STEP 3 - THE BOOTSTRAP
# =============================================================================
def BuildTrajectoryLabelArrays(EventFrame):
    """
    Per temperature -> per trajectory: int array of event labels
    (1 = Chain, 0 = Gas). This is the object resampled within each trajectory.
    """
    ByTemp = {}
    for Temp in Temperatures:
        Sub = EventFrame[EventFrame['Temperature'] == Temp]
        TrajArrays = []
        TrajIds = []
        for TrajId, Group in Sub.groupby('TrajectoryId', sort=True):
            Labels = (Group['Outcome'].values == 'Chain').astype(np.int64)
            TrajArrays.append(Labels)
            TrajIds.append(TrajId)
        ByTemp[Temp] = {'TrajIds': TrajIds, 'TrajArrays': TrajArrays}
    return ByTemp


def RunBootstrap(EventFrame, NResamplesLocal, Seed):
    """
    Trajectory-stratified bootstrap, per temperature.

    One resample:
      1. Draw len(TrajIds) (=4) trajectories WITH replacement.
      2. For each drawn trajectory, draw WITH replacement from its own event
         list as many events as it originally contained (a trajectory drawn
         twice contributes two independent event-resamples).
      3. Pool; PhiC = ChainCount / GasCount  (NaN if GasCount == 0).

    Returns (PhiCByTemp, NanCountByTemp). A single Generator (fixed seed) drives
    every temperature so the run is fully reproducible; rounds are paired across
    temperatures by index i (valid: the draws are independent between temps).
    """
    Rng = np.random.default_rng(Seed)
    ByTemp = BuildTrajectoryLabelArrays(EventFrame)

    PhiCByTemp = {}
    NanCountByTemp = {}
    for Temp in Temperatures:
        TrajArrays = ByTemp[Temp]['TrajArrays']
        NTraj = len(TrajArrays)
        Lengths = np.array([len(A) for A in TrajArrays])

        PhiC = np.empty(NResamplesLocal, dtype=float)
        NanCount = 0
        for i in range(NResamplesLocal):
            Drawn = Rng.integers(0, NTraj, size=NTraj)
            NChain = 0
            NGas = 0
            for D in Drawn:
                Arr = TrajArrays[D]
                if len(Arr) == 0:
                    continue
                Idx = Rng.integers(0, len(Arr), size=len(Arr))
                Sample = Arr[Idx]
                C = int(Sample.sum())
                NChain += C
                NGas += len(Sample) - C
            if NGas > 0:
                PhiC[i] = NChain / NGas
            else:
                PhiC[i] = np.nan
                NanCount += 1
        PhiCByTemp[Temp] = PhiC
        NanCountByTemp[Temp] = NanCount
        print(f"    Bootstrap {Temp} K: {NResamplesLocal} resamples, "
              f"{NanCount} NaN (zero-gas) rounds, "
              f"trajectory event counts = {Lengths.tolist()}", flush=True)

    return PhiCByTemp, NanCountByTemp


# =============================================================================
#                     STEP 4 - HEADLINE STATISTICS
# =============================================================================
def ComputeHeadlineStatistics(PhiCByTemp, NanCountByTemp, EventFrame, Summary):
    """Assemble every headline number requested in Step 4."""
    A873  = PhiCByTemp[873]
    A973  = PhiCByTemp[973]
    A1073 = PhiCByTemp[1073]
    A1273 = PhiCByTemp[1273]

    # Primary: 973 below BOTH 873 and 1073 (paired by round index, valid rounds only)
    ValidPrimary = ~(np.isnan(A973) | np.isnan(A873) | np.isnan(A1073))
    Primary = np.mean((A973[ValidPrimary] < A873[ValidPrimary]) &
                      (A973[ValidPrimary] < A1073[ValidPrimary]))

    Valid973_873 = ~(np.isnan(A973) | np.isnan(A873))
    Frac973lt873 = float(np.mean(A973[Valid973_873] < A873[Valid973_873]))

    Valid973_1073 = ~(np.isnan(A973) | np.isnan(A1073))
    Frac973lt1073 = float(np.mean(A973[Valid973_1073] < A1073[Valid973_1073]))

    ValidAll = ~(np.isnan(A873) | np.isnan(A973) | np.isnan(A1073) | np.isnan(A1273))
    GlobalMin = float(np.mean((A973[ValidAll] < A873[ValidAll]) &
                              (A973[ValidAll] < A1073[ValidAll]) &
                              (A973[ValidAll] < A1273[ValidAll])))

    PerTemp = {}
    for Temp in Temperatures:
        Arr = PhiCByTemp[Temp]
        Valid = Arr[~np.isnan(Arr)]
        ObsChain = Summary[Temp]['Chain']
        ObsGas = Summary[Temp]['CO'] + Summary[Temp]['CO2']
        Observed = (ObsChain / ObsGas) if ObsGas > 0 else float('nan')
        PerTemp[str(Temp)] = {
            'Median': float(np.median(Valid)),
            'P2_5':  float(np.percentile(Valid, 2.5)),
            'P97_5': float(np.percentile(Valid, 97.5)),
            'Observed': Observed,
            'ObservedChain': ObsChain,
            'ObservedGas': ObsGas,
            'NanRounds': int(NanCountByTemp[Temp]),
            'EventCount': int((EventFrame['Temperature'] == Temp).sum()),
        }

    Headline = {
        'Seed': RandomSeed,
        'NResamples': len(A973),
        'PrimaryFraction_973_below_873_and_1073': float(Primary),
        'Fraction_973_below_873': Frac973lt873,
        'Fraction_973_below_1073': Frac973lt1073,
        'Fraction_973_global_minimum': GlobalMin,
        'PrimaryValidRounds': int(np.sum(ValidPrimary)),
        'GlobalMinValidRounds': int(np.sum(ValidAll)),
        'NanRoundsByTemp': {str(T): int(NanCountByTemp[T]) for T in Temperatures},
        'PerTemperature': PerTemp,
    }
    return Headline


# =============================================================================
#                            STEP 5 - FIGURE
# =============================================================================
def MakeFigure(PhiCByTemp, Headline, OutPngPath, OutPdfPath):
    """Violin panels of resampled PhiC per temperature, sharing one axis, with
    the observed value and 95% interval marked."""
    plt.rcParams.update({'font.size': 11})
    Fig, Ax = plt.subplots(figsize=(7.2, 4.6))

    Colours = {873: '#1f77b4', 973: '#d62728', 1073: '#2ca02c', 1273: '#9467bd'}
    Positions = list(range(len(Temperatures)))
    DataSets = [PhiCByTemp[T][~np.isnan(PhiCByTemp[T])] for T in Temperatures]

    Parts = Ax.violinplot(DataSets, positions=Positions, showextrema=False,
                          widths=0.8)
    for Body, Temp in zip(Parts['bodies'], Temperatures):
        Body.set_facecolor(Colours[Temp])
        Body.set_alpha(0.45)
        Body.set_edgecolor(Colours[Temp])

    for Pos, Temp in zip(Positions, Temperatures):
        Stats = Headline['PerTemperature'][str(Temp)]
        # 95% bootstrap interval
        Ax.plot([Pos, Pos], [Stats['P2_5'], Stats['P97_5']],
                color=Colours[Temp], lw=2.5, solid_capstyle='round', zorder=3)
        # median
        Ax.plot(Pos, Stats['Median'], 'o', color=Colours[Temp],
                markersize=7, zorder=4)
        # observed value from Step 2
        Ax.plot(Pos, Stats['Observed'], marker='D', color='black',
                markersize=7, markerfacecolor='white', markeredgewidth=1.6,
                zorder=5)

    Ax.set_xticks(Positions)
    Ax.set_xticklabels([f"{T} K" for T in Temperatures])
    Ax.set_ylabel(r"$\phi_C = N_\mathrm{chain} / N_\mathrm{gasified}$")
    Ax.set_xlabel("Temperature")
    Ax.set_title("Trajectory-stratified bootstrap of the carbon retention ratio")
    Ax.axhline(0, color='0.85', lw=0.8, zorder=0)

    from matplotlib.lines import Line2D
    Legend = [
        Line2D([0], [0], marker='o', color='0.3', lw=0, markersize=7, label='Bootstrap median'),
        Line2D([0], [0], color='0.3', lw=2.5, label='95% bootstrap interval'),
        Line2D([0], [0], marker='D', color='black', lw=0, markerfacecolor='white',
               markersize=7, markeredgewidth=1.6, label='Observed (Table 3)'),
    ]
    Ax.legend(handles=Legend, loc='upper left', framealpha=0.9, fontsize=9)

    Fig.tight_layout()
    Fig.savefig(OutPngPath, dpi=300)
    Fig.savefig(OutPdfPath)
    plt.close(Fig)
    print(f"    Figure written: {OutPngPath}", flush=True)
    print(f"    Figure written: {OutPdfPath}", flush=True)


# =============================================================================
#           FOLLOW-UP STATISTICS  (from the cached resample arrays only)
# =============================================================================
def ComputeFollowUpStatistics():
    """
    Dominance and ordering statistics derived from the existing bootstrap arrays
    in BootstrapResults/ResampledPhiC.npz. No new resampling: paired resample
    index i is preserved across temperatures (arrays are never sorted/shuffled).

    Appends results under the "FollowUpStatistics" key of BootstrapResults.json
    and prints the markdown report. Returns the results dict.
    """
    NpzPath = os.path.join(ResultsDir, "ResampledPhiC.npz")
    Data = np.load(NpzPath)
    A873, A973, A1073, A1273 = Data['T873'], Data['T973'], Data['T1073'], Data['T1273']
    Arrays = {873: A873, 973: A973, 1073: A1073, 1273: A1273}

    # --- 3. Integrity ---
    Lengths = {T: int(Arrays[T].size) for T in Temperatures}
    NanCounts = {T: int(np.isnan(Arrays[T]).sum()) for T in Temperatures}

    # --- 1. Dominance P[phi_C < 1] (over non-NaN entries) ---
    Dominance = {}
    for T in Temperatures:
        Arr = Arrays[T]
        Valid = Arr[~np.isnan(Arr)]
        PLess = float(np.mean(Valid < 1.0))
        PMore = float(np.mean(Valid > 1.0))
        Dominance[str(T)] = {'P_below_1': PLess, 'P_above_1': PMore}

    # --- 2. Ordering probabilities (paired rounds, valid where all involved non-NaN) ---
    ValidABC = ~(np.isnan(A973) | np.isnan(A1073) | np.isnan(A1273))
    Monotonic = float(np.mean((A973[ValidABC] < A1073[ValidABC]) &
                              (A1073[ValidABC] < A1273[ValidABC])))

    ValidV = ValidABC & ~np.isnan(A873)
    VShape = float(np.mean((A873[ValidV] > A973[ValidV]) &
                           (A973[ValidV] < A1073[ValidV]) &
                           (A1073[ValidV] < A1273[ValidV])))

    ValidRef = ~(np.isnan(A1073) | np.isnan(A1273))
    Reference = float(np.mean(A1073[ValidRef] < A1273[ValidRef]))

    Ordering = {
        'Monotonic_973_1073_1273': Monotonic,
        'VShape_873_973_1073_1273': VShape,
        'Reference_1073_1273': Reference,
    }

    FollowUp = {
        'Source': 'ResampledPhiC.npz',
        'Seed': RandomSeed,
        'NResamples': Lengths[973],
        'DominanceProbability': Dominance,
        'OrderingProbability': Ordering,
        'ArrayLengths': {str(T): Lengths[T] for T in Temperatures},
        'NanCounts': {str(T): NanCounts[T] for T in Temperatures},
    }

    # --- Print markdown report (exact requested format) ---
    Out = []
    Out.append("# Follow-up statistics: dominance and ordering")
    Out.append("")
    Out.append(f"Source: ResampledPhiC.npz (10,000 paired resamples, seed {RandomSeed})")
    Out.append("")
    Out.append("## 1. Gasification dominance P[phi_C < 1]")
    Out.append("")
    Out.append("| T (K) | P[phi_C < 1] | P[phi_C > 1] |")
    Out.append("|------:|-------------:|-------------:|")
    for T in Temperatures:
        D = Dominance[str(T)]
        Out.append(f"| {str(T).center(5)} | {D['P_below_1']:>12.3f} | {D['P_above_1']:>12.3f} |")
    Out.append("")
    Out.append("## 2. Ordering probabilities (paired rounds)")
    Out.append("")
    Out.append("| Ordering | P |")
    Out.append("|---|---:|")
    Out.append(f"| phi_C(973) < phi_C(1073) < phi_C(1273)  (monotonic recovery) | {Monotonic:.3f} |")
    Out.append(f"| phi_C(873) > phi_C(973) < phi_C(1073) < phi_C(1273)  (full V-shape) | {VShape:.3f} |")
    Out.append(f"| phi_C(1073) < phi_C(1273)  (reference) | {Reference:.3f} |")
    Out.append("")
    Out.append("## 3. Integrity")
    Out.append("")
    Out.append("Array lengths: " + " / ".join(str(Lengths[T]) for T in Temperatures))
    Out.append("NaN counts:    " + " / ".join(str(NanCounts[T]) for T in Temperatures))
    print("\n".join(Out), flush=True)

    # --- Append to BootstrapResults.json ---
    JsonPath = os.path.join(ResultsDir, "BootstrapResults.json")
    if os.path.exists(JsonPath):
        with open(JsonPath, 'r') as F:
            Existing = json.load(F)
    else:
        Existing = {}
    Existing['FollowUpStatistics'] = FollowUp
    with open(JsonPath, 'w') as F:
        json.dump(Existing, F, indent=2)

    return FollowUp


# =============================================================================
#                                 DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Trajectory-stratified bootstrap of PhiC(T).")
    Parser.add_argument('--rebuild-events', action='store_true',
                        help="Force re-streaming of the xyz files even if the event-table cache exists.")
    Parser.add_argument('--nresamples', type=int, default=NResamples,
                        help="Bootstrap resamples per temperature (default 10000).")
    Parser.add_argument('--smoke', type=int, default=None,
                        help="Cap frames per xyz (plumbing test only; chain totals will NOT match Table 3).")
    Parser.add_argument('--jobs', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)),
                        help="Worker processes for the xyz chain pass (default min(6, cpus-2)).")
    Parser.add_argument('--follow-up', action='store_true',
                        help="Only compute the follow-up dominance/ordering statistics from the "
                             "cached ResampledPhiC.npz (no resampling, no xyz access), then exit.")
    Args = Parser.parse_args()

    os.makedirs(ResultsDir, exist_ok=True)
    EventTablePath = os.path.join(ResultsDir, "EventTable.csv")

    # ---- Follow-up-only mode: reuse the existing bootstrap arrays ----
    if Args.follow_up:
        ComputeFollowUpStatistics()
        return

    print("=" * 64)
    print("BOOTSTRAP VALIDATION OF THE CARBON RETENTION RATIO MINIMUM (973 K)")
    print(f"Seed = {RandomSeed} | Resamples = {Args.nresamples} | DataDir = {DataDir}")
    print("=" * 64, flush=True)

    # ---- Event table (cached) ----
    if Args.smoke is not None:
        print(f"\n[SMOKE MODE] capping {Args.smoke} frames/xyz - totals will not match Table 3.")
        EventFrame, OtherReport = BuildEventTable(MaxFrames=Args.smoke, Jobs=Args.jobs)
    elif Args.rebuild_events or not os.path.exists(EventTablePath):
        print(f"\nSTEP 1 - Building event table (streaming all xyz files, jobs={Args.jobs}) ...",
              flush=True)
        EventFrame, OtherReport = BuildEventTable(Jobs=Args.jobs)
        EventFrame.to_csv(EventTablePath, index=False)
        print(f"  Event table cached to {EventTablePath}", flush=True)
    else:
        print(f"\nSTEP 1 - Loading cached event table {EventTablePath}", flush=True)
        EventFrame = pd.read_csv(EventTablePath)
        OtherReport = {}

    # ---- Step 2: sanity check (hard guardrail on chain totals) ----
    print()
    Strict = Args.smoke is None
    Summary = SanityCheck(EventFrame, OtherReport, Strict=Strict)

    # ---- Step 3: bootstrap ----
    print("\nSTEP 3 - Trajectory-stratified bootstrap ...", flush=True)
    PhiCByTemp, NanCountByTemp = RunBootstrap(EventFrame, Args.nresamples, RandomSeed)

    # ---- Step 4: headline statistics ----
    print("\nSTEP 4 - Headline statistics ...", flush=True)
    Headline = ComputeHeadlineStatistics(PhiCByTemp, NanCountByTemp, EventFrame, Summary)

    # ---- Step 5: figure ----
    print("\nSTEP 5 - Figure ...", flush=True)
    MakeFigure(PhiCByTemp, Headline,
               os.path.join(ResultsDir, "PhiC_Bootstrap_Distributions.png"),
               os.path.join(ResultsDir, "PhiC_Bootstrap_Distributions.pdf"))

    # ---- Persist resample arrays + results ----
    np.savez_compressed(os.path.join(ResultsDir, "ResampledPhiC.npz"),
                        **{f"T{T}": PhiCByTemp[T] for T in Temperatures})
    with open(os.path.join(ResultsDir, "BootstrapResults.json"), 'w') as F:
        json.dump(Headline, F, indent=2)

    # ---- Step 6: report ----
    print("\n" + "=" * 64)
    print("STEP 6 - SUMMARY")
    print("=" * 64)
    print(f"Random seed              : {RandomSeed}")
    print(f"Resamples per temperature: {Args.nresamples}")
    print("Per-temperature event counts used (gas + chain):")
    for Temp in Temperatures:
        PT = Headline['PerTemperature'][str(Temp)]
        print(f"  {Temp} K: {PT['EventCount']} events "
              f"(chain={PT['ObservedChain']}, gas={PT['ObservedGas']}), "
              f"NaN rounds={PT['NanRounds']}")
    print(f"\nPRIMARY: P[ PhiC(973) < PhiC(873) AND PhiC(973) < PhiC(1073) ] = "
          f"{Headline['PrimaryFraction_973_below_873_and_1073']:.4f} "
          f"(over {Headline['PrimaryValidRounds']} valid rounds)")
    print(f"  P[ PhiC(973) < PhiC(873)  ]           = {Headline['Fraction_973_below_873']:.4f}")
    print(f"  P[ PhiC(973) < PhiC(1073) ]           = {Headline['Fraction_973_below_1073']:.4f}")
    print(f"  P[ 973 K is the global minimum ]      = {Headline['Fraction_973_global_minimum']:.4f}")
    print("\nPer-temperature PhiC (median [2.5, 97.5] percentile):")
    for Temp in Temperatures:
        PT = Headline['PerTemperature'][str(Temp)]
        print(f"  {Temp} K: {PT['Median']:.3f}  [{PT['P2_5']:.3f}, {PT['P97_5']:.3f}]"
              f"   observed={PT['Observed']:.3f}")
    print(f"\nResults written to: {ResultsDir}")
    print("=" * 64, flush=True)


if __name__ == "__main__":
    Main()
