"""
RateAnalysisAudit.py
=======================================================================
Full audit of the RateAnalysis CSV gas-removal logs against the raw xyz
trajectories (GROUND TRUTH).

For every one of the 16 trajectories:
  1. Stream the raw xyz frame by frame (~1 fs resolution, never loaded whole)
     counting C and O atoms per frame. Every drop in the C count is a real
     departure (atoms only leave the box via SLUSCHI removal).
  2. At each departure, parse the frames either side, identify the vanished
     carbon (nearest-neighbour matching, minimum image), and classify the
     departing molecule from its O coordination in the last frame present
     (O within ChemicalCOCutoff) cross-checked against the simultaneous O
     count drop.
  3. Parse the CSV 'Gas Removed' log (one molecule tuple per row) and match
     logged carbon events to xyz departures by time (within MatchWindowFs).
  4. Report, per trajectory: unlogged departures (with species and the CSV
     row where the entry belongs), phantom log rows (logged but no xyz
     departure), and species mismatches on matched events.

READ-ONLY throughout: neither the CSVs nor the xyz files are modified.
Output: RateAnalysisAuditResults/ (AuditFindings.csv, SUMMARY.md pieces
printed to console).

Usage:
    python RateAnalysisAudit.py            # all 16, 6 workers
    python RateAnalysisAudit.py --only 973_1
"""

import os
import re
import ast
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

ScriptDir = os.path.dirname(os.path.abspath(__file__))
DataDir = os.environ.get("ZRC_DATA_DIR", os.path.join(ScriptDir, "..", "data"))
ResultsDir = os.path.join(ScriptDir, "RateAnalysisAuditResults")

Temperatures = [873, 973, 1073, 1273]
SimIndices = [1, 2, 3, 4]

ChemicalCOCutoff = 1.7      # Angstrom; covalent C-O in a departing CO/CO2
MatchWindowFs = 500.0        # max |deletion - log row| for a match
LatticeRegex = re.compile(r'Lattice="([^"]+)"')
TimeRegex = re.compile(r'Time=(\S+)')


def StreamElementCounts(XyzPath):
    """Yield (TimeFs, NC, NO, RawAtomLines, CommentLine) per frame.

    RawAtomLines are kept only long enough for the caller to buffer the
    previous frame; nothing is accumulated.
    """
    with open(XyzPath) as Fh:
        while True:
            CountLine = Fh.readline()
            if not CountLine:
                return
            CountLine = CountLine.strip()
            if not CountLine:
                continue
            NAtoms = int(CountLine)
            Comment = Fh.readline()
            Lines = [Fh.readline() for _ in range(NAtoms)]
            NC = sum(1 for L in Lines if L.startswith('C'))
            NO = sum(1 for L in Lines if L.startswith('O'))
            Match = TimeRegex.search(Comment)
            TimeFs = float(Match.group(1)) if Match else float('nan')
            yield TimeFs, NC, NO, Lines, Comment


def ParseFrame(Lines, Comment):
    """Atom symbols + coordinates + lattice from buffered raw lines."""
    Symbols = np.empty(len(Lines), dtype='<U3')
    Coords = np.empty((len(Lines), 3))
    for I, L in enumerate(Lines):
        P = L.split()
        Symbols[I] = P[0]
        Coords[I] = [float(P[1]), float(P[2]), float(P[3])]
    LatMatch = LatticeRegex.search(Comment)
    Lattice = np.fromstring(LatMatch.group(1), sep=' ').reshape(3, 3)
    return Symbols, Coords, Lattice


def MinImageDistMatrix(A, B, Lattice):
    Inv = np.linalg.inv(Lattice)
    FracA = A @ Inv
    FracB = B @ Inv
    Diff = FracA[:, None, :] - FracB[None, :, :]
    Diff -= np.round(Diff)
    Cart = Diff @ Lattice
    return np.sqrt((Cart ** 2).sum(axis=2))


def ClassifyDeparture(PrevLines, PrevComment, CurrLines, CurrComment):
    """Identify the vanished carbon(s) and their O coordination in the last
    frame present. Returns a list of dicts (one per vanished C)."""
    PSym, PCoord, PLat = ParseFrame(PrevLines, PrevComment)
    CSym, CCoord, _ = ParseFrame(CurrLines, CurrComment)

    PrevC = PCoord[PSym == 'C']
    CurrC = CCoord[CSym == 'C']
    PrevO = PCoord[PSym == 'O']
    CurrO = CCoord[CSym == 'O']
    PrevZr = PCoord[PSym == 'Zr']

    Results = []
    if len(CurrC) == 0 or len(PrevC) == 0:
        return Results
    Dist = MinImageDistMatrix(PrevC, CurrC, PLat)
    NearestNext = Dist.min(axis=1)
    MissingIdx = np.where(NearestNext > 1.0)[0]      # ~1 fs frames: real C moves << 1 A

    # O atoms that also vanished this step (for cross-checking species and
    # for the molecule-integrity classification).
    VanishedOCount = 0
    VanishedOPos = np.zeros((0, 3))
    if len(PrevO) and len(CurrO):
        DistO = MinImageDistMatrix(PrevO, CurrO, PLat)
        MissingO = np.where(DistO.min(axis=1) > 1.0)[0]
        VanishedOCount = len(MissingO)
        VanishedOPos = PrevO[MissingO]
    elif len(PrevO):
        VanishedOCount = len(PrevO)
        VanishedOPos = PrevO

    for Mi in MissingIdx:
        CPos = PrevC[Mi][None, :]
        DistCO = MinImageDistMatrix(CPos, PrevO, PLat)[0] if len(PrevO) else np.array([])
        NBondedO = int((DistCO < ChemicalCOCutoff).sum())
        NearestOList = np.sort(DistCO)[:3].round(3).tolist() if len(DistCO) else []
        DistCZr = MinImageDistMatrix(CPos, PrevZr, PLat)[0] if len(PrevZr) else np.array([])
        NearestZr = float(DistCZr.min()) if len(DistCZr) else float('nan')
        Species = {0: 'C', 1: 'CO', 2: 'CO2'}.get(NBondedO, f'CO{NBondedO}?')
        # Molecule integrity: was the vanished C bonded to the O atom(s) that
        # vanished in the same step? Clean removal = every co-vanished O within
        # ChemicalCOCutoff of the vanished C.
        if len(VanishedOPos):
            DistToVanO = MinImageDistMatrix(CPos, VanishedOPos, PLat)[0]
            NVanOBonded = int((DistToVanO < ChemicalCOCutoff).sum())
            MutualDistances = np.sort(DistToVanO).round(3).tolist()
        else:
            NVanOBonded = 0
            MutualDistances = []
        CleanMolecule = (VanishedOCount > 0 and NVanOBonded == VanishedOCount)
        Results.append(dict(
            NBondedO=NBondedO, Species=Species, NearestODistances=NearestOList,
            NearestZrDistance=round(NearestZr, 3), VanishedOThisStep=VanishedOCount,
            CleanMolecule=bool(CleanMolecule), VanishedODistances=MutualDistances,
        ))
    return Results


def LoadLoggedCarbonEvents(CsvPath):
    """(RowTimeFs, SpeciesString, RawCell) for every C-bearing Gas Removed row."""
    Df = pd.read_csv(CsvPath)
    Events = []
    for _, Row in Df.iterrows():
        Cell = Row['Gas Removed']
        if not isinstance(Cell, str) or Cell.strip() in ('[]', ''):
            continue
        Molecules = ast.literal_eval(Cell)
        for Mol in Molecules:
            NC = sum(1 for A in Mol if A == 'C')
            NO = sum(1 for A in Mol if A == 'O')
            if NC == 0:
                continue
            Species = 'CO' if NO == 1 else ('CO2' if NO == 2 else f'C{NC}O{NO}')
            Events.append(dict(RowTimeFs=float(Row['Time (fs)']), Species=Species,
                               RawCell=Cell))
    return Df, Events


def AuditTrajectory(Args):
    Temp, Sim = Args
    Label = f"{Temp}_{Sim}"
    XyzPath = os.path.join(DataDir, f"{Label}.xyz")
    CsvPath = os.path.join(DataDir, f"RateAnalysis_{Temp}_{Sim}.csv")

    Departures = []
    PrevTime, PrevNC, PrevLines, PrevComment = None, None, None, None
    LastTime = 0.0
    for TimeFs, NC, NO, Lines, Comment in StreamElementCounts(XyzPath):
        if PrevNC is not None and NC < PrevNC:
            Info = ClassifyDeparture(PrevLines, PrevComment, Lines, Comment)
            for Rec in Info:
                Rec.update(TimeFs=TimeFs, PrevTimeFs=PrevTime)
                Departures.append(Rec)
        PrevTime, PrevNC, PrevLines, PrevComment = TimeFs, NC, Lines, Comment
        LastTime = TimeFs
    FinalNC = PrevNC

    Df, Logged = LoadLoggedCarbonEvents(CsvPath)
    RowTimes = Df['Time (fs)'].to_numpy()

    # Greedy nearest-in-time matching, one-to-one.
    UsedLog = set()
    for Dep in Departures:
        Best, BestGap = None, MatchWindowFs
        for Li, Log in enumerate(Logged):
            if Li in UsedLog:
                continue
            Gap = abs(Dep['TimeFs'] - Log['RowTimeFs'])
            if Gap < BestGap:
                Best, BestGap = Li, Gap
        Dep['MatchedLogIdx'] = Best
        Dep['MatchGapFs'] = round(BestGap, 1) if Best is not None else None
        if Best is not None:
            UsedLog.add(Best)
            Dep['LoggedSpecies'] = Logged[Best]['Species']
        else:
            Dep['LoggedSpecies'] = None

    Unlogged = [D for D in Departures if D['MatchedLogIdx'] is None]
    Phantoms = [L for Li, L in enumerate(Logged) if Li not in UsedLog]
    SpeciesMismatch = [D for D in Departures
                       if D['MatchedLogIdx'] is not None
                       and D['LoggedSpecies'] != D['Species']]

    # For each unlogged departure, the CSV row where the entry belongs:
    # the last row at or before the deletion frame.
    for D in Unlogged:
        Idx = int(np.searchsorted(RowTimes, D['TimeFs'], side='right')) - 1
        D['TargetRowTimeFs'] = float(RowTimes[Idx]) if Idx >= 0 else None
        D['TargetRowCurrent'] = Df.iloc[Idx]['Gas Removed'] if Idx >= 0 else None

    return dict(
        Label=Label, NDepartures=len(Departures), NLogged=len(Logged),
        FinalC=FinalNC, LastTimeFs=LastTime,
        Unlogged=Unlogged, Phantoms=Phantoms, SpeciesMismatch=SpeciesMismatch,
        AllDepartures=Departures,
    )


def Main():
    Parser = argparse.ArgumentParser()
    Parser.add_argument("--only", type=str, default=None)
    Parser.add_argument("--jobs", type=int, default=6)
    Args = Parser.parse_args()

    Tasks = [(T, S) for T in Temperatures for S in SimIndices]
    if Args.only:
        T, S = Args.only.split("_")
        Tasks = [(int(T), int(S))]

    os.makedirs(ResultsDir, exist_ok=True)

    if len(Tasks) > 1 and Args.jobs > 1:
        with Pool(Args.jobs) as P:
            Results = P.map(AuditTrajectory, Tasks)
    else:
        Results = [AuditTrajectory(T) for T in Tasks]

    Rows = []
    print(f"{'Traj':>8} {'xyzDep':>7} {'logged':>7} {'C@end':>6} "
          f"{'unlogged':>9} {'phantom':>8} {'specMis':>8}")
    for R in Results:
        Ok = (not R['Unlogged']) and (not R['Phantoms']) and (not R['SpeciesMismatch'])
        print(f"{R['Label']:>8} {R['NDepartures']:>7} {R['NLogged']:>7} "
              f"{R['FinalC']:>6} {len(R['Unlogged']):>9} {len(R['Phantoms']):>8} "
              f"{len(R['SpeciesMismatch']):>8}  {'OK' if Ok else '<-- FINDINGS'}")
        for D in R['Unlogged']:
            Rows.append(dict(Trajectory=R['Label'], Finding='UnloggedDeparture',
                             XyzDeletionFs=D['TimeFs'], Species=D['Species'],
                             NBondedO=D['NBondedO'],
                             NearestODistances=str(D['NearestODistances']),
                             NearestZrDistance=D['NearestZrDistance'],
                             TargetRowTimeFs=D['TargetRowTimeFs'],
                             TargetRowCurrent=D['TargetRowCurrent']))
        for L in R['Phantoms']:
            Rows.append(dict(Trajectory=R['Label'], Finding='PhantomLogRow',
                             XyzDeletionFs=None, Species=L['Species'],
                             TargetRowTimeFs=L['RowTimeFs'],
                             TargetRowCurrent=L['RawCell']))
        for D in R['SpeciesMismatch']:
            Rows.append(dict(Trajectory=R['Label'], Finding='SpeciesMismatch',
                             XyzDeletionFs=D['TimeFs'], Species=D['Species'],
                             LoggedSpecies=D['LoggedSpecies'],
                             NearestODistances=str(D['NearestODistances'])))

    Findings = pd.DataFrame(Rows)
    OutPath = os.path.join(ResultsDir, "AuditFindings.csv")
    Findings.to_csv(OutPath, index=False)
    print(f"\nFindings written: {OutPath} ({len(Findings)} rows)")
    if len(Findings):
        print(Findings.to_string(index=False))

    # Full per-departure integrity table (every deletion in every trajectory).
    IntRows = []
    for R in Results:
        for D in R['AllDepartures']:
            IntRows.append(dict(
                Trajectory=R['Label'], DeletionFs=D['TimeFs'],
                Species=D['Species'], VanishedO=D['VanishedOThisStep'],
                CleanMolecule=D['CleanMolecule'],
                VanishedODistances=str(D['VanishedODistances']),
                NearestZrDistance=D['NearestZrDistance'],
                Logged=D['MatchedLogIdx'] is not None,
            ))
    Integrity = pd.DataFrame(IntRows)
    IntPath = os.path.join(ResultsDir, "AllDeparturesIntegrity.csv")
    Integrity.to_csv(IntPath, index=False)
    NScrambled = int((~Integrity['CleanMolecule']).sum())
    print(f"\nIntegrity table written: {IntPath} ({len(Integrity)} departures, "
          f"{NScrambled} with non-molecular atom deletion)")
    if NScrambled:
        print(Integrity[~Integrity['CleanMolecule']].to_string(index=False))


if __name__ == "__main__":
    Main()
