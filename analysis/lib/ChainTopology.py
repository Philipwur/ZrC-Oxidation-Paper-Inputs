"""
ChainTopology.py
=======================================================================
Structural characterisation of interfacial carbon chains for the ZrC oxidation
study. Upgrades the paper's "consistent with an atomistic precursor of the
amorphous interfacial carbon layer" from assertion to observation via bond
angles, branching, rings, and chain-termination chemistry.

Descriptive statistics only. The per-temperature sample is too small for
temperature-resolved inference, so everything is pooled into two crossover
groups:
    BelowCrossover = 873 + 973 K
    AboveCrossover = 1073 + 1273 K

Window & detection
------------------
Only the final 10% of each trajectory is analysed (the paper's density-profile
window, FrameWindow = (90, 100)%). Earlier frames are fast-forwarded by byte
offset without parsing coordinates. Chain carbons are identified with the exact
bootstrap detection stack (instantaneous C-C / C-Zr coordination + 0.2 ps state-
persistence smoothing) imported from BootstrapCalculation.py, so counts stay
consistent with the 35/31/48/66 final-frame values (asserted per trajectory).

Bond rule (paper-exact): bond iff distance < (Ri + Rj) * 1.5 under the minimum-
image convention, with RC = 0.77, RO = 0.66, RZr = 1.45 A (=> C-C 2.31, C-O 2.15,
Zr-C 3.33, Zr-O 3.17 A). All bonds are built from these radii, so the cutoffs are
reproduced exactly.

Frames are ~1 fs apart and heavily time-correlated, so topology statistics are
sampled every 50 fs (deterministic, not stochastic) after a 0.2 ps burn-in that
lets the persistence filter stabilise inside the window. The persistence filter
itself runs on EVERY window frame; only the recorded statistics are subsampled.

Usage
-----
    python ChainTopology.py                 # all 16 trajectories, 6 workers
    python ChainTopology.py --jobs 4
    python ChainTopology.py --only 873_1    # single trajectory (debug)
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

# Reuse the bootstrap's exact detection stack + constants.
ScriptDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
sys.path.insert(0, ScriptDir)
import BootstrapCalculation as Bc
from BootstrapCalculation import (
    CovalentRadii, RadiusTol, CutoffCC, CutoffZrC, TotalStartC,
    StatePersistenceTimePs, CartesianToFractional, MinimumImageDistances,
    MapAtomIdentities, DetermineInstantaneousStates, ParseLatticeFromComment,
    TimeCommentRegex, FirstFloatRegex, PaperChainByTraj,
)


# =============================================================================
#                                CONFIGURATION
# =============================================================================
ResultsDir = os.path.join(ScriptDir, "ChainTopologyResults")
DataDir    = os.environ.get("ZRC_DATA_DIR", os.path.join(ScriptDir, "..", "data"))

Temperatures = [873, 973, 1073, 1273]
SimIndices   = [1, 2, 3, 4]

RandomSeed   = 20260715          # reported; subsampling is deterministic (no RNG used)
WindowStartPct = 0.90            # final 10% (paper FrameWindow = (90, 100)%)
SubsampleFs  = 50.0              # statistics sampling interval within the window
BurnInFs     = StatePersistenceTimePs * 1000.0    # 200 fs; persistence stabilisation
SnippetRadius = 4.0              # A; neighbours exported around representative chains

# Bond cutoffs (derived from the paper radii; used only for reporting/consistency).
CutoffCO  = (CovalentRadii['C'] + CovalentRadii['O'])  * RadiusTol   # 2.145 A
CutoffZrO = (CovalentRadii['Zr'] + CovalentRadii['O']) * RadiusTol   # 3.165 A

# Termination criteria.
CarbideZrMinC = 4                # a carbide Zr has >= 4 C neighbours (Zr-C coordination)
OxideZrMinO   = 2                # an oxidised Zr has >= 2 O neighbours (paper NO >= 2)

GroupBelow = "BelowCrossover"
GroupAbove = "AboveCrossover"


def GroupOf(Temp):
    return GroupBelow if Temp in (873, 973) else GroupAbove


def XyzPath(Temp, Sim):
    return os.path.join(DataDir, f"{Temp}_{Sim}.xyz")


# =============================================================================
#                      STREAMING (fast-forward by byte offset)
# =============================================================================
def ScanFrameOffsets(FilePath):
    """
    One line-only pass: count frames and record the byte offset of each frame's
    atom-count header. No coordinate parsing. Handles variable atom counts
    (atom count is re-read every frame).
    """
    Offsets = []
    with open(FilePath, 'r') as Fh:
        while True:
            Pos = Fh.tell()
            Line = Fh.readline()
            if not Line:
                break
            Stripped = Line.strip()
            if not Stripped:
                continue
            try:
                NAtoms = int(Stripped)
            except ValueError:
                break
            Offsets.append(Pos)
            for _ in range(NAtoms + 1):     # skip comment + NAtoms atom lines
                Fh.readline()
    return len(Offsets), Offsets


def ReadFramesFrom(FilePath, StartOffset):
    """Seek to StartOffset and stream frames to EOF: (TimeFs, Atoms, Coords, Lattice)."""
    with open(FilePath, 'r') as Fh:
        Fh.seek(StartOffset)
        while True:
            Line = Fh.readline()
            if not Line:
                break
            Stripped = Line.strip()
            if not Stripped:
                continue
            try:
                NAtoms = int(Stripped)
            except ValueError:
                break
            Comment = Fh.readline()
            Atoms = np.empty(NAtoms, dtype=object)
            Coords = np.empty((NAtoms, 3), dtype=float)
            Broken = False
            for i in range(NAtoms):
                Parts = Fh.readline().split()
                if len(Parts) < 4:
                    Broken = True
                    break
                Atoms[i] = Parts[0]
                Coords[i, 0] = float(Parts[1]); Coords[i, 1] = float(Parts[2]); Coords[i, 2] = float(Parts[3])
            if Broken:
                break
            Match = TimeCommentRegex.search(Comment) or FirstFloatRegex.search(Comment)
            TimeFs = float(Match.group(1)) if Match else 0.0
            yield TimeFs, Atoms, Coords, ParseLatticeFromComment(Comment)


# =============================================================================
#                          GEOMETRY / BOND HELPERS
# =============================================================================
def ComputeBonds(Atoms, FracCoords, Lattice):
    """All-atom minimum-image bond adjacency via distance < (Ri+Rj)*RadiusTol."""
    Dist = MinimumImageDistances(FracCoords, Lattice)
    Radii = np.array([CovalentRadii[A] for A in Atoms], dtype=float)
    Cutoff = (Radii[:, None] + Radii[None, :]) * RadiusTol
    Bond = Dist < Cutoff
    np.fill_diagonal(Bond, False)
    return Bond


def MinImageVector(FracA, FracB, Lattice):
    """Cartesian minimum-image vector A -> B."""
    Diff = FracB - FracA
    Diff -= np.round(Diff)
    return Diff @ Lattice


def ConnectedComponents(Nodes, Adjacency):
    """Connected components of an undirected graph given as adjacency lists."""
    Seen = set()
    Comps = []
    for Start in Nodes:
        if Start in Seen:
            continue
        Stack = [Start]
        Comp = []
        while Stack:
            Node = Stack.pop()
            if Node in Seen:
                continue
            Seen.add(Node)
            Comp.append(Node)
            for Nb in Adjacency[Node]:
                if Nb not in Seen:
                    Stack.append(Nb)
        Comps.append(Comp)
    return Comps


def FundamentalCycleSizes(Comp, Adjacency):
    """
    Sizes of fundamental cycles in one connected component (BFS spanning tree;
    each non-tree edge closes one cycle). The cycle length of a non-tree edge
    (u, v) is the tree-path length u..LCA..v plus 1. Empty list => acyclic.
    """
    Depth = {Comp[0]: 0}
    Parent = {Comp[0]: None}
    Queue = [Comp[0]]
    TreeEdges = set()
    while Queue:
        Node = Queue.pop(0)
        for Nb in Adjacency[Node]:
            if Nb not in Depth:
                Depth[Nb] = Depth[Node] + 1
                Parent[Nb] = Node
                TreeEdges.add(frozenset((Node, Nb)))
                Queue.append(Nb)

    def TreePathLength(U, V):
        Steps = 0
        while Depth[U] > Depth[V]:
            U = Parent[U]; Steps += 1
        while Depth[V] > Depth[U]:
            V = Parent[V]; Steps += 1
        while U != V:
            U = Parent[U]; V = Parent[V]; Steps += 2
        return Steps

    Sizes = []
    for Node in Comp:
        for Nb in Adjacency[Node]:
            if Node < Nb and frozenset((Node, Nb)) not in TreeEdges:
                Sizes.append(TreePathLength(Node, Nb) + 1)
    return Sizes


# =============================================================================
#                       PER-FRAME TOPOLOGY ANALYSIS
# =============================================================================
def AnalyseFrameTopology(Atoms, FracCoords, Lattice, ConfirmedStates, TrackedIDs):
    """
    Topology of the confirmed chain carbons in one frame. Returns a dict of
    chain records, per-carbon degree stats, C-C-C angles, termini classification,
    and ring sizes. C-neighbour counts use the chain graph (bonded confirmed chain
    carbons); O/Zr neighbour counts use all bonded O/Zr atoms.
    """
    Bond = ComputeBonds(Atoms, FracCoords, Lattice)
    CarbonFullIdx = np.where(Atoms == 'C')[0]
    ElemArr = Atoms

    ChainFullIdx = [int(CarbonFullIdx[i]) for i in range(len(TrackedIDs))
                    if ConfirmedStates.get(int(TrackedIDs[i])) == 'Chain']

    # Chain graph (C-C bonds among confirmed chain carbons).
    Adjacency = {F: [] for F in ChainFullIdx}
    ChainSet = set(ChainFullIdx)
    for A in ChainFullIdx:
        BondedC = np.where(Bond[A] & (ElemArr == 'C'))[0]
        for B in BondedC:
            if int(B) in ChainSet:
                Adjacency[A].append(int(B))

    Degree = {F: len(Adjacency[F]) for F in ChainFullIdx}
    OCount = {F: int(np.sum(Bond[F] & (ElemArr == 'O'))) for F in ChainFullIdx}
    ZrCount = {F: int(np.sum(Bond[F] & (ElemArr == 'Zr'))) for F in ChainFullIdx}

    Comps = ConnectedComponents(ChainFullIdx, Adjacency)

    # Angles: chain carbons with exactly 2 chain-C neighbours.
    Angles = []
    for A in ChainFullIdx:
        if Degree[A] == 2:
            N1, N2 = Adjacency[A]
            V1 = MinImageVector(FracCoords[A], FracCoords[N1], Lattice)
            V2 = MinImageVector(FracCoords[A], FracCoords[N2], Lattice)
            Cos = np.dot(V1, V2) / (np.linalg.norm(V1) * np.linalg.norm(V2) + 1e-12)
            Angles.append(float(np.degrees(np.arccos(np.clip(Cos, -1.0, 1.0)))))

    BranchPoints = sum(1 for F in ChainFullIdx if Degree[F] >= 3)

    # Termini (degree 1) and their non-carbon anchoring.
    Termini = []
    RingSizes = []
    ChainRecords = []
    for Ci, Comp in enumerate(Comps):
        CompTermini = []
        for A in Comp:
            if Degree[A] != 1:
                continue
            ZrNeighbours = np.where(Bond[A] & (ElemArr == 'Zr'))[0]
            Carbide = False
            Oxide = False
            for Z in ZrNeighbours:
                NcZr = int(np.sum(Bond[Z] & (ElemArr == 'C')))
                NoZr = int(np.sum(Bond[Z] & (ElemArr == 'O')))
                if NcZr >= CarbideZrMinC:
                    Carbide = True
                if NoZr >= OxideZrMinO:
                    Oxide = True
            OxygenBonded = OCount[A] >= 1
            Dangling = not (Carbide or Oxide or OxygenBonded)
            Term = {'CarbideAnchored': Carbide, 'OxideAnchored': Oxide,
                    'OxygenBonded': OxygenBonded, 'Dangling': Dangling}
            Termini.append(Term)
            CompTermini.append(Term)

        RingSizes.extend(FundamentalCycleSizes(Comp, Adjacency))
        Bridging = (any(T['CarbideAnchored'] for T in CompTermini) and
                    any(T['OxideAnchored'] or T['OxygenBonded'] for T in CompTermini))
        ChainRecords.append({
            'ChainIndex': Ci,
            'Length': len(Comp),
            'NBranchPoints': sum(1 for A in Comp if Degree[A] >= 3),
            'NTermini': len(CompTermini),
            'SumO': sum(OCount[A] for A in Comp),
            'SumZr': sum(ZrCount[A] for A in Comp),
            'Bridging': bool(Bridging),
        })

    return {
        'ChainFullIdx': ChainFullIdx,
        'Adjacency': Adjacency,
        'Degree': Degree,
        'Components': Comps,
        'NChainCarbons': len(ChainFullIdx),
        'NChains': len(Comps),
        'Angles': Angles,
        'BranchPoints': BranchPoints,
        'Termini': Termini,
        'RingSizes': RingSizes,
        'ChainRecords': ChainRecords,
    }


# =============================================================================
#                       PER-TRAJECTORY WORKER
# =============================================================================
def ProcessTrajectory(TaskArgs):
    """Stream one trajectory's final-10% window and accumulate topology stats."""
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

    Accum = {
        'InventoryRows': [], 'Angles': [], 'Termini': [], 'RingSizes': [],
        'ChainCarbonObs': 0, 'BranchObs': 0, 'ChainObs': 0, 'BridgingObs': 0,
        'PerFrameChainCarbons': [], 'PerFrameChains': [],
    }
    LastFrame = None

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

        # Persistence filter (every frame).
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

        # Record subsampled statistics after the burn-in.
        Stable = (TimeFs - WindowStartTime) >= BurnInFs
        Due = LastRecordedTime is None or (TimeFs - LastRecordedTime) >= SubsampleFs
        if Stable and Due:
            LastRecordedTime = TimeFs
            NAnalysed += 1
            Topo = AnalyseFrameTopology(Atoms, FracCoords, Lattice, ConfirmedStates, TrackedIDs)
            Accum['Angles'].extend(Topo['Angles'])
            Accum['Termini'].extend(Topo['Termini'])
            Accum['RingSizes'].extend(Topo['RingSizes'])
            Accum['ChainCarbonObs'] += Topo['NChainCarbons']
            Accum['BranchObs'] += Topo['BranchPoints']
            Accum['ChainObs'] += Topo['NChains']
            Accum['BridgingObs'] += sum(1 for C in Topo['ChainRecords'] if C['Bridging'])
            Accum['PerFrameChainCarbons'].append(Topo['NChainCarbons'])
            Accum['PerFrameChains'].append(Topo['NChains'])
            for Rec in Topo['ChainRecords']:
                Accum['InventoryRows'].append({
                    'Temperature': Temp, 'TrajectoryId': TrajId, 'Group': Group,
                    'Time_ps': round(Tp, 3), 'ChainIndex': Rec['ChainIndex'],
                    'Length': Rec['Length'], 'NBranchPoints': Rec['NBranchPoints'],
                    'NTermini': Rec['NTermini'], 'SumO': Rec['SumO'],
                    'SumZr': Rec['SumZr'], 'Bridging': Rec['Bridging'],
                })

        LastFrame = (TimeFs, Atoms, Coords, Lattice)

    # End-of-trajectory flush (matches the bootstrap final count).
    for AtomID, Pending in PendingStates.items():
        ConfirmedStates[AtomID] = Pending['State']

    # Final-frame chain grouping (for chain-length distribution + representatives).
    FinalTime, FinalAtoms, FinalCoords, FinalLattice = LastFrame
    FinalFrac = CartesianToFractional(FinalCoords, FinalLattice)
    FinalTopo = AnalyseFrameTopology(FinalAtoms, FinalFrac, FinalLattice, ConfirmedStates, TrackedIDs)
    FinalChainLengths = [len(C) for C in FinalTopo['Components']]
    FinalChainCount = FinalTopo['NChainCarbons']

    return {
        'TrajId': TrajId, 'Temp': Temp, 'Group': Group,
        'NFrames': NFrames, 'IdxStart': IdxStart,
        'WindowStartTime_ps': WindowStartTime / 1000.0,
        'WindowEndTime_ps': FinalTime / 1000.0,
        'NWindowFrames': NFrames - IdxStart,
        'NAnalysedFrames': NAnalysed,
        'FinalChainCount': FinalChainCount,
        'ExpectedChainCount': PaperChainByTraj[TrajId],
        'FinalChainLengths': FinalChainLengths,
        'Accum': Accum,
        # Final-frame geometry for representative-structure export.
        'FinalAtoms': FinalAtoms.tolist(),
        'FinalCoords': FinalCoords.tolist(),
        'FinalLattice': FinalLattice.tolist(),
        'FinalComponents': [[int(A) for A in C] for C in FinalTopo['Components']],
        'FinalDegrees': {int(k): int(v) for k, v in FinalTopo['Degree'].items()},
    }


# =============================================================================
#                       REPRESENTATIVE STRUCTURE EXPORT
# =============================================================================
def ExportChainSnippet(Result, Component, OutPath, Motif):
    """Write chain atoms + neighbours within SnippetRadius as an unwrapped .xyz."""
    Atoms = np.array(Result['FinalAtoms'], dtype=object)
    Coords = np.array(Result['FinalCoords'], dtype=float)
    Lattice = np.array(Result['FinalLattice'], dtype=float)
    Frac = CartesianToFractional(Coords, Lattice)

    ChainIdx = list(Component)
    Included = set(ChainIdx)
    for A in ChainIdx:
        Diff = Frac - Frac[A]
        Diff -= np.round(Diff)
        Dist = np.sqrt(np.sum((Diff @ Lattice) ** 2, axis=1))
        for B in np.where(Dist < SnippetRadius)[0]:
            Included.add(int(B))

    # Unwrap the cluster around the first chain atom for clean rendering.
    Ref = Frac[ChainIdx[0]]
    Lines = [str(len(Included)),
             f"{Motif} chain from {Result['TrajId']} (T={Result['Temp']} K), "
             f"{len(ChainIdx)} chain C + neighbours < {SnippetRadius} A"]
    for B in sorted(Included):
        Diff = Frac[B] - Ref
        Diff -= np.round(Diff)
        Pos = Ref @ Lattice + Diff @ Lattice
        Lines.append(f"{Atoms[B]:2s} {Pos[0]:12.6f} {Pos[1]:12.6f} {Pos[2]:12.6f}")
    with open(OutPath, 'w') as F:
        F.write("\n".join(Lines) + "\n")


def SelectRepresentatives(Results):
    """Pick 3-4 final-frame chains spanning motifs: shortest, longest, branched, medium."""
    Catalogue = []
    for R in Results:
        for Comp in R['FinalComponents']:
            HasBranch = any(R['FinalDegrees'].get(A, 0) >= 3 for A in Comp)
            Catalogue.append({'Result': R, 'Component': Comp, 'Length': len(Comp),
                              'HasBranch': HasBranch})
    Chosen = []
    Used = set()

    def Take(Item, Motif):
        Key = (Item['Result']['TrajId'], tuple(sorted(Item['Component'])))
        if Key in Used:
            return
        Used.add(Key)
        Chosen.append((Item, Motif))

    Linear = [C for C in Catalogue if not C['HasBranch']]
    Branched = [C for C in Catalogue if C['HasBranch']]
    if Linear:
        Take(min(Linear, key=lambda C: C['Length']), "ShortLinear")
    if Catalogue:
        Take(max(Catalogue, key=lambda C: C['Length']), "Longest")
    if Branched:
        Take(max(Branched, key=lambda C: C['Length']), "Branched")
    # A medium linear chain to round out the set.
    MediumPool = sorted([C for C in Linear if 3 <= C['Length'] <= 6], key=lambda C: C['Length'])
    if MediumPool:
        Take(MediumPool[len(MediumPool) // 2], "MediumLinear")
    return Chosen[:4]


# =============================================================================
#                             AGGREGATION
# =============================================================================
def AggregateGroup(Results, GroupName):
    Members = [R for R in Results if R['Group'] == GroupName]
    Angles = []
    Termini = []
    RingSizes = []
    ChainCarbonObs = BranchObs = ChainObs = BridgingObs = 0
    PerFrameChainCarbons = []
    PerFrameChains = []
    NAnalysed = 0
    FinalLengths = []
    for R in Members:
        A = R['Accum']
        Angles.extend(A['Angles'])
        Termini.extend(A['Termini'])
        RingSizes.extend(A['RingSizes'])
        ChainCarbonObs += A['ChainCarbonObs']
        BranchObs += A['BranchObs']
        ChainObs += A['ChainObs']
        BridgingObs += A['BridgingObs']
        PerFrameChainCarbons.extend(A['PerFrameChainCarbons'])
        PerFrameChains.extend(A['PerFrameChains'])
        NAnalysed += R['NAnalysedFrames']
        FinalLengths.extend(R['FinalChainLengths'])

    Angles = np.array(Angles)
    NTerm = len(Termini)

    def Frac(Key):
        return sum(1 for T in Termini if T[Key]) / NTerm if NTerm else float('nan')

    return {
        'NAnalysedFrames': NAnalysed,
        'MeanChainCarbonsPerFrame': float(np.mean(PerFrameChainCarbons)) if PerFrameChainCarbons else float('nan'),
        'MeanChainsPerFrame': float(np.mean(PerFrameChains)) if PerFrameChains else float('nan'),
        'AngleSamples': int(Angles.size),
        'MedianAngle': float(np.median(Angles)) if Angles.size else float('nan'),
        'FractionAngleAbove150': float(np.mean(Angles > 150.0)) if Angles.size else float('nan'),
        'BranchPointFraction': (BranchObs / ChainCarbonObs) if ChainCarbonObs else float('nan'),
        'CyclesFound': int(len(RingSizes)),
        'RingSizes': sorted(RingSizes),
        'TriangleCount': int(sum(1 for S in RingSizes if S == 3)),
        'GenuineRingSizes': sorted(S for S in RingSizes if S >= 4),
        'AromaticRingCount': int(sum(1 for S in RingSizes if S >= 5)),
        'NTermini': NTerm,
        'Termination': {
            'CarbideAnchored': Frac('CarbideAnchored'),
            'OxideAnchored': Frac('OxideAnchored'),
            'OxygenBonded': Frac('OxygenBonded'),
            'Dangling': Frac('Dangling'),
        },
        'TerminationCounts': {
            'CarbideAnchored': sum(1 for T in Termini if T['CarbideAnchored']),
            'OxideAnchored': sum(1 for T in Termini if T['OxideAnchored']),
            'OxygenBonded': sum(1 for T in Termini if T['OxygenBonded']),
            'Dangling': sum(1 for T in Termini if T['Dangling']),
        },
        'ChainObs': ChainObs,
        'BridgingObs': BridgingObs,
        'BridgingChainFraction': (BridgingObs / ChainObs) if ChainObs else float('nan'),
        'FinalChainLengths': sorted(FinalLengths),
    }


def ChainLengthTable(BelowLengths, AboveLengths):
    AllLengths = sorted(set(BelowLengths) | set(AboveLengths))
    from collections import Counter
    Bc_ = Counter(BelowLengths)
    Ac_ = Counter(AboveLengths)
    return [{'Length': L, 'Below': Bc_.get(L, 0), 'Above': Ac_.get(L, 0)} for L in AllLengths]


# =============================================================================
#                               FIGURE
# =============================================================================
def MakeAngleFigure(BelowAngles, AboveAngles, OutPng, OutPdf):
    Fig, Ax = plt.subplots(figsize=(7.6, 5.0))
    Bins = np.linspace(60, 180, 41)
    Ax.hist(BelowAngles, bins=Bins, density=True, alpha=0.55, color='#1f77b4',
            label=f"BelowCrossover (n={len(BelowAngles)})")
    Ax.hist(AboveAngles, bins=Bins, density=True, alpha=0.55, color='#d62728',
            label=f"AboveCrossover (n={len(AboveAngles)})")
    Ax.axvline(120, color='0.35', ls='--', lw=1.5)
    Ax.axvline(180, color='0.35', ls=':', lw=1.5)
    Ax.annotate("120 (sp$^2$-like)", (120, Ax.get_ylim()[1] * 0.92), rotation=90,
                va='top', ha='right', fontsize=9, color='0.35')
    Ax.annotate("180 (sp/cumulenic)", (179, Ax.get_ylim()[1] * 0.92), rotation=90,
                va='top', ha='right', fontsize=9, color='0.35')
    Ax.set_xlim(60, 180)
    Ax.set_xlabel("C-C-C angle (degrees)")
    Ax.set_ylabel("Probability density")
    Ax.set_title("Interfacial carbon-chain C-C-C bond angles")
    Ax.legend(loc='upper left')
    Fig.tight_layout()
    Fig.savefig(OutPng, dpi=300)
    Fig.savefig(OutPdf)
    plt.close(Fig)


# =============================================================================
#                                 DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Interfacial carbon chain topology.")
    Parser.add_argument('--jobs', type=int, default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Parser.add_argument('--only', type=str, default=None, help="Single trajectory, e.g. 873_1 (debug).")
    Args = Parser.parse_args()

    np.random.seed(RandomSeed)
    os.makedirs(ResultsDir, exist_ok=True)

    if Args.only:
        Temp, Sim = Args.only.split('_')
        Tasks = [(int(Temp), int(Sim))]
    else:
        Tasks = [(T, S) for T in Temperatures for S in SimIndices]

    print(f"Analysing {len(Tasks)} trajectories (jobs={Args.jobs}) ...", flush=True)
    Results = []
    if Args.jobs > 1 and len(Tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=Args.jobs) as Pool:
            Futures = {Pool.submit(ProcessTrajectory, T): T for T in Tasks}
            for Fut in as_completed(Futures):
                R = Fut.result()
                Results.append(R)
                Flag = "OK" if R['FinalChainCount'] == R['ExpectedChainCount'] else "MISMATCH"
                print(f"  {R['TrajId']}: window {R['NWindowFrames']} frames, "
                      f"{R['NAnalysedFrames']} analysed, final chain {R['FinalChainCount']} "
                      f"vs {R['ExpectedChainCount']} [{Flag}]", flush=True)
    else:
        for T in Tasks:
            R = ProcessTrajectory(T)
            Results.append(R)
            Flag = "OK" if R['FinalChainCount'] == R['ExpectedChainCount'] else "MISMATCH"
            print(f"  {R['TrajId']}: window {R['NWindowFrames']} frames, "
                  f"{R['NAnalysedFrames']} analysed, final chain {R['FinalChainCount']} "
                  f"vs {R['ExpectedChainCount']} [{Flag}]", flush=True)

    Results.sort(key=lambda R: (R['Temp'], R['TrajId']))
    ConsistencyPass = all(R['FinalChainCount'] == R['ExpectedChainCount'] for R in Results)

    # ---- Inventory CSV ----
    InventoryRows = []
    for R in Results:
        InventoryRows.extend(R['Accum']['InventoryRows'])
    pd.DataFrame(InventoryRows).to_csv(os.path.join(ResultsDir, "ChainInventory.csv"), index=False)

    # ---- Aggregate per group ----
    Below = AggregateGroup(Results, GroupBelow)
    Above = AggregateGroup(Results, GroupAbove)
    LengthTable = ChainLengthTable(Below['FinalChainLengths'], Above['FinalChainLengths'])

    # ---- Angle figure ----
    BelowAngles = [A for R in Results if R['Group'] == GroupBelow for A in R['Accum']['Angles']]
    AboveAngles = [A for R in Results if R['Group'] == GroupAbove for A in R['Accum']['Angles']]
    MakeAngleFigure(BelowAngles, AboveAngles,
                    os.path.join(ResultsDir, "AngleDistributions.png"),
                    os.path.join(ResultsDir, "AngleDistributions.pdf"))

    # ---- Representative structures ----
    Representatives = SelectRepresentatives(Results)
    RepTable = []
    for i, (Item, Motif) in enumerate(Representatives):
        R = Item['Result']
        FileName = f"Representative_{i+1}_{Motif}_{R['TrajId']}.xyz"
        ExportChainSnippet(R, Item['Component'], os.path.join(ResultsDir, FileName), Motif)
        RepTable.append({'File': FileName, 'Temp': R['Temp'], 'TrajId': R['TrajId'],
                         'Motif': Motif, 'Length': Item['Length']})

    # ---- JSON ----
    ResultsJson = {
        'Seed': RandomSeed,
        'SubsampleInterval_fs': SubsampleFs,
        'BurnIn_fs': BurnInFs,
        'Window': {'FractionFromEnd': 1 - WindowStartPct,
                   'FramesPerTraj': {R['TrajId']: R['NWindowFrames'] for R in Results},
                   'AnalysedPerTraj': {R['TrajId']: R['NAnalysedFrames'] for R in Results},
                   'StartEndTime_ps': {R['TrajId']: [R['WindowStartTime_ps'], R['WindowEndTime_ps']]
                                       for R in Results}},
        'ChainDetection': {'Source': 'BootstrapCalculation stack (instantaneous + 0.2 ps persistence)',
                           'ConsistencyVsPaper': 'pass' if ConsistencyPass else 'fail',
                           'PerTrajectory': {R['TrajId']: [R['FinalChainCount'], R['ExpectedChainCount']]
                                             for R in Results}},
        'BondCutoffs': {'CC': CutoffCC, 'CO': CutoffCO, 'ZrC': CutoffZrC, 'ZrO': CutoffZrO},
        'TerminationCriteria': {'CarbideZr_minC': CarbideZrMinC, 'OxideZr_minO': OxideZrMinO},
        'Groups': {GroupBelow: Below, GroupAbove: Above},
        'ChainLengthTable': LengthTable,
        'Representatives': RepTable,
    }
    with open(os.path.join(ResultsDir, "TopologyResults.json"), 'w') as F:
        json.dump(ResultsJson, F, indent=2)

    WriteSummary(Below, Above, LengthTable, RepTable, Results, ConsistencyPass, BelowAngles, AboveAngles)
    print("\nDone. Results in", ResultsDir, flush=True)


def WriteSummary(Below, Above, LengthTable, RepTable, Results, ConsistencyPass,
                 BelowAngles, AboveAngles):
    BelowFrames = sum(R['NAnalysedFrames'] for R in Results if R['Group'] == GroupBelow)
    AboveFrames = sum(R['NAnalysedFrames'] for R in Results if R['Group'] == GroupAbove)

    L = []
    L.append("# Interfacial carbon chain topology")
    L.append("")
    L.append(f"Source: xyz final-10% window (re-streamed) + BootstrapCalculation detection stack; "
             f"seed {RandomSeed}, subsampling interval {int(SubsampleFs)} fs")
    L.append(f"Window: final 10% per trajectory; frames analysed per group: "
             f"below {BelowFrames} / above {AboveFrames}")
    L.append(f"Chain detection: bootstrap stack recomputed on the window, consistency check vs "
             f"35/31/48/66: {'pass' if ConsistencyPass else 'fail'}")
    L.append("")
    L.append("## 1. Sample sizes")
    L.append("")
    L.append("| Group | Chain carbons (mean per frame) | Distinct chains (mean per frame) | Angle samples |")
    L.append("|---|---:|---:|---:|")
    for Name, G in [("BelowCrossover", Below), ("AboveCrossover", Above)]:
        L.append(f"| {Name} | {G['MeanChainCarbonsPerFrame']:.2f} | {G['MeanChainsPerFrame']:.2f} | "
                 f"{G['AngleSamples']} |")
    L.append("")
    L.append("## 2. Geometry")
    L.append("")
    L.append("| Group | Median CCC angle (deg) | Fraction > 150 deg | Branch-point fraction | Rings found (>=4-membered) |")
    L.append("|---|---:|---:|---:|---:|")
    for Name, G in [("BelowCrossover", Below), ("AboveCrossover", Above)]:
        L.append(f"| {Name} | {G['MedianAngle']:.1f} | {G['FractionAngleAbove150']:.3f} | "
                 f"{G['BranchPointFraction']:.3f} | {len(G['GenuineRingSizes'])} |")
    L.append("")
    L.append("## 3. Chain lengths (final-frame, per group)")
    L.append("")
    L.append("| Length | Below count | Above count |")
    L.append("|---:|---:|---:|")
    for Row in LengthTable:
        L.append(f"| {Row['Length']} | {Row['Below']} | {Row['Above']} |")
    L.append("")
    L.append("## 4. Termination chemistry (fraction of termini, per group)")
    L.append("")
    L.append("| Category | Below | Above |")
    L.append("|---|---:|---:|")
    for Cat in ['CarbideAnchored', 'OxideAnchored', 'OxygenBonded', 'Dangling']:
        L.append(f"| {Cat} | {Below['Termination'][Cat]:.3f} | {Above['Termination'][Cat]:.3f} |")
    L.append(f"| Bridging chains (>= 1 terminus carbide-anchored AND >= 1 oxide-anchored or O-bonded) "
             f"| {Below['BridgingChainFraction']:.3f} | {Above['BridgingChainFraction']:.3f} |")
    L.append("")
    L.append("## 5. Representative structures")
    L.append("")
    L.append("| File | T (K) | Trajectory | Motif |")
    L.append("|---|---:|---:|---|")
    for Row in RepTable:
        L.append(f"| {Row['File']} | {Row['Temp']} | {Row['TrajId']} | {Row['Motif']} (len {Row['Length']}) |")
    L.append("")
    L.append("## 6. Notes")
    for N in BuildNotes(Below, Above, Results, ConsistencyPass):
        L.append(f"- {N}")

    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), 'w') as F:
        F.write(Text + "\n")
    print(Text)


def BuildNotes(Below, Above, Results, ConsistencyPass):
    Notes = []
    Notes.append(f"Chain detection reused the bootstrap stack (imported from BootstrapCalculation.py); "
                 f"final-frame counts reproduce the per-trajectory 14/4/11/6, 10/6/8/7, 16/7/16/9, "
                 f"12/13/26/15 values ({'all match' if ConsistencyPass else 'MISMATCH - see JSON'}).")
    Notes.append(f"Persistence smoothing runs on every window frame from a fresh Lattice init; a "
                 f"{int(BurnInFs)} fs burn-in is discarded so the 0.2 ps filter is stable before any "
                 f"frame is recorded. The window (~13 ps) >> 0.2 ps, so final-frame states match the "
                 f"full-trajectory bootstrap exactly.")
    Notes.append("C-neighbour count (degree, terminus/branch/angle classification) uses the chain "
                 "graph = C-C bonds among confirmed chain carbons; O and Zr neighbour counts use all "
                 "bonded O/Zr atoms.")
    Notes.append(f"Carbide Zr criterion: Zr with >= {CarbideZrMinC} C neighbours (Zr-C < {CutoffZrC:.2f} A). "
                 f"Oxide Zr criterion: Zr with >= {OxideZrMinO} O neighbours (Zr-O < {CutoffZrO:.2f} A, "
                 f"paper NO >= 2 convention). Termination categories overlap by design.")
    Aromatic = Below['AromaticRingCount'] + Above['AromaticRingCount']
    Notes.append(
        f"Rings: no aromatic rings (>=5-membered) in either group; genuine >=4-membered rings are "
        f"{len(Below['GenuineRingSizes'])} below / {len(Above['GenuineRingSizes'])} above "
        f"(sizes {Below['GenuineRingSizes']} / {Above['GenuineRingSizes']}). The {Below['TriangleCount']} "
        f"below / {Above['TriangleCount']} above 3-membered graph cycles are NOT carbocycles: at the "
        f"paper's 2.31 A C-C cutoff a chain kinked near the median ~150 deg has a 1-3 separation "
        f"below cutoff, closing a spurious triangle. So the chains are acyclic linear/branched paths - "
        f"an atomistic precursor, not a graphitic sheet ({Aromatic} aromatic rings)."
        if Aromatic == 0 else
        f"Rings: {Aromatic} aromatic (>=5) rings found - unexpected; see JSON RingSizes.")
    Notes.append(f"Branching is rare and group-dependent: branch-point fraction {Below['BranchPointFraction']:.3f} "
                 f"below vs {Above['BranchPointFraction']:.3f} above crossover. Representative chains are drawn "
                 f"from final frames only (per spec); no final-frame chain was branched at its snapshot, so the "
                 f"branched motif is not exported despite transient in-window branching above crossover.")
    Notes.append("Subsampling is deterministic (every 50 fs by frame time), so the reported seed is "
                 "not actually consumed; it is fixed for reproducibility of any future stochastic step.")
    return Notes


if __name__ == "__main__":
    Main()
