"""
BoundaryConventions.py
=======================================================================
Settles the analysis-boundary conventions and finishes the gas-audit
addendum before the Refresh140 full refresh. Verification and one small
recomputation only - no new physics, no figure regeneration.

Steps
-----
1. Per-trajectory endpoints: total time, last frame with t <= 140.0 ps,
   gap to 140.0 ps, nearest frame if different. Confirms the nominal
   4 x 140.0 ps per temperature to within one MD frame.
2. Boundary event census from the RateAnalysis CSVs: gas events in
   (139.5, 140.0] ps, events > 140.0 ps (excluded by the closed-interval
   rule), events within 2 fs of the horizon.
3. Final-state frame rule: confirmed chain-carbon count (established
   detection stack: BootstrapCalculation state machine, 0.2 ps
   time-based persistence filter run on every raw frame) at the
   last-frame-<=-140.0 rule vs the nearest-frame rule.
4. Audit addendum: event-based sticking = GasOriginUptakes /
   (GasOriginUptakes + Bounces) with Wilson 95% CI, from
   GasSupplyAudit.json counts only; appended to GasSupplyAudit/SUMMARY.md
   under an ADDENDUM heading (the one permitted write outside this
   folder; idempotent - re-runs replace the addendum).

Implementation notes
--------------------
* The data folder is read-only; the xyz pass is a light seek-and-stream:
  a byte-offset estimate near 136.5 ps, resynced to a frame boundary,
  then ~3.5 ps of raw frames to just past 140.0 ps (>= 2.5 ps burn-in
  for the persistence filter, >> the 0.2 ps threshold, same convention
  the sibling window scripts rely on).
* Chain-state snapshots take confirmed states only; pending transitions
  are NOT flushed at the horizon (the end-of-trajectory flush in the
  bootstrap applies to trajectory end, not to a mid-trajectory horizon).
* Nothing stochastic; seed 20260715 quoted for consistency.

Conventions.json is the machine-readable record for the Refresh140
orchestrator; its keys are documented inline where they are written.
"""

import os
import re
import sys
import ast
import json
import argparse

import numpy as np

ScriptDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core")  # results and caches are kept beside the scripts in core/
sys.path.insert(0, ScriptDir)

from ChainTopology import ReadFramesFrom, XyzPath, Temperatures, SimIndices, DataDir
from BootstrapCalculation import (
    CartesianToFractional, MapAtomIdentities, DetermineInstantaneousStates,
    TotalStartC, StatePersistenceTimePs,
)

ResultsDir = os.path.join(ScriptDir, "BoundaryConventions")
AuditJsonPath = os.path.join(ScriptDir, "GasSupplyAudit", "GasSupplyAudit.json")
AuditSummaryPath = os.path.join(ScriptDir, "GasSupplyAudit", "SUMMARY.md")

RandomSeed = 20260715
HorizonFs = 140000.0
StreamStartTargetFs = 136500.0     # >= 2.5 ps burn-in before the horizon
SeekMarginBytes = 25 * 1024 * 1024
BoundaryWindowPs = (139.5, 140.0)
HorizonToleranceFs = 2.0
AddendumMarker = "## ADDENDUM (BoundaryConventions.py): event-based sticking"

TimeRegex = re.compile(r"Time\s*=\s*([\d.]+)")


# =============================================================================
#                      SEEK-AND-STREAM (light xyz pass)
# =============================================================================
def ResyncToFrame(Fh):
    """From the current (arbitrary) file position: discard the partial line,
    then scan to the next frame header (a bare-integer atom-count line whose
    following line carries Lattice=). Returns (ByteOffset, TimeFs) of that
    header, or None at EOF."""
    Fh.readline()
    while True:
        Pos = Fh.tell()
        Line = Fh.readline()
        if not Line:
            return None
        Stripped = Line.strip()
        if Stripped.isdigit():
            Comment = Fh.readline()
            if "Lattice=" in Comment:
                Match = TimeRegex.search(Comment)
                return (Pos, float(Match.group(1))) if Match else None


def FindOffsetAtTime(Path, TargetFs, TotalFs):
    """Frame-start byte offset whose time is <= TargetFs, close below it.
    Byte-fraction estimate minus a safety margin, stepping back if needed
    (later frames are larger, so the estimate can only land late)."""
    Size = os.path.getsize(Path)
    Offset = max(0, int(Size * (TargetFs / TotalFs)) - SeekMarginBytes)
    with open(Path, "r") as Fh:
        while True:
            if Offset == 0:
                return 0
            Fh.seek(Offset)
            Sync = ResyncToFrame(Fh)
            if Sync is not None and Sync[1] <= TargetFs:
                return Sync[0]
            Offset = max(0, Offset - SeekMarginBytes)


def ProcessTrajectoryBoundary(TaskArgs):
    """Steps 1 + 3 for one trajectory: endpoint frames around the horizon and
    confirmed chain-carbon counts under the two frame rules."""
    Temp, Sim, TotalPs = TaskArgs
    Path = XyzPath(Temp, Sim)
    StartOffset = FindOffsetAtTime(Path, StreamStartTargetFs, TotalPs * 1000.0)

    TrackedIDs = np.arange(TotalStartC)
    ConfirmedStates = {ID: "Lattice" for ID in range(TotalStartC)}
    PendingStates = {}
    PrevFracC = None
    FirstStreamedFs = None

    LastLeq = None            # (TimeFs, ChainCount) at last frame with t <= horizon
    FirstAfter = None         # (TimeFs, ChainCount) at first frame with t > horizon
    NFrames = 0

    for TimeFs, Atoms, Coords, Lattice in ReadFramesFrom(Path, StartOffset):
        if FirstStreamedFs is None:
            FirstStreamedFs = TimeFs
        FracCoords = CartesianToFractional(Coords, Lattice)
        CurrFracC = FracCoords[Atoms == "C"]
        if PrevFracC is not None:
            TrackedIDs = MapAtomIdentities(PrevFracC, CurrFracC, TrackedIDs, Lattice)
        else:
            if len(CurrFracC) < len(TrackedIDs):
                TrackedIDs = TrackedIDs[:len(CurrFracC)]
        PrevFracC = CurrFracC

        RawStates, _ = DetermineInstantaneousStates(FracCoords, Atoms, Lattice)
        CurrentFrameStates = {int(TrackedIDs[i]): RawStates[i]
                              for i in range(len(RawStates))}
        Tp = TimeFs / 1000.0
        for AtomID in range(TotalStartC):
            RawState = CurrentFrameStates.get(AtomID, "Gas")
            if RawState == ConfirmedStates[AtomID]:
                PendingStates.pop(AtomID, None)
                continue
            if AtomID in PendingStates and PendingStates[AtomID]["State"] == RawState:
                if Tp - PendingStates[AtomID]["StartTime"] >= StatePersistenceTimePs:
                    ConfirmedStates[AtomID] = RawState
                    del PendingStates[AtomID]
            else:
                PendingStates[AtomID] = {"State": RawState, "StartTime": Tp}

        NFrames += 1
        ChainCount = sum(1 for S in ConfirmedStates.values() if S == "Chain")
        if TimeFs <= HorizonFs:
            LastLeq = (TimeFs, ChainCount)
        else:
            FirstAfter = (TimeFs, ChainCount)
            break

    if LastLeq is None:
        raise RuntimeError(f"{Temp}_{Sim}: no frame at or below the horizon "
                           f"was streamed (seek landed past it?)")

    # Nearest-frame rule: the closer of last-<= and first-after.
    if FirstAfter is not None and (abs(FirstAfter[0] - HorizonFs)
                                   < abs(LastLeq[0] - HorizonFs)):
        Nearest = FirstAfter
    else:
        Nearest = LastLeq

    BurnInPs = (LastLeq[0] - FirstStreamedFs) / 1000.0
    return {
        "Temp": Temp, "Sim": Sim, "TotalPs": TotalPs,
        "LastLeqFs": LastLeq[0], "GapFs": HorizonFs - LastLeq[0],
        "FirstAfterFs": FirstAfter[0] if FirstAfter else None,
        "NearestFs": Nearest[0],
        "ChainCountLastLeq": LastLeq[1],
        "ChainCountNearest": Nearest[1],
        "FrameRuleAgree": bool(LastLeq[1] == Nearest[1]),
        "BurnInPs": BurnInPs, "NStreamedFrames": NFrames,
    }


# =============================================================================
#                     STEP 2 - BOUNDARY EVENT CENSUS
# =============================================================================
def RateAnalysisPath(Temp, Sim):
    return os.path.join(DataDir, f"RateAnalysis_{Temp}_{Sim}.csv")


def CountBoundaryEvents(Temp, Sim):
    """Gas events (removed molecules) from the RateAnalysis CSV, classified
    against the horizon. Each entry of a row's 'Gas Removed' list is one
    event, timestamped at the row's Time (fs)."""
    import pandas as pd
    Df = pd.read_csv(RateAnalysisPath(Temp, Sim))
    InWindow, AfterHorizon, NearHorizon = 0, 0, 0
    for _, Row in Df.iterrows():
        Removed = ast.literal_eval(str(Row["Gas Removed"]))
        if not Removed:
            continue
        TimeFs = float(Row["Time (fs)"])
        TimePs = TimeFs / 1000.0
        Count = len(Removed)
        if BoundaryWindowPs[0] < TimePs <= BoundaryWindowPs[1]:
            InWindow += Count
        if TimePs > BoundaryWindowPs[1]:
            AfterHorizon += Count
        if abs(TimeFs - HorizonFs) <= HorizonToleranceFs:
            NearHorizon += Count
    return {"InWindow": InWindow, "AfterHorizon": AfterHorizon,
            "NearHorizon": NearHorizon}


# =============================================================================
#                STEP 4 - EVENT-BASED STICKING (audit addendum)
# =============================================================================
def WilsonInterval(K, N, Z=1.959964):
    """Wilson score 95% CI for a binomial proportion."""
    if N == 0:
        return float("nan"), float("nan")
    P = K / N
    Denom = 1.0 + Z * Z / N
    Centre = (P + Z * Z / (2 * N)) / Denom
    Half = (Z / Denom) * np.sqrt(P * (1 - P) / N + Z * Z / (4 * N * N))
    return Centre - Half, Centre + Half


def BuildStickingAddendum(Audit):
    Rows = {}
    for Temp in Temperatures:
        Entry = Audit["PerTemperature"][str(Temp)]
        Rows[Temp] = {}
        for Window in ("0-40", "40-140"):
            Census = Entry["BounceCensus"][Window]
            Uptakes = Census["UptakesGasOrigin"]
            Bounces = Census["Bounces"]
            N = Uptakes + Bounces
            P = Uptakes / N if N else float("nan")
            Lo, Hi = WilsonInterval(Uptakes, N)
            Flux = Entry["Sticking"][Window]
            Rows[Temp][Window] = {
                "Uptakes": Uptakes, "Bounces": Bounces, "N": N,
                "Probability": P, "CI95": [Lo, Hi],
                "FluxBased": Flux["Probability"],
                "FluxBasedSigma": Flux["PoissonSigma"],
                "Discrepancy": P - Flux["Probability"] if N else float("nan"),
            }
    return Rows


def AddendumText(Sticking):
    L = []
    L.append(AddendumMarker)
    L.append("")
    L.append("Event-based effective sticking probability = GasOriginUptakes / "
             "(GasOriginUptakes + Bounces), Wilson 95% CI; computed by "
             "BoundaryConventions.py from the counts above (no re-streaming). "
             "Flux-based values from Section 4 repeated for comparison; "
             "discrepancy = event-based minus flux-based.")
    L.append("")
    L.append("| T (K) | Window | Event-based [95% CI] | Flux-based | Discrepancy |")
    L.append("|---:|---|---|---|---:|")
    for Temp in Temperatures:
        for Window in ("0-40", "40-140"):
            E = Sticking[Temp][Window]
            L.append(
                f"| {Temp} | {Window} | {E['Probability']:.3f} "
                f"[{E['CI95'][0]:.3f}, {E['CI95'][1]:.3f}] "
                f"(n={E['N']}) | {E['FluxBased']:.3f} +/- "
                f"{E['FluxBasedSigma']:.3f} | {E['Discrepancy']:+.3f} |")
    L.append("")
    L.append("The two estimators share the numerator concept but different "
             "denominators (arrival events counted directly vs Hertz-Knudsen "
             "impingements from the windowed mean J_free); their disagreement "
             "sizes the denominator-model uncertainty, not a physics change.")
    return "\n".join(L)


def ApplyAddendum(Sticking):
    """Append (or replace) the addendum section in GasSupplyAudit/SUMMARY.md."""
    if not os.path.exists(AuditSummaryPath):
        return False
    with open(AuditSummaryPath, "r") as F:
        Text = F.read()
    if AddendumMarker in Text:
        Text = Text[:Text.index(AddendumMarker)].rstrip() + "\n"
    Text = Text.rstrip() + "\n\n" + AddendumText(Sticking) + "\n"
    with open(AuditSummaryPath, "w") as F:
        F.write(Text)
    return True


# =============================================================================
#                                REPORTING
# =============================================================================
def WriteOutputs(Boundary, Census, Sticking, AddendumApplied, GatePass, Audit):
    PerTraj = {f"{B['Temp']}_{B['Sim']}": B for B in Boundary}
    MaxSpacing = {K: V["MaxSpacing_fs"] for K, V in
                  Audit["Meta"]["PotimAudit"]["PerTrajectory"].items()}
    WithinOneFrame = {K: bool(PerTraj[K]["GapFs"] <= MaxSpacing[K])
                      for K in PerTraj}
    AllWithinOneFrame = all(WithinOneFrame.values())
    FrameRuleAllAgree = all(B["FrameRuleAgree"] for B in Boundary)

    Conventions = {
        "Seed": RandomSeed,
        "Horizon_ps": HorizonFs / 1000.0,
        "GatePass_16of16_at_140ps": GatePass,
        "IntervalRule": ("closed at the horizon: events with t <= 140.0 ps "
                        "are included; t > 140.0 ps excluded"),
        "FrameRule": {
            "Declared": "last frame with t <= 140.0 ps",
            "NearestFrameEquivalent": FrameRuleAllAgree,
            "Note": ("chain-carbon counts verified identical under both rules "
                     "per Step 3" if FrameRuleAllAgree else
                     "RULES DISAGREE - see Disagreements; decide before refresh"),
        },
        "PersistenceFilter": {
            "Threshold_ps": StatePersistenceTimePs,
            "TimeBased": True,
            "EvaluatedOn": ("every raw MD frame (variable ~0.7-1.8 fs spacing), "
                            "NOT the 50 fs audit grid"),
            "HorizonSnapshotRule": ("confirmed states only; pending transitions "
                                    "are not flushed at the horizon"),
            "Source": "BootstrapCalculation.ExtractChainEvents state machine",
        },
        "OneFrameCriterion": ("gap from last frame <= 140.0 ps to the horizon "
                              "must not exceed that trajectory's max raw frame "
                              "spacing (PotimAudit.MaxSpacing_fs)"),
        "NominalTimeAccurateWithinOneFrame": AllWithinOneFrame,
        "Endpoints": {
            K: {"Total_ps": PerTraj[K]["TotalPs"],
                "LastLeqHorizon_fs": PerTraj[K]["LastLeqFs"],
                "GapToHorizon_fs": PerTraj[K]["GapFs"],
                "NearestFrame_fs": PerTraj[K]["NearestFs"],
                "MaxRawSpacing_fs": MaxSpacing[K],
                "WithinOneFrame": WithinOneFrame[K],
                "ChainCountLastLeq": PerTraj[K]["ChainCountLastLeq"],
                "ChainCountNearest": PerTraj[K]["ChainCountNearest"],
                "BurnIn_ps": PerTraj[K]["BurnInPs"]}
            for K in sorted(PerTraj)
        },
        "BoundaryEventCensus": {
            f"{T}_{S}": Census[(T, S)] for T in Temperatures for S in SimIndices
        },
        "EventBasedSticking": {
            str(T): {W: Sticking[T][W] for W in ("0-40", "40-140")}
            for T in Temperatures
        },
        "AddendumAppliedToAuditSummary": AddendumApplied,
        "AuditJsonSource": os.path.relpath(AuditJsonPath, ScriptDir),
    }
    with open(os.path.join(ResultsDir, "Conventions.json"), "w") as F:
        json.dump(Conventions, F, indent=2)
    WriteSummary(Boundary, Census, Sticking, AddendumApplied, GatePass,
                 WithinOneFrame, AllWithinOneFrame, FrameRuleAllAgree)


def WriteSummary(Boundary, Census, Sticking, AddendumApplied, GatePass,
                 WithinOneFrame, AllWithinOneFrame, FrameRuleAllAgree):
    L = []
    L.append("# Boundary conventions and audit addendum")
    L.append("")
    L.append(f"Gate: 16/16 >= 140.0 ps: {'yes' if GatePass else 'no'}")
    L.append("## 1. Endpoints")
    L.append("| Trajectory | Total (ps) | Last frame <= 140.0 (ps) | Gap to 140.0 (fs) |")
    L.append("|---|---:|---:|---:|")
    for B in sorted(Boundary, key=lambda X: (X["Temp"], X["Sim"])):
        L.append(f"| {B['Temp']}_{B['Sim']} | {B['TotalPs']:.3f} | "
                 f"{B['LastLeqFs'] / 1000.0:.5f} | {B['GapFs']:.2f} |")
    L.append(f"Nominal 560.0 ps per temperature accurate to within one frame: "
             f"{'yes' if AllWithinOneFrame else 'no'}")
    L.append("")
    L.append("## 2. Boundary events")
    L.append("| T (K) | Events in (139.5, 140.0] | Events > 140.0 (excluded) | "
             "Within 2 fs of horizon |")
    L.append("|---:|---:|---:|---:|")
    for Temp in Temperatures:
        InW = sum(Census[(Temp, S)]["InWindow"] for S in SimIndices)
        After = sum(Census[(Temp, S)]["AfterHorizon"] for S in SimIndices)
        Near = sum(Census[(Temp, S)]["NearHorizon"] for S in SimIndices)
        L.append(f"| {Temp} | {InW} | {After} | {Near} |")
    L.append("")
    L.append("## 3. Final-state frame rule")
    L.append(f"Chain counts identical under last-frame vs nearest-frame rule: "
             f"{'yes' if FrameRuleAllAgree else 'no'}")
    if not FrameRuleAllAgree:
        L.append("")
        L.append("| Trajectory | Chain count (last <= 140.0) | Chain count (nearest) |")
        L.append("|---|---:|---:|")
        for B in sorted(Boundary, key=lambda X: (X["Temp"], X["Sim"])):
            if not B["FrameRuleAgree"]:
                L.append(f"| {B['Temp']}_{B['Sim']} | {B['ChainCountLastLeq']} | "
                         f"{B['ChainCountNearest']} |")
    L.append("")
    L.append(f"## 4. Event-based sticking (addendum applied to audit summary: "
             f"{'yes' if AddendumApplied else 'no'})")
    L.append("| T (K) | 0-40 ps (event-based) | 40-140 ps (event-based) | "
             "40-140 ps (flux-based, for comparison) |")
    L.append("|---:|---:|---:|---:|")
    for Temp in Temperatures:
        E0 = Sticking[Temp]["0-40"]
        E1 = Sticking[Temp]["40-140"]
        L.append(f"| {Temp} | {E0['Probability']:.3f} "
                 f"[{E0['CI95'][0]:.3f}, {E0['CI95'][1]:.3f}] | "
                 f"{E1['Probability']:.3f} "
                 f"[{E1['CI95'][0]:.3f}, {E1['CI95'][1]:.3f}] | "
                 f"{E1['FluxBased']:.3f} +/- {E1['FluxBasedSigma']:.3f} |")
    L.append("")
    L.append("## 5. Notes")
    L.append("- Persistence filter confirmed TIME-BASED (0.2 ps threshold "
             "compared against physical frame times), evaluated on every raw "
             "MD frame - NOT on the 50 fs audit grid; the grid is only the "
             "gas-audit sampling. Answer to 'time-based on grid': time-based "
             "yes, on-grid no.")
    L.append("- Horizon snapshots use confirmed states only; pending "
             "transitions are not flushed at 140.0 ps (the bootstrap's flush "
             "applies to trajectory end). This is the convention the refresh "
             "must reuse for any mid-trajectory state count.")
    L.append("- 'Within one MD frame' evaluated per trajectory against its own "
             "max raw frame spacing (PotimAudit.MaxSpacing_fs, 0.7-1.8 fs "
             "band); the frame-rule check streams from ~136.5 ps with >= 2.5 "
             "ps of persistence burn-in (>> 0.2 ps), the same window "
             "convention the sibling scripts validated against the full "
             "bootstrap.")
    L.append("- Boundary gas events use RateAnalysis row timestamps (cycle "
             "boundaries, ~64-80 fs apart); one event per removed molecule in "
             "the row's 'Gas Removed' list.")
    L.append("- Event-based sticking uses gas-origin uptakes and bounces only "
             "(surface-born transient O-O pairs excluded on both sides), "
             "Wilson 95% CI; the flux-based comparison inherits the audit's "
             "Hertz-Knudsen denominator. Their difference is "
             "denominator-model uncertainty, not new physics.")
    L.append("- Nothing here blocks the refresh unless Section 3 reports a "
             "frame-rule disagreement or Section 1 reports a gap exceeding "
             "one frame; both conditions are recorded in Conventions.json "
             "(FrameRule.NearestFrameEquivalent, "
             "NominalTimeAccurateWithinOneFrame).")
    L.append(f"- Seed {RandomSeed} quoted for consistency; nothing stochastic. "
             "Data folder untouched; the only write outside "
             "BoundaryConventions/ is the audit-summary addendum.")
    Text = "\n".join(L)
    with open(os.path.join(ResultsDir, "SUMMARY.md"), "w") as F:
        F.write(Text + "\n")
    print(Text)


# =============================================================================
#                                  DRIVER
# =============================================================================
def Main():
    Parser = argparse.ArgumentParser(description="Boundary conventions + audit addendum.")
    Parser.add_argument("--jobs", type=int,
                        default=max(1, min(6, (os.cpu_count() or 2) - 2)))
    Args = Parser.parse_args()

    os.makedirs(ResultsDir, exist_ok=True)
    with open(AuditJsonPath) as F:
        Audit = json.load(F)

    EndTimes = Audit["Meta"]["TrajectoryEndTimes_ps"]
    GatePass = all(V >= HorizonFs / 1000.0 for V in EndTimes.values())
    if not GatePass:
        Short = {K: V for K, V in EndTimes.items() if V < HorizonFs / 1000.0}
        print(f"STOP: gate failed; short trajectories: {Short}")
        return 1

    Tasks = [(T, S, EndTimes[f"{T}_{S}"]) for T in Temperatures for S in SimIndices]
    print(f"Steps 1+3: seek-and-stream around the horizon (jobs={Args.jobs}) ...",
          flush=True)
    Boundary = []
    if Args.jobs > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=Args.jobs) as Pool:
            Futures = {Pool.submit(ProcessTrajectoryBoundary, T): T for T in Tasks}
            for Fut in as_completed(Futures):
                B = Fut.result()
                Boundary.append(B)
                print(f"  {B['Temp']}_{B['Sim']}: gap {B['GapFs']:.2f} fs, "
                      f"chain {B['ChainCountLastLeq']} vs {B['ChainCountNearest']}, "
                      f"burn-in {B['BurnInPs']:.1f} ps", flush=True)
    else:
        for T in Tasks:
            B = ProcessTrajectoryBoundary(T)
            Boundary.append(B)
            print(f"  {B['Temp']}_{B['Sim']}: gap {B['GapFs']:.2f} fs, "
                  f"chain {B['ChainCountLastLeq']} vs {B['ChainCountNearest']}, "
                  f"burn-in {B['BurnInPs']:.1f} ps", flush=True)

    print("Step 2: RateAnalysis boundary census ...", flush=True)
    Census = {(T, S): CountBoundaryEvents(T, S)
              for T in Temperatures for S in SimIndices}

    print("Step 4: event-based sticking addendum ...", flush=True)
    Sticking = BuildStickingAddendum(Audit)
    AddendumApplied = ApplyAddendum(Sticking)

    WriteOutputs(Boundary, Census, Sticking, AddendumApplied, GatePass, Audit)
    print("\nDone. Results in", ResultsDir, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(Main())
