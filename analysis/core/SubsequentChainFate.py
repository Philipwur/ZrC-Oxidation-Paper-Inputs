"""Count subsequent carbon fates without fitting a kinetic or lifetime model.

Uses the existing full-resolution, persistence-filtered chain histories and
audited molecular departures. Each carbon enters once, at its FIRST confirmed
chain entry, regardless of its final state. Reports endpoint counts and counts
after a common follow-up (default 20 ps), with late entries explicitly excluded
from BOTH numerator and denominator of the latter. Standard library only.

Run from any directory:
    python SubsequentChainFate.py
    python SubsequentChainFate.py --followup-ps 40 --output-dir SubsequentChainFate40ps

Relative output paths are resolved beside this script. Inputs are read-only;
only this analysis's output files are written. Methods.tex documents the count
definitions. No confidence intervals, significance tests, or lifetime fits.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TEMPERATURES = (873, 973, 1073, 1273)
HORIZON_PS = 140.0
PERSISTENCE_PS = 0.2
# Confirmed Gas starts 0.2 ps after disappearance, to within one raw MD step
# (at most 1.8 fs in the audited data). This is an ID-matching check, not a
# new chemical criterion. Ambiguous/missing matches stop the analysis.
RAW_STEP_TOLERANCE_PS = 0.002
EPS = 1e-8
FATES = ("Chain", "OtherPresent", "Gas", "ArtifactRemoval")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def match_departures(record, departures):
    """Attach the existing physical-departure audit to persistent carbon IDs."""
    matched = {}
    for departure in departures:
        time = float(departure["DeletionFs"]) / 1000.0
        check(math.isfinite(time), "Non-finite departure time")
        candidates = [atom for atom, confirmed in record["FirstGasPs"].items()
                      if PERSISTENCE_PS - EPS <= float(confirmed) - time
                      <= PERSISTENCE_PS + RAW_STEP_TOLERANCE_PS + EPS]
        check(len(candidates) == 1, f"Departure at {time} ps has {len(candidates)} ID matches")
        atom = candidates[0]
        check(atom not in matched, f"Repeated departure for atom {atom}")
        check(departure["CleanMolecule"] in ("True", "False"), "Invalid audit classification")
        matched[atom] = {
            "Time_ps": time,
            "Kind": "Gas" if departure["CleanMolecule"] == "True" else "ArtifactRemoval",
            "ConfirmationDelay_ps": float(record["FirstGasPs"][atom]) - time,
        }
    check(set(matched) == set(record["FirstGasPs"]), "Departure audit and Gas IDs differ")
    return matched


def chain_intervals(record):
    """Use exact cached confirmation times; do not reclassify bonds or frames."""
    intervals = {atom: [] for atom in record["FirstChainPs"]}
    end = float(record["HorizonFrameFs"]) / 1000.0
    for episode in record["ClosedEpisodes"]:
        atom = str(episode["AtomID"])
        start, stop = float(episode["Start_ps"]), float(episode["End_ps"])
        check(atom in intervals and 0 <= start < stop <= end + EPS, "Invalid closed chain episode")
        intervals[atom].append((start, stop))
    for atom, start in record["OpenEpisodes"].items():
        check(atom in intervals and 0 <= float(start) <= end, "Invalid open chain episode")
        intervals[atom].append((float(start), None))
    for atom, episodes in intervals.items():
        episodes.sort(key=lambda item: item[0])
        check(bool(episodes), f"No chain episode for atom {atom}")
        check(abs(episodes[0][0] - record["FirstChainPs"][atom]) < EPS,
              f"First chain entry disagrees with episodes for atom {atom}")
        for previous, current in zip(episodes, episodes[1:]):
            check(previous[1] is not None and previous[1] <= current[0],
                  f"Overlapping chain episodes for atom {atom}")
    return intervals


def fate_at(time, episodes, departure):
    # Actual removal takes precedence over the state filter's 0.2 ps delay.
    if departure is not None and departure["Time_ps"] <= time + EPS:
        return departure["Kind"]
    if any(start <= time + EPS and (stop is None or time < stop - EPS)
           for start, stop in episodes):
        return "Chain"
    return "OtherPresent"


def carbon_row(trajectory, atom, record, episodes, departure, followup):
    first = float(record["FirstChainPs"][atom])
    end = float(record["HorizonFrameFs"]) / 1000.0
    check(departure is None or first < departure["Time_ps"], "Chain entry after removal")
    target = first + followup
    eligible = target <= end + EPS
    status = fate_at(target, episodes, departure) if eligible else "InsufficientFollowup"
    if status == "ArtifactRemoval":
        eligible = False
    first_exit = episodes[0][1]
    if departure is not None:
        first_exit = min(first_exit if first_exit is not None else math.inf, departure["Time_ps"])
    return {
        "Temperature": int(trajectory.split("_")[0]), "Trajectory": trajectory,
        "AtomID": int(atom), "FirstChain_ps": first,
        "ObservationEnd_ps": end, "AvailableFollowup_ps": end - first,
        "FirstChainExit_ps": first_exit,
        "Departure_ps": None if departure is None else departure["Time_ps"],
        "DepartureKind": None if departure is None else departure["Kind"],
        "EndpointFate": fate_at(end, episodes, departure),
        "Followup_ps": followup, "FollowupEligible": eligible,
        "FollowupFate": status,
        "ContinuouslyChainedAtFollowup":
            (first_exit is None or target < first_exit - EPS) if eligible else None,
        "ChainEpisodes": len(episodes),
    }


def summarise(label, rows, all_gas, artifacts, trajectory_count):
    endpoint = Counter(row["EndpointFate"] for row in rows)
    fixed = Counter(row["FollowupFate"] for row in rows)
    eligible = sum(row["FollowupEligible"] for row in rows)
    known_endpoint = len(rows) - endpoint["ArtifactRemoval"]
    result = {
        "Group": label, "Trajectories": trajectory_count, "EverChained": len(rows),
        "EndpointKnown": known_endpoint,
        **{f"Endpoint{fate}": endpoint[fate] for fate in FATES},
        "Followup_ps": rows[0]["Followup_ps"] if rows else None,
        "FollowupEligible": eligible,
        "ExcludedLateEntry": fixed["InsufficientFollowup"],
        "ExcludedArtifactBeforeFollowup": fixed["ArtifactRemoval"],
        **{f"Followup{fate}": fixed[fate] for fate in FATES[:3]},
        "FollowupContinuousChain": sum(row["ContinuouslyChainedAtFollowup"] is True for row in rows),
        "AllMolecularGasCarbons": all_gas,
        "GasCarbonsWithPriorChain": endpoint["Gas"],
        "AllArtifactRemovals": artifacts,
    }
    for fate in FATES[:3]:
        result[f"Endpoint{fate}Fraction"] = endpoint[fate] / known_endpoint if known_endpoint else None
        result[f"Followup{fate}Fraction"] = fixed[fate] / eligible if eligible else None
    result["FractionGasWithPriorChain"] = endpoint["Gas"] / all_gas if all_gas else None
    check(sum(endpoint.values()) == len(rows), "Endpoint counts do not sum")
    check(sum(fixed[f] for f in FATES[:3]) == eligible, "Follow-up counts do not sum")
    check(eligible + fixed["InsufficientFollowup"] + fixed["ArtifactRemoval"] == len(rows),
          "Follow-up exclusions do not sum")
    return result


def write_csv(path, rows):
    check(bool(rows), f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def methods_text(followup):
    return rf"""Each carbon atom was followed from its first confirmed entry into the
chain state, using the existing atom identities and 0.2~ps persistence
criterion. At the simulation endpoint, it was classified as chain carbon,
carbon still present in another state, or gasified carbon identified by the
audited molecular-removal record. Known erroneous atom removals were excluded
from physical-fate fractions and reported separately. Because chain formation
occurred at different times, we additionally counted these states at a common
follow-up of {followup:g}~ps after first chain entry, including only atoms for
which that interval fell within the recorded trajectory and preceded any
erroneous removal. Late entries were excluded regardless of their observed
outcome. Continuous chain occupancy over this interval was distinguished from
chain occupancy at its endpoint, which can include re-entry. Counts and
fractions were reported for each trajectory and pooled over the four
trajectories at each temperature. Gasified atoms with any prior confirmed
chain membership were also counted. These are descriptive finite-observation
statistics; no lifetime distribution or kinetic model was fitted.
"""


def write_summary(path, summaries, per_trajectory, rows, followup):
    lines = ["# Subsequent fate after first confirmed chain entry", "",
             f"Common follow-up: {followup:g} ps. Nominal simulation endpoint: 140 ps.", "",
             "The follow-up interval is an analysis setting, not an estimated lifetime. "
             "Counts are descriptive; no significance or long-term stability is inferred.", "",
             "## Full available observation (unequal time since chain entry)", "",
             "| T (K) | Ever chained | Chain at end | Other present | Gas at end | Artifact excluded | Gas with prior chain / all gas |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        lines.append(f"| {s['Group']} | {s['EverChained']} | {s['EndpointChain']} | "
                     f"{s['EndpointOtherPresent']} | {s['EndpointGas']} | {s['EndpointArtifactRemoval']} | "
                     f"{s['GasCarbonsWithPriorChain']}/{s['AllMolecularGasCarbons']} |")
    lines += ["", f"## Common follow-up of {followup:g} ps", "",
              "| T (K) | Eligible | Late entries excluded | Artifact excluded | Chain | Other present | Gas | Continuously chained |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for s in summaries:
        lines.append(f"| {s['Group']} | {s['FollowupEligible']} | {s['ExcludedLateEntry']} | "
                     f"{s['ExcludedArtifactBeforeFollowup']} | {s['FollowupChain']} | "
                     f"{s['FollowupOtherPresent']} | {s['FollowupGas']} | {s['FollowupContinuousChain']} |")
    lines += ["", "## Individual trajectories", "",
              "| Trajectory | Ever chained | End: chain / other / gas | Eligible at common follow-up | Follow-up: chain / other / gas |",
              "| --- | ---: | --- | ---: | --- |"]
    for s in per_trajectory:
        lines.append(f"| {s['Group']} | {s['EverChained']} | "
                     f"{s['EndpointChain']} / {s['EndpointOtherPresent']} / {s['EndpointGas']} | "
                     f"{s['FollowupEligible']} | {s['FollowupChain']} / {s['FollowupOtherPresent']} / {s['FollowupGas']} |")
    followups = [r["AvailableFollowup_ps"] for r in rows]
    lines += ["", "## Interpretation and reproducibility", "",
              f"Available observation after chain entry ranges from {min(followups):.2f} to {max(followups):.2f} ps.", "",
              "- OtherPresent means present but not in the confirmed chain state; it does not establish a return to the carbide lattice.",
              "- Gas includes a later molecular departure even if the atom first left its chain for another present state. Last chain-exit type alone is not its final fate.",
              "- Endpoint fractions use known physical fates; fixed-follow-up fractions use the eligible cohort. Gas-with-prior-chain fractions use all audited molecular gas carbons. CSV files include these denominators.",
              "- The fixed-follow-up cohort contains sufficiently early chain entries. It does not estimate the fate of late-forming chains or correct differences in interface age between temperatures.",
              "- Chain occupancy at the follow-up time allows re-entry; continuous occupancy does not. Each atom is counted once, not once per episode.",
              "- Four trajectories per temperature are shown separately. Pooled atom fractions are not evidence of independent atom-level replication or temperature significance.",
              "- Existing classification, identity tracking, and gas-removal conventions are inherited. This analysis does not revalidate the raw AIMD trajectories.",
              "- Clean-molecule departure audit entries are matched uniquely to persistent IDs using the existing 0.2 ps Gas-confirmation delay (plus at most 0.002 ps raw-step tolerance). All 157 departures, including five artifacts, must match; the 152 molecular departures and retained chains must agree with EventTableV2.csv.",
              "- No source files are changed. RunInfo.json records input hashes. Methods.tex contains a short methods paragraph.", "",
              "To check the observation-window choice, rerun with another explicitly reported "
              "--followup-ps value and a separate --output-dir; do not choose a window because its result is preferred.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--followup-ps", type=float, default=20.0)
    parser.add_argument("--output-dir", type=Path, default=Path("SubsequentChainFateResults"))
    args = parser.parse_args()
    check(math.isfinite(args.followup_ps) and 0 < args.followup_ps <= HORIZON_PS,
          "Follow-up must be finite and in (0, 140] ps")
    output = args.output_dir if args.output_dir.is_absolute() else SCRIPT_DIR / args.output_dir
    inputs = [SCRIPT_DIR / "RateAnalysisAuditResults" / "AllDeparturesIntegrity.csv",
              SCRIPT_DIR / "Refresh140" / "Events" / "EventTableV2.csv"]
    departures, ledger = [read_csv(p) for p in inputs]
    atoms, audit_rows, per_trajectory = [], [], []
    for temperature in TEMPERATURES:
        for sim in range(1, 5):
            trajectory = f"{temperature}_{sim}"
            state_path = SCRIPT_DIR / "Refresh140" / "FrameCache" / f"{trajectory}_states_140ps.json"
            inputs.append(state_path)
            record = json.loads(state_path.read_text(encoding="utf-8"))
            check(record["Temp"] == temperature and record["Sim"] == sim, "State-file identity mismatch")
            end = float(record["HorizonFrameFs"]) / 1000.0
            check(HORIZON_PS - RAW_STEP_TOLERANCE_PS <= end <= HORIZON_PS, "Unexpected observation endpoint")
            check(set(record["HorizonStates"]) == {str(i) for i in range(48)}, "Expected 48 tracked carbon atoms")
            matched = match_departures(record, [d for d in departures if d["Trajectory"] == trajectory])
            intervals = chain_intervals(record)
            local = [carbon_row(trajectory, atom, record, intervals[atom], matched.get(atom), args.followup_ps)
                     for atom in sorted(intervals, key=int)]
            gas_ids = {atom for atom, state in record["HorizonStates"].items() if state == "Gas"}
            check(gas_ids == set(matched), f"Final Gas IDs inconsistent in {trajectory}")
            retained = {int(atom) for atom, state in record["HorizonStates"].items() if state == "Chain"}
            check(retained == set(record["RetainedIDs"]) == {int(a) for a in record["OpenEpisodes"]},
                  f"Retained-chain IDs inconsistent in {trajectory}")
            check(len(local) == record["CumulativeEverChain"], "Ever-chain count mismatch")
            n_gas = sum(d["Kind"] == "Gas" for d in matched.values())
            n_artifacts = len(matched) - n_gas
            events = [e for e in ledger if e["TrajectoryId"] == trajectory]
            check(n_gas == sum(e["Outcome"] == "Gas" for e in events), f"Gas ledger mismatch in {trajectory}")
            check(len(retained) == sum(e["Outcome"] == "Chain" for e in events), f"Chain ledger mismatch in {trajectory}")
            check(sum(r["EndpointFate"] == "Chain" for r in local) == len(retained), "Endpoint-chain mismatch")
            atoms.extend(local)
            per_trajectory.append(summarise(trajectory, local, n_gas, n_artifacts, 1))
            for atom, departure in matched.items():
                audit_rows.append({"Trajectory": trajectory, "AtomID": int(atom), **departure,
                                   "EverChained": atom in intervals})
    summaries = []
    for temperature in TEMPERATURES:
        selected = [r for r in atoms if r["Temperature"] == temperature]
        trajectories = [r for r in per_trajectory if r["Group"].startswith(f"{temperature}_")]
        summaries.append(summarise(temperature, selected,
                         sum(r["AllMolecularGasCarbons"] for r in trajectories),
                         sum(r["AllArtifactRemovals"] for r in trajectories), len(trajectories)))
    check(len(audit_rows) == len(departures) == 157, "Unexpected full departure-audit size")
    check(sum(r["AllArtifactRemovals"] for r in summaries) == 5, "Expected five disclosed artifacts")
    check(sum(r["AllMolecularGasCarbons"] for r in summaries) == 152, "Expected 152 audited gas carbons")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("Carbons.csv", atoms), ("PerTrajectory.csv", per_trajectory),
                       ("PerTemperature.csv", summaries), ("AuditMatches.csv", audit_rows)):
        write_csv(output / name, rows)
    write_summary(output / "SUMMARY.md", summaries, per_trajectory, atoms, args.followup_ps)
    (output / "Methods.tex").write_text(methods_text(args.followup_ps), encoding="utf-8")
    metadata = {"Followup_ps": args.followup_ps, "NominalHorizon_ps": HORIZON_PS,
                "IndependentTrajectoriesPerTemperature": 4,
                "Analysis": "Descriptive first-entry cohort counts; no fitted model or atom-level uncertainty",
                "InputSHA256": {str(p.relative_to(SCRIPT_DIR)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                "ScriptSHA256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "Validation": "Unique audit matches; endpoint and event-ledger counts; disjoint fate totals passed"}
    (output / "RunInfo.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    for s in summaries:
        print(f"{s['Group']} K: {s['EverChained']} ever chained; endpoint "
              f"{s['EndpointChain']} chain, {s['EndpointOtherPresent']} other, {s['EndpointGas']} gas; "
              f"{s['FollowupEligible']} eligible at {args.followup_ps:g} ps")
    print(f"Validated all 16 trajectories. Results: {output}")


if __name__ == "__main__":
    main()
