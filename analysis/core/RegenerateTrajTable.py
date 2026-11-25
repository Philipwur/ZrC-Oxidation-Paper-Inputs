"""
RegenerateTrajTable.py
=======================================================================
Regenerates Supplementary Table S1 (per-trajectory raw kinetic outputs)
from the audited sources.

Columns and sources:
  CO Des / CO2 Des : EventTableV2.csv (audited, artifact-corrected,
                     truncated at the 140.0 ps horizon)
  O Des            : CO + 2*CO2       (derived)
  C Des            : CO + CO2         (derived)
  Scale O          : PhiTimeseries.csv NScaleO at the last grid point
                     <= 140.0 ps (persistence-filtered bookkeeping)
  C Chains         : EventTableV2.csv chain commitments (= confirmed
                     chain occupancy at the horizon, the main-text
                     Table 2 convention)
  Avg. PP / J-bar  : GasSupplyAudit.json WindowStats["0-140"]["free"]
                     (J_free convention, the audit's Table 1 candidate)

Sanity gates: per-temperature gas totals must reproduce 31/47/38/36
and chain totals 35/31/48/69; PhiTimeseries NChainC(140) is checked
against the event-table chain counts and any mismatch is printed.

Output: RegenerateTrajTableResults/ (Rows.tex = tabular body,
Values.json, console summary). All inputs read-only.
"""

import os
import json

import numpy as np
import pandas as pd

ScriptDir = os.path.dirname(os.path.abspath(__file__))
EventPath = os.path.join(ScriptDir, "Refresh140", "Events", "EventTableV2.csv")
PhiPath = os.path.join(ScriptDir, "phiTimeseries", "PhiTimeseries.csv")
AuditPath = os.path.join(ScriptDir, "GasSupplyAudit", "GasSupplyAudit.json")
OutDir = os.path.join(ScriptDir, "RegenerateTrajTableResults")

Temperatures = [873, 973, 1073, 1273]
Sims = [1, 2, 3, 4]
GasTotals = {873: 31, 973: 47, 1073: 38, 1273: 36}
ChainTotals = {873: 35, 973: 31, 1073: 48, 1273: 69}


def Main():
    os.makedirs(OutDir, exist_ok=True)
    Ev = pd.read_csv(EventPath)
    Phi = pd.read_csv(PhiPath)
    Audit = json.load(open(AuditPath))["PerTrajectory"]

    Rows = []
    Values = {}
    for T in Temperatures:
        GasSum, ChainSum = 0, 0
        for S in Sims:
            Traj = f"{T}_{S}"
            Sub = Ev[Ev.TrajectoryId == Traj]
            CO = int(((Sub.Outcome == "Gas") & (Sub.Species == "CO")).sum())
            CO2 = int(((Sub.Outcome == "Gas") & (Sub.Species == "CO2")).sum())
            Chains = int((Sub.Outcome == "Chain").sum())
            ODes, CDes = CO + 2 * CO2, CO + CO2
            GasSum += CDes
            ChainSum += Chains

            Tr = Phi[Phi.TrajectoryId == Traj]
            Tr = Tr[Tr.Time_ps <= 140.0 + 1e-9].sort_values("Time_ps")
            ScaleO = int(Tr.NScaleO.iloc[-1])
            NChainPhi = int(Tr.NChainC.iloc[-1])
            if NChainPhi != Chains:
                print(f"NOTE {Traj}: NChainC(140) = {NChainPhi} vs "
                      f"event-table chains {Chains} (table uses events)")

            W = Audit[Traj]["WindowStats"]["0-140"]["free"]
            PP = W["P_bar_Mean"]
            J = W["J_Mean"] / 1e27

            Rows.append(f"{T}K\\_{S} & {CO} & {CO2} & {ODes} & {CDes} & "
                        f"{ScaleO} & {Chains} & {PP:.1f} & {J:.1f} \\\\")
            Values[Traj] = dict(CO=CO, CO2=CO2, ODes=ODes, CDes=CDes,
                                ScaleO=ScaleO, Chains=Chains,
                                PP_bar=round(PP, 1), J_1e27=round(J, 1))
        assert GasSum == GasTotals[T], f"{T} K gas total {GasSum}"
        assert ChainSum == ChainTotals[T], f"{T} K chain total {ChainSum}"
        Rows.append("MIDRULE")

    Body = "\n".join(Rows[:-1]).replace("MIDRULE", "\\midrule")
    open(os.path.join(OutDir, "Rows.tex"), "w").write(Body + "\n")
    json.dump(Values, open(os.path.join(OutDir, "Values.json"), "w"),
              indent=1)
    print(Body)
    print("\nGates passed: gas totals", list(GasTotals.values()),
          "chain totals", list(ChainTotals.values()))
    print("Written:", OutDir)


if __name__ == "__main__":
    Main()
