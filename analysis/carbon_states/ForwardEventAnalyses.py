"""Everything in the paper's commitment kinetics that depends on the chain
event, rerun with the forward chain event.

Submitted chain event   a carbon's LAST entry into the chain it still occupies
                        at 140 ps (defined by looking back from the end)
Forward chain event     a carbon's FIRST confirmed carbon bond, whatever
                        happens to it afterwards

Both come from the paper's own audited records (core/Refresh140 and
WeibullAudit.rebuild_subjects), on the paper's clocks: gas at the logged
desorption time, chain entries at confirmation. Nothing in core/ is
modified. Its functions are imported and pointed at a new event table.

Every analysis runs twice, on the submitted events and on the forward events.
The submitted run must reproduce the numbers printed in the submitted paper,
which is the check that this script repeats the paper's calculations.

 0  event table with the forward chain event, in the paper's format
 1  Weibull shapes over 40 to 140 ps (pooled, whole-cell resamples, single
    trajectories, other start times)
 2  bootstrap ordering fractions of the gas and chain rates
 3  paired bootstrap of the commitment and net-mass activation energies
 4  flux-normalised activation energy
 5  naive-rate comparison and conversion percentages
 6  sensitivity table (event definition, cutoffs, persistence, Zr rule)
 7  transitions between the five states per temperature
 8  liberation numbers for the text (first and lasting liberation, lifetime
    and fate of non-lattice carbon)
 9  sensitivity of commitment and liberation to the state definitions
    (Supplementary Table S3c): DefinitionSensitivity.csv and .md
10  liberations that are not reversed (Section 2.3 of the paper):
    LiberationNotReversed.md and .json

Run:  python ForwardEventAnalyses.py      (about three minutes)
Output: ForwardEventResults/  (SUMMARY.md, csv and json files)
"""

import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Paths as RP  # noqa: E402
import CommitmentCompetingRisks as production  # noqa: E402  (core)
import EaBootstrap as EB  # noqa: E402
import SquareModel as SQ  # noqa: E402
import WeibullAudit as WA  # noqa: E402
from SurvivalAnalysis import FitExponentialRate  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = RP.results_dir(__file__, "ForwardEventResults")
SUBMITTED_TABLE = RP.REWRITE_DIR / "Refresh140" / "Events" / "EventTableV2.csv"
TEMPS = list(RP.TEMPERATURES)
INV_T = 1.0 / np.array(TEMPS, float)
KB_EV = 8.617333e-5
STARTS = (30.0, 35.0, 40.0, 45.0, 50.0)
CHANNELS = ("total", "gas", "chain")


def arrhenius(rates):
    slope, intercept = np.polyfit(INV_T, np.log(rates), 1)
    resid = np.log(rates) - (intercept + slope * INV_T)
    r2 = 1 - np.sum(resid ** 2) / np.sum((np.log(rates) - np.log(rates).mean()) ** 2)
    return dict(Ea_eV=float(-slope * KB_EV), R2=float(r2), Slope=float(slope), Intercept=float(intercept))


# ----------------------------------------------------------------------------- 0
def forward_event_table():
    """Per-carbon records with both definitions, and the forward event table."""
    events = pd.read_csv(SUBMITTED_TABLE)
    subjects, _ = WA.rebuild_subjects(events)            # 768 carbons, audited, production clocks
    first = subjects.FirstChain.to_numpy(float)
    duration, cause = subjects.Duration.to_numpy().copy(), subjects.Cause.to_numpy().copy()
    bonded_first = np.isfinite(first) & (first < duration)
    subjects["ForwardDuration"] = np.where(bonded_first, first, duration)
    subjects["ForwardCause"] = np.where(bonded_first, 2, cause)
    assert not (bonded_first & (subjects.Kind == "ArtifactRemoval")).any()   # artifact list stays valid

    gas = events[events.Outcome == "Gas"].copy()
    later_gas = subjects[bonded_first & (subjects.Cause == 1)]              # bonded first, released later
    drop = []
    for _, row in later_gas.iterrows():
        hit = gas[(gas.TrajectoryId == row.Trajectory) & np.isclose(gas.Time_ps, row.Duration, atol=1e-6)
                  & ~gas.index.isin(drop)]
        assert len(hit) >= 1, row
        drop.append(hit.index[0])
    chain = subjects[subjects.ForwardCause == 2]
    table = pd.concat([gas.drop(index=drop),
                       pd.DataFrame(dict(Temperature=chain.Temperature, TrajectoryId=chain.Trajectory, Outcome="Chain",
                                         Species="Chain", Time_ps=chain.ForwardDuration))], ignore_index=True)
    table = table.sort_values(["Temperature", "TrajectoryId", "Outcome", "Time_ps"]).reset_index(drop=True)
    # the production risk-set builder must give back exactly these records
    for temp in TEMPS:
        for traj, (d, c) in production.BuildSubjects(table, temp).items():
            mine = subjects[subjects.Trajectory == traj]
            assert Counter(zip(np.round(d, 8), c)) == Counter(zip(np.round(mine.ForwardDuration, 8), mine.ForwardCause)), traj
    table.to_csv(OUT / "EventTable_Forward.csv", index=False)
    subjects.to_csv(OUT / "Subjects_BothDefinitions.csv", index=False)
    ever = np.isfinite(first) & (first <= duration + 1e-9)             # every carbon that forms a bond while observed
    assert int(ever.sum()) == int((subjects.ForwardCause == 2).sum())
    history = dict(FirstBonds=int(ever.sum()),
                   StillChainAt140=int((ever & (subjects.Cause == 2)).sum()),
                   ReleasedAsGasLater=int((bonded_first & (subjects.Cause == 1)).sum()),
                   BondBrokenNoGas=int((bonded_first & (subjects.Kind == "HorizonCensored")).sum()),
                   EndpointChainsWithEarlierFirstEntry=int(((subjects.Cause == 2) & (subjects.Duration - first > 1e-9)).sum()))
    return events, table, subjects, history


# ----------------------------------------------------------------------- 1 to 5
def kinetics(table, label, out):
    """Items 1 to 5 for one event table, with the paper's own routines."""
    out.mkdir(exist_ok=True)
    by_temp = {t: production.BuildSubjects(table, t) for t in TEMPS}
    res = dict(Label=label)

    # full-run rates, channels, Arrhenius (SI Table S3a, Fig. 3)
    fits, rows = {}, []
    for t in TEMPS:
        d = np.concatenate([by_temp[t][k][0] for k in by_temp[t]])
        c = np.concatenate([by_temp[t][k][1] for k in by_temp[t]])
        fits[t] = {}
        for channel in CHANNELS:
            e = WA.event_mask(c, channel)
            f = FitExponentialRate(d, e.astype(float))
            fits[t][channel] = f
            rows.append(dict(Temperature=t, Channel=channel, Events=f["Events"], Exposure_ps=f["Exposure_ps"],
                             Rate_e3=1e3 * f["Lambda"], CI_Lower_e3=1e3 * f["CI_Lower"], CI_Upper_e3=1e3 * f["CI_Upper"]))
        rows.append(dict(Temperature=t, Channel="censored", Events=int((c == 0).sum())))
        assert np.isclose(fits[t]["gas"]["Lambda"] + fits[t]["chain"]["Lambda"], fits[t]["total"]["Lambda"])
    res["Rates"] = pd.DataFrame(rows)
    res["Arrhenius"] = {ch: arrhenius([fits[t][ch]["Lambda"] for t in TEMPS]) for ch in CHANNELS}

    # 5 naive rate (reacted carbon kept in the denominator) and conversion
    n_events = [fits[t]["total"]["Events"] for t in TEMPS]
    res["Naive"] = dict(Conversion=[n / 192 for n in n_events],
                        Ea_eV=arrhenius([n / (192 * 140.0) for n in n_events])["Ea_eV"])

    # 4 flux-normalised
    audit = json.loads((RP.REWRITE_DIR / "GasSupplyAudit" / "GasSupplyAudit.json").read_text())
    flux = [audit["PerTemperature"][str(t)]["WindowStats"]["0-140"]["free"]["J_TempMean"] for t in TEMPS]
    res["FluxNormalised_Ea_eV"] = arrhenius([fits[t]["total"]["Lambda"] / j for t, j in zip(TEMPS, flux)])["Ea_eV"]

    # 2 ordering fractions, the production loop and random stream
    rng = np.random.default_rng(production.Seed)
    keys = {t: list(by_temp[t].keys()) for t in TEMPS}
    n_boot = production.NBootRates
    lam = {t: {ch: np.empty(n_boot) for ch in CHANNELS} for t in TEMPS}
    for b in range(n_boot):
        for t in TEMPS:
            d_list, c_list = [], []
            for ti in rng.choice(4, size=4, replace=True):
                d, c = by_temp[t][keys[t][ti]]
                idx = rng.integers(0, len(d), size=len(d))
                d_list.append(d[idx]); c_list.append(c[idx])
            dd, cc = np.concatenate(d_list), np.concatenate(c_list)
            lam[t]["gas"][b] = (cc == 1).sum() / dd.sum()
            lam[t]["chain"][b] = (cc == 2).sum() / dd.sum()
            lam[t]["total"][b] = (cc > 0).sum() / dd.sum()
    p = lambda cond: float(np.mean(cond))  # noqa: E731
    res["Ordering"] = dict(
        GasAboveChain_973=p(lam[973]["gas"] > lam[973]["chain"]),
        ChainAboveGas_1273=p(lam[1273]["chain"] > lam[1273]["gas"]),
        ChainRises_973_to_1273=p(lam[1273]["chain"] > lam[973]["chain"]),
        GasRises_873_to_973=p(lam[973]["gas"] > lam[873]["gas"]),
        Gas973_above_1073=p(lam[973]["gas"] > lam[1073]["gas"]),
        Gas973_above_1273=p(lam[973]["gas"] > lam[1273]["gas"]),
        TotalStrictlyIncreasing=p((lam[973]["total"] > lam[873]["total"]) & (lam[1073]["total"] > lam[973]["total"])
                                  & (lam[1273]["total"] > lam[1073]["total"])))
    ea_boot = {ch: -(np.log(np.stack([lam[t][ch] for t in TEMPS], 1)) - np.log(np.stack([lam[t][ch] for t in TEMPS], 1)).mean(1, keepdims=True))
               @ (INV_T - INV_T.mean()) / np.sum((INV_T - INV_T.mean()) ** 2) * KB_EV for ch in CHANNELS}
    res["ChannelEaBootstrap"] = {ch: dict(SD=float(np.nanstd(v, ddof=1)), CI95=[float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))])
                                 for ch, v in ea_boot.items()}

    # 3 paired bootstrap with the net-mass activation energy: the loop and random stream of
    # EaBootstrap.Main. Its gate against the stored net rates is not applied here, because
    # the net-mass input file was regenerated after that script last ran (see SUMMARY.md).
    subjects = EB.BuildSubjects(table)
    mass = EB.PerTrajectoryMassRate(pd.read_csv(EB.PhiPath))
    point_net = [sum(mass[f"{t}_{s}"] for s in EB.Sims) for t in TEMPS]
    notes = json.loads(Path(EB.NotesPath).read_text())
    stored_net = [notes["SyntheticTGA"]["Slopes"][str(t)]["Slope_uPerPs"] for t in TEMPS]
    rng = np.random.default_rng(EB.Seed)
    ea_commit, ea_mass = np.empty(EB.NBoot), np.empty(EB.NBoot)
    for b in range(EB.NBoot):
        lam_b, net_b = [], []
        for t in TEMPS:
            d_list, c_list, net = [], [], 0.0
            for ti in rng.choice(4, size=4, replace=True):
                traj = f"{t}_{EB.Sims[ti]}"
                d, c = subjects[traj]
                idx = rng.integers(0, len(d), size=len(d))
                d_list.append(d[idx]); c_list.append(c[idx])
                net += mass[traj]
            dd, cc = np.concatenate(d_list), np.concatenate(c_list)
            lam_b.append(max((cc > 0).sum(), 1) / dd.sum())
            net_b.append(net)
        ea_commit[b], ea_mass[b] = EB.EaOf(lam_b), EB.EaOf(net_b)

    def stats(v):
        return dict(Mean=float(v.mean()), SD=float(v.std(ddof=1)), CI95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    res["EaBootstrap"] = dict(PointEaCommit_eV=float(res["Arrhenius"]["total"]["Ea_eV"]), PointEaMass_eV=float(EB.EaOf(point_net)),
                              BootEaCommit=stats(ea_commit), BootEaMass=stats(ea_mass), BootDiff=stats(ea_mass - ea_commit),
                              P_MassGreaterCommit=float(np.mean(ea_mass > ea_commit)),
                              PointNetRates=point_net, StoredNetRates=stored_net)

    # 1 Weibull shapes after the initial transient, restarted clock
    weib, per_traj, scan, at_risk = [], [], [], {}
    for t in TEMPS:
        cells = list(by_temp[t].values())
        all_d, all_c = np.concatenate([x[0] for x in cells]), np.concatenate([x[1] for x in cells])
        at_risk[t] = int((all_d > 40).sum())
        boot = {ch: [] for ch in CHANNELS}
        for pick in product(range(4), repeat=4):                       # all 256 ordered whole-cell resamples
            d = np.concatenate([cells[i][0] for i in pick]); c = np.concatenate([cells[i][1] for i in pick])
            keep = d > 40
            for ch in CHANNELS:
                boot[ch].append(WA.fit_independent(d[keep] - 40, WA.event_mask(c[keep], ch))["Rho"])
        for ch in CHANNELS:
            keep = all_d > 40
            f = WA.fit_independent(all_d[keep] - 40, WA.event_mask(all_c[keep], ch))
            weib.append(dict(Temperature=t, Channel=ch, Events=f["Events"], Rho=f["Rho"], LRT_p=f["LRT_p"],
                             Lower=float(np.quantile(boot[ch], .025)), Upper=float(np.quantile(boot[ch], .975)),
                             FractionBelowOne=float(np.mean(np.array(boot[ch]) < 1))))
            for start in STARTS:
                keep = all_d > start
                f = WA.fit_independent(all_d[keep] - start, WA.event_mask(all_c[keep], ch))
                scan.append(dict(Temperature=t, Channel=ch, Start_ps=start, AtRisk=int(keep.sum()), Rho=f["Rho"]))
        for name, (d, c) in by_temp[t].items():
            keep = d > 40
            f = WA.fit_independent(d[keep] - 40, c[keep] > 0)
            per_traj.append(dict(Temperature=t, Trajectory=name, Events=f["Events"], Rho=f["Rho"]))
    res["Weibull"], res["WeibullPerTrajectory"], res["WeibullStartScan"], res["AtRisk40"] = (
        pd.DataFrame(weib), pd.DataFrame(per_traj), pd.DataFrame(scan), at_risk)
    for key in ("Rates", "Weibull", "WeibullPerTrajectory", "WeibullStartScan"):
        res[key].to_csv(out / f"{key}.csv", index=False)
    return res


# ----------------------------------------------------------------------------- 8
def per_atom_episodes(**rule):
    ledger = pd.read_csv(SQ.LEDGER)
    for temperature, cell in SQ.CELLS:
        name = f"{temperature}_{cell}"
        artifacts = set(ledger[(ledger.Trajectory == name) & (ledger.Kind == "ArtifactRemoval")].AtomID)
        times, states, _ = SQ.build(name, artifacts, **rule)
        counts, time_in, records, episodes = SQ.tally(times, states)
        by_atom = {}
        for atom, state, t0, t1, nxt in episodes:
            by_atom.setdefault(atom, []).append((state, t0, t1, nxt))
        yield temperature, name, by_atom, {r["AtomID"]: r["Final"] for r in records}, counts, time_in


def liberation_numbers():
    """First and lasting liberation, and the lifetime and fate of non-lattice
    carbon, under the final two-property rules."""
    off = (SQ.UND, SQ.CHU, SQ.GAS)                                      # without Zr coordination
    atoms, stays = [], []
    for temperature, name, by_atom, final, _, _ in per_atom_episodes():
        for atom, eps in by_atom.items():
            stop = eps[-1][2]
            lib = next((t0 for s, t0, _, _ in eps if s in (SQ.UND, SQ.CHU)), None)
            com = next(((s, t0) for s, t0, _, _ in eps if s in (SQ.CHL, SQ.CHU, SQ.GAS)), None)
            flags = [s in off for s, _, _, _ in eps]
            excursions = int(sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1])))
            lasting = None
            if flags[-1] and final[atom] != "ArtifactCensored":
                i = len(flags) - 1
                while i > 0 and flags[i - 1]:
                    i -= 1
                lasting = eps[i][1]
            atoms.append(dict(Temperature=temperature, Trajectory=name, AtomID=atom, Stop_ps=stop, Final=final[atom],
                              Lib_ps=lib, Lasting_ps=lasting, Commit_ps=com[1] if com else None,
                              CommitTo=("Gas" if com[0] == SQ.GAS else "Chain") if com else "", Excursions=excursions))
            for s, t0, t1, nxt in eps:
                if s == SQ.UND:
                    stays.append(dict(Temperature=temperature, Trajectory=name, AtomID=atom, Start_ps=t0, Lifetime_ps=t1 - t0,
                                      Fate={SQ.LAT: "Returns to lattice", SQ.GAS: "Gas", SQ.CHU: "Bonds", SQ.CHL: "Bonds"}.get(nxt, "Open at end")))
    atoms, stays = pd.DataFrame(atoms), pd.DataFrame(stays)
    atoms.to_csv(OUT / "Liberation_Atoms.csv", index=False)
    stays.to_csv(OUT / "NonLattice_Episodes.csv", index=False)

    rows = []
    for t, a in atoms.groupby("Temperature"):
        for label, col in (("First liberation", "Lib_ps"), ("Lasting liberation", "Lasting_ps")):
            event = a[col].notna()
            f = FitExponentialRate(a[col].fillna(a.Stop_ps).to_numpy(float), event.to_numpy(float))
            rows.append(dict(Temperature=t, Event=label, Carbons=f["Events"], Rate_e3=1e3 * f["Lambda"],
                             CI_Lower_e3=1e3 * f["CI_Lower"], CI_Upper_e3=1e3 * f["CI_Upper"]))
    rates = pd.DataFrame(rows)
    rates.to_csv(OUT / "Liberation_Rates.csv", index=False)
    fits = {label: arrhenius(rates[rates.Event == label].Rate_e3.to_numpy() / 1e3) for label in ("First liberation", "Lasting liberation")}

    lib = atoms[atoms.Lib_ps.notna()]
    lib_first = lib[lib.Commit_ps.isna() | (lib.Lib_ps < lib.Commit_ps)]
    bonded_first = atoms[(atoms.CommitTo == "Chain") & (atoms.Lib_ps.isna() | (atoms.Commit_ps < atoms.Lib_ps))]
    summary = dict(
        FirstLiberated=int(len(lib)), LastingLiberated=int(atoms.Lasting_ps.notna().sum()),
        LiberatedButCoordinatedAt140=int((lib.Final.isin(["Lattice", "ChainInLattice"])).sum()),
        LiberatedOnceOnly=int((lib.Excursions == 1).sum()), LiberatedMoreThanOnce=int((lib.Excursions > 1).sum()),
        MedianExcursionsPerLiberatedCarbon=float(lib.Excursions.median()), MaxExcursions=int(lib.Excursions.max()),
        MedianFirstToLastingDelay_ps=float((lib.Lasting_ps - lib.Lib_ps).dropna().median()),
        MedianDelayAmongRepeatLeavers_ps=float((lib.Lasting_ps - lib.Lib_ps)[lib.Excursions > 1].dropna().median()),
        Delay90AmongRepeatLeavers_ps=float((lib.Lasting_ps - lib.Lib_ps)[lib.Excursions > 1].dropna().quantile(.9)),
        ShareLastingAtFirstLiberation=float(((lib.Lasting_ps - lib.Lib_ps).dropna() < 1e-9).mean()),
        NonLatticeEpisodes=int(len(stays)), Fates=stays.Fate.value_counts().to_dict(),
        MedianLifetime_ps=float(stays.Lifetime_ps.median()), Lifetime90_ps=float(stays.Lifetime_ps.quantile(.9)),
        LongestLifetime_ps=float(stays.Lifetime_ps.max()),
        MedianLifetimeByFate_ps={k: float(v) for k, v in stays.groupby("Fate").Lifetime_ps.median().items()},
        MedianLifetimeByTemperature_ps={int(k): float(v) for k, v in stays.groupby("Temperature").Lifetime_ps.median().items()},
        ReturnShareByTemperature={int(k): float(v) for k, v in stays.groupby("Temperature").Fate.apply(lambda s: (s == "Returns to lattice").mean()).items()},
        LiberatedBeforeCommitment=int(len(lib_first)),
        MedianLiberationToCommitment_ps=float((lib_first.Commit_ps - lib_first.Lib_ps).dropna().median()),
        BondedBeforeLiberation=int(len(bonded_first)),
        BondedFirstLaterLiberated=int(bonded_first.Lib_ps.notna().sum()),
        MedianBondToLiberation_ps=float((bonded_first.Lib_ps - bonded_first.Commit_ps).dropna().median()),
        Arrhenius=fits)
    (OUT / "Liberation_Summary.json").write_text(json.dumps(summary, indent=1))
    return rates, summary


# ----------------------------------------------------------------------- 6 and 7
VARIANTS = [("Final rules", dict()),
            ("C-C cutoff 2.08 A", dict(c_k=0)), ("C-C cutoff 2.54 A", dict(c_k=2)),
            ("Persistence 0.1 ps", dict(hold=0.1)), ("Persistence 0.4 ps", dict(hold=0.4)),
            ("Zr-C cutoff 3.00 A", dict(zr_k=0)), ("Zr-C cutoff 3.66 A", dict(zr_k=2)),
            ("Threshold of four Zr", dict(min_zr=4)),
            ("Any Zr counted (submitted rule)", dict(count="any", chain_hold=None)),
            ("0.5 ps for the Zr coordination of bonded carbon", dict(chain_hold=0.5))]


def sensitivity():
    rows = []
    for label, rule in VARIANTS:
        acc = {t: dict(gas=0, chain=0, com_t=0.0, lib=0, lib_t=0.0) for t in TEMPS}
        split = Counter()
        for temperature, _, by_atom, final, _, _ in per_atom_episodes(**rule):
            a = acc[temperature]
            for atom, eps in by_atom.items():
                stop = eps[-1][2]
                lib = next((t0 for s, t0, _, _ in eps if s in (SQ.UND, SQ.CHU)), None)
                com = next(((s, t0) for s, t0, _, _ in eps if s in (SQ.CHL, SQ.CHU, SQ.GAS)), None)
                if com:
                    a["gas" if com[0] == SQ.GAS else "chain"] += 1
                a["com_t"] += com[1] if com else stop
                a["lib"] += lib is not None
                a["lib_t"] += stop if lib is None else lib
                split[final[atom]] += 1
        rate = {k: np.array([acc[t][k] / acc[t]["com_t"] for t in TEMPS]) for k in ("gas", "chain")}
        total, lib = rate["gas"] + rate["chain"], np.array([acc[t]["lib"] / acc[t]["lib_t"] for t in TEMPS])
        rows.append(dict(Variant=label, GasFirst="/".join(str(acc[t]["gas"]) for t in TEMPS),
                         ChainFirst="/".join(str(acc[t]["chain"]) for t in TEMPS),
                         Commitment_Ea=arrhenius(total)["Ea_eV"], Gas_Ea=arrhenius(rate["gas"])["Ea_eV"],
                         Chain_Ea=arrhenius(rate["chain"])["Ea_eV"], Liberated=sum(acc[t]["lib"] for t in TEMPS),
                         Liberation_Ea=arrhenius(lib)["Ea_eV"], LiberationBelowCommitment=bool((lib < total).all()),
                         ChainInLattice140=split["ChainInLattice"], ChainNonLattice140=split["ChainUnder"],
                         Gas140=split["Gas"], Lattice140=split["Lattice"], NonLattice140=split["Undercoordinated"]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Sensitivity.csv", index=False)
    return table


def transitions():
    base = SQ.analyse()
    names = ["Lattice", "Free carbon", "Chain (in lattice)", "Chain (free)", "Gas"]
    rows = []
    for t in TEMPS + [None]:
        counts, time_in = SQ.pooled(base["per_cell"], t)
        for i in range(4):
            for j in range(5):
                if i != j and SQ.pooled(base["per_cell"])[0][i, j] > 0:
                    rows.append(dict(Temperature="All" if t is None else t, From=names[i], To=names[j], Count=int(counts[i, j]),
                                     TimeInFrom_ps=float(time_in[i]), Rate_e3=1e3 * counts[i, j] / time_in[i]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "Transitions.csv", index=False)
    return table


# ------------------------------------------------------------------------ report
PRINTED = dict(  # numbers printed in the submitted paper, for the reproduction check
    Rates={873: (3.04, 1.43, 1.61), 973: (3.86, 2.33, 1.54), 1073: (4.25, 1.88, 2.37), 1273: (5.84, 2.00, 3.84)},
    Ea=0.15, Naive=0.11, Flux=0.13, Conversion=(0.34, 0.55), BootSD=0.06, BootCI=(0.03, 0.28), Diff=0.07,
    DiffCI=(-0.06, 0.18), PMass=0.86, AtRisk=(171, 165, 170, 156),
    Ordering=dict(GasAboveChain_973=0.94, ChainAboveGas_1273=0.99, ChainRises_973_to_1273=0.999, GasRises_873_to_973=0.98,
                  Gas973_above_1073=0.81, Gas973_above_1273=0.73, TotalStrictlyIncreasing=0.50),
    Weibull={873: ((0.85, 0.60, 2.06), (0.99, 0.66, 1.92), (0.79, 0.55, 2.19)), 973: ((0.76, 0.70, 0.80), (0.62, 0.51, 0.72), (0.95, 0.70, 1.46)),
             1073: ((0.85, 0.60, 1.05), (0.63, 0.45, 1.05), (1.03, 0.71, 1.56)), 1273: ((0.72, 0.57, 0.92), (0.51, 0.34, 1.68), (0.78, 0.68, 0.95))},
    BelowOnePerTrajectory=(2, 4, 2, 3))


def check_reproduction(res):
    """The submitted events must give the printed numbers. Returns failures."""
    bad = []

    def near(name, got, want, tol):
        if abs(got - want) > tol:
            bad.append(f"{name}: got {got:.4f}, printed {want}")
    r = res["Rates"]
    for t in TEMPS:
        for ch, want in zip(CHANNELS, PRINTED["Rates"][t]):
            near(f"rate {t} {ch}", float(r[(r.Temperature == t) & (r.Channel == ch)].Rate_e3.iloc[0]), want, 0.0051)
    near("Ea", res["Arrhenius"]["total"]["Ea_eV"], PRINTED["Ea"], 0.0051)
    near("naive Ea", res["Naive"]["Ea_eV"], PRINTED["Naive"], 0.0051)
    near("flux Ea", res["FluxNormalised_Ea_eV"], PRINTED["Flux"], 0.0051)
    near("conversion 873", res["Naive"]["Conversion"][0], PRINTED["Conversion"][0], 0.0051)
    near("conversion 1273", res["Naive"]["Conversion"][3], PRINTED["Conversion"][1], 0.0051)
    eb = res["EaBootstrap"]
    near("bootstrap SD", eb["BootEaCommit"]["SD"], PRINTED["BootSD"], 0.0051)
    near("bootstrap CI low", eb["BootEaCommit"]["CI95"][0], PRINTED["BootCI"][0], 0.0051)
    near("bootstrap CI high", eb["BootEaCommit"]["CI95"][1], PRINTED["BootCI"][1], 0.0051)
    near("difference", eb["BootDiff"]["Mean"], PRINTED["Diff"], 0.0101)
    near("P(mass > commitment)", eb["P_MassGreaterCommit"], PRINTED["PMass"], 0.0051)
    for key, want in PRINTED["Ordering"].items():
        near(key, res["Ordering"][key], want, 0.0051 if want != 0.999 else 0.0006)
    w = res["Weibull"]
    for t in TEMPS:
        for ch, want in zip(CHANNELS, PRINTED["Weibull"][t]):
            row = w[(w.Temperature == t) & (w.Channel == ch)].iloc[0]
            for name, got, target in zip(("rho", "lower", "upper"), (row.Rho, row.Lower, row.Upper), want):
                near(f"Weibull {t} {ch} {name}", float(got), target, 0.0051)
    if tuple(res["AtRisk40"][t] for t in TEMPS) != PRINTED["AtRisk"]:
        bad.append(f"at risk at 40 ps: {res['AtRisk40']}")
    below = tuple(int((g.Rho < 1).sum()) for _, g in res["WeibullPerTrajectory"].groupby("Temperature"))
    if below != PRINTED["BelowOnePerTrajectory"]:
        bad.append(f"single-trajectory shapes below one: {below}")
    return bad


def fmt_rate(r, t, ch):
    row = r[(r.Temperature == t) & (r.Channel == ch)].iloc[0]
    return f"{row.Rate_e3:.2f} [{row.CI_Lower_e3:.2f}, {row.CI_Upper_e3:.2f}]"


def write_summary(old, new, history, lib_rates, lib, sens, trans, failures):
    L = ["# Commitment kinetics with the forward chain event", "",
         "Chain event: a carbon's first confirmed carbon bond (was: its last entry into the chain it still occupies at 140 ps). "
         "Gas event: release as CO or CO2 without an earlier carbon bond. Everything else is as in the submitted paper, "
         "computed with the paper's own routines on the new event table (`EventTable_Forward.csv`).", "",
         "## Reproduction check", "",
         ("The submitted events reproduce every number checked against the submitted paper (rates, activation energies, "
          "ordering fractions, bootstrap intervals, Weibull shapes and ranges, populations at 40 ps).") if not failures else
         "The submitted events reproduce the numbers checked against the submitted paper (rates, activation energies, ordering "
         "fractions, bootstrap intervals, Weibull shapes and ranges, populations at 40 ps), with these exceptions:\n\n"
         + "\n".join(f"- {f}" for f in failures), "",
         "One input has moved since the submission, independent of the chain event. The net-mass input file "
         f"(`phiTimeseries/PhiTimeseries.csv`) was regenerated after the paired bootstrap was last run. "
         f"Net mass rates now compute to {', '.join(f'{v:.2f}' for v in old['EaBootstrap']['PointNetRates'])} u/ps against the stored "
         f"{', '.join(f'{v:.2f}' for v in old['EaBootstrap']['StoredNetRates'])}. The net-mass activation energy is "
         f"{old['EaBootstrap']['PointEaMass_eV']:.4f} eV against the stored 0.2202 eV. Both round to 0.22 eV. The cause was not investigated here.",
         "", "## What the new event changes in the records", "",
         f"- {history['FirstBonds']} carbons form a carbon bond. {history['StillChainAt140']} are in a chain at 140 ps, "
         f"{history['ReleasedAsGasLater']} are released as gas later, and {history['BondBrokenNoGas']} lose the bond and remain in the solid.",
         f"- Of the 183 chain carbons at 140 ps, {history['EndpointChainsWithEarlierFirstEntry']} have a first bond earlier than the entry used in the submission.",
         "", "## SI Table S3(a): full-run rates (10^-3 ps^-1), counts in brackets after the rate interval", "",
         "| T (K) | Outcome | Submitted | New |", "|---:|---|---|---|"]
    for t in TEMPS:
        for ch, name in zip(CHANNELS, ("Total committed", "Gas", "Chains")):
            n_old = int(old["Rates"][(old["Rates"].Temperature == t) & (old["Rates"].Channel == ch)].Events.iloc[0])
            n_new = int(new["Rates"][(new["Rates"].Temperature == t) & (new["Rates"].Channel == ch)].Events.iloc[0])
            L.append(f"| {t} | {name} | {fmt_rate(old['Rates'], t, ch)} ({n_old}) | {fmt_rate(new['Rates'], t, ch)} ({n_new}) |")
        c_old = int(old["Rates"][(old["Rates"].Temperature == t) & (old["Rates"].Channel == "censored")].Events.iloc[0])
        c_new = int(new["Rates"][(new["Rates"].Temperature == t) & (new["Rates"].Channel == "censored")].Events.iloc[0])
        L.append(f"| {t} | Censored | {c_old} | {c_new} |")
    L += ["", "## Activation energies (eV)", "", "| Quantity | Submitted | New |", "|---|---|---|"]
    for ch, name in zip(CHANNELS, ("Total commitment", "Gas channel", "Chain channel")):
        L.append(f"| {name} | {old['Arrhenius'][ch]['Ea_eV']:.3f} (R2 {old['Arrhenius'][ch]['R2']:.3f}) | "
                 f"{new['Arrhenius'][ch]['Ea_eV']:.3f} (R2 {new['Arrhenius'][ch]['R2']:.3f}) |")

    def boot(r):
        b = r["EaBootstrap"]
        return (f"s.d. {b['BootEaCommit']['SD']:.3f}, 95% [{b['BootEaCommit']['CI95'][0]:.3f}, {b['BootEaCommit']['CI95'][1]:.3f}]",
                f"{b['BootDiff']['Mean']:.3f}, 95% [{b['BootDiff']['CI95'][0]:.3f}, {b['BootDiff']['CI95'][1]:.3f}], P(mass larger) {b['P_MassGreaterCommit']:.2f}")
    L += [f"| Total commitment, two-level bootstrap | {boot(old)[0]} | {boot(new)[0]} |",
          f"| Net mass minus commitment | {boot(old)[1]} | {boot(new)[1]} |",
          f"| Net mass (unchanged) | {old['EaBootstrap']['PointEaMass_eV']:.3f}, s.d. {old['EaBootstrap']['BootEaMass']['SD']:.3f} | "
          f"{new['EaBootstrap']['PointEaMass_eV']:.3f}, s.d. {new['EaBootstrap']['BootEaMass']['SD']:.3f} |",
          f"| Flux-normalised commitment | {old['FluxNormalised_Ea_eV']:.3f} | {new['FluxNormalised_Ea_eV']:.3f} |",
          f"| Naive rate (reacted carbon kept in the denominator) | {old['Naive']['Ea_eV']:.4f} | {new['Naive']['Ea_eV']:.4f} |",
          f"| Conversion at 873 and 1273 K | {old['Naive']['Conversion'][0]:.0%} and {old['Naive']['Conversion'][3]:.0%} | "
          f"{new['Naive']['Conversion'][0]:.0%} and {new['Naive']['Conversion'][3]:.0%} |",
          f"| Gas channel, bootstrap s.d. | {old['ChannelEaBootstrap']['gas']['SD']:.3f} | {new['ChannelEaBootstrap']['gas']['SD']:.3f} |",
          f"| Chain channel, bootstrap s.d. | {old['ChannelEaBootstrap']['chain']['SD']:.3f} | {new['ChannelEaBootstrap']['chain']['SD']:.3f} |",
          "", "## Ordering fractions of the full-run rates (10,000 two-level resamples)", "", "| Comparison | Submitted | New |", "|---|---:|---:|"]
    names = dict(GasAboveChain_973="Gas above chain at 973 K", ChainAboveGas_1273="Chain above gas at 1273 K",
                 ChainRises_973_to_1273="Chain rate rises from 973 to 1273 K", GasRises_873_to_973="Gas rate rises from 873 to 973 K",
                 Gas973_above_1073="Gas at 973 K above 1073 K", Gas973_above_1273="Gas at 973 K above 1273 K",
                 TotalStrictlyIncreasing="Total strictly increasing over all four temperatures")
    for key, name in names.items():
        L.append(f"| {name} | {old['Ordering'][key]:.1%} | {new['Ordering'][key]:.1%} |")
    L += ["", "## SI Table S3(b): Weibull shapes over 40 to 140 ps, restarted clock", "",
          "Shape with the central 95% range over all 256 ordered whole-trajectory resamples.", "",
          "| T (K) | Channel | Submitted | New | New, likelihood-ratio p |", "|---:|---|---|---|---:|"]
    for t in TEMPS:
        for ch in CHANNELS:
            o = old["Weibull"][(old["Weibull"].Temperature == t) & (old["Weibull"].Channel == ch)].iloc[0]
            n = new["Weibull"][(new["Weibull"].Temperature == t) & (new["Weibull"].Channel == ch)].iloc[0]
            L.append(f"| {t} | {ch} | {o.Rho:.2f} [{o.Lower:.2f}, {o.Upper:.2f}] | {n.Rho:.2f} [{n.Lower:.2f}, {n.Upper:.2f}] | {n.LRT_p:.3g} |")

    def below(r):
        return ", ".join(str(int((g.Rho < 1).sum())) for _, g in r["WeibullPerTrajectory"].groupby("Temperature"))

    def scan_range(r, t, ch="total"):
        s = r["WeibullStartScan"]
        v = s[(s.Temperature == t) & (s.Channel == ch)].Rho
        return f"{v.min():.2f} to {v.max():.2f}"
    L += ["", f"- Carbon still uncommitted at 40 ps: submitted {', '.join(str(old['AtRisk40'][t]) for t in TEMPS)}; new {', '.join(str(new['AtRisk40'][t]) for t in TEMPS)}.",
          f"- Single trajectories with a total shape below one (of four, per temperature): submitted {below(old)}; new {below(new)}.",
          "- Total shape over start times of 30, 35, 40, 45 and 50 ps: " + "; ".join(
              f"{t} K submitted {scan_range(old, t)}, new {scan_range(new, t)}" for t in TEMPS) + ".",
          "", "## Liberation, for the text (final two-property rules)", "",
          "State names: Lattice, Free carbon (no Zr coordination, no carbon bond), Chain (in lattice), Chain (free).", "",
          "First liberation is a carbon's first loss of Zr coordination, bonded or not. Lasting liberation is the start of the "
          "final stretch without Zr coordination, for carbon that is non-lattice or gas at 140 ps. It is defined by looking back and is given for comparison only.", "",
          "| T (K) | First liberation, rate (carbons) | Lasting liberation, rate (carbons) |", "|---:|---|---|"]
    for t in TEMPS:
        a = lib_rates[(lib_rates.Temperature == t) & (lib_rates.Event == "First liberation")].iloc[0]
        b = lib_rates[(lib_rates.Temperature == t) & (lib_rates.Event == "Lasting liberation")].iloc[0]
        L.append(f"| {t} | {a.Rate_e3:.2f} [{a.CI_Lower_e3:.2f}, {a.CI_Upper_e3:.2f}] ({int(a.Carbons)}) | "
                 f"{b.Rate_e3:.2f} [{b.CI_Lower_e3:.2f}, {b.CI_Upper_e3:.2f}] ({int(b.Carbons)}) |")
    f1, f2 = lib["Arrhenius"]["First liberation"], lib["Arrhenius"]["Lasting liberation"]
    fates = lib["Fates"]
    n_ep = lib["NonLatticeEpisodes"]
    L += ["", f"- Activation energy: first liberation {f1['Ea_eV']:.3f} eV (R2 {f1['R2']:.3f}), lasting liberation {f2['Ea_eV']:.3f} eV (R2 {f2['R2']:.3f}).",
          f"- {lib['FirstLiberated']} carbons are liberated at least once. {lib['LastingLiberated']} are without Zr coordination at 140 ps or left as gas. "
          f"{lib['LiberatedButCoordinatedAt140']} are Zr-coordinated again at 140 ps.",
          f"- {lib['LiberatedOnceOnly']} leave once and stay out. {lib['LiberatedMoreThanOnce']} leave more than once "
          f"(median {lib['MedianExcursionsPerLiberatedCarbon']:.0f} departure per liberated carbon, at most {lib['MaxExcursions']}). "
          f"For {lib['ShareLastingAtFirstLiberation']:.0%} of lastingly liberated carbons the first departure is the lasting one. "
          f"Among carbons that leave more than once, the median delay from first to lasting departure is "
          f"{lib['MedianDelayAmongRepeatLeavers_ps']:.1f} ps (90th percentile {lib['Delay90AmongRepeatLeavers_ps']:.1f} ps).",
          f"- Free carbon (no Zr coordination and no carbon bond): {n_ep} episodes. Median lifetime {lib['MedianLifetime_ps']:.2f} ps, "
          f"90th percentile {lib['Lifetime90_ps']:.1f} ps, longest {lib['LongestLifetime_ps']:.1f} ps.",
          "- Fate of those episodes: " + ", ".join(f"{k.lower()} {v} ({v / n_ep:.0%})" for k, v in fates.items()) + ".",
          "- Median lifetime by fate (ps): " + ", ".join(f"{k.lower()} {v:.2f}" for k, v in lib["MedianLifetimeByFate_ps"].items()) + ".",
          "- Median lifetime by temperature (ps): " + ", ".join(f"{k} K {v:.2f}" for k, v in lib["MedianLifetimeByTemperature_ps"].items()) + ".",
          "- Share returning to the lattice by temperature: " + ", ".join(f"{k} K {v:.0%}" for k, v in lib["ReturnShareByTemperature"].items()) + ".",
          f"- {lib['LiberatedBeforeCommitment']} carbons are liberated before they commit (median {lib['MedianLiberationToCommitment_ps']:.1f} ps from first liberation to commitment). "
          f"{lib['BondedBeforeLiberation']} bond before any liberation, and {lib['BondedFirstLaterLiberated']} of those are liberated later "
          f"(median {lib['MedianBondToLiberation_ps']:.1f} ps after the bond).",
          "", "## Sensitivity to definitions (state model rebuilt for each variant, its own clock)", "",
          "| Variant | Gas first | Chain first | Commitment Ea | Gas Ea | Chain Ea | Liberated | Liberation Ea | Liberation slower at all T | Chain (in lattice) / Chain (free) at 140 ps |",
          "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for _, r in sens.iterrows():
        L.append(f"| {r.Variant} | {r.GasFirst} | {r.ChainFirst} | {r.Commitment_Ea:.3f} | {r.Gas_Ea:.3f} | {r.Chain_Ea:.3f} | {r.Liberated} | "
                 f"{r.Liberation_Ea:.3f} | {'yes' if r.LiberationBelowCommitment else 'no'} | {r.ChainInLattice140} / {r.ChainNonLattice140} |")
    L += ["", "Event-definition variants on the paper's clock are the two columns of the tables above (submitted and new).",
          "", "## Transitions between the five states (final rules)", "",
          "| From | To | " + " | ".join(f"{t} K" for t in TEMPS) + " | All |", "|---|---|" + "---:|" * 5]
    pairs = trans[trans.Temperature == "All"][["From", "To"]].itertuples(index=False)
    for a, b in pairs:
        sub = trans[(trans.From == a) & (trans.To == b)].set_index("Temperature")
        L.append(f"| {a} | {b} | " + " | ".join(f"{int(sub.Count[t])} ({sub.Rate_e3[t]:.2f})" for t in TEMPS) + f" | {int(sub.Count['All'])} ({sub.Rate_e3['All']:.2f}) |")
    L += ["", "Counts, with the rate per carbon in the starting state in brackets (10^-3 ps^-1).", ""]
    (OUT / "SUMMARY.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


def main():
    events, table, subjects, history = forward_event_table()
    old = kinetics(events, "submitted", OUT / "Submitted")
    failures = check_reproduction(old)
    new = kinetics(table, "forward", OUT / "Forward")
    lib_rates, lib = liberation_numbers()
    sens = sensitivity()
    trans = transitions()
    (OUT / "Results.json").write_text(json.dumps(dict(
        History=history, ReproductionFailures=failures,
        **{k: {kk: v[kk] for kk in ("Arrhenius", "Naive", "FluxNormalised_Ea_eV", "Ordering", "ChannelEaBootstrap", "EaBootstrap", "AtRisk40")}
           for k, v in (("Submitted", old), ("Forward", new))}), indent=1, default=float))
    write_summary(old, new, history, lib_rates, lib, sens, trans, failures)
    print("\nReproduction failures:", failures or "none")
    definition_sensitivity()
    liberation_not_reversed()


# -----------------------------------------------------------------------------
# Sensitivity of commitment and liberation to the state definitions (Supplementary Table S3c)
# -----------------------------------------------------------------------------

OFF = (SQ.UND, SQ.CHU, SQ.GAS)             # without Zr coordination
TABLE_VARIANTS = [("Adopted definitions", dict()),
            ("C--C cutoff 2.08 A", dict(c_k=0)), ("C--C cutoff 2.54 A", dict(c_k=2)),
            ("Zr--C cutoff 3.00 A", dict(zr_k=0)), ("Zr--C cutoff 3.66 A", dict(zr_k=2)),
            ("Coordinated with four or more Zr", dict(min_zr=4)),
            ("Any Zr counted, not only original neighbours", dict(count="any")),
            ("Persistence 0.1 ps", dict(hold=0.1)), ("Persistence 0.4 ps", dict(hold=0.4)),
            ("Persistence 0.5 ps for Zr coordination of bonded carbon", dict(chain_hold=0.5))]


def table_variant(rule):
    acc = {t: dict(gas=0, chain=0, com_t=0.0, held=0, held_t=0.0) for t in TEMPS}
    counts = np.zeros((5, 5))
    final_all = {}
    for temperature, name, by_atom, final, c, _ in per_atom_episodes(**rule):
        counts += c
        a = acc[temperature]
        for atom, eps in by_atom.items():
            stop = eps[-1][2]
            final_all[final[atom]] = final_all.get(final[atom], 0) + 1
            com = next(((s, t0) for s, t0, _, _ in eps if s in (SQ.CHL, SQ.CHU, SQ.GAS)), None)
            if com:
                a["gas" if com[0] == SQ.GAS else "chain"] += 1
            a["com_t"] += com[1] if com else stop
            flags = [s in OFF for s, _, _, _ in eps]
            last = None
            if flags[-1] and final[atom] != "ArtifactCensored":
                i = len(flags) - 1
                while i > 0 and flags[i - 1]:
                    i -= 1
                last = eps[i][1]
            a["held"] += last is not None
            a["held_t"] += stop if last is None else last
    n = lambda i, j: int(counts[i, j])
    # every move between a coordinated and an uncoordinated state, including the rare
    # ones that change both properties at once
    to_lib = sum(n(i, j) for i in (SQ.LAT, SQ.CHL) for j in OFF)
    back = sum(n(i, j) for i in (SQ.UND, SQ.CHU) for j in (SQ.LAT, SQ.CHL))
    com = np.array([(acc[t]["gas"] + acc[t]["chain"]) / acc[t]["com_t"] for t in TEMPS])
    lib = np.array([acc[t]["held"] / acc[t]["held_t"] for t in TEMPS])
    held = sum(acc[t]["held"] for t in TEMPS)
    assert held == to_lib - back, (held, to_lib, back)
    return dict(GasFirst=sum(acc[t]["gas"] for t in TEMPS), ChainFirst=sum(acc[t]["chain"] for t in TEMPS),
                Commitment_Ea=arrhenius(com)["Ea_eV"], ToLiberated=to_lib, Reversed=back, NotReversed=held,
                NotReversed_Ea=arrhenius(lib)["Ea_eV"], LiberationBelowCommitment=bool((lib < com).all()),
                **{f"Com_{t}": 1e3 * v for t, v in zip(TEMPS, com)}, **{f"Lib_{t}": 1e3 * v for t, v in zip(TEMPS, lib)},
                Lattice140=final_all.get("Lattice", 0), Free140=final_all.get("Undercoordinated", 0),
                Chain140=final_all.get("ChainInLattice", 0) + final_all.get("ChainUnder", 0), Gas140=final_all.get("Gas", 0))


def definition_sensitivity():
    rows = [dict(Variant=label, **table_variant(rule)) for label, rule in TABLE_VARIANTS]
    sub = json.loads((OUT / "Results.json").read_text())["Submitted"]
    old = pd.read_csv(OUT / "Submitted" / "Rates.csv")
    rows.append(dict(Variant="Chain event of the submitted version", GasFirst=int(old[old.Channel == "gas"].Events.sum()),
                     ChainFirst=int(old[old.Channel == "chain"].Events.sum()), Commitment_Ea=sub["Arrhenius"]["total"]["Ea_eV"]))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "DefinitionSensitivity.csv", index=False)
    L = ["# Sensitivity to the state definitions", "", "Written by ForwardEventAnalyses.py. Sums over the sixteen trajectories.", "",
         "| Variant | Gas first | Chain first | Commitment Ea (eV) | To a liberated state | Reversed | Not reversed | Ea of those (eV) | Below commitment at every T | Lattice, free, chain, gas at 140 ps |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rows:
        if "ToLiberated" in r:
            L.append(f"| {r['Variant']} | {r['GasFirst']} | {r['ChainFirst']} | {r['Commitment_Ea']:.3f} | {r['ToLiberated']} | {r['Reversed']} | "
                     f"{r['NotReversed']} | {r['NotReversed_Ea']:.3f} | {'yes' if r['LiberationBelowCommitment'] else 'no'} | "
                     f"{r['Lattice140']}, {r['Free140']}, {r['Chain140']}, {r['Gas140']} |")
        else:
            L.append(f"| {r['Variant']} | {r['GasFirst']} | {r['ChainFirst']} | {r['Commitment_Ea']:.3f} | | | | | | |")
    (OUT / "DefinitionSensitivity.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


# -----------------------------------------------------------------------------
# Liberations that are not reversed (Section 2.3 of the paper)
# -----------------------------------------------------------------------------

N_BOOT = 10000
LIBERATION_RULES = [("One persistence time (0.2 ps)", dict(chain_hold=None)), ("With the 0.5 ps rule for bonded carbon", dict(chain_hold=0.5))]


def liberation_atom_table(rule):
    paper = pd.read_csv(OUT / "Subjects_BothDefinitions.csv")[["Trajectory", "AtomID", "ForwardDuration", "ForwardCause"]]
    rows, counts = [], np.zeros((5, 5))
    for temperature, name, by_atom, final, c, _ in per_atom_episodes(**rule):
        counts += c
        for atom, eps in by_atom.items():
            flags = [s in OFF for s, _, _, _ in eps]
            first = next((t0 for (s, t0, _, _), f in zip(eps, flags) if f), None)
            last = None
            if flags[-1] and final[atom] != "ArtifactCensored":
                i = len(flags) - 1
                while i > 0 and flags[i - 1]:
                    i -= 1
                last = eps[i][1]
            rows.append(dict(Temperature=temperature, Trajectory=name, AtomID=atom, Stop_ps=eps[-1][2], Final=final[atom],
                             First_ps=first, NotReversed_ps=last))
    atoms = pd.DataFrame(rows).merge(paper, on=["Trajectory", "AtomID"], validate="one_to_one")
    atoms = atoms.sort_values(["Temperature", "Trajectory", "AtomID"]).reset_index(drop=True)
    assert len(atoms) == 768 and (atoms.groupby("Trajectory").size() == 48).all()
    return atoms, counts


def not_reversed_numbers(rule):
    atoms, counts = liberation_atom_table(rule)
    n = lambda i, j: int(counts[i, j])
    to_liberated = n(SQ.LAT, SQ.UND) + n(SQ.CHL, SQ.CHU)
    reversed_ = n(SQ.UND, SQ.LAT) + n(SQ.CHU, SQ.CHL) + n(SQ.CHU, SQ.LAT) + n(SQ.UND, SQ.CHL)
    series = {"Not reversed": (atoms.NotReversed_ps.fillna(atoms.Stop_ps), atoms.NotReversed_ps.notna()),
              "First": (atoms.First_ps.fillna(atoms.Stop_ps), atoms.First_ps.notna()),
              "Commitment": (atoms.ForwardDuration, atoms.ForwardCause > 0)}
    rates = {k: np.array([e[atoms.Temperature == t].sum() / d[atoms.Temperature == t].sum() for t in TEMPS]) for k, (d, e) in series.items()}
    fits = {k: arrhenius(v) for k, v in rates.items()}

    rng = np.random.default_rng(RP.SEED)
    inv_t = 1.0 / np.array(TEMPS, float)
    x = inv_t - inv_t.mean()
    logs = {k: np.empty((N_BOOT, 4)) for k in series}
    for j, t in enumerate(TEMPS):
        pick_cell = rng.integers(0, 4, size=(N_BOOT, 4))
        pick = (pick_cell[:, :, None], rng.integers(0, 48, size=(N_BOOT, 4, 48)))
        mask = (atoms.Temperature == t).to_numpy()
        for k, (d, e) in series.items():
            dd, ee = d.to_numpy(float)[mask].reshape(4, 48), e.to_numpy(float)[mask].reshape(4, 48)
            logs[k][:, j] = np.log(ee[pick].sum((1, 2)) / dd[pick].sum((1, 2)))
    ea = {k: -(v - v.mean(1, keepdims=True)) @ x / np.sum(x ** 2) * KB_EV for k, v in logs.items()}

    def stats(v):
        return dict(SD=float(v.std(ddof=1)), CI95=[float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))])
    held = atoms[atoms.NotReversed_ps.notna()]
    return dict(
        TransitionsToLiberated=to_liberated, Reversed=reversed_, NotReversed=to_liberated - reversed_,
        NotReversedCarbons=int(len(held)), FirstLiberatedCarbons=int(atoms.First_ps.notna().sum()),
        Fate=held.Final.value_counts().to_dict(),
        ChainSwitches=dict(Exits=n(SQ.CHL, SQ.CHU), Returns=n(SQ.CHU, SQ.CHL)),
        IntoChain=dict(FromLattice=n(SQ.LAT, SQ.CHL), Total=n(SQ.LAT, SQ.CHL) + n(SQ.UND, SQ.CHU)),
        Final=atoms.Final.value_counts().to_dict(),
        Rates_e3={k: [round(1e3 * r, 3) for r in v] for k, v in rates.items()},
        Ea_eV={k: round(f["Ea_eV"], 4) for k, f in fits.items()}, R2={k: round(f["R2"], 4) for k, f in fits.items()},
        Bootstrap={k: stats(v) for k, v in ea.items()},
        NotReversedMinusCommitment=stats(ea["Not reversed"] - ea["Commitment"]),
        ShareBelowCommitmentAtAllT=float((logs["Not reversed"] < logs["Commitment"]).all(1).mean()),
        BelowCommitmentAtAllT=bool((rates["Not reversed"] < rates["Commitment"]).all()))


def liberation_not_reversed():
    res = {label: not_reversed_numbers(rule) for label, rule in LIBERATION_RULES}
    (OUT / "LiberationNotReversed.json").write_text(json.dumps(res, indent=1))
    L = ["# Liberations that are not reversed", "",
         "Written by ForwardEventAnalyses.py. The first column is the definition used in the paper "
         "(0.2 ps persistence for every change of state). The second is the earlier definition with 0.5 ps for the "
         "Zr coordination of bonded carbon, kept for comparison.", "",
         "| Quantity | " + " | ".join(res) + " |", "|---|" + "---:|" * len(res)]

    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(r) for r in res.values()) + " |")
    row("Transitions to a liberated state", lambda r: str(r["TransitionsToLiberated"]))
    row("Reversed", lambda r: str(r["Reversed"]))
    row("Not reversed", lambda r: str(r["NotReversed"]))
    row("Carbons with a liberation that is not reversed", lambda r: str(r["NotReversedCarbons"]))
    row("of which gas, chain, free carbon at 140 ps", lambda r: f"{r['Fate'].get('Gas', 0)}, {r['Fate'].get('ChainUnder', 0)}, {r['Fate'].get('Undercoordinated', 0)}")
    row("Carbons liberated at least once", lambda r: str(r["FirstLiberatedCarbons"]))
    row("Chain (in lattice) to Chain (free), and back", lambda r: f"{r['ChainSwitches']['Exits']}, {r['ChainSwitches']['Returns']}")
    row("Chain (in lattice), Chain (free) at 140 ps", lambda r: f"{r['Final'].get('ChainInLattice', 0)}, {r['Final'].get('ChainUnder', 0)}")
    row("Transitions into a chain from lattice carbon", lambda r: f"{r['IntoChain']['FromLattice']} of {r['IntoChain']['Total']}")
    for k in ("Not reversed", "First", "Commitment"):
        row(f"{k}: rates (10^-3 ps^-1) at {', '.join(map(str, TEMPS))} K", lambda r, k=k: ", ".join(f"{v:.2f}" for v in r["Rates_e3"][k]))
        row(f"{k}: activation energy (eV), R2", lambda r, k=k: f"{r['Ea_eV'][k]:.3f}, {r['R2'][k]:.3f}")
        row(f"{k}: bootstrap s.d., 95% interval (eV)", lambda r, k=k: f"{r['Bootstrap'][k]['SD']:.3f}, {r['Bootstrap'][k]['CI95'][0]:.3f} to {r['Bootstrap'][k]['CI95'][1]:.3f}")
    row("Not reversed minus commitment, 95% interval (eV)", lambda r: f"{r['NotReversedMinusCommitment']['CI95'][0]:.3f} to {r['NotReversedMinusCommitment']['CI95'][1]:.3f}")
    row("Below commitment at every temperature", lambda r: "yes" if r["BelowCommitmentAtAllT"] else "no")
    row("Share of resamples below commitment at every temperature", lambda r: f"{r['ShareBelowCommitmentAtAllT']:.1%}")
    L += ["", f"Two-level bootstrap, {N_BOOT} resamples, seed {RP.SEED}, same draws for all three events."]
    (OUT / "LiberationNotReversed.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
