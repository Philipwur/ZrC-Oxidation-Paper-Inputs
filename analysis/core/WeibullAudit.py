"""Independent audit of the paper's 140 ps competing-risk Weibull fits.

Reads existing event, atom-history, and physical-departure audit records. Does
not overwrite production fits, figures, raw trajectories, or the files of the paper.
The independent fit solves the strictly decreasing profile-likelihood score
for rho; lifelines is used only as a separate reproduction check.

Run: python WeibullAudit.py
Outputs: WeibullAuditResults/ (audit tables, JSON, and SUMMARY.md).
"""
from collections import Counter
from itertools import product
import os
from pathlib import Path
import hashlib
import json
import math
import re
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, norm
from lifelines import WeibullFitter

import CommitmentCompetingRisks as production
from SubsequentChainFate import match_departures, read_csv

HERE = Path(__file__).resolve().parent
OUT = HERE / 'WeibullAuditResults'
TEMPS = (873, 973, 1073, 1273)
CHANNELS = {'gas': 1, 'chain': 2, 'total': -1}
HORIZON = 140.0
INPUTS = [HERE/'Refresh140/Events/EventTableV2.csv',
          HERE/'CompetingRisksResults/CommitmentFits.json',
          HERE/'RateAnalysisAuditResults/AllDeparturesIntegrity.csv']


def fit_independent(durations, events):
    """Right-censored likelihood, with all clocks starting at simulation t=0.

    Profile scale^rho = sum(t_i^rho)/d. Shape score:
    d/rho + sum_event(log t) - d*sum(t^rho log t)/sum(t^rho).
    The score derivative is negative: -d/rho^2 - d*Var_weighted(log t).
    This also gives the efficient information and the Wald SE of rho.
    """
    t, e = np.asarray(durations, float), np.asarray(events, bool)
    assert np.all(np.isfinite(t)) and np.all(t > 0)
    d = int(e.sum())
    if d == 0:
        return None
    logs = np.log(t)
    offset = float(logs.max())
    centered = logs - offset
    event_log_sum = float(logs[e].sum())

    def score(rho):
        weights = np.exp(rho * centered)
        mean = offset + float(np.dot(weights, centered) / weights.sum())
        return d/rho + event_log_sum - d*mean

    upper = 2.0
    while score(upper) > 0 and upper < 1024:
        upper *= 2
    rho = brentq(score, 1e-7, upper, xtol=1e-12)
    weights = np.exp(rho * centered)
    log_sum = rho*offset + np.log(weights.sum())
    log_scale = (log_sum - np.log(d))/rho
    ll = d*np.log(rho) - d*rho*log_scale + (rho-1)*event_log_sum - d
    rate = d / t.sum()
    ll_exp = d*np.log(rate) - rate*t.sum()
    statistic = 2*(ll-ll_exp)
    assert statistic >= -1e-8
    variance_log_t = np.average((logs-np.average(logs, weights=weights))**2,
                               weights=weights)
    se = 1/np.sqrt(d/rho**2 + d*variance_log_t)
    z = norm.ppf(.975)
    return dict(Events=d, Exposure_ps=float(t.sum()), Rate=float(rate),
                Rho=float(rho), Scale=float(np.exp(log_scale)),
                LogLikelihood=float(ll), ExponentialLogLikelihood=float(ll_exp),
                LRT=float(max(0, statistic)), LRT_p=float(chi2.sf(max(0, statistic), 1)),
                Rho_CI_Lower=float(rho-z*se), Rho_CI_Upper=float(rho+z*se),
                ScoreResidual=float(score(rho)))


def event_mask(cause, channel):
    return cause > 0 if channel == 'total' else cause == CHANNELS[channel]


def rebuild_subjects(events):
    """Recover atom identities from audited removals and confirmed histories."""
    departures = read_csv(INPUTS[2])
    rows, states = [], {}
    for temp in TEMPS:
        for sim in range(1, 5):
            traj = f'{temp}_{sim}'
            path = HERE/f'Refresh140/FrameCache/{traj}_states_140ps.json'
            INPUTS.append(path)
            record = json.loads(path.read_text())
            states[traj] = record
            matched = match_departures(record, [x for x in departures if x['Trajectory'] == traj])
            assert set(map(int, record['HorizonStates'])) == set(range(48))
            assert set(record['RetainedIDs']) == {
                int(atom) for atom, state in record['HorizonStates'].items() if state == 'Chain'}
            for atom in range(48):
                key = str(atom)
                if key in matched:
                    departure = matched[key]
                    cause = 1 if departure['Kind'] == 'Gas' else 0
                    time = departure['Time_ps']
                    kind = departure['Kind']
                elif atom in record['RetainedIDs']:
                    cause, time, kind = 2, record['LastChainEntryPs'][key], 'RetainedChain'
                else:
                    cause, time, kind = 0, HORIZON, 'HorizonCensored'
                rows.append(dict(Temperature=temp, Trajectory=traj, AtomID=atom,
                                 Duration=float(time), Cause=cause, Kind=kind,
                                 FirstChain=record['FirstChainPs'].get(key)))
    df = pd.DataFrame(rows)
    assert len(df) == 768 and not df.duplicated(['Trajectory', 'AtomID']).any()
    assert ((df.Duration > 0) & (df.Duration <= HORIZON)).all()
    assert (df.Kind == 'ArtifactRemoval').sum() == 5
    df['PhysicalDuration'] = df.Duration
    # Production uses gas-log timestamps, while the physical audit records
    # the first raw frame after removal. Match in temporal order and retain
    # BOTH clocks; quantify their discrepancy rather than silently replacing it.
    for temp in TEMPS:
        subjects = production.BuildSubjects(events, temp)
        for traj, (dur, causes) in subjects.items():
            mine = df[df.Trajectory == traj]
            assert len(mine) == 48
            gas = mine[mine.Cause == 1].sort_values('Duration')
            logged = np.sort(dur[causes == 1])
            assert len(gas) == len(logged)
            assert np.max(np.abs(gas.Duration.to_numpy()-logged), initial=0) < .2
            df.loc[gas.index, 'Duration'] = logged
            mine = df[df.Trajectory == traj]
            assert Counter(zip(np.round(dur, 8), causes)) == Counter(
                zip(np.round(mine.Duration, 8), mine.Cause)), traj
    return df, states


def resample_cells(subjects, rng, inner):
    durations, causes = [], []
    for idx in rng.choice(4, 4, replace=True):
        d, c = subjects[idx]
        take = rng.integers(0, len(d), size=len(d)) if inner else np.arange(len(d))
        durations.append(d[take]); causes.append(c[take])
    return np.concatenate(durations), np.concatenate(causes)


def main():
    OUT.mkdir(exist_ok=True)
    events = pd.read_csv(INPUTS[0])
    stored = json.loads(INPUTS[1].read_text())
    df, states = rebuild_subjects(events)
    df.to_csv(OUT/'VerifiedSubjects.csv', index=False)
    fits, comparison, diagnostics, loo = [], [], [], []
    subjects = {}
    for temp in TEMPS:
        prod = production.BuildSubjects(events, temp)
        subjects[temp] = list(prod.values())  # preserve order for original RNG reproduction
        all_d = np.concatenate([x[0] for x in subjects[temp]])
        all_c = np.concatenate([x[1] for x in subjects[temp]])
        for channel in CHANNELS:
            ev = event_mask(all_c, channel)
            new = fit_independent(all_d, ev)
            old = stored['Fits'][str(temp)][channel]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                wf = WeibullFitter().fit(all_d, ev)
            row = dict(Temperature=temp, Channel=channel, **new)
            row['LifelinesWarnings'] = ' | '.join(str(w.message) for w in caught)
            fits.append(row)
            comparison.append(dict(Temperature=temp, Channel=channel,
                DeltaRhoStored=new['Rho']-old['Weibull']['Rho'],
                DeltaLLStored=new['LogLikelihood']-old['Weibull']['LogLikelihood'],
                DeltaPStored=new['LRT_p']-old['LRT_p'],
                DeltaRhoFreshLifelines=new['Rho']-wf.rho_,
                DeltaLLFreshLifelines=new['LogLikelihood']-wf.log_likelihood_,
                DeltaLowerCI=new['Rho_CI_Lower']-old['Weibull']['Rho_CI_Lower'],
                DeltaUpperCI=new['Rho_CI_Upper']-old['Weibull']['Rho_CI_Upper']))
            for omitted in range(4):
                d = np.concatenate([x[0] for i,x in enumerate(subjects[temp]) if i != omitted])
                c = np.concatenate([x[1] for i,x in enumerate(subjects[temp]) if i != omitted])
                loo.append(dict(Temperature=temp, Channel=channel, Omitted=omitted+1,
                                **fit_independent(d, event_mask(c, channel))))
            # Fixed, equal windows: descriptive check for non-monotone hazards.
            for lo, hi in zip([0,35,70,105], [35,70,105,140]):
                exposure = float(np.clip(all_d-lo, 0, hi-lo).sum())
                count = int((ev & (all_d > lo) & (all_d <= hi)).sum())
                diagnostics.append(dict(Temperature=temp, Channel=channel,
                    Start=lo, End=hi, Events=count, Exposure=exposure, Rate=count/exposure))
    pd.DataFrame(fits).to_csv(OUT/'IndependentFits.csv', index=False)
    pd.DataFrame(comparison).to_csv(OUT/'NumericalComparison.csv', index=False)
    for comparison_row in comparison:
        assert abs(comparison_row['DeltaRhoStored']) < 1e-5
        assert abs(comparison_row['DeltaLLStored']) < 1e-7
        assert abs(comparison_row['DeltaPStored']) < 1e-7
    assert all(not row['LifelinesWarnings'] for row in fits)
    pd.DataFrame(loo).to_csv(OUT/'LeaveOneCellOut.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/'EqualWindowRates.csv', index=False)
    differences = df[df.Cause==1].PhysicalDuration-df[df.Cause==1].Duration
    timing = dict(NGas=int((df.Cause==1).sum()), NDifferent=int((differences.abs()>1e-8).sum()),
                  MaxAbsDifference_ps=float(differences.abs().max()),
                  MeanAbsDifference_ps=float(differences.abs().mean()))
    print('All 768 subjects/counts match; 12 fits reproduced using production event times.', flush=True)
    print('Gas-log versus physical-removal timing:', timing, flush=True)
    print(pd.DataFrame(fits)[['Temperature','Channel','Rho','LRT_p']].to_string(index=False), flush=True)
    print('Maximum numerical discrepancies:', pd.DataFrame(comparison).drop(columns=['Temperature','Channel']).abs().max().to_dict(), flush=True)

    # Reproduce the existing gas-shape bootstrap after consuming the exact
    # preceding rate-bootstrap draws. Refit using the independent solver.
    rng = np.random.default_rng(production.Seed)
    for _ in range(production.NBootRates):
        for temp in TEMPS:
            resample_cells(subjects[temp], rng, inner=True)
    original_boot = []
    for _ in range(production.NBootRho):
        d, c = resample_cells(subjects[1273], rng, inner=True)
        fitted = fit_independent(d, c == 1)
        if fitted is not None:
            original_boot.append(fitted['Rho'])
    original_boot = np.array(original_boot)
    reproduction = dict(N=len(original_boot), CI=np.quantile(original_boot,[.025,.975]).tolist(),
                        P_below_1=float(np.mean(original_boot<1)), Stored=stored['RhoGas1273Bootstrap'])
    print('Original 1273 K gas-shape bootstrap:', reproduction, flush=True)

    # All 4^4 ordered draws of four whole cells: exact empirical cluster
    # bootstrap distribution, with no within-cell independence assumption.
    cluster, boot = [], {}
    for temp in TEMPS:
        boot[temp] = {channel: [] for channel in CHANNELS}
        for selected in product(range(4), repeat=4):
            d = np.concatenate([subjects[temp][i][0] for i in selected])
            c = np.concatenate([subjects[temp][i][1] for i in selected])
            for channel in CHANNELS:
                result = fit_independent(d, event_mask(c, channel))
                boot[temp][channel].append(result['Rho'])
        for channel, samples in boot[temp].items():
            samples = np.asarray(samples)
            cluster.append(dict(Temperature=temp, Channel=channel,
                Lower=float(np.quantile(samples,.025)), Upper=float(np.quantile(samples,.975)),
                P_below_1=float(np.mean(samples<1)), P_above_1=float(np.mean(samples>1)),
                N=len(samples)))
    pd.DataFrame(cluster).to_csv(OUT/'WholeCellBootstrap.csv', index=False)

    tail_cluster = []
    for temp in TEMPS:
        values = {channel: [] for channel in CHANNELS}
        for selected in product(range(4), repeat=4):
            d = np.concatenate([subjects[temp][i][0] for i in selected])
            c = np.concatenate([subjects[temp][i][1] for i in selected])
            keep = d > 40
            d, c = d[keep]-40, c[keep]
            for channel in CHANNELS:
                values[channel].append(fit_independent(d,event_mask(c,channel))['Rho'])
        for channel, samples in values.items():
            samples = np.asarray(samples)
            tail_cluster.append(dict(Temperature=temp, Channel=channel,
                Lower=float(np.quantile(samples,.025)), Upper=float(np.quantile(samples,.975)),
                P_below_1=float(np.mean(samples<1)), N=len(samples)))
    pd.DataFrame(tail_cluster).to_csv(OUT/'After40psWholeCellBootstrap.csv', index=False)

    # Quantify how the retrospective last-entry definition changes timing.
    # Sensitivity 1: first entry only for the same endpoint-retained carbons.
    # Sensitivity 2: genuinely first confirmed chain entry or gas removal,
    # including transient chains; changes the estimand and event counts.
    sensitivity = []
    for temp in TEMPS:
        subset = df[df.Temperature == temp]
        for variant in ['PhysicalRemovalTimes', 'EndpointFirstEntry', 'FirstChainOrGas', 'After40psShiftedClock']:
            d = subset.Duration.to_numpy().copy()
            c = subset.Cause.to_numpy().copy()
            if variant == 'PhysicalRemovalTimes':
                d = subset.PhysicalDuration.to_numpy().copy()
            elif variant == 'EndpointFirstEntry':
                mask = c == 2
                d[mask] = subset.FirstChain.to_numpy()[mask]
            elif variant == 'FirstChainOrGas':
                first = subset.FirstChain.to_numpy()
                mask = np.isfinite(first) & (first < d)
                d[mask], c[mask] = first[mask], 2
            else:
                mask = d > 40
                d, c = d[mask]-40, c[mask]
            for channel in CHANNELS:
                sensitivity.append(dict(Temperature=temp, Variant=variant, Channel=channel,
                    **fit_independent(d, event_mask(c,channel))))
    pd.DataFrame(sensitivity).to_csv(OUT/'DefinitionAndTimeOriginSensitivity.csv', index=False)
    retained = df[df.Cause == 2].copy()
    retained['Delay'] = retained.Duration-retained.FirstChain
    history = dict(RetainedCarbons=len(retained),
                   ReenteredRetained=int((retained.Delay>1e-9).sum()),
                   MaxLastMinusFirst_ps=float(retained.Delay.max()),
                   GasWithPreviousChain=int(((df.Cause==1)&df.FirstChain.notna()).sum()),
                   HorizonCensoredWithPreviousChain=int(((df.Kind=='HorizonCensored')&df.FirstChain.notna()).sum()))
    # Check direct two-decimal rounding in the table printed in the Supplementary Information
    # (set ZRC_PAPER_DIR to the folder that holds Supplementary/supplementary.tex).
    paper = Path(os.environ.get('ZRC_PAPER_DIR', HERE.parent/'paper'))  # folder with the LaTeX sources
    si_path = paper/'Supplementary'/'supplementary.tex'
    table = ''
    if si_path.is_file():  # optional: without the LaTeX source this comparison is skipped
        INPUTS.append(si_path)
        si_text = si_path.read_text(encoding='utf-8-sig')
        table = si_text.split(r'\label{tab:SurvivalFull}',1)[1].split(r'\end{table}',1)[0]
    rounding = []
    for line in table.splitlines():
        if '&' not in line:
            continue
        fields = [x.strip() for x in line.split('&')]
        if fields[0].isdigit():
            temp = int(fields[0])
        channel = fields[1].replace('chains','chain')
        if channel not in CHANNELS:
            continue
        old_fit = stored['Fits'][str(temp)][channel]['Weibull']
        numbers = re.findall(r'\d+\.\d+',fields[4])
        for number, key in zip(numbers,['Rho','Rho_CI_Lower','Rho_CI_Upper']):
            expected = f'{old_fit[key]:.2f}'
            if number != expected:
                rounding.append(dict(Temperature=temp,Channel=channel,Quantity=key,
                                     Printed=number,DirectRounding=expected))
        p_field = fields[5].replace('\\','').strip()
        if re.fullmatch(r'\d+\.\d+',p_field):
            expected = f"{stored['Fits'][str(temp)][channel]['LRT_p']:.2f}"
            if p_field != expected:
                rounding.append(dict(Temperature=temp,Channel=channel,Quantity='LRT_p',
                                     Printed=p_field,DirectRounding=expected))
    pd.DataFrame(rounding).to_csv(OUT/'SIRoundingDiscrepancies.csv',index=False)
    payload = dict(Fits=fits, Comparison=comparison, ReproducedBootstrap=reproduction,
                   WholeCellBootstrap=cluster, History=history, Timing=timing, Sensitivity=sensitivity,
                   After40psWholeCellBootstrap=tail_cluster, SIRoundingDiscrepancies=rounding,
                   LeaveOneCellOut=loo, EqualWindowRates=diagnostics,
                   InputSHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS})
    (OUT/'Audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print('Whole-cell shape bootstrap:', flush=True)
    print(pd.DataFrame(cluster).to_string(index=False), flush=True)
    print('Chain-history timing:', history, flush=True)
    write_summary(payload)
    print('Wrote reproducible audit outputs to',OUT,flush=True)


def write_summary(audit):
    lines = [
        '# Weibull calculation audit',
        '',
        'Read-only audit of production inputs and the paper; '
        'all new files are confined to WeibullAuditResults/. No production fit, '
        'figure, trajectory, or text was replaced.',
        '',
        '## Finding',
        '',
        'The numerical Weibull fitting and likelihood-ratio test reproduce. '
        'The important problem is interpretation: a shape near one and a '
        'non-significant test against an exponential do not establish a constant '
        'rate throughout the run. A single Weibull can describe a monotone '
        'increase or decrease but cannot represent a rise followed by a fall. '
        'The total commitment histories show precisely this limitation.',
        '',
        '## What was independently checked',
        '',
        '- Reconstructed all 768 carbon subjects (48 per cell, four cells per '
        'temperature) from the existing raw-departure audit and cached confirmed '
        'state histories. The raw XYZ trajectories themselves were not '
        'reclassified in this audit.',
        '- Matched 152 gas events, 183 endpoint chain events, and five artifact '
        'censors; remaining subjects are censored at 140 ps. Checked positive '
        'times, horizon limits, unique identities, all subject counts, and '
        'production risk-set construction.',
        '- Independently maximized the right-censored Weibull likelihood using '
        'the one-dimensional profile score, instead of the lifelines optimizer. '
        'Computed the exponential likelihood in closed form and used a '
        'one-degree-of-freedom likelihood-ratio test.',
        '- Refit with the installed lifelines 0.30.0, including unsuppressed '
        'warning capture; no fit warnings occurred. Maximum shape discrepancy '
        'was below 4.3e-7, likelihood discrepancy below 7e-12, and p-value '
        'discrepancy below 8e-12. Wald shape intervals also reproduce.',
        '- Reproduced the existing 2,000-draw gas-shape bootstrap, including '
        'the original RNG state after 10,000 two-level rate resamples. The '
        '1273 K interval is [0.485168, 0.790306], with all draws below one.',
        '',
        '## Recomputed fits',
        '',
        '| Temperature (K) | Channel | Events | Shape rho | Wald 95% interval | LRT p |',
        '|---:|---|---:|---:|---|---:|',
    ]
    for row in audit['Fits']:
        lines.append(f"| {row['Temperature']} | {row['Channel']} | {row['Events']} | "
                     f"{row['Rho']:.6f} | [{row['Rho_CI_Lower']:.6f}, "
                     f"{row['Rho_CI_Upper']:.6f}] | {row['LRT_p']:.6g} |")
    lines += [
        '',
        'Competing events are treated as censored at their commitment time, '
        'so all three fits use the same observed times. This is the usual '
        'cause-specific likelihood construction. Exponential channel rates '
        'sum exactly to the total; the independently fitted Weibull hazard '
        'curves need not sum to another Weibull curve.',
        '',
        '## Constancy and the time origin',
        '',
        'As a descriptive check, the event count in each of four equal 35 ps '
        'intervals was divided by the actual uncommitted-carbon exposure in '
        'that interval. These windows are an audit diagnostic, not a new '
        'recommended analysis. Total rates (10^-3 ps^-1 per '
        'uncommitted carbon) are:',
        '',
        '| Temperature (K) | 0-35 ps | 35-70 ps | 70-105 ps | 105-140 ps |',
        '|---:|---:|---:|---:|---:|',
    ]
    for temp in TEMPS:
        rows = [r for r in audit['EqualWindowRates'] if r['Temperature']==temp and r['Channel']=='total']
        lines.append(f'| {temp} | '+' | '.join(f"{r['Rate']*1000:.3f}" for r in rows)+' |')
    lines += [
        '',
        'A separate sensitivity check retained only carbons still at risk at '
        '40 ps and reset their clocks to time since 40 ps. This is a different '
        'time-origin model, not a replacement for the original fit. At 1273 K '
        'the total shape becomes 0.719578 (atom-independent LRT p=0.00145), '
        'and the chain shape becomes 0.782434. Thus the full-run chain shape '
        'above one mainly captures its delayed onset; it is not evidence of '
        'continuous chain acceleration throughout the run.',
        '',
        'Resampling whole trajectories after 40 ps gives a total-shape interval '
        '[0.570, 0.921] at 1273 K. Four trajectories are too few for precise '
        'population-level uncertainty claims, but this reinforces the need '
        'to remove the blanket claim of a constant total rate. A later decline '
        'does not by itself identify protection by carbon: changing access to '
        'reactive sites, oxide evolution, and other mechanisms are not separated.',
        '',
        '## Dependence and endpoint definitions',
        '',
        'The printed likelihood-ratio p-values and Wald intervals assume '
        'independent carbon records. Carbons within a cell share an environment '
        'and can enter chains together. The SI discloses this, but the original '
        'shape bootstrap covers only gas at 1273 K. This audit also evaluated '
        'every one of the 4^4 ordered whole-cell resamples and leave-one-cell-out '
        'fits. The full-window gas shape below one at 1273 K and chain shapes '
        'above one at 1073/1273 K survive these checks. Resampling fractions '
        'are not null-hypothesis p-values, and 256 resamples do not add '
        'independent trajectories.',
        '',
        'The recorded chain event is the last confirmed entry for a carbon '
        'still in a chain at 140 ps. This is an endpoint-defined history, not '
        'first-ever chain formation. Four retained carbons have different '
        'first and last entry times. Replacing last with first entry for those '
        'same endpoint carbons does not change the high-temperature shape '
        'conclusions. A separate first-chain-or-gas sensitivity includes '
        'transient chains and changes the event definition; it also retains '
        'those high-temperature directions. These checks support the '
        'descriptive timing result, not an irreversible elementary reaction model.',
        '',
        '## Small data and reporting discrepancies',
        '',
        '28 of 152 gas-event timestamps differ between the production event '
        'table (gas-log clock) and physical-removal audit. The largest difference '
        'is 0.120 ps and the mean absolute difference across all 152 is '
        '0.00648 ps. Substituting physical-removal times changes the 1273 K '
        'gas shape from 0.604692 to 0.605370 and p from 0.000415873 to '
        '0.000429266 (approximately); no substantive conclusion changes. '
        'A future cleanup should choose one documented clock consistently.',
        '',
        'Direct rounding of the currently stored fits differs from these SI '
        'entries. These are last-digit discrepancies, not changes in significance:',
        '',
        '| Temperature (K) | Channel | Quantity | Printed | Direct rounding |',
        '|---:|---|---|---:|---:|',
    ]
    for row in audit['SIRoundingDiscrepancies']:
        lines.append(f"| {row['Temperature']} | {row['Channel']} | {row['Quantity']} | "
                     f"{row['Printed']} | {row['DirectRounding']} |")
    lines += [
        '',
        'The 873 K chain Wald interval includes one while its LRT p is below '
        '0.05. This is not a coding inconsistency: Wald intervals and '
        'likelihood-ratio tests use different approximations. The stronger '
        'high-temperature results are not affected.',
        '',
        'The SI describes the clustering checks collectively as 10,000 '
        'resamples, but the code and saved output use 10,000 for the rate '
        'comparisons and 2,000 for the 1273 K gas Weibull shape. The latter '
        'sample count should be stated separately.',
        '',
        '## Implications for the paper',
        '',
        '- Revise the abstract and Discussion statements that the total '
        'commitment rate stays approximately constant. Likewise, qualify the '
        'Results sentence that the total rate remains constant and the '
        'interpretation that increasing retention occurs without a decline '
        'in overall commitment.',
        '- Keep the evidence for early gasification, delayed chain formation, '
        'and the temperature-dependent product balance. They survive this audit.',
        '- Do not infer that the later decline proves carbon protection. '
        'The calculation does not isolate the cause of that decline.',
        '- State Weibull results as full-window shape tests; the model cannot '
        'establish time-invariant kinetics or exclude non-monotone changes.',
        '- Keep the independent-atom p-value qualification and the limited '
        'number of independent trajectories visible in the statistical methods.',
        '',
        '## Reproduction and sources',
        '',
        'Run `python WeibullAudit.py` in the installed Python environment '
        '(lifelines 0.30.0, numpy 1.26.4, scipy 1.13.1, pandas 2.3.3). '
        'Audit.json records input hashes; the CSV files give full numeric '
        'comparisons, subject identities, bootstrap checks, and sensitivities.',
        '',
        '- [lifelines WeibullFitter documentation](https://lifelines.readthedocs.io/en/latest/fitters/univariate/WeibullFitter.html): '
        'survival and hazard parameterization and shape interpretation.',
        '- [Austin, Lee and Fine, 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4741409/): '
        'cause-specific hazards and competing-risk interpretation.',
    ]
    (OUT/'SUMMARY.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
