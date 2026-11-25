"""Descriptive trajectory review using existing audited events and frame caches.

No new kinetic fitting or production changes. Identify the six original carbon
planes, count products by starting plane, inspect simultaneous chain pairs, and
compare commitment with existing bound-oxygen and gas-supply histories.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

HERE = Path(__file__).resolve().parent
OUT = HERE / 'IndependentTrajectoryResults'
EVENTS = HERE / 'WeibullAuditResults/VerifiedSubjects.csv'
PHI = HERE / 'phiTimeseries/PhiTimeseries.csv'
SUPPLY = HERE / 'GasSupplyAudit/GasSupplyAudit.json'
TEMPS = (873, 973, 1073, 1273)
COLORS = {'total':'#31343b', 'gas':'#ca682c', 'chain':'#24769a',
          'oxygen':'#7455a2', 'free':'#407958', 'far':'#9aad9a'}
DEPTH_COLORS = ['#d7a256', '#4e96a9', '#79619b']
INPUTS = [EVENTS, PHI, SUPPLY]


def frame(z, index):
    lo, hi = z['FrameStarts'][index:index+2]
    tl, th = z['TrackedStarts'][index:index+2]
    return (z['Species'][lo:hi], z['Coords'][lo:hi].astype(float),
            z['Lattices'][index].astype(float), z['Tracked'][tl:th])


def starting_planes(z):
    species, coords, lattice, ids = frame(z, 0)
    frac = (coords[species == 2] @ np.linalg.inv(lattice))[:, 0] % 1
    assert len(frac) == 48 and len(set(ids)) == 48
    ordered = np.sort(frac)
    gaps = np.diff(np.r_[ordered, ordered[0]+1])
    origin = ordered[(np.argmax(gaps)+1) % len(ordered)]
    # Distance along the normal to constant fractional-x planes; unwrap at
    # the vacuum gap. This avoids arbitrary Cartesian wrapping at x=0.
    height = 1/np.linalg.norm(np.linalg.inv(lattice)[:, 0])
    position = ((frac-origin) % 1)*height
    order = np.argsort(position)
    labels = []
    for threshold in (.5, 1., 1.5):
        plane = np.empty(48, dtype=int)
        plane[order] = np.cumsum(np.r_[False, np.diff(position[order]) > threshold])
        assert np.array_equal(np.bincount(plane), [8]*6)
        labels.append(plane)
    assert all(np.array_equal(labels[0], x) for x in labels[1:])
    plane = labels[1]
    return pd.DataFrame({'AtomID':ids, 'InitialPlane':plane,
                         'InitialDepth':np.minimum(plane, 5-plane),
                         'InitialNormalPosition_A':position})


def interval_mean(times, values, lo, hi):
    t, v = np.asarray(times), np.asarray(values)
    inner = t[(t > lo) & (t < hi)]
    grid = np.r_[lo, inner, hi]
    return float(np.trapz(np.interp(grid, t, v), grid)/(hi-lo))


def tied_pairs(z, d):
    rows = []
    times = z['Times']/1000
    for time, group in d[d.Cause == 2].groupby('Duration'):
        if len(group) < 2:
            continue
        index = min(np.searchsorted(times, time), len(times)-1)
        species, coords, lattice, ids = frame(z, index)
        frac = coords[species == 2] @ np.linalg.inv(lattice)
        indices = [int(np.flatnonzero(ids == atom)[0]) for atom in group.AtomID]
        difference = frac[indices][:,None,:]-frac[indices][None,:,:]
        difference -= np.round(difference)
        distance = np.linalg.norm(difference @ lattice, axis=2)
        np.fill_diagonal(distance, np.inf)
        connected = bool((distance < 2.31).any(axis=1).all())
        rows.append(dict(Trajectory=d.Trajectory.iloc[0], Time_ps=time,
            Atoms=len(group), AtomIDs=','.join(map(str, group.AtomID)),
            FrameTime_ps=float(times[index]), BondedWithinGroup=connected,
            NearestWithinGroup_A=float(distance.min())))
    return rows


def moving_mean(series, n=41):
    # Two-ps display average on the existing 50-fs grid. No smoothing of events.
    return pd.Series(series).rolling(n, center=True, min_periods=1).mean().to_numpy()


def cumulative(ax, d, cause, color, label=None, lw=1.5):
    take = d.Cause > 0 if cause == 0 else d.Cause == cause
    times = np.sort(d.loc[take, 'Duration'])
    ax.step(np.r_[0,times,140], np.r_[0,np.arange(1,len(times)+1),len(times)],
            where='post', color=color, lw=lw, label=label)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
        'axes.titlesize':11, 'axes.labelsize':10, 'xtick.labelsize':8,
        'ytick.labelsize':8, 'axes.spines.top':False, 'axes.spines.right':False,
        'axes.edgecolor':'#999999', 'axes.linewidth':.6, 'pdf.fonttype':42,
        'text.color':'#252b33', 'axes.labelcolor':'#252b33'})


def focus_figure(data, histories, summaries):
    fig, axes = plt.subplots(4,4,figsize=(13.2,10.8),sharex=True,
                             gridspec_kw={'height_ratios':[1.2,1.,1.,1.]})
    fig.subplots_adjust(left=.10,right=.98,bottom=.115,top=.85,wspace=.16,hspace=.28)
    fig.suptitle('Four cells, different oxidation histories',x=.10,y=.973,
                 ha='left',fontsize=19,fontweight='bold')
    fig.text(.10,.941,'1273 K | Exact carbon commitment times, original lattice planes and oxygen histories',fontsize=11)
    handles = [Line2D([0],[0],color=COLORS[k],lw=2,label=label) for k,label in
               [('total','Total commitment'),('gas','Gas'),('chain','Retained chain')]]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.093,.918),ncol=3,frameon=False)
    labels = ['Committed carbon\n(atoms)', 'Starting carbon\nplane depth',
              'Oxygen in solid\n(atoms)', 'Estimated gas flux\n'+r'($10^{29}$ m$^{-2}$ s$^{-1}$)']
    for col in range(4):
        name=f'1273_{col+1}';d=data[data.Trajectory==name];h=histories[name]
        r=next(x for x in summaries if x['Trajectory']==name)
        axes[0,col].set_title(f'Cell {col+1}'+(' *' if r['ArtifactCensors'] else ''),pad=10,fontweight='bold')
        for cause,key in [(0,'total'),(1,'gas'),(2,'chain')]:
            cumulative(axes[0,col],d,cause,COLORS[key],lw=1.8 if cause==0 else 1.3)
        axes[0,col].set_ylim(0,40);axes[0,col].set_yticks([0,10,20,30,40])
        axes[0,col].text(.035,.94,f"{r['Gas']} gas + {r['Chain']} chain",transform=axes[0,col].transAxes,
                         va='top',fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=2))
        ranks = {int(a):j for dep,g in d.groupby('InitialDepth') for j,a in enumerate(sorted(g.AtomID))}
        for cause,key,marker in [(1,'gas','o'),(2,'chain','^')]:
            part=d[d.Cause==cause]
            y=part.InitialDepth.to_numpy()+np.array([(ranks[int(a)]-7.5)/32 for a in part.AtomID])
            axes[1,col].scatter(part.Duration,y,c=COLORS[key],marker=marker,s=23,linewidths=.3,edgecolors='white',zorder=3)
        for y in [.5,1.5]:axes[1,col].axhline(y,color='#dddddd',lw=.7)
        axes[1,col].set_ylim(2.45,-.45);axes[1,col].set_yticks([0,1,2]);axes[1,col].set_yticklabels(['Outer','Second','Central'])
        t=h['Time_ps'].to_numpy()
        axes[2,col].plot(t,h.NScaleO,color=COLORS['oxygen'],alpha=.20,lw=.5)
        axes[2,col].plot(t,moving_mean(h.NScaleO),color=COLORS['oxygen'],lw=1.5)
        axes[2,col].set_ylim(0,75);axes[2,col].set_yticks([0,20,40,60])
        axes[3,col].plot(h.GasTime_ps,moving_mean(h.Jfree),color=COLORS['free'],lw=1.3,label='Free O2')
        axes[3,col].plot(h.GasTime_ps,moving_mean(h.Jfar),color=COLORS['far'],lw=1.1,ls='--',label='Far O2')
        axes[3,col].set_ylim(0,8);axes[3,col].set_yticks([0,2,4,6,8]);axes[3,col].set_xlabel('Time (ps)')
        if col==3:axes[3,col].legend(frameon=False,fontsize=8,loc='upper right')
        for row in range(4):
            ax=axes[row,col];ax.set_xlim(0,140);ax.set_xticks([0,35,70,105,140])
            ax.grid(axis='y',color='#ededed',lw=.7);ax.set_axisbelow(True)
            if col==0:ax.set_ylabel(labels[row])
            else:ax.tick_params(labelleft=False)
    fig.text(.10,.068,'Each marker is one carbon. Depth refers to its starting plane, not its position at commitment.',fontsize=9)
    fig.text(.10,.047,'Oxygen curves use a 2 ps display average; commitment is unsmoothed. Gas flux is an estimated supply, not oxygen arrival at the interface.',fontsize=8.7)
    fig.text(.10,.026,'* Cell 3 has three previously audited erroneous carbon removals. They are excluded from commitments, but their dynamical effects remain.',fontsize=8.7)
    return fig


def overview_figure(data):
    fig,axes=plt.subplots(4,4,figsize=(13.2,10.8),sharex=True,sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.87,wspace=.17,hspace=.25)
    fig.suptitle('Commitment in all sixteen trajectories',x=.10,y=.969,ha='left',fontsize=19,fontweight='bold')
    fig.text(.10,.938,'Rows: temperature | Columns: independent cell | Same axes and event definition throughout',fontsize=11)
    handles=[Line2D([0],[0],color=COLORS[k],lw=2,label=l) for k,l in [('total','Total'),('gas','Gas'),('chain','Retained chain')]]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.094,.921),ncol=3,frameon=False)
    for row,temp in enumerate(TEMPS):
        for col in range(4):
            d=data[data.Trajectory==f'{temp}_{col+1}'];ax=axes[row,col]
            for cause,key in [(0,'total'),(1,'gas'),(2,'chain')]:cumulative(ax,d,cause,COLORS[key],lw=1.6 if cause==0 else 1.2)
            if row==0:ax.set_title(f'Cell {col+1}',fontweight='bold')
            if col==0:ax.set_ylabel(f'{temp} K\nCommitted carbon')
            if row==3:ax.set_xlabel('Time (ps)')
            counts=d.Cause.value_counts();ax.text(.04,.92,f"{counts.get(1,0)} gas + {counts.get(2,0)} chain",transform=ax.transAxes,va='top',fontsize=9)
            ax.set_xlim(0,140);ax.set_ylim(0,40);ax.set_xticks([0,35,70,105,140]);ax.set_yticks([0,10,20,30,40]);ax.grid(axis='y',color='#ededed',lw=.6)
    fig.text(.10,.046,'Steps count gas desorption and entry into chains retained at 140 ps. Plateaus do not by themselves establish passivation.',fontsize=9)
    return fig


def origin_figure(data):
    fig,axes=plt.subplots(1,3,figsize=(13.2,8.5),sharey=True)
    fig.subplots_adjust(left=.11,right=.98,bottom=.14,top=.82,wspace=.14)
    fig.suptitle('Where did the committed carbon start?',x=.11,y=.96,ha='left',fontsize=19,fontweight='bold')
    fig.text(.11,.922,'Six initial carbon planes, grouped symmetrically into three depths (16 carbon atoms per depth per cell)',fontsize=10.5)
    names=[f'{t}_{s}' for t in TEMPS for s in range(1,5)]
    for ax,cause,title in zip(axes,[0,1,2],['Total commitment','Gas','Retained chain']):
        left=np.zeros(16)
        for depth,label,color in zip(range(3),['Outer','Second','Central'],DEPTH_COLORS):
            values=[]
            for name in names:
                d=data[data.Trajectory==name];event=d.Cause>0 if cause==0 else d.Cause==cause
                values.append(int((event&(d.InitialDepth==depth)).sum()))
            ax.barh(np.arange(16),values,left=left,height=.71,color=color,label=label)
            left+=values
        ax.set_title(title,fontweight='bold');ax.set_xlim(0,40);ax.set_xticks([0,10,20,30,40]);ax.set_xlabel('Carbon atoms at 140 ps')
        ax.set_yticks(np.arange(16));ax.set_yticklabels([n.replace('_',' K / cell ') for n in names]);ax.invert_yaxis()
        for line in [3.5,7.5,11.5]:ax.axhline(line,color='#cccccc',lw=.6)
        ax.set_axisbelow(True);ax.grid(axis='x',color='#eeeeee',lw=.6)
    axes[0].legend(loc='lower left',bbox_to_anchor=(-.02,1.06),frameon=False,ncol=3)
    fig.text(.11,.078,'151 of 152 gasified carbons originated in the outer planes; this does not specify where they were when gasified.',fontsize=10)
    fig.text(.11,.045,'Artifact removals (2 in 973 K / cell 1, 3 in 1273 K / cell 3) are excluded. Initial depth is provenance, not a reconstructed moving interface.',fontsize=9)
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    events=pd.read_csv(EVENTS);phi=pd.read_csv(PHI);gas=json.loads(SUPPLY.read_text())
    tables=[];pairs=[];summaries=[];window_rows=[];histories={}
    # Import an existing classifier solely to verify selected oxygen-cache values.
    import Refresh140 as production
    for name,d in events.groupby('Trajectory',sort=False):
        cache=HERE/f'Refresh140/FrameCache/{name}_grid50fs_140ps.npz';INPUTS.append(cache)
        with np.load(cache) as z:
            initial=starting_planes(z)
            joined=d.merge(initial,on='AtomID',validate='one_to_one')
            tables.append(joined);new_pairs=tied_pairs(z,joined);pairs.extend(new_pairs)
            h=phi[phi.TrajectoryId==name].sort_values('Time_ps').reset_index(drop=True).copy()
            np.testing.assert_allclose(h.Time_ps,z['Times']/1000,atol=1e-10)
            for index in [0,700,1400,1800,2100,2700,2800]:
                sp,co,la,_=frame(z,index)
                assert production.ScaleOxygenCount(sp,co,la)==h.NScaleO.iloc[index]
            series=gas['PerTrajectory'][name]['Series']
            assert len(series['Times_ps'])==len(h)
            h['GasTime_ps']=series['Times_ps'];h['Jfree']=np.asarray(series['Jfree_m2s'])/1e29;h['Jfar']=np.asarray(series['Jfar_m2s'])/1e29
            histories[name]=h
            gas_events=joined[joined.Cause==1];chain_events=joined[joined.Cause==2];all_events=joined[joined.Cause>0]
            r=dict(Trajectory=name,Temperature=int(joined.Temperature.iloc[0]),
                Gas=len(gas_events),Chain=len(chain_events),Total=len(all_events),
                ArtifactCensors=int((joined.Kind=='ArtifactRemoval').sum()),
                FirstChain_ps=float(chain_events.Duration.min()),LastCommitment_ps=float(all_events.Duration.max()),
                LastGas_ps=float(gas_events.Duration.max()),TiedChainAtoms=sum(x['Atoms'] for x in new_pairs))
            for depth,label in enumerate(['Outer','Second','Central']):
                r[label+'Committed']=int((all_events.InitialDepth==depth).sum())
                r[label+'Gas']=int((gas_events.InitialDepth==depth).sum())
                r[label+'Chain']=int((chain_events.InitialDepth==depth).sum())
            for lo,hi in [(0,35),(35,70),(70,105),(105,140),(35,87.5),(87.5,140),(90,95),(135,140)]:
                row=dict(Trajectory=name,Start_ps=lo,End_ps=hi,
                    GasEvents=int(((gas_events.Duration>lo)&(gas_events.Duration<=hi)).sum()),
                    ChainEvents=int(((chain_events.Duration>lo)&(chain_events.Duration<=hi)).sum()),
                    MeanBoundO=interval_mean(h.Time_ps,h.NScaleO,lo,hi),
                    MeanJfree1e29=interval_mean(h.GasTime_ps,h.Jfree,lo,hi),
                    MeanJfar1e29=interval_mean(h.GasTime_ps,h.Jfar,lo,hi))
                window_rows.append(row)
                if (lo,hi)==(90,95):r['BoundO_90to95']=row['MeanBoundO']
                if (lo,hi)==(135,140):r['BoundO_135to140']=row['MeanBoundO']
                if (lo,hi)==(35,87.5):r['Jfree_early']=row['MeanJfree1e29'];r['Jfar_early']=row['MeanJfar1e29']
                if (lo,hi)==(87.5,140):r['Jfree_late']=row['MeanJfree1e29'];r['Jfar_late']=row['MeanJfar1e29']
            summaries.append(r)
    data=pd.concat(tables,ignore_index=True)
    assert len(data)==768 and data[data.Cause==1].shape[0]==152 and data[data.Cause==2].shape[0]==183
    assert all(x['BondedWithinGroup'] for x in pairs)
    data.to_csv(OUT/'CarbonOrigins.csv',index=False)
    pd.DataFrame(pairs).to_csv(OUT/'SimultaneousChainPairs.csv',index=False)
    pd.DataFrame(summaries).to_csv(OUT/'TrajectorySummary.csv',index=False)
    pd.DataFrame(window_rows).to_csv(OUT/'OxygenAndEventWindows.csv',index=False)
    style()
    figures=[focus_figure(data,histories,summaries),overview_figure(data),origin_figure(data)]
    with PdfPages(OUT/'IndependentTrajectories.pdf') as pdf:
        for fig,name in zip(figures,['1273K_Trajectories','AllTrajectoryCommitments','CarbonStartingPlanes']):
            pdf.savefig(fig,facecolor='white');fig.savefig(OUT/f'{name}.png',dpi=175,facecolor='white');plt.close(fig)
    metadata=dict(InputSHA256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUTS},
        OriginalPlaneCheck='Six planes of 8 carbon atoms each in all 16 cells; invariant at gap thresholds 0.5,1.0,1.5 A',
        OxygenCacheCheck='112 selected frames reclassified with the production function; all matched',
        SimultaneousChainGroups=len(pairs),SimultaneousChainAtoms=sum(x['Atoms'] for x in pairs),
        AllTiedGroupsBonded=True,EventDefinition='Gas desorption or last confirmed chain entry for carbon retained at 140 ps',
        Changes='Only this script and IndependentTrajectoryResults; no production edits')
    (OUT/'Run.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    write_report(data,summaries,pairs)
    print(pd.DataFrame(summaries).query('Temperature==1273').to_string(index=False))
    print('All-temperature gas by starting depth:',data[data.Cause==1].groupby('InitialDepth').size().to_dict())
    print('All-temperature chains by starting depth:',data[data.Cause==2].groupby('InitialDepth').size().to_dict())


def write_report(data,summaries,pairs):
    lines=['# Independent trajectory review','',
        'Descriptive follow-up to the commitment slowdown check. No additional kinetic '
        'models or hypothesis tests were fitted. The production outputs are unchanged.','',
        '## Main observations','',
        '| 1273 K cell | Gas | Chains | Total | Outer / second / central origin | Last commitment (ps) |',
        '|---:|---:|---:|---:|---|---:|']
    for r in summaries:
        if r['Temperature']==1273:
            lines.append(f"| {r['Trajectory'].split('_')[1]} | {r['Gas']} | {r['Chain']} | {r['Total']} | "
                f"{r['OuterCommitted']} / {r['SecondCommitted']} / {r['CentralCommitted']} | {r['LastCommitment_ps']:.2f} |")
    for cause,label in [(1,'gas'),(2,'retained chains')]:
        counts=data[data.Cause==cause].groupby('InitialDepth').size().reindex(range(3),fill_value=0)
        lines+=['',f"Across all temperatures, carbon committed to {label} has outer / second / central counts "
                f"{counts.iloc[0]} / {counts.iloc[1]} / {counts.iloc[2]}."]
    lines+=['',
        'Gasification overwhelmingly involves atoms that began in the original outer '
        'carbon planes. Deeper starting planes contribute mainly retained chains. '
        'This is a statement about atomic provenance, not the location of gasification '
        'or proof that gasification at depth is impossible. The one second-plane '
        'gas event is atom 32 in 1273_4 at 119.28 ps.','',
        'At 1273 K, cells 1 and 2 commit 21 and 22 carbons; cells 3 and 4 commit 34 '
        'and 28. The difference mainly comes from retained chains and conversion '
        'of carbon that started in deeper planes. Cell 1 has no commitments after '
        '89.021 ps, but its oxygen count continues to rise.','',
        '| Cell | Bound O, 90-95 ps | Bound O, 135-140 ps | Early free flux | Late free flux |',
        '|---:|---:|---:|---:|---:|']
    for r in summaries:
        if r['Temperature']==1273:
            lines.append(f"| {r['Trajectory'].split('_')[1]} | {r['BoundO_90to95']:.2f} | {r['BoundO_135to140']:.2f} | {r['Jfree_early']:.3f} | {r['Jfree_late']:.3f} |")
    lines+=['','Flux units: 10^29 m^-2 s^-1. Early=35-87.5 ps; late=87.5-140 ps. '
        'All four 1273 K cells have higher estimated free-O2 flux late. '
        'The far-O2 estimate also does not show a common late decline (cell 4 is nearly unchanged). '
        'There is no common shortage of gas-phase oxygen supply in these histories. '
        'These estimates do not measure delivery through the oxide to reactive carbon; '
        'feedback between reaction, gas population and dosing prevents a causal inference.','',
        f"{sum(x['Atoms'] for x in pairs)} of 183 chain commitments occur in "
        f"{len(pairs)} exactly simultaneous pairs. Every pair is mutually bonded "
        'at the first cached frame at/after confirmation (C-C cutoff 2.31 A). '
        'At 1273 K this accounts for 38 of 69 chain commitments. '
        'These are coupled carbon records rather than independent elementary events. '
        'No new pair-based rate is introduced.','',
        '## Interpretation','',
        'The most useful insight is that the population remaining uncommitted changes '
        'in starting depth and likely accessibility as conversion proceeds. '
        'Simply counting the remaining carbon cannot remove these differences. '
        'The observed decline in gas commitment can therefore include exhaustion '
        'of the original outer-carbon population, alongside changes in local '
        'chemistry and oxygen access. This does not isolate a cause.','',
        'This depth difference does not remove the temperature-dependent '
        'partitioning observation. Among committed carbon from the original '
        'outer planes, the chain fraction is 15/(47+15)=24% at 973 K and '
        '26/(35+26)=43% at 1273 K. These are descriptive product fractions, '
        'not a new significance test. The whole-slab shift also includes '
        'a larger contribution from deeper original planes at high temperature.','',
        'The individual histories do not show a single shared stopping time or '
        'a universal late decrease. Bursts, pauses and continued deeper-plane '
        'commitment coexist across cells. Oxygen uptake and carbon commitment '
        'can separate in time. Thus a carbon-commitment plateau should not '
        'be called cessation of oxidation.','',
        'Cell 3 at 1273 K has three known erroneous carbon removals, and cell 1 '
        'at 973 K has two. These atoms are censored, excluded from all product '
        'counts and separately retained in CarbonOrigins.csv. Their physical '
        'effects on subsequent dynamics cannot be removed analytically. '
        'In particular, the most converted 1273 K cell is not a clean mechanistic '
        'control for the less converted cells.','',
        '## Short, reproducible methods','',
        '1. Use the audited gas/retained-chain event ledger and existing 50 fs '
        'frame caches; do not redefine any commitment.',
        '2. At the first frame, unwrap the slab-normal fractional coordinate '
        'across the largest vacuum gap. Separate its six carbon planes at '
        'gaps greater than 1 A. Every plane has eight carbons; thresholds '
        '0.5 and 1.5 A give identical assignments for all cells. Pair the '
        'two outer planes, the next two, and the central two. These are '
        'initial crystallographic planes, not manually drawn final regions.',
        '3. Join plane assignments to the original tracked atom IDs. Count '
        'gas, chain and censored outcomes by initial plane depth.',
        '4. For exactly tied chain timestamps, check the existing 2.31 A '
        'minimum-image C-C bond criterion at the first cached frame at/after '
        'confirmation. Do not merge them into a new event definition.',
        '5. Reuse existing network-bound oxygen counts and free/far oxygen '
        'supply estimates. Independently reclassify seven cached frames per '
        'cell to verify oxygen counts (112 exact matches). Window means '
        'integrate the sampled values in physical time. Two-ps centered '
        'averages are used for display only.', '',
        '## Reproduce','', '`python IndependentTrajectoryReview.py`','',
        'Run.json stores input hashes. IndependentTrajectories.pdf contains '
        'the 1273 K histories, all sixteen commitment curves, and product '
        'counts by starting plane. CSVs contain all identities and summaries.','']
    (OUT/'SUMMARY.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
