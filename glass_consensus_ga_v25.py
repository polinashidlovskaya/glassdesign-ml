"""GlassDesign ML 2.5 candidate: 2x2 factorial GA search.

All four arms query both GlassDesign and GlassNet on identical-size batches.
Two factors: global/adaptive family choice, GD-only/dual-consensus feedback.
No fitting, no access to training/held-out data. The research novelty is the
online use of two frozen predictors, not a claim of experimental validation.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from glass_hybrid_v21 import _feasible, _penalty, _project, _key

GN_NAMES={
    'Density_g_cm3':'GN_Density_g_cm3',
    'RefractiveIndex':'GN_RefractiveIndex',
    'Tg_C':'GN_Tg_C',
    'Tsoft_C':'GN_Tsoft_C',
}
PROPS=tuple(GN_NAMES)


def _gn_view(frame):
    return frame[list(GN_NAMES.values())].rename(columns={v:k for k,v in GN_NAMES.items()})


def scored(frame, criteria, feedback):
    """Fitness for minimization, no access to unseen predictions."""
    gd_pen=_penalty(frame,criteria,True)
    gd_pass=_feasible(frame,criteria,True)
    if feedback=='both':
        net=_gn_view(frame)
        gn_pen=_penalty(net,criteria,True)
        gn_pass=_feasible(net,criteria,True)
        violation=np.maximum(gd_pen,gn_pen)+.25*(gd_pen+gn_pen)
        is_hit=gd_pass & gn_pass
    elif feedback=='gd':
        violation=gd_pen.copy()
        is_hit=gd_pass
    else: raise ValueError('feedback must be gd or both')
    # Small tie-break in fully feasible region, same coefficient for all arms.
    close=np.zeros(len(frame),float)
    for name,(low,high) in criteria.items():
        if low is None or high is None:continue
        center=(float(low)+float(high))*.5
        span=max(float(high)-float(low),1e-4)
        close+=np.abs(frame[name].to_numpy(float)-center)/span
        if feedback=='both':
            close+=np.abs(frame[GN_NAMES[name]].to_numpy(float)-center)/span
    if feedback=='both': close*=.5
    return violation+.02*close, is_hit


def generate(families, pilot, gd_model, gn_model, oxides, criteria, seed,
             family_policy='adaptive', feedback='both', budget=120, batch_size=16):
    """Return exactly `budget` new, unique, bounded compositions.

    gd_model and gn_model are immutable oracle wrappers with `.rows` count.
    Both are called on every new candidate by *all* arms, avoiding an unfair
    difference in model-evaluation budgets. All models see the same compositions
    within each arm and different arms receive the same pilot.
    """
    from glass_core import sample_composition
    import glass_benchmark_v21 as bm
    if family_policy not in ('global','adaptive') or feedback not in ('gd','both'):
        raise ValueError('Bad factorial arm')
    if budget<=0 or batch_size<=0 or not families or len(pilot)==0:
        raise ValueError('Invalid budget or pilot')
    if not set(PROPS).issubset(pilot) or not set(GN_NAMES.values()).issubset(pilot):
        raise ValueError('Pilot must contain both prediction systems')
    rng=np.random.default_rng(int(seed)+42100)
    oxides=list(oxides)
    oxide_idx={n:i for i,n in enumerate(oxides)}
    family_names=list(families)
    bounds={f:(list(spec),np.array([v[0] for v in spec.values()],float),
               np.array([v[1] for v in spec.values()],float)) for f,spec in families.items()}
    pop=pilot.copy().reset_index(drop=True)
    pop['_fitness'],pop['_feedback_pass']=scored(pop,criteria,feedback)
    seen={_key(x) for x in pilot[oxides].to_numpy(float)}
    pilot_hit={f:float(pop.loc[pop.Glass_family.eq(f),'_feedback_pass'].mean()) for f in family_names}
    pilot_pen={f:float(pop.loc[pop.Glass_family.eq(f),'_fitness'].min()) for f in family_names}
    nnew={f:0 for f in family_names}
    nhit={f:0 for f in family_names}
    next_id=int(pilot.Candidate_ID.max())+1
    produced=0
    rounds=0
    records=[]

    def choose_bandit():
        n=sum(nnew.values())
        val=[]
        for f in family_names:
            prior=(nhit[f]+2*pilot_hit[f]+1)/(nnew[f]+4)
            explore=.35*math.sqrt(math.log(n+len(family_names)+1)/(nnew[f]+1))
            prox=.10/(1+pilot_pen[f])
            val.append(prior+explore+prox)
        z=np.array(val)
        p=np.exp((z-z.max())/.18)
        return family_names[int(rng.choice(len(family_names),p=p/p.sum()))]

    def family_choice():
        if family_policy=='adaptive':return choose_bandit()
        return family_names[int(rng.integers(len(family_names)))]

    def tournament(pool):
        if pool.empty:raise RuntimeError('Missing parents')
        choices=rng.integers(0,len(pool),size=3)
        fits=pool['_fitness'].to_numpy(float)
        return pool.iloc[int(min(choices,key=lambda i:(fits[int(i)],int(i))))]

    while produced<budget:
        k=min(batch_size,budget-produced)
        batch=[]
        tries=0
        while len(batch)<k:
            tries+=1
            if tries>30000:raise RuntimeError('No unique recipes available')
            if rng.random()<.18:
                family=family_choice()
                comp=sample_composition(families[family],rng)
                child=np.array([comp[o] for o in bounds[family][0]],float)
            else:
                if family_policy=='adaptive' and rng.random()<.25:
                    family=family_choice()
                    first=tournament(pop.loc[pop.Glass_family.eq(family)].reset_index(drop=True))
                else:
                    first=tournament(pop)
                    family=str(first.Glass_family)
                names,lo,hi=bounds[family]
                child=first[names].to_numpy(float).copy()
                mates=pop.loc[pop.Glass_family.eq(family)].reset_index(drop=True)
                if len(mates)>1 and rng.random()<.80:
                    second=tournament(mates)[names].to_numpy(float)
                    alpha=float(rng.uniform(.08,.92))
                    child=alpha*child+(1-alpha)*second
                if rng.random()<.92:
                    scale=float(rng.choice([1.5,3.,6.,11.],p=[.2,.4,.3,.1]))
                    child+=rng.normal(0,scale,size=len(child))
                child=_project(child,lo,hi)
            names,lo,hi=bounds[family]
            full=np.zeros(len(oxides),float)
            for n,v in zip(names,child):full[oxide_idx[n]]=v
            signature=_key(full)
            if signature in seen:continue
            seen.add(signature)
            row={o:float(full[i]) for i,o in enumerate(oxides)}
            row.update(Glass_family=family,Candidate_ID=next_id,
                       Search_stage=f'{family_policy}_{feedback}_GA_364',
                       Search_round=rounds)
            next_id+=1
            batch.append(row)
        frame=pd.DataFrame(batch)
        before_gd=gd_model.rows
        before_gn=gn_model.rows
        gd=gd_model(frame.copy())
        net=gn_model(frame.copy())
        if gd_model.rows-before_gd!=k or gn_model.rows-before_gn!=k:
            raise RuntimeError('Unequal model evaluation budgets')
        if not gd.Candidate_ID.equals(frame.Candidate_ID) or not net.Candidate_ID.equals(frame.Candidate_ID):
            raise RuntimeError('Candidate alignment mismatch')
        for p in PROPS:frame[p]=gd[p].to_numpy(float)
        for p in GN_NAMES.values():frame[p]=net[p].to_numpy(float)
        frame['GD_pass']=_feasible(frame,criteria,True)
        frame['GN_pass']=_feasible(_gn_view(frame),criteria,True)
        frame['Both_pass']=frame.GD_pass & frame.GN_pass
        frame['Goal_pass']=frame.GD_pass
        frame['Goal_penalty']=_penalty(frame,criteria,True)
        frame['Delta_T_C']=frame.Tsoft_C-frame.Tg_C
        frame['_fitness'],frame['_feedback_pass']=scored(frame,criteria,feedback)
        for family,g in frame.groupby('Glass_family'):
            nnew[family]+=len(g)
            nhit[family]+=int(g['_feedback_pass'].sum())
        records.append(frame)
        produced+=k
        rounds+=1
        union=pd.concat([pop,frame],ignore_index=True)
        union=union.sort_values(['_fitness','Candidate_ID'],kind='stable')
        champs=union.drop_duplicates('Glass_family',keep='first')
        rest=union.loc[~union.Candidate_ID.isin(champs.Candidate_ID)]
        pop=pd.concat([champs,rest.head(max(0,len(pilot)-len(champs)))],ignore_index=True)
    out=pd.concat(records,ignore_index=True).drop(columns=['_fitness','_feedback_pass'])
    bm._check_output(out,oxides,budget,pilot,criteria,True)
    for family,part in out.groupby('Glass_family'):
        names,lo,hi=bounds[family]
        X=part[names].to_numpy(float)
        if (X<lo-1e-5).any() or (X>hi+1e-5).any():
            raise RuntimeError('Composition outside family constraints')
    if (out.GD_pass & out.GN_pass).sum()!=out.Both_pass.sum():
        raise RuntimeError('Consensus mask inconsistent')
    return out
