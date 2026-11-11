"""Scientific-aware, goal-centered and composition-diverse TOP-15 presentation.

No model training or alteration of success flags. Screening tier is heuristic,
not calibrated prediction confidence. The shortlist is a display ordering only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROPERTIES = ('Density_g_cm3','RefractiveIndex','Tg_C','Tsoft_C')
GN = {'Density_g_cm3':'GN_Density_g_cm3', 'RefractiveIndex':'GN_RefractiveIndex',
      'Tg_C':'GN_Tg_C', 'Tsoft_C':'GN_Tsoft_C'}
GRADE_COST = {'A':0., 'B':0.12,'C':0.38,'D':0.55}


def prepare_candidates(good, science, criteria, oxide_columns, literature=None):
    if good.empty:
        return good.copy(), []
    if 'Run_Unique_ID' not in good or good.Run_Unique_ID.duplicated().any():
        raise ValueError('Нет уникальных идентификаторов кандидатов')
    candidates=good.copy().reset_index(drop=True)
    ids=candidates.Run_Unique_ID.astype(str)
    if science is not None and not science.empty:
        if science.Run_Unique_ID.duplicated().any():
            raise ValueError('Дубли научного рейтинга')
        sci=science.copy().set_index(science.Run_Unique_ID.astype(str))
        t=sci.reindex(ids)
        candidates['Scientific_tier_403']=t['Screening_tier'].fillna('').astype(str).str[:1].to_numpy()
    else:
        candidates['Scientific_tier_403']=''
    grades=candidates.Scientific_tier_403.map(GRADE_COST).fillna(0.27).to_numpy(float)
    center_components=[]
    weights=[]
    for prop in PROPERTIES:
        if prop not in criteria or prop not in candidates:
            continue
        lo,hi=criteria[prop]
        if lo is None or hi is None or float(hi)<=float(lo):
            continue
        center=(float(lo)+float(hi))*0.5
        scale=float(hi)-float(lo)
        gd=pd.to_numeric(candidates[prop],errors='coerce').to_numpy(float)
        gncol=GN[prop]
        if gncol in candidates:
            gn=pd.to_numeric(candidates[gncol],errors='coerce').to_numpy(float)
            # Equal treatment: optimize a consensus prediction inside user bounds.
            proximity=(np.abs(gd-center)+np.abs(gn-center)) / (2*scale)
        else:
            proximity=np.abs(gd-center)/scale
        center_components.append(proximity)
        weights.append(1.)
    if center_components:
        center_score=np.mean(np.vstack(center_components),axis=0)
    else:
        center_score=np.zeros(len(candidates))
    center_score=np.where(np.isfinite(center_score),center_score,100.)
    # A/C labels constrain priority, but they are not probabilities of success.
    literature_cost=np.full(len(candidates),0.05,dtype=float)
    if literature is not None and not literature.empty and {'Run_Unique_ID','Property','Distance_halfL1_mol'}.issubset(literature):
        distances=literature.groupby(['Run_Unique_ID','Property']).Distance_halfL1_mol.min().unstack()
        typical=distances.median(axis=1).reindex(ids).to_numpy(float)
        literature_cost=np.where(np.isfinite(typical),np.clip(typical/80.,0.,0.1),0.1)
    # Literature distance indicates presence of measured relatives, not proof of
    # model precision or glass formation. It is only a small ranking tie-breaker.
    base=center_score + grades + literature_cost
    candidates['Distance_to_user_interval_centers_403']=center_score
    candidates['Priority_penalty_403']=grades
    candidates['Literature_proximity_penalty_403']=literature_cost
    candidates['Base_priority_403']=base
    oxides=[o for o in oxide_columns if o in candidates]
    if not oxides:
        raise ValueError('Нет столбцов оксидов для отбора разнообразных составов')
    X=candidates[oxides].to_numpy(dtype=float)
    if not np.isfinite(X).all() or not np.allclose(X.sum(axis=1),100.,atol=.15):
        raise ValueError('Нарушена сумма молярных процентов')
    return candidates,oxides


def recommend_top_diverse(good, science, criteria, oxide_columns, n=15,
                          min_separation_mol=3.,literature=None):
    """Greedy, deterministic 15 with center proximity, tier, families, diversity.

    No fake A confidence or missing diagnostic imputation: unknown grade costs
    more than B but less than C. Family bonus is small, not a hard constraint.
    """
    data,oxides=prepare_candidates(good,science,criteria,oxide_columns,literature=literature)
    if data.empty:
        return data,data
    n=min(int(n),len(data))
    X=data[oxides].to_numpy(float)
    base=data.Base_priority_403.to_numpy(float)
    fam=data.Glass_family.astype(str).to_numpy() if 'Glass_family' in data else np.array(['']*len(data))
    chosen=[]
    chosen_families=set()
    remaining=set(range(len(data)))
    family_counts={}
    # Include a genuinely different chemical family when its best candidate is
    # reasonably close to the overall target, without claiming that all
    # families are equally well supported experimentally.
    best_global=float(np.min(base))
    if n>=len(set(fam)) and n>=4:
        distinct=sorted(set(fam),key=lambda family: min(base[i] for i in remaining if fam[i]==family))
        for family in distinct:
            available=[i for i in remaining if fam[i]==family]
            pick=min(available,key=lambda i:(base[i],str(data.iloc[i].Run_Unique_ID)))
            if base[pick] <= best_global+0.25:
                chosen.append(pick)
                chosen_families.add(family)
                family_counts[family]=1
                remaining.remove(pick)
            if len(chosen)>=n:break
    max_family=max(2,int(np.ceil(n*0.6))) if len(set(fam))>1 else n
    while len(chosen)<n:
        proposals=[]
        candidate_pool=[i for i in remaining if family_counts.get(fam[i],0)<max_family]
        if not candidate_pool:candidate_pool=sorted(remaining)
        for i in candidate_pool:
            separation = (min(.5*np.abs(X[i]-X[j]).sum() for j in chosen)
                          if chosen else np.inf)
            duplicate_cost = (0.20 * max(0.,min_separation_mol-separation)/min_separation_mol
                              if chosen else 0.)
            same_family_cost=0.006 * family_counts.get(fam[i],0)
            key=(base[i]+duplicate_cost+same_family_cost,base[i],str(data.iloc[i].Run_Unique_ID))
            proposals.append((key,i,separation))
        _,best,_=min(proposals,key=lambda t:t[0])
        chosen.append(best)
        chosen_families.add(fam[best]);remaining.remove(best)
        family_counts[fam[best]]=family_counts.get(fam[best],0)+1
    first=data.iloc[chosen].sort_values(['Base_priority_403','Run_Unique_ID'],kind='stable').copy().reset_index(drop=True)
    first.insert(0,'Display_rank_403',np.arange(1,len(first)+1))
    leftover=data.drop(index=chosen).sort_values(['Base_priority_403','Run_Unique_ID'],kind='stable').reset_index(drop=True)
    if len(first)!=n or len(leftover)+len(first)!=len(data):
        raise AssertionError('Пропали составы при формировании главной подборки')
    if set(first.Run_Unique_ID)&set(leftover.Run_Unique_ID):
        raise AssertionError('Составы повторяются между основным списком и остальными')
    return first,leftover
