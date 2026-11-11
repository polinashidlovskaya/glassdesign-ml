"""GlassDesign ML step 385: an additive, auditable scientific screening layer.

Does NOT fit/retrain models, edit model files or claim glass formation.
Works with existing step384 screened CSV or directly with a step381 run.
Requires original project data only for live fresh screening.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROPERTIES = ('Density_g_cm3', 'RefractiveIndex', 'Tg_C', 'Tsoft_C')
DISTANCES = {
    'Density_g_cm3': 'Nearest_rho_halfL1_mol',
    'RefractiveIndex': 'Nearest_n_halfL1_mol',
    'Tg_C': 'Nearest_Tg_halfL1_mol',
    'Tsoft_C': 'Nearest_Tsoft_halfL1_mol',
}
OXIDE_PATTERN = re.compile(r'^[A-Z][a-z]?\d*O\d*$')
# These cutoffs are transparent operational heuristics, NOT validated error bounds.
REF_DIST_MOL = 5.0
LOCAL_TG_JUMP_C = 5.0
TWO_MODEL_TG_GAP_C = 10.0
GN_SOFT_GAP_C = 5.0
DIVERSITY_HALF_L1_MOL = 2.0


def _oxide_cols(df):
    return [c for c in df.columns if OXIDE_PATTERN.fullmatch(str(c))]


def _validate_table(df, label):
    needed = {'Candidate_ID', 'Tg_C', 'Tsoft_C', 'GN_Tg_C', 'GN_Tsoft_C'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{label}: отсутствуют поля {sorted(missing)}')
    if df.empty or df.Candidate_ID.isna().any() or df.Candidate_ID.duplicated().any():
        raise ValueError(f'{label}: пустой или повторяющийся Candidate_ID')
    oxides = _oxide_cols(df)
    x = df[oxides].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(x).all() or (x < -1e-7).any() or not np.allclose(x.sum(axis=1), 100, atol=1e-4, rtol=0):
        raise ValueError(f'{label}: некорректные мольные доли (сумма должна быть 100%)')
    return oxides


def verify_and_combine(source: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Safely join 384 audit with 381 export; prevent accidental wrong-run ID joins."""
    ox = _validate_table(source, 'Исходный CSV')
    _validate_table(audit, 'Аудит')
    if set(source.Candidate_ID) != set(audit.Candidate_ID) or len(source) != len(audit):
        raise ValueError('Разные Candidate_ID: аудит относится к другому запуску')
    a = audit.set_index('Candidate_ID').sort_index()
    s = source.set_index('Candidate_ID').sort_index()
    for c in ['Tg_C', 'Tsoft_C', 'GN_Tg_C', 'GN_Tsoft_C']:
        if not np.allclose(s[c], a[c], rtol=0, atol=2e-4, equal_nan=False):
            raise ValueError(f'Аудит и исходный CSV не совпадают: поле {c}. Остановлено.')
    # Verify ALL oxides; absent fields in a compact audit count as zero.
    all_oxides=sorted(set(ox)|set(_oxide_cols(audit)))
    for c in all_oxides:
        left=s[c].to_numpy(float) if c in s else np.zeros(len(s))
        right=a[c].to_numpy(float) if c in a else np.zeros(len(a))
        if not np.allclose(left,right,rtol=0,atol=2e-4):
            raise ValueError(f'Несовпадение составов разных запусков: оксид {c}. Остановлено.')
    if 'Run_Unique_ID' in a and 'Run_Unique_ID' in s:
        if not (a.Run_Unique_ID.astype(str).to_numpy() == s.Run_Unique_ID.astype(str).to_numpy()).all():
            raise ValueError('Run_Unique_ID не совпадает: нельзя объединять запуски')
    new_cols = [c for c in source.columns if c not in audit.columns]
    enriched = a.join(s[new_cols], how='left')
    return enriched.reset_index()


def screen_reliability(df: pd.DataFrame, target_n=1.83) -> pd.DataFrame:
    """Append evidence descriptors and heuristic screening tiers; no probability/risk score."""
    df = df.copy()
    required = set(DISTANCES.values()) | {
        'Candidate_ID', 'Tg_C', 'Tsoft_C',
        'GN_Tg_C', 'GN_Tsoft_C', 'Max_local_Tg_jump_C'
    }
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f'Отсутствует результат аудита шага 384: {sorted(miss)}')
    for col in list(DISTANCES.values())+['Max_local_Tg_jump_C','GN_Tg_C','GN_Tsoft_C','Tg_C','Tsoft_C']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    for name, col in DISTANCES.items():
        desc = {'Density_g_cm3':'rho','RefractiveIndex':'n','Tg_C':'Tg','Tsoft_C':'Tsoft'}[name]
        df['Reference_within_5mol_'+desc] = np.where(df[col].isna(), 'Не проверено',
                np.where(df[col] <= REF_DIST_MOL, 'Да', 'Нет'))
    df['Tg_model_disagreement_C'] = abs(df.Tg_C - df.GN_Tg_C)
    df['Tsoft_model_disagreement_C'] = abs(df.Tsoft_C - df.GN_Tsoft_C)
    df['GN_Tsoft_minus_Tg_C'] = df.GN_Tsoft_C - df.GN_Tg_C
    df['Tg_local_jump_warning'] = df.Max_local_Tg_jump_C > LOCAL_TG_JUMP_C
    df['GN_Tsoft_definition_check'] = df.GN_Tsoft_minus_Tg_C < GN_SOFT_GAP_C
    df['Tg_two_model_warning'] = df.Tg_model_disagreement_C > TWO_MODEL_TG_GAP_C
    df['Tsoft_reference_weak'] = df[DISTANCES['Tsoft_C']].isna() | (df[DISTANCES['Tsoft_C']] > REF_DIST_MOL)
    if 'RefractiveIndex' in df:
        df['Distance_to_YAG_n_assumed'] = abs(pd.to_numeric(df.RefractiveIndex, errors='coerce') - float(target_n))
    else:
        df['Distance_to_YAG_n_assumed'] = np.nan
    known = df[list(DISTANCES.values())+['Max_local_Tg_jump_C','GN_Tg_C']].notna().all(axis=1)
    allrefs = df[list(DISTANCES.values())].le(REF_DIST_MOL).all(axis=1)
    ok = known & allrefs & (~df.Tg_local_jump_warning) & (~df.Tg_two_model_warning)
    fatal_local = known & df.Tg_local_jump_warning
    df['Screening_tier'] = np.select(
        [~known, fatal_local, ok],
        ['D — неполная диагностика', 'C — проверить нестабильный Tg',
         'A — приоритет пробного стекловарения'],
        default='B — нужны дополнительные проверки')
    tier_order = {'A — приоритет пробного стекловарения': 0,
                  'B — нужны дополнительные проверки': 1,
                  'C — проверить нестабильный Tg': 2,
                  'D — неполная диагностика': 3}
    df['_tier_order'] = df.Screening_tier.map(tier_order)
    def reasons(row):
        msgs=[]
        if pd.isna(row[DISTANCES['Tsoft_C']]):
            msgs.append('Tsoft: близость к данным не проверена')
        elif row[DISTANCES['Tsoft_C']] > REF_DIST_MOL:
            msgs.append('Tsoft: нет аналога ≤5 мол.%')
        if row.Tg_local_jump_warning:
            msgs.append('Tg: локальный скачок >5 °C')
        if row.Tg_two_model_warning:
            msgs.append('Tg: расхождение моделей >10 °C')
        if row.GN_Tsoft_definition_check:
            msgs.append('GlassNet: Tsoft почти равна Tg, нужна проверка определений')
        if not msgs: msgs.append('Формальных предупреждений нет; эксперимент необходим')
        return '; '.join(msgs)
    df['Scientific_warnings'] = df.apply(reasons, axis=1)
    return df.sort_values(
        ['_tier_order','Nearest_Tsoft_halfL1_mol','Tg_model_disagreement_C',
         'Distance_to_YAG_n_assumed','Candidate_ID'],
        ascending=[True,True,True,True,True], kind='stable',na_position='last'
    ).drop(columns='_tier_order').reset_index(drop=True)


def diverse_shortlist(ranked: pd.DataFrame, max_items=6, min_distance=2.0) -> pd.DataFrame:
    """Select compositionally different representatives; never infer existence of glass."""
    oxides = _oxide_cols(ranked)
    if not oxides:
        raise ValueError('Нет оксидов для оценки разнообразия')
    chosen=[]
    for i,row in ranked.iterrows():
        v=row[oxides].to_numpy(float)
        if not chosen:
            chosen.append(i)
        else:
            dist=min(.5 * np.abs(v-ranked.loc[j, oxides].to_numpy(float)).sum() for j in chosen)
            if dist >= min_distance-1e-9:
                chosen.append(i)
        if len(chosen) == max_items: break
    return ranked.loc[chosen].reset_index(drop=True)


def run_live_audit(root, candidates: pd.DataFrame, max_active_oxides=6):
    """Run reusable per-property diagnostics for any new step381 candidate batch.

    Reruns ONLY inference with frozen models, with no fit. Proximity uses potentially
    training-set literature records, so MUST NOT be treated as independent validation.
    """
    root=Path(root)
    for directory in (root/'GlassDesignML',root/'GlassDesignML_Upgrade'):
        if str(directory) not in sys.path:
            sys.path.insert(0,str(directory))
    from glass_science_lite_311 import near_analogues
    from glass_benchmark_v21 import FrozenPredictor
    feat=json.loads((root/'GlassDesignML/model_features.json').read_text(encoding='utf-8'))
    oxides=sorted(set.intersection(*(set(f) for f in feat.values())))
    out=candidates.copy().reset_index(drop=True)
    if out.empty: raise ValueError('Нет составов для диагностики')
    missing=[x for x in oxides if x not in out]
    for x in missing:out[x]=0.0
    x=out[oxides].to_numpy(float)
    if not np.isfinite(x).all() or not np.allclose(x.sum(axis=1),100,atol=1e-4):
        raise ValueError('Неверные концентрации исходных составов')
    issues={}
    for p,col in DISTANCES.items():
        _,ds,problems=near_analogues(out,{p:(None,None)},root,max_half_l1=REF_DIST_MOL,chunk_size=4000)
        out[col]=pd.to_numeric(pd.Series(ds),errors='coerce')
        if problems: issues[p]=problems
    gd=FrozenPredictor(root,feat,oxides)
    def inputframe(data, start=0):
        return pd.DataFrame(data,columns=oxides).assign(Candidate_ID=np.arange(start,start+len(data)))
    base=gd(inputframe(x))
    for prop in PROPERTIES:
        if prop in candidates:
            delta=float(np.max(np.abs(base[prop].to_numpy(float)-candidates[prop].to_numpy(float))))
            if delta>1e-2:
                raise RuntimeError(f'{prop}: свежий прогноз отличается от CSV на {delta:.3f}; возможно, модель или версия изменены. Аудит остановлен.')
    changed=[]
    parents=[]
    truncated=np.zeros(len(out),bool)
    for i,v in enumerate(x):
        active=np.where(v>0.5)[0]
        # A cap keeps time predictable for many-component systems; flag limitations.
        if len(active)>max_active_oxides:
            truncated[i]=True
            active=active[np.argsort(v[active])[-max_active_oxides:]]
        local=[]
        for donor in active:
            for receiver in active:
                if donor==receiver or v[donor]<0.05: continue
                vv=v.copy();vv[donor]-=0.05;vv[receiver]+=0.05
                local.append(vv)
        if local:
            parents.extend([i]*len(local));changed.extend(local)
    drift=np.full(len(out),np.nan)
    if changed:
        arr=np.array(changed,float)
        pred=gd(inputframe(arr,100000))
        diff=np.abs(pred.Tg_C.to_numpy(float)-base.Tg_C.to_numpy(float)[parents])
        drift=np.zeros(len(out))
        np.maximum.at(drift,parents,diff)
    out['Max_local_Tg_jump_C']=drift
    out['Local_Tg_axes_tested']=np.bincount(parents,minlength=len(out))
    out['Local_Tg_check_limited_to_top_6']=truncated
    out['Live_diagnostic_issues']=str(issues) if issues else ''
    return out,issues


def save_report(folder, ranked, shortlist, input_fingerprint=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    p=folder/'scientific_screen_v385.csv';q=folder/'scientific_shortlist_v385.csv'
    ranked.to_csv(p,index=False,encoding='utf-8-sig')
    shortlist.to_csv(q,index=False,encoding='utf-8-sig')
    info={'version':385,'rows':int(len(ranked)),'shortlist_rows':int(len(shortlist)),
          'scientific_limits':'Heuristic screening; cannot prove glass formation, chemical compatibility, or independent prediction accuracy.',
          'reference_dist_mol':REF_DIST_MOL,'local_tg_jump_threshold_C':LOCAL_TG_JUMP_C,
          'two_model_tg_gap_threshold_C':TWO_MODEL_TG_GAP_C,
          'gn_soft_gap_warning_C':GN_SOFT_GAP_C,'diversity_half_l1_mol':DIVERSITY_HALF_L1_MOL,
          'data_sources':'Original step381 and per-property step384 audit; reference proximity may use TRAIN records',
          'source_fingerprint':input_fingerprint}
    (folder/'scientific_screen_v385_manifest.json').write_text(
        json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return p,q
