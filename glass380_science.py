"""Optional model-based glass-formation and reference-distance diagnostics.

Never considers absence of a tool, timeouts or an empty search to be positive
experimental evidence. Runs once on ALL feasible candidate recipes, if requested.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _signature(df,criteria):
    payload=df.to_json(orient='split',double_precision=12)
    payload+=json.dumps(criteria,ensure_ascii=False,sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def check_all(result,root,oxide_columns=None,force=False,candidates=None):
    """Return scientific status for every feasible recipe, with explicit unknowns.

    This is optional and can take time due to GlassPy and reference datasets.
    """
    root=Path(root)
    folder=Path(result['folder'])
    near_misses=result['tables']['both_pass_recipes'].empty
    if candidates is None:
        candidates=result['tables']['both_pass_recipes']
    candidates=candidates.copy().reset_index(drop=True)
    criteria=result['criteria']
    source=candidates[['Run_Unique_ID','Candidate_ID','Glass_family']].copy()
    sig=_signature(candidates,criteria)
    csv=folder/('science_status_nearest.csv' if near_misses else 'science_status.csv')
    meta=folder/('science_status_nearest.meta.json' if near_misses else 'science_status.meta.json')
    if not force and csv.exists() and meta.exists():
        old=json.loads(meta.read_text(encoding='utf-8'))
        if old.get('signature')==sig:
            previous=pd.read_csv(csv,encoding='utf-8-sig')
            if len(previous)==len(candidates) and list(previous.Run_Unique_ID)==list(source.Run_Unique_ID):
                return previous, old.get('issues',[])
    if oxide_columns is None:
        from glass380_engine import load_catalog
        oxide_columns=load_catalog(root)[1]
    # All missing statuses stay unknown (never assume False).
    statuses=pd.DataFrame({'Run_Unique_ID':source.Run_Unique_ID,
                           'Candidate_ID':source.Candidate_ID,
                           'Near_analogue_within_5_mol':pd.Series([None]*len(source),dtype='object'),
                           'Nearest_max_halfL1_mol':np.full(len(source),np.nan),
                           'Nearest_any_halfL1_mol':np.full(len(source),np.nan),
                           'Nearest_any_source':pd.Series([None]*len(source),dtype='object'),
                           'VITRIFY_score':np.full(len(source),np.nan),
                           'Tsoft_ge_Tg':(candidates.Tsoft_C.to_numpy(float)>=
                                          candidates.Tg_C.to_numpy(float))})
    issues=[]
    code_dirs=[root/'GlassDesignML',root/'GlassDesignML_Upgrade']
    for p in code_dirs:
        if str(p) not in sys.path:sys.path.insert(0,str(p))
    try:
        scientific=importlib.import_module('glass_science_lite_311')
    except Exception as e:
        issues.append('Научный модуль отсутствует: '+str(e))
        scientific=None
    if scientific is not None and len(candidates):
        try:
            flags,distances,issue,by_property=scientific.near_analogues(
                candidates,criteria,root,return_distances=True)
            if len(flags)!=len(candidates) or len(distances)!=len(candidates):
                raise ValueError('Ответ для аналогов имеет неверную длину')
            statuses['Near_analogue_within_5_mol']=flags
            statuses['Nearest_max_halfL1_mol']=distances
            # The nearest match across any measured-property dataset is distinct
            # from the 5 mol.% criterion, which must hold for every selected one.
            labels={'Density_g_cm3':'Плотность','RefractiveIndex':'Преломление',
                    'Tg_C':'Tg','Tsoft_C':'Tsoft'}
            nearest_any=np.full(len(candidates),np.inf,dtype=float)
            nearest_source=[None]*len(candidates)
            for prop,values in by_property.items():
                if values is None:
                    continue
                arr=np.asarray(values,dtype=float)
                if arr.shape!=(len(candidates),):
                    raise ValueError('Некорректный массив расстояний '+prop)
                statuses['Nearest_'+prop+'_halfL1_mol']=arr
                improved=np.isfinite(arr)&(arr<nearest_any)
                for i in np.flatnonzero(improved):
                    nearest_source[i]=labels.get(prop,prop)
                nearest_any=np.minimum(nearest_any,arr)
            statuses['Nearest_any_halfL1_mol']=np.where(
                np.isfinite(nearest_any),nearest_any,np.nan)
            statuses['Nearest_any_source']=nearest_source
            if issue:issues.append('Экспериментальные аналоги: '+str(issue))
        except Exception as e:
            issues.append('Аналоги не проверены: '+repr(e))
        try:
            worker=next((p for p in (
                root/'GlassDesignML/glass_vitrify_worker_311.py',
                root/'GlassDesignML_Upgrade/glass_vitrify_worker_311.py') if p.is_file()),None)
            if worker is None:
                issues.append('VITRIFY не проверен: отсутствует glass_vitrify_worker_311.py')
            else:
                scores,issue=scientific.glass_formation_isolated(
                    candidates,oxide_columns,worker,timeout=90)
                if scores is not None and len(scores)==len(candidates):
                    statuses['VITRIFY_score']=scores
                else:
                    issues.append('VITRIFY не проверен: неверная длина/пустой результат')
                if issue:issues.append('VITRIFY: '+str(issue))
        except Exception as e:
            issues.append('VITRIFY не проверен: '+repr(e))
    statuses.to_csv(csv,index=False,encoding='utf-8-sig')
    meta.write_text(
        json.dumps({'signature':sig,'issues':issues,'n':len(statuses)},ensure_ascii=False,indent=2),
        encoding='utf-8')
    combined=candidates.merge(statuses.drop(columns='Candidate_ID'),on='Run_Unique_ID',
                              how='left',validate='one_to_one')
    combined.to_csv(folder/('nearest_science.csv' if near_misses else 'both_pass_science.csv'),index=False,encoding='utf-8-sig')
    return statuses, issues