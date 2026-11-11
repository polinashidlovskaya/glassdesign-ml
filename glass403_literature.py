"""Universal per-candidate literature evidence beyond a fixed 5 mol.% cutoff.

Only original SciGlass records with a verified KOD -> primary RefID -> Reference
chain may be displayed as attributed sources. Distant references are *not*
called close analogues; no DOI invented, frozen models never modified.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from glass398_universal_literature import (
    PROPERTIES, load_measurement_index, load_archive, build_archive_lookup,
    check_source_link_for_id, read_reviewed_catalog, composition_label,
)
from glass399_integrated import find_index, find_reviewed_catalog

MODULE_VERSION='403.1'
OUTPUT='sciglass_literature_step403.csv'
META='literature_meta_step403.json'


def distance_description(distance, near=5.):
    d=float(distance)
    if d<=near+0.001:
        return 'Близкий измеренный аналог'
    if d<=15:
        return 'Похожая система, но состав отличается заметно'
    if d<=30:
        return 'Литературный пример с существенно отличающимся составом'
    return 'Удалённый литературный пример: нельзя переносить измеренные свойства'


def _ensure_input(eligible):
    if eligible.empty:return eligible.copy()
    if eligible.Run_Unique_ID.isna().any() or eligible.Run_Unique_ID.duplicated().any():
        raise ValueError('Повторяющиеся идентификаторы состава')
    if 'Input_Row' in eligible:raise ValueError('Input_Row — зарезервированный столбец')
    recipes=eligible.copy().reset_index(drop=True)
    recipes.insert(0,'Input_Row',np.arange(1,len(recipes)+1,dtype=int))
    return recipes


def search_documented_nearest(recipes, arrays, lookup, reviewed,
                              top_k=3, near_threshold=5.,
                              max_candidates_checked=2000, progress=None):
    """Search all oxide compositions, all four properties, irrespective of family.

    For compositions totalling 100%, 1/2 L1=100-sum(min(x_i,y_i)).
    Computes distances in only the nonzero user oxides, no 569-oxide
    temporary arrays per candidate. Source provenance must fully verify.
    """
    if not 1<=int(top_k)<=10:raise ValueError('Число аналогов должно быть 1–10')
    X,R,Y=arrays['X'],arrays['R'],arrays['Y']
    if len(R)!=len(recipes):raise ValueError('Кандидаты и индекс разной длины')
    if not np.allclose(np.sum(X,axis=1),100.,atol=.05):
        raise ValueError('SciGlass должен хранить нормированные молярные концентрации')
    cache={}
    output=[]
    for i in range(len(recipes)):
        rec=recipes.iloc[i]
        v=R[i]
        if not np.allclose(v.sum(),100.,atol=.05):
            raise ValueError('Неверная сумма молярных процентов кандидата')
        dist=np.full(len(X),100.,dtype=np.float32)
        for j in np.flatnonzero(v>0.00001):
            # Exact half-L1 distance identity for two 100 mol.% vectors.
            dist-=np.minimum(X[:,j],v[j])
        np.maximum(dist,0,out=dist)
        for prop,(prop_ru,unit,predfield) in PROPERTIES.items():
            eligible=np.flatnonzero(np.isfinite(Y[prop]))
            if not len(eligible):continue
            # Full sorting is fast (up to ~100k values); deterministic ties.
            order=eligible[np.argsort(dist[eligible],kind='stable')]
            seen_kods=set()
            selected=[]
            first_journal=None
            for idx in order[:max_candidates_checked]:
                sid=str(arrays['ids'][int(idx)])
                if sid not in cache:
                    cache[sid]=check_source_link_for_id(
                        sid,arrays['auth'][int(idx)],arrays['years'][int(idx)],lookup,reviewed)
                source=cache[sid]
                kod=source['KOD_verified']
                if not source['Source_archive_confirmed'] or not kod or not source['Source_URL']:
                    continue
                if kod in seen_kods:continue
                seen_kods.add(kod)
                bib=str(source['Source_bibliography'] or '')
                source_kind=('Патент' if 'patent' in bib.lower() else
                             ('Научная статья' if str(source['Source_title']).strip() else
                              'Библиографическая запись'))
                record=(int(idx),source,source_kind)
                if source_kind=='Научная статья' and first_journal is None:
                    first_journal=record
                if len(selected)<int(top_k):selected.append(record)
                # Guarantee at least one peer-published record when supported by
                # verified SciGlass source data, even if nearest hits are patents.
                if len(selected)>=top_k and first_journal is not None:break
            if selected and first_journal is not None and not any(
                    r[2]=='Научная статья' for r in selected):
                if all(r[0]!=first_journal[0] for r in selected):
                    selected[-1]=first_journal
            # Sort by REAL chemical distance; replacement may be farther.
            selected.sort(key=lambda a:(float(dist[a[0]]),str(arrays['ids'][a[0]])))
            for rank,(idx,source,source_kind) in enumerate(selected,1):
                sid=str(arrays['ids'][idx])
                value=float(Y[prop][idx])
                if prop in ('Tg','TdilatometricSoftening'):value-=273.15
                d=float(dist[idx])
                output.append({
                    'Input_Row':int(rec.Input_Row),
                    'Run_Unique_ID':str(rec.Run_Unique_ID),
                    'Candidate_ID':str(rec.Candidate_ID),
                    'Property':prop, 'Property_RU':prop_ru,
                    'Rank':rank,
                    'Distance_halfL1_mol':round(d,5),
                    'Measured_value':round(value,6),
                    'Unit':unit,
                    'SciGlass_ID':sid,
                    'SciGlass_author':arrays['auth'][idx],
                    'SciGlass_year':arrays['years'][idx],
                    'Source_kind':source_kind,
                    'Analogue_similarity_status':distance_description(d,near_threshold),
                    'Is_close_at_5mol':d<=near_threshold+.001,
                    'Candidate_composition_mol':composition_label(v,arrays['oxide_columns']),
                    'Analogue_composition_mol':composition_label(X[idx],arrays['oxide_columns']),
                    **source,
                })
        if progress is not None:
            progress(i+1,len(recipes))
    result=pd.DataFrame(output)
    if not result.empty:
        result.sort_values(['Input_Row','Property','Rank'],inplace=True,kind='stable')
        if result.duplicated(['Input_Row','Property','Rank']).any():
            raise AssertionError('Повторены ранги литературных аналогов')
        if not result.Source_archive_confirmed.astype(bool).all():
            raise AssertionError('Неподтверждённая публикация попала в результаты')
        if not result.Source_URL.astype(str).str.startswith('https://').all():
            raise AssertionError('Недопустимая ссылка на первоисточник')
        if not set(result.Run_Unique_ID.astype(str)).issubset(set(recipes.Run_Unique_ID.astype(str))):
            raise AssertionError('Источник привязан к чужому составу')
    return result


def _fingerprint(result):
    return result.get('input_fingerprint')


def compute_literature_403(result, root, report_dir, progress=None,
                           top_k=3, near_threshold=5.):
    root=Path(root);out=Path(report_dir);out.mkdir(parents=True,exist_ok=True)
    eligible=result['tables']['both_pass_recipes']
    if eligible.empty:return pd.DataFrame()
    recipes=_ensure_input(eligible)
    indexed=load_measurement_index(find_index(root),recipes)
    archive_dir=root/'GlassDesignML_Upgrade'/'SciGlass_Universal_Literature_step398'/'cache'/'archive'
    archive,information=load_archive(archive_dir,force_download=False)
    lookup=build_archive_lookup(archive)
    reviewed_file=find_reviewed_catalog(root)
    reviewed=read_reviewed_catalog(reviewed_file) if reviewed_file is not None else {}
    rows=search_documented_nearest(recipes,indexed,lookup,reviewed,
                                  top_k=top_k,near_threshold=near_threshold,
                                  progress=progress)
    if len(rows):
        keys=eligible[['Run_Unique_ID','Candidate_ID']].astype(str).drop_duplicates()
        check=rows[['Run_Unique_ID','Candidate_ID']].astype(str).drop_duplicates()
        verify=check.merge(keys,on=['Run_Unique_ID','Candidate_ID'],how='left',indicator=True)
        if not verify._merge.eq('both').all():
            raise AssertionError('Литература другого запуска')
    tmp=out/(OUTPUT+'.tmp')
    rows.to_csv(tmp,index=False,encoding='utf-8-sig')
    tmp.replace(out/OUTPUT)
    meta={
        'version':MODULE_VERSION,
        'input_fingerprint':_fingerprint(result),
        'input_rows':len(recipes),
        'linked_candidates':int(rows.Run_Unique_ID.nunique()) if len(rows) else 0,
        'linked_records':len(rows),
        'near_records_5mol':int(rows.Is_close_at_5mol.sum()) if len(rows) else 0,
        'different_documents':int(rows.KOD_verified.nunique()) if len(rows) else 0,
        'index_file':str(find_index(root)),
        'source_archive_hashes':{k:v['sha256'] for k,v in information.items()},
        'source_catalog_397':str(reviewed_file) if reviewed_file else None,
        'note':'Distant references do not establish similar physical behavior; no independent validation.',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
    }
    (out/(META+'.tmp')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/(META+'.tmp')).replace(out/META)
    return rows


def load_cached_literature_403(report_dir, fingerprint):
    d=Path(report_dir);meta=d/META;csv=d/OUTPUT
    if not (meta.is_file() and csv.is_file()):return None
    try:
        info=json.loads(meta.read_text(encoding='utf-8'))
        if info.get('version')!=MODULE_VERSION or info.get('input_fingerprint')!=fingerprint:
            return None
        return pd.read_csv(csv,encoding='utf-8-sig',dtype={'SciGlass_ID':str,'KOD_verified':str,'Primary_RefID':str})
    except (ValueError,KeyError,OSError):
        return None
