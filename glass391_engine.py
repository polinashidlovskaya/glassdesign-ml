"""GlassDesign ML — temperature-property-aligned engine + visibility gate, step 391.

This module NEVER retrains or rewrites original GlassDesign models, original app.py,
source datasets or locked test. Backend algorithms are taken unchanged from
user's ORIGINAL working step378 archive, with hashes recorded in every run.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROPS = ('Density_g_cm3', 'RefractiveIndex', 'Tg_C', 'Tsoft_C')
GN = {p: 'GN_' + p for p in PROPS}
PILOT_PER_FAMILY = 8
NEW_BUDGET = 120
MAX_FAMILIES = 4
SEP_MOL = 2.0
BACKENDS = ('glass_benchmark_v21', 'glass_consensus_ga_v25',
            'glass_consensus_pilot_v27', 'glass_pilot_rank_only',
            'GlassDesignML_step364_dual_feedback')


def discover_root():
    env = os.environ.get('GLASSDESIGN_ROOT')
    candidates = [Path(env)] if env else []
    candidates += sorted(Path('/content').glob('glassdesign_mount_*/MyDrive/YAG_Inverse_Design'))
    candidates += sorted(Path('/content').glob('glassdesign_drive_*/MyDrive/YAG_Inverse_Design'))
    candidates += [Path('/content/drive/MyDrive/YAG_Inverse_Design')]
    return next((p for p in candidates if (p/'GlassDesignML/model_features.json').is_file()
                 and (p/'GlassDesignML/glass_families.json').is_file()), None)


def load_catalog(root):
    root = Path(root)
    folder = root/'GlassDesignML'
    feat = json.loads((folder/'model_features.json').read_text(encoding='utf-8'))
    if len(feat) != 4:
        raise ValueError('Каталог признаков должен содержать ровно четыре модели')
    oxides = sorted(set.intersection(*(set(x) for x in feat.values())))
    catalog = json.loads((folder/'glass_families.json').read_text(encoding='utf-8'))
    catalog = catalog.get('families', catalog)
    if not oxides or not isinstance(catalog,dict):
        raise ValueError('Некорректная конфигурация оксидов и химических семейств')
    return feat, oxides, catalog


def _bounds(v, ox):
    if not isinstance(v, (tuple,list)) or len(v)!=2:
        raise ValueError(f'{ox}: нужны две границы мол.%')
    lo,hi = map(float,v)
    if not np.isfinite(lo) or not np.isfinite(hi) or not 0 <= lo <= hi <= 100:
        raise ValueError(f'{ox}: диапазон должен быть в пределах 0–100 мол.%')
    return (lo,hi)


def validate_criteria(criteria):
    if not isinstance(criteria,dict) or not criteria or not set(criteria).issubset(PROPS):
        raise ValueError('Выбери хотя бы одно поддерживаемое свойство')
    ans={}
    for p, limits in criteria.items():
        if not isinstance(limits, (tuple,list)) or len(limits)!=2:
            raise ValueError(f'Для {p} нужны две границы (от, до)')
        lo,hi=limits
        lo = None if lo is None else float(lo)
        hi = None if hi is None else float(hi)
        if lo is None and hi is None: raise ValueError(f'{p}: обе границы пустые')
        if lo is not None and not np.isfinite(lo): raise ValueError(f'{p}: неверная нижняя граница')
        if hi is not None and not np.isfinite(hi): raise ValueError(f'{p}: неверная верхняя граница')
        if lo is not None and hi is not None and lo>hi: raise ValueError(f'{p}: от > до')
        if p in ('Density_g_cm3','RefractiveIndex') and any(x is not None and x<=0 for x in (lo,hi)):
            raise ValueError(f'{p}: значение должно быть положительным')
        ans[p]=(lo,hi)
    return ans


def select_families(all_families, mode='auto', selected_names=None,
                    manual_bounds=None, excluded_oxides=(), oxides=None):
    oxides = set(oxides or [])
    excluded=set(excluded_oxides or [])
    if not excluded.issubset(oxides):
        raise ValueError('В исключениях указан неподдерживаемый оксид')
    if mode=='manual':
        spec={k:_bounds(v,k) for k,v in (manual_bounds or {}).items()}
        if len(spec)<2:raise ValueError('Для собственного состава нужно минимум два оксида')
        if not set(spec).issubset(oxides):raise ValueError('В собственном составе есть неподдерживаемые оксиды')
        if set(spec)&excluded:raise ValueError('Оксид одновременно указан и исключён')
        if sum(v[0] for v in spec.values()) > 100+1e-9 or sum(v[1] for v in spec.values()) < 100-1e-9:
            raise ValueError('Суммарные пределы концентраций не допускают 100 мол.%')
        # This family is evaluated directly by the unchanged GA backend.
        return {'Пользовательская система':spec}
    if mode=='families':
        selected=set(selected_names or [])
        if not selected:raise ValueError('Выбери хотя бы одно семейство')
        unknown=selected-set(all_families)
        if unknown:raise ValueError('Неизвестное семейство: '+', '.join(sorted(unknown)))
    elif mode=='auto':selected=set(all_families)
    else:raise ValueError('Неизвестный режим задания состава')
    families={k:v for k,v in sorted(all_families.items())
              if k in selected and not (set(v)&excluded)}
    if not families:
        raise ValueError('После исключения оксидов не осталось подходящих семейств')
    return families


def load_backends(folder=None):
    """Require the original *tested* GA code, never auto-reimplement it silently."""
    folder = Path(folder) if folder else Path(__file__).resolve().parent
    for name in BACKENDS:
        if not (folder/(name+'.py')).is_file():
            raise FileNotFoundError('Не найден оригинальный модуль шага 378: '+name+'.py. '
                                    'Загрузи архив GlassDesignML_step378_universal_generator.zip.')
    if str(folder) not in sys.path:sys.path.insert(0,str(folder))
    return {n:importlib.import_module(n) for n in BACKENDS}


def predicted_pass(df,criteria,prefix=''):
    ok=np.ones(len(df),bool)
    for p,(lo,hi) in criteria.items():
        x=pd.to_numeric(df[prefix+p],errors='coerce').to_numpy(float)
        ok &= np.isfinite(x)
        if lo is not None:ok &= x>=lo-1e-9
        if hi is not None:ok &= x<=hi+1e-9
    tg=pd.to_numeric(df[prefix+'Tg_C'],errors='coerce').to_numpy(float)
    ts=pd.to_numeric(df[prefix+'Tsoft_C'],errors='coerce').to_numpy(float)
    ok &= np.isfinite(tg)&np.isfinite(ts)&(ts>=tg-1e-9)
    return ok


def score_violation(df,criteria):
    scales={'Density_g_cm3':.5,'RefractiveIndex':.05,'Tg_C':50.,'Tsoft_C':50.}
    penalties=[]
    for pref in ('','GN_'):
        for prop,(low,high) in criteria.items():
            x=df[pref+prop].to_numpy(dtype=float)
            scale=max(scales[prop],high-low if high is not None and low is not None else 0)
            if low is not None:penalties.append(np.maximum(low-x,0)/scale)
            if high is not None:penalties.append(np.maximum(x-high,0)/scale)
        penalties.append(np.maximum(df[pref+'Tg_C'].to_numpy(dtype=float)-
                                    df[pref+'Tsoft_C'].to_numpy(dtype=float),0)/50)
    return np.maximum.reduce(penalties) if penalties else np.zeros(len(df))


def center_distance(df,criteria):
    accum=np.zeros(len(df)); cnt=0
    for p,(lo,hi) in criteria.items():
        if lo is not None and hi is not None:
            for pref in ('','GN_'):
                accum+=np.abs(df[pref+p].to_numpy(float)-(lo+hi)*.5)/max(hi-lo,1e-6)
                cnt+=1
    return accum/cnt if cnt else accum


def unique_check(df,oxides,tag):
    if df.Candidate_ID.isna().any() or df.Candidate_ID.duplicated().any():
        raise AssertionError(f'{tag}: повторяющиеся/пустые Candidate_ID')
    if not set(oxides).issubset(df.columns):raise AssertionError(f'{tag}: отсутствуют оксиды')
    x=df[oxides].to_numpy(dtype=float)
    if not np.isfinite(x).all() or (x< -1e-7).any() or not np.allclose(x.sum(axis=1),100,atol=1e-5):
        raise AssertionError(f'{tag}: некорректная сумма оксидов')
    if df[oxides].round(7).duplicated().any():raise AssertionError(f'{tag}: дубли химического состава')


def validate_chemistry(df,families,oxides,tag):
    unique_check(df,oxides,tag)
    if not set(df.Glass_family).issubset(families):
        raise AssertionError(f'{tag}: недопустимое семейство')
    for name,g in df.groupby('Glass_family'):
        limits=families[name]
        for ox,(lo,hi) in limits.items():
            xs=g[ox].to_numpy(dtype=float)
            if (xs<lo-1e-5).any() or (xs>hi+1e-5).any():
                raise AssertionError(f'{tag}: {name} / {ox}: неверные химические границы')
        forbidden=set(oxides)-set(limits)
        if forbidden and (np.abs(g[sorted(forbidden)].to_numpy(float))>1e-6).any():
            raise AssertionError(f'{tag}: {name} содержит запрещённый оксид')
    for p in PROPS:
        for pref in ('','GN_'):
            if not np.isfinite(df[pref+p].to_numpy(dtype=float)).all():
                raise AssertionError(f'{tag}: отсутствует прогноз {pref+p}')


def append_flags(df,criteria):
    df=df.copy()
    df['GD_pass']=predicted_pass(df,criteria)
    df['GN_pass']=predicted_pass(df,criteria,prefix='GN_')
    df['Both_pass']=df.GD_pass & df.GN_pass
    df['Max_normalized_violation']=score_violation(df,criteria)
    df['Center_distance']=center_distance(df,criteria)
    return df


def representatives(eligible,oxides,count=15,per_family=5,separation=SEP_MOL):
    """Stable, rule-based diverse shortlist; no misleading quality-probability score."""
    ranked=eligible.sort_values(['Center_distance','Glass_family','Candidate_ID'],kind='stable')
    chosen=[]; vectors=[]; counts={}
    for idx,row in ranked.iterrows():
        name=row['Glass_family']
        if counts.get(name,0)>=per_family:continue
        v=row[oxides].to_numpy(dtype=float)
        if vectors and min(.5*np.abs(v-prev).sum() for prev in vectors)<separation-1e-9:continue
        chosen.append(idx);vectors.append(v);counts[name]=counts.get(name,0)+1
        if len(chosen)>=count:break
    return ranked.loc[chosen].reset_index(drop=True).copy()


def fix_global_ids(pilot,new,others):
    """Fix original 378 collision permanently in EVERY output of step380."""
    _new=new.copy();_pilot=pilot.copy()
    if not _pilot.Candidate_ID.is_unique or not _new.Candidate_ID.is_unique:
        raise AssertionError('ID не уникальны внутри стадий')
    old=_new.Candidate_ID.astype(int).to_numpy()
    start=int(_pilot.Candidate_ID.max())+1
    mapping=dict(zip(map(int,old),range(start,start+len(old))))
    if len(mapping)!=len(old):raise AssertionError('ID внутри новых составов повторяются')
    _pilot.insert(_pilot.columns.get_loc('Candidate_ID')+1,
                  'Run_Unique_ID',['P%06d'%int(i) for i in _pilot.Candidate_ID])
    def renumber(df):
        d=df.copy()
        d.insert(d.columns.get_loc('Candidate_ID')+1,
                 'Source_Candidate_ID',d.Candidate_ID.astype(int))
        d['Candidate_ID']=d.Source_Candidate_ID.map(mapping).astype(int)
        d.insert(d.columns.get_loc('Candidate_ID')+1,
                 'Run_Unique_ID',['N%06d'%int(i) for i in d.Candidate_ID])
        return d
    result={name:renumber(df) for name,df in others.items()}
    result['all_new_recipes']=renumber(_new)
    if set(_pilot.Candidate_ID) & set(result['all_new_recipes'].Candidate_ID):
        raise AssertionError('ID пересекаются после исправления')
    return _pilot,result,mapping


def _hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def run(criteria, *, root=None, mode='auto', selected_names=None,manual_bounds=None,
        excluded_oxides=(), seed=71, budget=NEW_BUDGET, max_families=MAX_FAMILIES,
        backend_dir=None, gd_factory=None, gn_factory=None, predictor_callback=None,
        out_dir=None):
    """Universal chemistry search with original Step378 GA backend and NEW stable export.

    Predictor factories support ONLY tests; defaults load frozen real models.
    """
    criteria=validate_criteria(criteria)
    if not isinstance(budget,int) or budget<1 or not isinstance(max_families,int) or max_families<1:
        raise ValueError('Неверный бюджет поиска')
    root=Path(root) if root is not None else discover_root()
    if root is None:raise FileNotFoundError('Не подключён Google Drive YAG_Inverse_Design')
    feature,oxides,catalog=load_catalog(root)
    backend=load_backends(backend_dir)
    bm=backend['glass_benchmark_v21']
    ga=backend['glass_consensus_ga_v25']
    rankmod=backend['glass_consensus_pilot_v27']
    pilot_rank=backend['glass_pilot_rank_only']
    glassnet=backend['GlassDesignML_step364_dual_feedback']
    all_families,rejected=bm.wide_family_filter(catalog,oxides,max_components=12)
    families=select_families(all_families,mode,selected_names,manual_bounds,excluded_oxides,oxides)
    k=min(max_families,len(families))
    if budget<k:raise ValueError('Бюджет меньше числа выбранных семейств')
    if str(root/'GlassDesignML') not in sys.path:sys.path.insert(0,str(root/'GlassDesignML'))
    if str(root/'GlassDesignML_Upgrade') not in sys.path:sys.path.insert(0,str(root/'GlassDesignML_Upgrade'))
    gd=bm.FrozenPredictor(root,feature,oxides) if gd_factory is None else gd_factory(root,feature,oxides)
    from glass391_glassnet import FrozenGlassNetAligned391
    gn=FrozenGlassNetAligned391(oxides) if gn_factory is None else gn_factory(oxides)
    tg=time.perf_counter()
    pg0,gn0=gd.rows,gn.rows
    pilot=bm.prepare_pilot(families,oxides,gd,seed*100003,PILOT_PER_FAMILY)
    gp=gn(pilot.copy())
    if not pilot.Candidate_ID.reset_index(drop=True).equals(gp.Candidate_ID.reset_index(drop=True)):
        raise AssertionError('Pilot: ID GD и GlassNet не соответствуют друг другу')
    for col in GN.values():pilot[col]=gp[col].to_numpy(dtype=float)
    if gd.rows-pg0!=len(pilot) or gn.rows-gn0!=len(pilot):
        raise AssertionError('Пилот не совпал по бюджету моделей')
    validate_chemistry(pilot,families,oxides,'pilot')
    best_families,ranking=rankmod.dual_ranking(pilot,families,k,criteria,
                                               pilot_rank.family_pilot_ranking)
    best_families=dict(sorted(best_families.items()))
    starting=pilot.loc[pilot.Glass_family.isin(best_families)].copy().reset_index(drop=True)
    if len(starting)!=k*PILOT_PER_FAMILY:raise AssertionError('Выбранный пилот неполон')
    pg1,gn1=gd.rows,gn.rows
    new=ga.generate(best_families,starting,gd,gn,oxides,criteria,seed,
                    family_policy='adaptive',feedback='both',budget=budget,batch_size=16)
    if gd.rows-pg1!=budget or gn.rows-gn1!=budget or len(new)!=budget:
        raise AssertionError('Отклонение от бюджета новых прогнозов')
    validate_chemistry(new,best_families,oxides,'new')
    pilot=append_flags(pilot,criteria)
    new=append_flags(new,criteria)
    # Audit-only GlassNet Littleton predictions were already computed by the same
    # GlassNet calls. Never use this property for search or pass/fail scoring.
    pilot=gn.attach_littleton(pilot)
    new=gn.attach_littleton(new)
    for thermal_frame in (pilot,new):
        thermal_frame["TdilatometricSoftening_C"] = thermal_frame["Tsoft_C"].to_numpy(float)
        thermal_frame["GN_TdilatometricSoftening_C"] = thermal_frame["GN_Tsoft_C"].to_numpy(float)
    from glass391_recommendations import annotate_recommendations, recommendation_counts
    new=annotate_recommendations(new,criteria)
    eligible=new.loc[new.Recommendation_visible].copy().sort_values(['Center_distance','Candidate_ID'],kind='stable')
    excluded=new.loc[~new.Recommendation_visible].copy()
    short=representatives(eligible,oxides)
    nearest=new.sort_values(['Max_normalized_violation','Center_distance','Candidate_ID'],
                            kind='stable').head(30).copy()
    overlap_before=len(set(pilot.Candidate_ID)&set(new.Candidate_ID))
    pilot,export,mapping=fix_global_ids(pilot,new,{
        'both_pass_recipes':eligible,
        'excluded_from_recommendations':excluded,
        'selected_representatives':short,
        'nearest_30':nearest})
    allnew=export['all_new_recipes']
    rejected=export['excluded_from_recommendations']
    good=export['both_pass_recipes']
    subset=export['selected_representatives']
    if not good.Both_pass.all() or not good.Recommendation_visible.all() or not set(subset.Candidate_ID).issubset(good.Candidate_ID):
        raise AssertionError('Результат отобран некорректно')
    if len(allnew)!=len(good)+len(rejected) or rejected.Recommendation_visible.any():
        raise AssertionError('Recommendation gate does not partition the full candidate export')
    elapsed=time.perf_counter()-tg
    if out_dir is None:
        name='run_'+datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'_'+str(seed)
        dest=root/'GlassDesignML_Upgrade'/'Temperature_Screen_step391'/name
    else:dest=Path(out_dir)
    if dest.exists():raise FileExistsError('Папка результатов уже существует: '+str(dest))
    dest.mkdir(parents=True)
    tables={'pilot_all':pilot,'all_new_recipes':allnew,'both_pass_recipes':good,
            'excluded_from_recommendations':rejected,
            'selected_representatives':subset,'nearest_30':export['nearest_30'],
            'family_pilot_ranking':ranking}
    for label,df in tables.items():df.to_csv(dest/(label+'.csv'),index=False,encoding='utf-8-sig')
    mappingframe=pd.DataFrame({'Source_Candidate_ID':list(mapping),
                               'Candidate_ID':list(mapping.values())})
    mappingframe.to_csv(dest/'candidate_id_mapping.csv',index=False,encoding='utf-8-sig')
    summary={'pilot':len(pilot),'new':len(allnew),'gd_pass':int(allnew.GD_pass.sum()),
             'glassnet_pass':int(allnew.GN_pass.sum()),'both_pass':len(good),
             'selected_representatives':len(subset),'successful_families':int(good.Glass_family.nunique()),
             'selected_families':list(best_families),'available_families':len(families),
             'original_ids_in_both_stages':overlap_before,'unique_id_range':[int(allnew.Candidate_ID.min()),int(allnew.Candidate_ID.max())],
             'seconds':round(elapsed,3),**recommendation_counts(allnew)}
    manifest={'step':391,'method':'Top4_Baseline_GA_dual_feedback','seed':int(seed),
              'mode':mode,'criteria':{p:list(v) for p,v in criteria.items()},
              'selected_names':sorted(selected_names or []),
              'manual_bounds':{k:list(v) for k,v in (manual_bounds or {}).items()} or None,
              'excluded_oxides':sorted(excluded_oxides or []),'pilot_per_family':PILOT_PER_FAMILY,
              'budget':budget,'max_families':max_families,'summary':summary,
              'backend_sha256':{n:_hash(Path(backend_dir or Path(__file__).resolve().parent)/(n+'.py')) for n in BACKENDS},
              'configuration_sha256':{str(p.relative_to(root)):_hash(p) for p in
                        (root/'GlassDesignML/model_features.json',
                         root/'GlassDesignML/glass_families.json',
                         root/'GlassDesignML/model_manifest.json') if p.is_file()},
              'temperature_property_GD':'TdilatometricSoftening',
              'temperature_property_GN':'TdilatometricSoftening',
              'legacy_temperature_key':'Tsoft_C / GN_Tsoft_C contain dilatometric softening in degrees C',
              'mapping_version':'step391 aligned Td and explicit recommendation gating',
              'recommendation_policy':'Only Both_pass=True (including Td>=Tg for BOTH models) is visible in recommendations; excluded retained with reasons',
              'littleton_glassnet_property':'GN_Littleton_Tsoft_C is stored for audit only; it never affects optimization',
              'limitation':'Consensus of two predictors optimized in search; NOT independent or experimental validation.',
              'created_at_utc':datetime.now(timezone.utc).isoformat()}
    (dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'summary':summary,'tables':tables,'manifest':manifest,'folder':dest,
            'oxides':oxides,'criteria':criteria}


def open_saved(folder):
    folder=Path(folder)
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('step') not in (380,381,390,391):raise ValueError('Это не сохранённый запуск финальной версии')
    names=['pilot_all','all_new_recipes','both_pass_recipes',
           'selected_representatives','nearest_30','family_pilot_ranking']
    if manifest['step']==391:
        names.append('excluded_from_recommendations')
    tables={n:pd.read_csv(folder/(n+'.csv'),encoding='utf-8-sig') for n in names}
    for n in ('all_new_recipes','both_pass_recipes','selected_representatives'):
        if tables[n].Candidate_ID.duplicated().any():raise ValueError('CSV: повторяющийся ID')
    if set(tables['pilot_all'].Candidate_ID)&set(tables['all_new_recipes'].Candidate_ID):
        raise ValueError('CSV: пересечение ID пилота и новых составов')
    if manifest['step']==391:
        whole=tables['all_new_recipes']; recs=tables['both_pass_recipes']
        rejected=tables['excluded_from_recommendations']
        if len(whole)!=len(recs)+len(rejected):
            raise ValueError('Corrupt recommendation split')
        if not recs.Recommendation_visible.astype(bool).all() or rejected.Recommendation_visible.astype(bool).any():
            raise ValueError('Inconsistent visibility in saved exports')
    return {'summary':manifest['summary'],'tables':tables,'manifest':manifest,
            'folder':folder,'criteria':manifest['criteria']}


def latest_saved(root):
    paths=[Path(root)/'GlassDesignML_Upgrade'/n for n in ('Temperature_Screen_step391','Property_Aligned_step390','Final_Generator_step381','Final_Generator_step380')]
    runs=sorted((p for folder in paths for p in folder.glob('run_*') if (p/'manifest.json').is_file()),reverse=True)
    if not runs:raise FileNotFoundError('Пока нет сохранённых запусков версии 380')
    return open_saved(runs[0])
