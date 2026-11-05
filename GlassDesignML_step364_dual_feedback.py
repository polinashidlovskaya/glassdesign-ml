"""Step364 GlassDesign ML 2.5: four-arm factorial online dual-model optimization.

New universal tasks G/H/I, development evaluation. No experiment results are
precomputed. SAME GD and GN prediction budget for all four arms, including
GD-only control arms. Models and source data are read-only. No fit().
"""
from __future__ import annotations
import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

TASKS={
 'G_mid_density_optics':{
     'Density_g_cm3':(3.4,4.7),'RefractiveIndex':(1.70,1.88),
     'Tg_C':(390,620),'Tsoft_C':(460,760)},
 'H_high_index_process':{
     'Density_g_cm3':(4.3,6.5),'RefractiveIndex':(1.82,2.05),
     'Tg_C':(330,570),'Tsoft_C':(410,720)},
 'I_light_thermostable':{
     'Density_g_cm3':(2.4,3.8),'RefractiveIndex':(1.49,1.73),
     'Tg_C':(450,690),'Tsoft_C':(520,800)},
}
SEEDS=(7,17,37)
BUDGET=120
BATCH=16
PILOT_PER_FAMILY=8
MAX_FAMILIES=8
ARMS=(('global','gd'),('adaptive','gd'),('global','both'),('adaptive','both'))


def find_root():
    c=[Path('/content/glassdesign_mount_824a9be5/MyDrive/YAG_Inverse_Design')]
    c+=sorted(Path('/content').glob('glassdesign_mount_*/MyDrive/YAG_Inverse_Design'))
    c+=sorted(Path('/content').glob('glassdesign_drive_*/MyDrive/YAG_Inverse_Design'))
    c.append(Path('/content/drive/MyDrive/YAG_Inverse_Design'))
    return next((p for p in c if (p/'GlassDesignML/model_features.json').is_file()
                 and (p/'GlassDesignML_Upgrade/glass_universal_pipeline_v3.py').is_file()),None)

class FrozenGlassNet:
    def __init__(self, oxides):
        from glasspy.predict import GlassNet
        self.network=GlassNet()
        self.oxides=list(oxides)
        self.rows=0
        self.calls=0
    def __call__(self,frame):
        ox=self.oxides
        x=frame[ox].to_numpy(float)
        if not np.isfinite(x).all() or (x<0).any() or not np.allclose(x.sum(axis=1),100.,atol=1e-5):
            raise RuntimeError('GlassNet input is not valid mol.%')
        active=[o for o in ox if (frame[o].to_numpy(float)>0).any()]
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',category=FutureWarning)
            raw=self.network.predict(frame[active].copy())
        if not isinstance(raw,pd.DataFrame):raw=pd.DataFrame(raw)
        need=['Density293K','RefractiveIndex','Tg','Tsoft']
        if not set(need).issubset(raw) or len(raw)!=len(frame):
            raise RuntimeError('GlassNet output missing columns')
        val=raw[need].apply(pd.to_numeric,errors='raise').reset_index(drop=True)
        if not np.isfinite(val.to_numpy(float)).all():raise RuntimeError('GlassNet returned NaN')
        out=pd.DataFrame({'Candidate_ID':frame.Candidate_ID.to_numpy()})
        out['GN_Density_g_cm3']=val.Density293K.to_numpy(float)
        out['GN_RefractiveIndex']=val.RefractiveIndex.to_numpy(float)
        out['GN_Tg_C']=val.Tg.to_numpy(float)-273.15
        out['GN_Tsoft_C']=val.Tsoft.to_numpy(float)-273.15
        self.rows+=len(frame); self.calls+=1
        return out


def diversity_count(part,oxides):
    valid=part.loc[part.Both_pass.astype(bool)].sort_values(
        ['Goal_penalty','Glass_family','Candidate_ID'],kind='stable')
    if not len(valid):return 0
    chosen=[]
    for x in valid[oxides].to_numpy(float):
        if not chosen or np.min(.5*np.sum(np.abs(np.asarray(chosen)-x),axis=1))>=2.-1e-9:
            chosen.append(x.copy())
    return len(chosen)


def factorial_effects(per_run):
    f=per_run.copy()
    f['adaptive']=f.Family_policy.eq('adaptive').astype(int)
    f['consensus']=f.Feedback.eq('both').astype(int)
    metrics=['Both_pass','Diverse_both_2mol','Families_both']
    effects=[]
    for metric in metrics:
        d=f.pivot(index=['Task','Seed'],columns=['adaptive','consensus'],values=metric)
        assert len(d)==len(TASKS)*len(SEEDS) and d.shape[1]==4
        v00=d[(0,0)]; v10=d[(1,0)]; v01=d[(0,1)]; v11=d[(1,1)]
        e_ad=.5*((v10-v00)+(v11-v01))
        e_cn=.5*((v01-v00)+(v11-v10))
        interaction=v11-v10-v01+v00
        for label,vec in [('Adaptive_family',e_ad),('Consensus_feedback',e_cn),('Interaction',interaction),
                          ('Final_minus_GA_consensus',v11-v01)]:
            effects.append({'Metric':metric,'Effect':label,'Mean_delta':float(vec.mean()),
                            'Wins':int((vec>0).sum()),'Ties':int((vec==0).sum()),
                            'Losses':int((vec<0).sum())})
    return pd.DataFrame(effects)


def main():
    root=find_root()
    if root is None:raise FileNotFoundError('Google Drive не подключён или проект не найден')
    here=Path(__file__).resolve().parent
    for path in (here,root/'GlassDesignML',root/'GlassDesignML_Upgrade'):
        if str(path) not in sys.path:sys.path.insert(0,str(path))
    import glass_benchmark_v21 as bm
    import glass_hybrid_v21 as hy
    import glass_consensus_ga_v25 as opt
    from glass_universal_pipeline_v3 import family_pilot_ranking
    feature_path=root/'GlassDesignML/model_features.json'
    features=json.loads(feature_path.read_text(encoding='utf-8'))
    oxides=sorted(set.intersection(*(set(x) for x in features.values())))
    catalog=json.loads((root/'GlassDesignML/glass_families.json').read_text(encoding='utf-8'))
    catalog=catalog.get('families',catalog)
    families,rejected=bm.wide_family_filter(catalog,oxides,max_components=12)
    if len(families)!=17:raise RuntimeError(f'Ожидалось 17 семейств, найдено {len(families)}')
    gd=bm.FrozenPredictor(root,features,oxides)
    gn=FrozenGlassNet(oxides)
    print('=== GLASSDESIGN ML 2.5 — ШАГ 364 ===')
    print('НОВЫЕ универсальные задачи G/H/I; факторный эксперимент 2×2.')
    print('Во всех четырёх алгоритмах GD и GlassNet вызываются для каждого состава.')
    print('Каждому: 136 одинаковых pilot + 120 новых композиций на модель.')
    print('Нет fit(), чтения обучающих CSV, VALIDATION_GROUP или TEST_LOCKED.')
    print('Семейств:',len(families),'| Оксидов:',len(oxides),'| Условия:',len(TASKS)*len(SEEDS)*len(ARMS))
    out=root/'GlassDesignML_Upgrade/Consensus_GA_Development_step364'
    if (out/'per_run_364.csv').exists():
        raise FileExistsError('Результаты шага 364 уже существуют. Не перезаписываем.')
    out.mkdir(parents=True,exist_ok=True)
    names=['GlassDesignML_step364_dual_feedback.py','glass_consensus_ga_v25.py',
           'glass_hybrid_v21.py','glass_benchmark_v21.py','glass_core.py']
    hashfiles={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in names}
    lock={'step':364,'status':'development_not_experimental_validation',
          'tasks':{t:{k:list(v) for k,v in d.items()} for t,d in TASKS.items()},
          'seeds':list(SEEDS),'arms':list(map(list,ARMS)),
          'pilot_per_family':PILOT_PER_FAMILY,'max_families':MAX_FAMILIES,
          'new_budget_each_oracle':BUDGET,'batch_size':BATCH,'sources_sha256':hashfiles,
          'note':'All four arms see equal GD and GN composition-evaluation budgets; GD-only arms discard GN feedback.'}
    lock_file=out/'locked_protocol_364.json'
    if lock_file.exists():
        if json.loads(lock_file.read_text(encoding='utf-8'))!=lock:
            raise RuntimeError('Предыдущий протокол шага 364 отличается: остановка')
    else:
        lock_file.write_text(json.dumps(lock,indent=2,ensure_ascii=False),encoding='utf-8')
    # Cache identical pre-task pilot for each seed, avoiding redundant oracle calls.
    pilot_cache={}
    for seed in SEEDS:
        pilot=bm.prepare_pilot(families,oxides,gd,int(seed)*100003,PILOT_PER_FAMILY)
        expected=len(families)*PILOT_PER_FAMILY
        if len(pilot)!=expected:raise RuntimeError('Wrong pilot size')
        preds=gn(pilot)
        for name in opt.GN_NAMES.values():pilot[name]=preds[name].to_numpy(float)
        pilot_cache[seed]=pilot
    results=[]
    reports=[]
    clocks=[]
    for task,criteria in TASKS.items():
        for seed in SEEDS:
            pilot=pilot_cache[seed].copy()
            pilot['Goal_pass']=hy._feasible(pilot,criteria,True)
            pilot['Goal_penalty']=hy._penalty(pilot,criteria,True)
            chosen,_=family_pilot_ranking(pilot,families,MAX_FAMILIES)
            if len(chosen)!=MAX_FAMILIES:raise RuntimeError('Too few families')
            initial=pilot.loc[pilot.Glass_family.isin(chosen)].copy().reset_index(drop=True)
            if len(initial)!=MAX_FAMILIES*PILOT_PER_FAMILY:
                raise RuntimeError('Wrong selected pilot size')
            print(f'\n{task}, seed={seed}: selected {list(chosen)}',flush=True)
            for fam_policy,feedback in ARMS:
                label=f'{fam_policy}_{feedback}'
                g0,n0=gd.rows,gn.rows
                t0=time.perf_counter()
                rows=opt.generate(chosen,initial,gd,gn,oxides,criteria,seed,
                                  family_policy=fam_policy,feedback=feedback,
                                  budget=BUDGET,batch_size=BATCH)
                secs=time.perf_counter()-t0
                if gd.rows-g0!=BUDGET or gn.rows-n0!=BUDGET:
                    raise RuntimeError('Unequal GD/GlassNet budgets!')
                rows['Task']=task;rows['Seed']=seed;rows['Method']=label
                results.append(rows)
                reports.append({'Task':task,'Seed':seed,'Method':label,
                  'Family_policy':fam_policy,'Feedback':feedback,
                  'N_new':BUDGET,'GD_pass':int(rows.GD_pass.sum()),
                  'GN_pass':int(rows.GN_pass.sum()),'Both_pass':int(rows.Both_pass.sum()),
                  'Diverse_both_2mol':int(diversity_count(rows,oxides)),
                  'Families_both':int(rows.loc[rows.Both_pass,'Glass_family'].nunique()),
                  'GD_rows':int(gd.rows-g0),'GN_rows':int(gn.rows-n0),
                  'Seconds':float(secs)})
                print(f'   {label}: BOTH {int(rows.Both_pass.sum())}/120; '
                      f'diverse {diversity_count(rows,oxides)}; '
                      f'families {rows.loc[rows.Both_pass,"Glass_family"].nunique()}; '
                      f'{secs:.2f}s',flush=True)
    all_new=pd.concat(results,ignore_index=True)
    per_run=pd.DataFrame(reports)
    if len(per_run)!=36 or len(all_new)!=4320:
        raise RuntimeError('Incomplete factorial run')
    if not per_run[['GD_rows','GN_rows']].eq(BUDGET).all().all():
        raise RuntimeError('Inconsistent budgets')
    summary=per_run.groupby('Method',as_index=False).agg(
        Runs=('Both_pass','size'),Mean_GD=('GD_pass','mean'),
        Mean_GN=('GN_pass','mean'),Mean_both=('Both_pass','mean'),
        Mean_diverse=('Diverse_both_2mol','mean'),
        Mean_families=('Families_both','mean'),Mean_seconds=('Seconds','mean'))
    by_task=per_run.groupby(['Task','Method'],as_index=False).agg(
        Mean_both=('Both_pass','mean'),Mean_diverse=('Diverse_both_2mol','mean'),
        Mean_families=('Families_both','mean'))
    effects=factorial_effects(per_run)
    all_new.to_csv(out/'all_new_compositions_364.csv',index=False,encoding='utf-8-sig')
    per_run.to_csv(out/'per_run_364.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(out/'summary_364.csv',index=False,encoding='utf-8-sig')
    by_task.to_csv(out/'by_task_364.csv',index=False,encoding='utf-8-sig')
    effects.to_csv(out/'factorial_effects_364.csv',index=False,encoding='utf-8-sig')
    meta={'step':364,'python':platform.python_version(),
          'pilot_shared_per_seed':136,'pilot_selected_per_task_seed':64,
          'new_budget_per_arm':BUDGET,'new_GD_and_GN_per_arm':BUDGET,
          'final_validation':False,'uses_experimental_labels':False,
          'models_retrained':False,
          'primary_metric':'Both_pass GD AND GlassNet plus Tsoft>=Tg for each oracle',
          'caveat':'Both predictors may share training data and property-definition differences; consensus is not experimental truth.',
          'warning':'This is a new feedback-information setting; do not compare its computational cost directly with posthoc GN benchmarks unless counting GN calls.'}
    (out/'methodology_364.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== ШАГ 364: ЧЕТЫРЕ ВАРИАНТА 2×2 ===')
    display(summary.round(3))
    print('\n=== ШАГ 364: ПО НОВЫМ ЗАДАЧАМ ===')
    display(by_task.round(3))
    print('\n=== ШАГ 364: ФАКТОРНЫЕ ЭФФЕКТЫ ===')
    display(effects.round(3))
    print('\nШАГ 364 ЗАВЕРШЁН:',out)
    print('Существующие модели/приложение/обучающие данные и TEST_LOCKED не изменены.')
    return summary,by_task,effects

if __name__=='__main__':
    main()
