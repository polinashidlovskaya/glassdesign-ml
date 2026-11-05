"""GlassDesign ML 2.1 locked-protocol comparative benchmark.

Uses only frozen joblib predictors and synthetic candidate compositions.
Never reads any original training/evaluation CSV, TEST_LOCKED or YAG records.
Compares random, existing NSGA-II, and 2.1 hybrid under shared pilot and
identical NEW composition prediction budgets. A priori task definitions.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import platform

import joblib
import numpy as np
import pandas as pd

PROPS=('Density_g_cm3','RefractiveIndex','Tg_C','Tsoft_C')
KEYS={
    'Плотность':'Density_g_cm3',
    'Показатель преломления':'RefractiveIndex',
    'Температура стеклования':'Tg_C',
    'Температура размягчения':'Tsoft_C'
}
# A priori, universal optical/thermal targets; no YAG objectives.
TASKS={
    'A_mid_optical':{
        'Density_g_cm3':(3.2,4.5),'RefractiveIndex':(1.60,1.78)},
    'B_high_index':{
        'Density_g_cm3':(4.4,6.0),'RefractiveIndex':(1.78,2.02),'Tsoft_C':(None,750)},
    'C_low_temp':{
        'Density_g_cm3':(2.6,5.0),'Tg_C':(320,570),'Tsoft_C':(410,720)},
}


def find_root():
    cands=[Path('/content/glassdesign_mount_824a9be5/MyDrive/YAG_Inverse_Design')]
    cands+=sorted(Path('/content').glob('glassdesign_mount_*/MyDrive/YAG_Inverse_Design'))
    cands+=sorted(Path('/content').glob('glassdesign_drive_*/MyDrive/YAG_Inverse_Design'))
    cands.append(Path('/content/drive/MyDrive/YAG_Inverse_Design'))
    return next((p for p in cands if (p/'GlassDesignML/model_features.json').is_file() and
                 (p/'GlassDesignML_Upgrade/glass_nsga2_universal.py').is_file()),None)


def select_paths(root,manifest):
    base=root/'GlassDesignML'
    fallback={
        'Плотность':[base/'models/density_extra_trees_final.joblib',root.parent/'density_extra_trees_final.joblib'],
        'Показатель преломления':[base/'models/refractive_index_extra_trees_final_v2.joblib',root.parent/'refractive_index_extra_trees_final_v2.joblib'],
        'Температура стеклования':[base/'models/rf_tg_final.joblib',root/'Tg/rf_tg_final.joblib'],
        'Температура размягчения':[base/'models/extra_trees_softening_final.joblib',root/'Softening/extra_trees_softening_final.joblib']
    }
    paths={}
    for key,items in fallback.items():
        filenames=[base/'models'/manifest[key]['filename'],*items]
        found=next((p for p in filenames if p.is_file()),None)
        if found is None:
            raise FileNotFoundError(f'{key}: cannot find model, paths: {filenames}')
        paths[key]=found
    return paths


def wide_family_filter(catalog, oxides, max_components=12):
    """Extend original max-7 family screening to 2..12 components.

    All four algorithms get the SAME extended catalogue. The original NSGA-II
    optimizer itself accepts variable n_var; its pipeline wrapper rejected 8+.
    """
    from glass_core import validate_bounds
    valid={}
    rejected=[]
    for name,meta in catalog.items():
        bounds=meta.get('bounds',meta) if isinstance(meta,dict) else meta
        try:
            checked=validate_bounds(bounds,oxides)
            if not 2<=len(checked)<=max_components:
                raise ValueError('Component number outside 2..%d'%max_components)
            valid[name]=checked
        except (ValueError,TypeError,KeyError,AttributeError) as exc:
            rejected.append({'Glass_family':name,'Reason':str(exc)})
    if not valid:
        raise RuntimeError('No supported families')
    return valid,pd.DataFrame(rejected,columns=['Glass_family','Reason'])


class FrozenPredictor:
    """Load four frozen estimators once. Count predictions per method in wrapper."""
    def __init__(self,root,feature_dict,oxides):
        self.oxides=list(oxides)
        manifest=json.loads((root/'GlassDesignML/model_manifest.json').read_text(encoding='utf-8'))
        paths=select_paths(root,manifest)
        self.models=[]
        for label,col in KEYS.items():
            package=joblib.load(paths[label])
            model=package.get('model') if isinstance(package,dict) else package
            if model is None or not callable(getattr(model,'predict',None)):
                raise TypeError(f'{label}: invalid joblib')
            own=list(map(str,model.feature_names_in_)) if hasattr(model,'feature_names_in_') else None
            feat=list(map(str,feature_dict[label]))
            if own is not None and own != feat:
                raise ValueError(f'{label}: features in estimator disagree with manifest')
            if isinstance(package,dict) and 'features' in package and list(package['features'])!=feat:
                raise ValueError(f'{label}: artifact feature order differs')
            if getattr(model,'n_features_in_',len(feat))!=len(feat):
                raise ValueError(f'{label}: n_features inconsistent')
            if set(self.oxides)-set(feat):
                raise ValueError(f'{label}: common oxides not supported')
            if hasattr(model,'n_jobs'):
                model.n_jobs=1  # in-memory tuning only
            self.models.append((label,col,model,feat,own is not None,str(paths[label])))
        self.calls=0
        self.rows=0
    def __call__(self,df):
        if not isinstance(df,pd.DataFrame) or not len(df):
            raise ValueError('Empty candidate table')
        X=df[self.oxides].to_numpy(float)
        if not np.isfinite(X).all() or (X<-1e-8).any() or not np.allclose(X.sum(axis=1),100,atol=1e-5):
            raise ValueError('Each oxide composition must sum to 100 mol.%')
        out=pd.DataFrame({'Candidate_ID':df['Candidate_ID'].to_numpy()})
        for label,col,model,feat,has_names,path in self.models:
            inp=df.reindex(columns=feat,fill_value=0.0).astype(float).div(100.)
            if not has_names:
                inp=inp.to_numpy()
            pred=np.asarray(model.predict(inp),float).reshape(-1)
            if len(pred)!=len(df) or not np.isfinite(pred).all():
                raise RuntimeError(f'{label}: invalid predictions')
            out[col]=pred
        self.calls+=1
        self.rows+=len(df)
        return out


def prepare_pilot(families,oxides,predictor,seed,count=8):
    from glass_core import generate_compositions
    blocks=[]
    for i,(name,bounds) in enumerate(families.items()):
        raw=generate_compositions({name:bounds},oxides,count,random_seed=seed+7919*i)
        blocks.append(raw)
    pilot=pd.concat(blocks,ignore_index=True)
    pilot['Candidate_ID']=np.arange(1,len(pilot)+1)
    pred=predictor(pilot)
    for p in PROPS:
        pilot[p]=pred[p].to_numpy(float)
    return pilot


def random_baseline(families,pilot,predictor,oxides,criteria,budget,seed,consistency=True):
    from glass_core import sample_composition
    from glass_hybrid_v21 import _feasible,_penalty,_key
    rng=np.random.default_rng(seed)
    names=list(families)
    seen={_key(x) for x in pilot[oxides].to_numpy(float)}
    records=[]
    max_attempts=1000*budget
    tries=0
    while len(records)<budget:
        tries+=1
        if tries>max_attempts:
            raise RuntimeError('Random baseline could not fill budget')
        name=names[len(records)%len(names)]
        c=sample_composition(families[name],rng)
        row={ox:float(c.get(ox,0)) for ox in oxides}
        k=_key([row[ox] for ox in oxides])
        if k in seen:
            continue
        seen.add(k)
        row.update({'Glass_family':name,'Candidate_ID':int(pilot['Candidate_ID'].max())+1+len(records),
                    'Search_stage':'Random','Search_round':0})
        records.append(row)
    batch=pd.DataFrame(records)
    pred=predictor(batch)
    for prop in PROPS:
        batch[prop]=pred[prop].to_numpy(float)
    batch['Goal_pass']=_feasible(batch,criteria,consistency)
    batch['Goal_penalty']=_penalty(batch,criteria,consistency)
    batch['Delta_T_C']=batch['Tsoft_C']-batch['Tg_C']
    return batch


def adaptive_legacy_baseline(families,pilot,predictor,oxides,criteria,budget,seed,consistency=True):
    """Fair-budget wrapper using the EXISTING glass_adaptive_search mutation kernel.

    Not the app's entire three-stage adaptive_multiple_families function;
    shares exactly the same pilot and NEW prediction budget as other methods.
    """
    from glass_adaptive_search import generate_adaptive_batch
    from glass_hybrid_v21 import _feasible,_penalty,_key
    rng=np.random.default_rng(seed)
    names=list(families)
    states={name:pilot.loc[pilot['Glass_family'].eq(name)].copy() for name in names}
    for name,frame in states.items():
        frame['Constraint_penalty']=_penalty(frame,criteria,consistency)
    seen={_key(x) for x in pilot[oxides].to_numpy(float)}
    new=[]
    next_id=int(pilot['Candidate_ID'].max())+1
    plans={name:budget//len(names)+int(i<budget%len(names)) for i,name in enumerate(names)}
    round_ix=0
    while any(value>0 for value in plans.values()):
        blocks=[]
        for name in names:
            count=min(8,plans[name])
            if count==0:
                continue
            history=states[name]
            block=generate_adaptive_batch(history,families[name],oxides,count,rng,
                                          exploration_fraction=0.30,family_name=name)
            if len(block)!=count:
                raise RuntimeError(f'Adaptive legacy failed budget for {name}')
            for x in block[oxides].to_numpy(float):
                key=_key(x)
                if key in seen:
                    raise RuntimeError(f'Legacy adaptive duplicate composition in {name}')
                seen.add(key)
            block['Candidate_ID']=np.arange(next_id,next_id+len(block))
            next_id+=len(block)
            block['Search_stage']='Adaptive-legacy'
            block['Search_round']=round_ix
            blocks.append(block)
            plans[name]-=count
        batch=pd.concat(blocks,ignore_index=True)
        pred=predictor(batch)
        for prop in PROPS:
            batch[prop]=pred[prop].to_numpy(float)
        batch['Goal_pass']=_feasible(batch,criteria,consistency)
        batch['Goal_penalty']=_penalty(batch,criteria,consistency)
        batch['Delta_T_C']=batch['Tsoft_C']-batch['Tg_C']
        batch['Constraint_penalty']=batch['Goal_penalty']
        for name in names:
            part=batch.loc[batch['Glass_family'].eq(name)]
            if len(part):
                states[name]=pd.concat((states[name],part),ignore_index=True)
        new.append(batch)
        round_ix+=1
    return pd.concat(new,ignore_index=True)


def diversity_metric(df,oxides,threshold=2.0):
    """Greedy unique feasible designs separated >=threshold mol.% half-L1.
    Sorting by constraint penalty then family stabilizes interpretation.
    """
    hits=df.loc[df['Goal_pass']].sort_values(['Goal_penalty','Glass_family','Candidate_ID'])
    kept=[]
    for x in hits[oxides].to_numpy(float):
        if not kept or np.all(np.sum(np.abs(np.asarray(kept)-x),axis=1)*.5>=threshold-1e-9):
            kept.append(x.copy())
    return len(kept)


def _check_output(df,oxides,n,pilot,criteria,consistency):
    from glass_hybrid_v21 import _feasible,_key
    if len(df)!=n or df['Candidate_ID'].duplicated().any():
        raise AssertionError('Invalid budget or candidate ID')
    X=df[oxides].to_numpy(float)
    initial=pilot[oxides].to_numpy(float)
    if not np.isfinite(X).all() or (X<0).any() or not np.allclose(X.sum(axis=1),100,atol=1e-5):
        raise AssertionError('Invalid composition')
    keys=[_key(x) for x in X]
    if len(set(keys))!=len(keys):
        raise AssertionError('Repeated new composition')
    if set(keys)&{_key(x) for x in initial}:
        raise AssertionError('New composition duplicates shared pilot')
    for col in PROPS:
        if not np.isfinite(df[col].to_numpy(float)).all():
            raise AssertionError('Invalid prediction')
    actual=_feasible(df,criteria,consistency)
    if not np.array_equal(actual,df['Goal_pass'].to_numpy(bool)):
        raise AssertionError('Feasibility inconsistent')


def _metrics(df,oxides,task,method,seed,pilot_len,elapsed,calls):
    from glass_hybrid_v21 import _penalty
    hits=df.loc[df['Goal_pass']]
    diverse=diversity_metric(df,oxides,threshold=2.0)
    return {'Task':task,'Method':method,'Seed':seed,
            'Pilot_shared':pilot_len,'New_budget':len(df),'Total_budget':pilot_len+len(df),
            'New_passed':len(hits),'New_success_pct':100*len(hits)/len(df),
            'Diverse_passed_2mol':diverse,
            'Diverse_passed_pct':100*diverse/len(df),
            'Distinct_successful_families':hits['Glass_family'].nunique(),
            'Median_penalty_new':float(df['Goal_penalty'].median()),
            'Model_calls_new':calls,'Wall_seconds_incl_predictions':elapsed}


def run_benchmark(root,stage_dir,seed_values=(7,17,37),new_budget=120,
                  pilot_per_family=8,max_families=8,tasks=None):
    """Execute frozen, paired benchmark and save trace files and summary.
    Use same pilot, tasks and new-budget for all 3 algorithms.
    """
    from glass_universal_pipeline_v3 import family_pilot_ranking,violation
    from glass_nsga2_universal import optimize_nsga2,feasibility
    from glass_hybrid_v21 import optimize_hybrid_v21,_penalty
    import pymoo
    root=Path(root)
    stage_dir=Path(stage_dir)
    stage_dir.mkdir(parents=True,exist_ok=True)
    features=json.loads((root/'GlassDesignML/model_features.json').read_text(encoding='utf-8'))
    oxides=sorted(set.intersection(*(set(v) for v in features.values())))
    catalog=json.loads((root/'GlassDesignML/glass_families.json').read_text(encoding='utf-8'))
    if isinstance(catalog.get('families'),dict):
        catalog=catalog['families']
    families,rejected=wide_family_filter(catalog,oxides)
    if len(families)<2:
        raise RuntimeError('Not enough available families')
    if new_budget < max_families*2:
        raise ValueError('Budget too low')
    tasks=dict(tasks or TASKS)
    script_dir=Path(__file__).resolve().parent
    code_hashes={name:hashlib.sha256((script_dir/name).read_bytes()).hexdigest()
                 for name in ('glass_hybrid_v21.py','glass_benchmark_v21.py')}
    protocol={'tasks':tasks,'seeds':list(seed_values),'new_budget':new_budget,
              'pilot_per_family':pilot_per_family,'max_families':max_families,
              'candidate_code_hashes':code_hashes}
    protocol_file=stage_dir/'locked_protocol_step356.json'
    current_json=json.dumps(protocol,ensure_ascii=False,sort_keys=True)
    if protocol_file.exists():
        previous=json.loads(protocol_file.read_text(encoding='utf-8'))
        if json.dumps(previous,ensure_ascii=False,sort_keys=True)!=current_json:
            raise RuntimeError('Benchmark protocol differs from existing frozen run. Use a different output folder.')
    else:
        protocol_file.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    predictor=FrozenPredictor(root,features,oxides)
    outputs=[]
    run_diagnostics=[]
    print('Свежая проверка рабочих моделей:',[(k,len(f)) for k,_,_,f,_,_ in predictor.models])
    print('Семейств после проверки:',len(families),'| бюджет пилота:',len(families)*pilot_per_family)
    print('Задач:',len(tasks),'| повторов:',len(seed_values),'| новых составов/метод:',new_budget)
    rejected.to_csv(stage_dir/'rejected_families.csv',index=False,encoding='utf-8-sig')
    for task,criteria in tasks.items():
        for seed in seed_values:
            print(f'\n=== {task}; seed={seed} ===',flush=True)
            pilot=prepare_pilot(families,oxides,predictor,int(seed)*100003,pilot_per_family)
            pilot['Goal_pass']=feasibility(pilot,criteria,True)
            pilot['Goal_penalty']=violation(pilot,criteria,True)
            chosen,rank=family_pilot_ranking(pilot,families,max_families)
            selected=pilot.loc[pilot['Glass_family'].isin(chosen)].copy()
            # Persist shared pilot in single file, not copied per algorithm.
            tag=f'{task}_s{seed}'
            pilot.to_csv(stage_dir/f'pilot_{tag}.csv',index=False,encoding='utf-8-sig')
            rank.to_csv(stage_dir/f'family_rank_{tag}.csv',index=False,encoding='utf-8-sig')
            print('Выбрано семейств:',list(chosen),flush=True)
            for method in ('Random','Adaptive-legacy','NSGA-II','Hybrid-2.1'):
                start_calls=predictor.calls
                start_rows=predictor.rows
                started=time.perf_counter()
                if method=='Random':
                    new=random_baseline(chosen,selected,predictor,oxides,criteria,new_budget,seed+500,True)
                    diagnostics=pd.DataFrame()
                elif method=='Adaptive-legacy':
                    new=adaptive_legacy_baseline(chosen,selected,predictor,oxides,criteria,new_budget,seed+700,True)
                    diagnostics=pd.DataFrame()
                elif method=='NSGA-II':
                    new,diagnostics=optimize_nsga2(chosen,selected,predictor,oxides,
                            criteria,new_budget,seed=seed+1000,consistency=True,max_generations=130)
                    new['Goal_penalty']=_penalty(new,criteria,True)
                else:
                    new,diagnostics=optimize_hybrid_v21(chosen,selected,predictor,oxides,
                            criteria,new_budget,seed=seed+2000,consistency=True,
                            batch_size=8,exploration_floor=2)
                elapsed=time.perf_counter()-started
                used=predictor.rows-start_rows
                if used!=new_budget:
                    raise AssertionError(f'{method}: actually predicted {used} rows, expected {new_budget}')
                _check_output(new,oxides,new_budget,pilot,criteria,True)
                new['Benchmark_method']=method
                new['Benchmark_task']=task
                new['Benchmark_seed']=seed
                new.to_csv(stage_dir/f'candidates_{tag}_{method.replace("-","_")}.csv',index=False,encoding='utf-8-sig')
                if not diagnostics.empty:
                    diagnostics.to_csv(stage_dir/f'allocation_{tag}_{method.replace("-","_")}.csv',index=False,encoding='utf-8-sig')
                row=_metrics(new,oxides,task,method,seed,len(pilot),elapsed,predictor.calls-start_calls)
                outputs.append(row)
                pd.DataFrame(outputs).to_csv(stage_dir/'all_runs_live.csv',index=False,encoding='utf-8-sig')
                print(f'  {method}: {row["New_passed"]}/{new_budget} попаданий; '
                      f'разнообразных ≥2 мол.%: {row["Diverse_passed_2mol"]}; '
                      f'семейств: {row["Distinct_successful_families"]}',flush=True)
    allruns=pd.DataFrame(outputs)
    allruns.to_csv(stage_dir/'benchmark_all_runs.csv',index=False,encoding='utf-8-sig')
    report=allruns.groupby('Method',as_index=False).agg(
        Runs=('New_passed','count'),Mean_hits=('New_passed','mean'),
        Median_hits=('New_passed','median'),
        Mean_diverse_2mol=('Diverse_passed_2mol','mean'),
        Mean_families=('Distinct_successful_families','mean'),
        Mean_success_pct=('New_success_pct','mean'))
    report.to_csv(stage_dir/'benchmark_summary.csv',index=False,encoding='utf-8-sig')
    paired=allruns.pivot_table(index=['Task','Seed'],columns='Method',
                               values=['New_passed','Diverse_passed_2mol'])
    comparison=[]
    for baseline in ('NSGA-II','Adaptive-legacy','Random'):
        for metric in ('New_passed','Diverse_passed_2mol'):
            delta=(paired[(metric,'Hybrid-2.1')]-paired[(metric,baseline)])
            comparison.append({'Against':baseline,'Metric':metric,
                'N_paired':len(delta),'Mean_delta':float(delta.mean()),
                'Median_delta':float(delta.median()),
                'Wins':int((delta>0).sum()),'Ties':int((delta==0).sum()),
                'Losses':int((delta<0).sum())})
    comparison=pd.DataFrame(comparison)
    comparison.to_csv(stage_dir/'paired_comparison.csv',index=False,encoding='utf-8-sig')
    per_task=allruns.groupby(['Task','Method'],as_index=False).agg(
        N_seeds=('Seed','count'),Mean_hits=('New_passed','mean'),
        SD_hits=('New_passed','std'),Mean_diverse=('Diverse_passed_2mol','mean'),
        Mean_families=('Distinct_successful_families','mean'))
    per_task.to_csv(stage_dir/'benchmark_by_task.csv',index=False,encoding='utf-8-sig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for column,axis_label,filename in [
        ('New_passed','Number of predicted in-target compositions','figure_hits_356.png'),
        ('Diverse_passed_2mol','Predicted in-target compositions >=2 mol% apart','figure_diversity_356.png'),
        ('Distinct_successful_families','Families with at least one in-target composition','figure_family_coverage_356.png')]:
        plt.figure(figsize=(9.5,4.8))
        for method in ('Random','Adaptive-legacy','NSGA-II','Hybrid-2.1'):
            part=allruns.loc[allruns.Method.eq(method)]
            mean=part.groupby('Task')[column].mean().reindex(list(tasks))
            plt.plot(range(len(tasks)),mean.to_numpy(),marker='o',label=method)
        plt.xticks(range(len(tasks)),list(tasks),rotation=10)
        plt.ylabel(axis_label)
        plt.title('GlassDesign ML: matched model-evaluation budgets')
        plt.legend()
        plt.grid(axis='y',alpha=0.2)
        plt.tight_layout()
        plt.savefig(stage_dir/filename,dpi=220)
        plt.close()
    relevant=[root/'GlassDesignML_Upgrade/glass_nsga2_universal.py',
              root/'GlassDesignML_Upgrade/glass_universal_pipeline_v3.py',
              root/'GlassDesignML/glass_predictor.py']
    manifest={
        'benchmark_type':'A priori computational surrogate-guided comparison; NOT external measured validation',
        'version':'GlassDesign ML 2.1 experimental candidate',
        'seed_values':list(seed_values),'new_budget_per_algorithm':new_budget,
        'pilot_per_family':pilot_per_family,'all_pilot_family_count':len(families),
        'selected_family_count':min(max_families,len(families)),
        'target_tasks':tasks,'methods':['Random','Adaptive-legacy','NSGA-II','Hybrid-2.1'],
        'model_paths':{k:path for k,_,_,_,_,path in predictor.models},
        'original_source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in relevant},
        'candidate_code_hashes':code_hashes,
        'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
        'pymoo':getattr(pymoo,'__version__','unknown'),
        'source_of_models':'Previously trained; no .fit() and no training data read',
        'family_extension':'Shared catalog expands original 2-7 to 2-12 oxides, allowing 8-oxide families for ALL algorithms',
        'data_isolation':'No TRAIN_FIT / VALIDATION_GROUP / TEST_LOCKED accessed',
        'caveats':['Optimizing the same prediction model used for hit scoring can reward model exploitation.',
            'No experimental verification, no claim of global superiority.',
            'Only internal optimizers tested here; external methods require separate reproducible adaptation.',
            'Legacy adaptive kernel is wrapped for identical shared-pilot conditions (not the complete app three-stage adaptive search).',
            'Same shared pilot and equal NEW evaluations; discovery budget includes the pilot.',
            'Diverse designs separated by >=2 mol.% half-L1 are not necessarily experimentally novel.']}
    (stage_dir/'benchmark_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return allruns,report,comparison
