"""Lightweight per-recipe evidence. Not external experimental validation.

No global scientific model and no UI decisions live here.
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

OXIDE_PATTERN = re.compile(r'^[A-Z][a-z]?\d*O\d*$')
SOURCES = {
    'Density_g_cm3': ('Source_Data/density_dataset_clean.csv', 'Density293K'),
    'RefractiveIndex': ('GlassDesignML_Upgrade/refractive_training_reconstructed.csv','RefractiveIndex'),
    'Tg_C': ('Tg/tg_train_clean.csv','Tg_C'),
    'Tsoft_C': ('Softening/softening_train_clean.csv','Tsoft_C'),
}


def near_analogues(candidates, criteria, root, max_half_l1=5.0, chunk_size=7000):
    """For EACH criterion find nearest measured recipe in relevant dataset.

    Output: None if reference wasn't checked, True if <= 5mol% for ALL
    properties selected, False otherwise. Distances are half-L1 (molar %).
    For fully measured reference entries only.
    """
    root=Path(root)
    oxides=[x for x in candidates.columns if OXIDE_PATTERN.fullmatch(str(x))]
    if not oxides:
        raise ValueError('В составе нет столбцов оксидов')
    cand=candidates[oxides].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
    if (not np.isfinite(cand).all() or (cand < 0).any()
            or not np.allclose(cand.sum(axis=1),100.,atol=0.03,rtol=0)):
        raise ValueError('Неверные значения мол.% у кандидатов')
    distances={}
    failures={}
    for prop in criteria:
        filename,target=SOURCES[prop]
        source=root / filename
        if not source.is_file():
            failures[prop]='Нет экспериментальной базы'
            distances[prop]=None
            continue
        try:
            columns=list(pd.read_csv(source,nrows=0).columns)
            refox=[x for x in columns if OXIDE_PATTERN.fullmatch(str(x))]
            if target not in columns or len(refox)<2:
                raise ValueError('В экспериментальном справочнике нет измерения или оксидов')
            allox=sorted(set(oxides)|set(refox))
            indices={x:i for i,x in enumerate(allox)}
            q=np.zeros((len(candidates),len(allox)),dtype=np.float32)
            for j,ox in enumerate(oxides):q[:,indices[ox]]=cand[:,j]
            best=np.full(len(candidates),np.inf,dtype=float)
            nvalid=0
            for part in pd.read_csv(source,usecols=refox+[target],chunksize=chunk_size,low_memory=False):
                data=part[refox].apply(pd.to_numeric,errors='coerce').to_numpy(np.float32)
                y=pd.to_numeric(part[target],errors='coerce').to_numpy(float)
                sums=data.sum(axis=1,dtype=np.float64)
                valid=(np.isfinite(data).all(axis=1)&(data>=0).all(axis=1)&
                       np.isfinite(y)&(y>0)&
                       (np.isclose(sums,1.,atol=0.015,rtol=0)|
                        np.isclose(sums,100.,atol=1.5,rtol=0)))
                if not valid.any():continue
                data=data[valid]
                frac=np.isclose(sums[valid],1.,atol=0.015,rtol=0)
                data[frac]*=100.
                nvalid+=len(data)
                ref=np.zeros((len(data),len(allox)),dtype=np.float32)
                for j,ox in enumerate(refox):ref[:,indices[ox]]=data[:,j]
                # Blocks keep RAM controlled; scipy native implementation
                best=np.minimum(best,cdist(q,ref,metric='cityblock').min(axis=1)*0.5)
            if not nvalid:
                raise ValueError('В справочнике нет корректных измерений')
            distances[prop]=best
        except (OSError,ValueError,KeyError,MemoryError) as e:
            distances[prop]=None
            failures[prop]=str(e)
    statuses=[]
    nearest=[]
    for i in range(len(candidates)):
        values=[v[i] for v in distances.values() if v is not None]
        if len(values)!=len(criteria):
            statuses.append(None)
            nearest.append(float(max(values)) if values else None)
        else:
            distance=float(max(values))
            nearest.append(distance)
            statuses.append(distance<=max_half_l1)
    return statuses,nearest,failures


def glass_formation_isolated(candidates, oxides, worker_file, timeout=80):
    """Run VITRIFY in a child process with timeout and memory cap.
    Failure means 'not checked' not negative glass-forming prediction.
    """
    with tempfile.TemporaryDirectory(prefix='glassdesign_vitrify_') as tmp:
        inp=Path(tmp)/'input.csv'
        out=Path(tmp)/'out.json'
        cols=[x for x in oxides if x in candidates.columns and
              pd.to_numeric(candidates[x],errors='coerce').fillna(0).abs().gt(1e-8).any()]
        if not cols:return [None]*len(candidates),'Нет ненулевых оксидов'
        candidates[cols].to_csv(inp,index=False)
        env=os.environ.copy()
        env.update({'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                    'MKL_NUM_THREADS':'1','MALLOC_ARENA_MAX':'2','NUMEXPR_MAX_THREADS':'1'})
        # On Linux, cap worker virtual address space as an OOM safeguard.
        def limit_memory():
            try:
                import resource
                memory=int(5.5*1024**3)
                resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
            except Exception:pass
        try:
            done=subprocess.run([sys.executable,str(worker_file),str(inp),str(out)],
                capture_output=True,text=True,timeout=timeout,env=env,
                preexec_fn=limit_memory if os.name=='posix' else None)
            if done.returncode!=0:
                reason=(done.stderr or done.stdout or 'worker завершился').strip()
                return [None]*len(candidates),reason[-600:]
            values=json.loads(out.read_text(encoding='utf-8'))
            if len(values)!=len(candidates):
                raise ValueError('Неправильное количество прогнозов стеклообразования')
            scores=[float(x) for x in values]
            if not np.isfinite(scores).all() or not all(0<=x<=1 for x in scores):
                raise ValueError('Некорректные значения VITRIFY')
            return scores,None
        except (subprocess.TimeoutExpired, OSError, ValueError) as e:
            return [None]*len(candidates),str(e)[-600:]


def consistent(row):
    values=[row.get(p,np.nan) for p in ['Density_g_cm3','RefractiveIndex','Tg_C','Tsoft_C']]
    if not all(pd.notna(x) and np.isfinite(float(x)) for x in values):
        return None
    return float(values[0])>0 and float(values[1])>0 and float(values[3])>=float(values[2])
