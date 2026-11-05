"""GlassDesign 2.1 experimental optimizer: constrained, adaptive-family hybrid search.

Always evaluates exactly `new_budget` previously unseen compositions; pilot
compositions are provided and never charged as new. This is an untrained search
algorithm, not a new regression model and not a proven improvement.

No YAG chemistry, no hidden target preferences, no TEST_LOCKED access.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

PROPERTIES = ('Density_g_cm3', 'RefractiveIndex', 'Tg_C', 'Tsoft_C')
SCALES = {'Density_g_cm3': 0.5, 'RefractiveIndex': 0.05, 'Tg_C': 50., 'Tsoft_C': 50.}


def _project(v, lower, upper):
    if lower.sum() > 100 + 1e-7 or upper.sum() < 100 - 1e-7:
        raise ValueError('Infeasible composition bounds')
    left, right = float(np.min(v - upper) - 101), float(np.max(v - lower) + 101)
    for _ in range(75):
        mid = (left + right) * 0.5
        if np.clip(v - mid, lower, upper).sum() > 100:
            left = mid
        else:
            right = mid
    ans = np.clip(v - (left + right) * 0.5, lower, upper)
    return ans


def _feasible(df, criteria, consistency):
    ok = np.ones(len(df), bool)
    for key, (low, high) in criteria.items():
        a = df[key].to_numpy(float)
        ok &= np.isfinite(a)
        if low is not None:
            ok &= a >= float(low) - 1e-9
        if high is not None:
            ok &= a <= float(high) + 1e-9
    if consistency:
        ok &= df['Tsoft_C'].to_numpy(float) + 1e-9 >= df['Tg_C'].to_numpy(float)
    return ok


def _penalty(df, criteria, consistency):
    total = np.zeros(len(df), float)
    for name, (lo, hi) in criteria.items():
        x = df[name].to_numpy(float)
        if lo is not None:
            total += np.maximum(float(lo)-x, 0)/SCALES[name]
        if hi is not None:
            total += np.maximum(x-float(hi), 0)/SCALES[name]
    if consistency:
        total += np.maximum(df['Tg_C'].to_numpy(float)-df['Tsoft_C'].to_numpy(float),0)/50.
    return total


def _key(v):
    return tuple(np.round(np.asarray(v, float), 6))


class _Family:
    def __init__(self, name, bounds, seed_rows, oxides, rng):
        self.name = name
        self.names = list(bounds)
        self.lo = np.asarray([bounds[o][0] for o in self.names], float)
        self.hi = np.asarray([bounds[o][1] for o in self.names], float)
        self.rng = rng
        self.nnew = 0
        self.nhits = 0
        self.best_gain = 0.
        self.best_pilot = np.inf
        self.best = np.inf
        self.pool = seed_rows.copy().reset_index(drop=True)
        self.seen = {_key(v) for v in self.pool[self.names].to_numpy(float)}
        if not len(self.pool) or not set(self.names).issubset(self.pool):
            raise ValueError(f'{name}: no usable pilot compositions')
        if len(self.seen) != len(self.pool):
            raise ValueError(f'{name}: duplicated pilot compositions')
        self.oxides = list(oxides)

    def _uniform(self):
        # Random simplex projection is not uniform in simplex, but baseline uses
        # the same sampler; this is exploratory sampling with valid constraints.
        return _project(self.rng.uniform(self.lo, self.hi), self.lo, self.hi)

    def _mutate(self, elites, scale):
        if len(elites) >= 2 and self.rng.random() < 0.65:
            p, q = self.rng.choice(len(elites), 2, replace=False)
            alpha = self.rng.uniform(0.05, .95)
            parent = alpha*elites[p] + (1-alpha)*elites[q]
        else:
            parent = elites[self.rng.integers(len(elites))].copy()
        child = parent + self.rng.normal(0, scale, len(parent))
        if len(parent)>1:
            # Exact mass-conserving transfer in addition to projection.
            i,j = self.rng.choice(len(parent), 2, replace=False)
            d = self.rng.uniform(-scale, scale)
            child[i] += d
            child[j] -= d
        return _project(child, self.lo, self.hi)

    def propose(self, count, criteria, consistency, global_seen):
        source = self.pool.copy()
        source['_penalty'] = _penalty(source,criteria,consistency)
        source['_hit'] = _feasible(source,criteria,consistency)
        # Once feasibility is reached, preserve diversity rather than only
        # optimizing a single point. Elites include both good and close-to-goal.
        source = source.sort_values(['_penalty', '_hit'], ascending=[True,False])
        elite_n = max(3,min(len(source), math.ceil(.25*len(source))))
        elites = source.head(elite_n)[self.names].to_numpy(float)
        existing = source[self.names].to_numpy(float)
        result=[]
        accepted_keys=set()
        attempts=0
        # Oversample without model calls. Select a mixture of promising search
        # neighbourhoods and far-away compositions to protect diversity.
        pool_target=max(count*5, 30)
        candidates=[]
        while len(candidates)<pool_target and attempts < max(2000,pool_target*60):
            attempts += 1
            r=self.rng.random()
            if r < 0.25:
                x=self._uniform()
            elif r < 0.60:
                x=self._mutate(elites,scale=2.0)
            elif r < 0.86:
                x=self._mutate(elites,scale=6.0)
            else:
                x=self._mutate(elites,scale=12.0)
            k=_key(x)
            full = np.zeros(len(self.oxides), float)
            full[[self.oxides.index(o) for o in self.names]] = x
            if _key(full) in global_seen or k in accepted_keys:
                continue
            candidates.append(x)
            accepted_keys.add(k)
        if len(candidates)<count:
            raise RuntimeError(f'{self.name}: exhausted proposals {len(candidates)} < {count}')
        c=np.asarray(candidates)
        # Prefer diverse compositions among elites, without using labels of
        # unseen candidates. Distances are half-L1 in mol.%.
        points=existing.copy()
        chosen=[]
        available=np.ones(len(c),bool)
        for index in range(count):
            dist=np.abs(c[:,None,:]-points[None,:,:]).sum(axis=2).min(axis=1)*.5
            dist[~available]=-np.inf
            if index == 0:
                # Propose the nearest perturbation half the time, and a novel
                # family point otherwise; seeded deterministic stochasticity.
                if self.rng.random()<0.50:
                    target=elites[0]
                    nearest=np.abs(c-target).sum(axis=1)*.5
                    nearest[~available]=np.inf
                    chosen_idx=int(np.argmin(nearest))
                else:
                    chosen_idx=int(np.argmax(dist))
            elif index % 4 == 3:
                chosen_idx=int(np.argmax(dist))
            else:
                # Penalize nearly identical compositions, but exploit locally.
                target=elites[self.rng.integers(len(elites))]
                neard=np.abs(c-target).sum(axis=1)*.5
                score=-neard + np.minimum(dist,3)*0.7
                score[~available]=-np.inf
                chosen_idx=int(np.argmax(score))
            x=c[chosen_idx]
            chosen.append(x)
            available[chosen_idx]=False
            points=np.vstack((points,x))
        for x in chosen:
            full = np.zeros(len(self.oxides), float)
            full[[self.oxides.index(o) for o in self.names]] = x
            global_seen.add(_key(full))
            self.seen.add(_key(x))
        return np.asarray(chosen)

    def update(self, new_rows, criteria, consistency):
        p=_penalty(new_rows,criteria,consistency)
        hit=_feasible(new_rows,criteria,consistency)
        self.nnew+=len(new_rows)
        self.nhits+=int(hit.sum())
        new_best=float(np.min(p))
        gain=max(0.,self.best-new_best)
        self.best_gain+=min(gain,5.)
        self.best=min(self.best,new_best)
        self.pool=pd.concat((self.pool,new_rows),ignore_index=True)


def optimize_hybrid_v21(families, initial, predictor, oxide_columns, criteria,
                        new_budget=120, seed=3561, consistency=True,
                        batch_size=8, exploration_floor=2):
    """Return (new_predictions, budget_diagnostics) with *exactly* new_budget.

    Adaptively allocates batches using an upper-confidence score with a nonzero
    exploration floor per family. Pilot rows must have properties from the SAME
    frozen predictor. No training, no validation labels, no GlassNet data.
    """
    if not isinstance(families,dict) or not families:
        raise ValueError('No families')
    if not criteria or set(criteria)-set(PROPERTIES):
        raise ValueError('Invalid criteria')
    if not set(PROPERTIES).issubset(initial.columns):
        raise ValueError('Pilot missing property columns')
    if new_budget < max(1,exploration_floor)*len(families):
        raise ValueError('Budget too small for every family to be explored')
    if int(batch_size)<1:
        raise ValueError('batch_size must be positive')
    if initial['Candidate_ID'].duplicated().any():
        raise ValueError('Pilot Candidate_ID not unique')
    oxide_columns=list(oxide_columns)
    rng=np.random.default_rng(seed)
    states={}
    global_seen=set()
    for name,bounds in families.items():
        row=initial.loc[initial['Glass_family'].eq(name)].copy()
        if len(row)<4:
            raise ValueError(f'{name}: need four pilot rows')
        state=_Family(name,bounds,row,oxide_columns,rng)
        state.best_pilot=float(np.min(_penalty(row,criteria,consistency)))
        state.best=state.best_pilot
        states[name]=state
        for x in row[oxide_columns].to_numpy(float):
            global_seen.add(_key(x))
    samples=[]
    diary=[]
    next_id=int(max(initial['Candidate_ID'].max(),0))+1
    remain=int(new_budget)
    round_index=0
    # Explicit initial family exploration before adaptive rewards.
    while remain>0:
        required=[k for k,s in states.items() if s.nnew<exploration_floor]
        if required:
            name=required[0]
            n=min(int(batch_size),exploration_floor-states[name].nnew,remain)
            rule='floor'
        else:
            total=sum(s.nnew for s in states.values())
            scores={}
            for k,s in states.items():
                # Empirical success with shrinkage; novelty bonus via UCB.
                hit_rate=(s.nhits+0.5)/(s.nnew+2.)
                ucb=0.50*math.sqrt(math.log(total+len(states)+1)/(s.nnew+1))
                progress=0.12*min(s.best_gain,5)/(1+s.nnew)
                # Reward near target during no-hit tasks to avoid zero-only bandit.
                proximity=0.30/(1+s.best)
                scores[k]=hit_rate+ucb+progress+proximity
            name=max(scores,key=lambda k:(scores[k],-list(states).index(k)))
            n=min(int(batch_size),remain)
            rule='adaptive_ucb'
        state=states[name]
        raw=state.propose(n,criteria,consistency,global_seen)
        rows=[]
        for v in raw:
            row={ox:0.0 for ox in oxide_columns}
            row.update({ox:float(a) for ox,a in zip(state.names,v)})
            row.update({'Glass_family':name,'Candidate_ID':next_id,
                        'Search_stage':'Hybrid_v21','Search_round':round_index})
            next_id+=1
            rows.append(row)
        frame=pd.DataFrame(rows)
        prediction=predictor(frame.copy())
        if not isinstance(prediction,pd.DataFrame) or len(prediction)!=len(frame):
            raise RuntimeError('Predictor shape mismatch')
        if 'Candidate_ID' in prediction.columns:
            if not prediction['Candidate_ID'].is_unique:
                raise RuntimeError('Predictor duplicate Candidate_ID')
            prediction=prediction.set_index('Candidate_ID').loc[frame['Candidate_ID']]
        values=prediction[list(PROPERTIES)].to_numpy(float)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite predictions')
        for j,col in enumerate(PROPERTIES):
            frame[col]=values[:,j]
        frame['Goal_pass']=_feasible(frame,criteria,consistency)
        frame['Goal_penalty']=_penalty(frame,criteria,consistency)
        frame['Delta_T_C']=frame['Tsoft_C']-frame['Tg_C']
        state.update(frame,criteria,consistency)
        samples.append(frame)
        remain-=len(frame)
        diary.append({'Round':round_index,'Glass_family':name,'Allocation_rule':rule,
                      'New_this_round':len(frame),'New_cumulative':new_budget-remain,
                      'Passed_this_round':int(frame['Goal_pass'].sum()),
                      'Family_new_cumulative':state.nnew,
                      'Family_passed_cumulative':state.nhits,
                      'Family_best_penalty':state.best})
        round_index+=1
    result=pd.concat(samples,ignore_index=True)
    if len(result)!=new_budget or result['Candidate_ID'].duplicated().any():
        raise AssertionError('Budget / Candidate_ID mismatch')
    X=result[oxide_columns].to_numpy(float)
    if not np.isfinite(X).all() or (X<-1e-9).any() or not np.allclose(X.sum(axis=1),100,atol=1e-5):
        raise AssertionError('Invalid composition balance')
    if len(set(_key(x) for x in X)) != len(result):
        raise AssertionError('Repeated composition')
    return result,pd.DataFrame(diary)
