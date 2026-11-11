"""GlassDesign ML step 391 — separate recommendations from diagnostic candidates.

A composition failing either model's physical ordering Td >= Tg is not shown
among recommendations. It remains in the full, auditable candidate export.
No glass-formation probability or physical material impossibility is implied.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

TOL_C = 1e-9
PROPS = ('Density_g_cm3','RefractiveIndex','Tg_C','Tsoft_C')


def annotate_recommendations(frame, criteria):
    """Return copy annotated with transparent mutually exclusive statuses.

    Both_pass is the existing consensus decision (includes Td>=Tg), not overwritten.
    Assumes Tsoft_C and GN_Tsoft_C contain aligned *dilatometric* softening.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError('Expected DataFrame')
    required = {'Both_pass','GD_pass','GN_pass','Tg_C','Tsoft_C',
                'GN_Tg_C','GN_Tsoft_C','Candidate_ID'}
    required.update({p for p in criteria})
    required.update({'GN_'+p for p in criteria})
    missing = required - set(frame.columns)
    if missing:
        raise ValueError('Нет столбцов: ' + ', '.join(sorted(missing)))
    df=frame.copy()
    if df['Candidate_ID'].duplicated().any():
        raise ValueError('Повторяющиеся Candidate_ID')
    for p in ('Tg_C','Tsoft_C','GN_Tg_C','GN_Tsoft_C'):
        if not np.isfinite(pd.to_numeric(df[p],errors='coerce').to_numpy(float)).all():
            raise ValueError('Нечисловой или отсутствующий прогноз: ' + p)
    gd_delta=df['Tsoft_C'].to_numpy(float)-df['Tg_C'].to_numpy(float)
    gn_delta=df['GN_Tsoft_C'].to_numpy(float)-df['GN_Tg_C'].to_numpy(float)
    df['GD_Td_minus_Tg_C']=gd_delta
    df['GN_Td_minus_Tg_C']=gn_delta
    df['GD_Td_ge_Tg']=gd_delta>=-TOL_C
    df['GN_Td_ge_Tg']=gn_delta>=-TOL_C
    # Exact numerical constraints of the current run (not inherited from another run).
    gd_range=np.ones(len(df),bool)
    gn_range=np.ones(len(df),bool)
    for prop,(lo,hi) in criteria.items():
        gd=df[prop].to_numpy(dtype=float)
        gn=df['GN_'+prop].to_numpy(dtype=float)
        gd_range &= np.isfinite(gd)
        gn_range &= np.isfinite(gn)
        if lo is not None:
            gd_range &= gd>=float(lo)-TOL_C
            gn_range &= gn>=float(lo)-TOL_C
        if hi is not None:
            gd_range &= gd<=float(hi)+TOL_C
            gn_range &= gn<=float(hi)+TOL_C
    df['GD_numeric_ranges_pass']=gd_range
    df['GN_numeric_ranges_pass']=gn_range
    expected_gd=gd_range & df['GD_Td_ge_Tg'].to_numpy(bool)
    expected_gn=gn_range & df['GN_Td_ge_Tg'].to_numpy(bool)
    if not np.array_equal(df['GD_pass'].to_numpy(bool),expected_gd):
        raise AssertionError('GD_pass does not match current criteria and Td>=Tg')
    if not np.array_equal(df['GN_pass'].to_numpy(bool),expected_gn):
        raise AssertionError('GN_pass does not match current criteria and Td>=Tg')
    expected=expected_gd & expected_gn
    if not np.array_equal(df['Both_pass'].to_numpy(bool),expected):
        raise AssertionError('Both_pass does not match aligned physical criteria')
    df['Recommendation_visible']=expected
    reasons=[]
    for i in range(len(df)):
        why=[]
        if not bool(df['GD_Td_ge_Tg'].iloc[i]):
            why.append('GlassDesign: Td ниже Tg')
        if not bool(df['GN_Td_ge_Tg'].iloc[i]):
            why.append('GlassNet: Td ниже Tg')
        if not bool(gd_range[i]):
            why.append('GlassDesign: вне числовых требований')
        if not bool(gn_range[i]):
            why.append('GlassNet: вне числовых требований')
        reasons.append('; '.join(why))
    df['Exclusion_reason']=reasons
    df['Recommendation_status']=np.where(expected,'Рекомендация для дальнейшего скрининга',
                                          'Не рекомендовать — оставить для диагностики')
    if df.loc[expected,'Exclusion_reason'].astype(bool).any():
        raise AssertionError('Recommendation has exclusion reason')
    if not (df.loc[~expected,'Exclusion_reason'].astype(bool)).all():
        raise AssertionError('Missing exclusion reason')
    return df


def recommendation_counts(frame):
    """Counts can overlap: a row can violate multiple checks."""
    return {
        'recommendations_visible':int(frame['Recommendation_visible'].sum()),
        'excluded_total':int((~frame['Recommendation_visible']).sum()),
        'excluded_GD_Td_below_Tg':int((~frame['GD_Td_ge_Tg']).sum()),
        'excluded_GN_Td_below_Tg':int((~frame['GN_Td_ge_Tg']).sum()),
        'excluded_GD_numeric_ranges':int((~frame['GD_numeric_ranges_pass']).sum()),
        'excluded_GN_numeric_ranges':int((~frame['GN_numeric_ranges_pass']).sum()),
    }
