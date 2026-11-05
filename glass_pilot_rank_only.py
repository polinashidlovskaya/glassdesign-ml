"""Exact extracted family_pilot_ranking from existing universal pipeline.
No model training, dependencies on original GA or pymoo.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

def family_pilot_ranking(pilot, available, max_families):
    # Все семейства получают одинаковое число pilot-прогнозов.
    ranking = (pilot.groupby('Glass_family', as_index=False)
               .agg(Pilot_tested=('Candidate_ID', 'count'),
                    Pilot_passed=('Goal_pass', 'sum'),
                    Best_penalty=('Goal_penalty', 'min'),
                    Median_penalty=('Goal_penalty', 'median')))
    ranking['Smoothed_pass_rate'] = ((ranking['Pilot_passed'] + 1.0) /
                                      (ranking['Pilot_tested'] + 2.0))
    ranking['Base_score'] = (2 * ranking['Smoothed_pass_rate']
                             + .30 / (1.0 + ranking['Best_penalty'])
                             + .15 / (1.0 + ranking['Median_penalty']))
    ranking['Selected'] = False
    chosen = []
    # Небольшая добавка за разнообразие семейств. Не жёсткое правило.
    while len(chosen) < min(max_families, len(available)):
        variants = ranking.loc[~ranking['Glass_family'].isin(chosen)].copy()
        if chosen:
            def distance(name):
                A = set(available[name])
                return min(1.0 - len(A & set(available[old])) / len(A | set(available[old]))
                           for old in chosen)
            variants['Diversity_bonus'] = variants['Glass_family'].map(distance) * .10
        else:
            variants['Diversity_bonus'] = 0.0
        variants['Selection_score'] = variants['Base_score'] + variants['Diversity_bonus']
        variants = variants.sort_values(['Selection_score', 'Pilot_passed', 'Glass_family'],
                                        ascending=[False, False, True])
        chosen.append(variants.iloc[0]['Glass_family'])
    ranking['Selected'] = ranking['Glass_family'].isin(chosen)
    ranking = ranking.sort_values(['Selected', 'Base_score'], ascending=[False, False])
    return {k: available[k] for k in chosen}, ranking
