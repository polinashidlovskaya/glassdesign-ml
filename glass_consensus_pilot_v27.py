"""GlassDesign ML 2.7 - consensus-aligned PILOT family selection.

Changes family preselection ONLY. Optimizer is unchanged v2.5.
Uses the same already-evaluated 136 pilot compositions under both frozen oracles.
"""
import numpy as np
import pandas as pd
from glass_hybrid_v21 import _penalty, _feasible
from glass_consensus_ga_v25 import _gn_view


def dual_ranking(pilot, all_families, max_families, criteria, legacy_ranker):
    """Identical family-ranking rule, with the dual-model search objective.

    The original ranker expects Goal_pass, Goal_penalty. Align them with
    the dual-feedback search metric while leaving ranker weights identical.
    """
    gn = _gn_view(pilot)
    gp = _penalty(pilot, criteria, True)
    np_ = _penalty(gn, criteria, True)
    dual = pilot.copy()
    dual['Goal_pass'] = _feasible(pilot, criteria, True) & _feasible(gn, criteria, True)
    dual['Goal_penalty'] = np.maximum(gp, np_) + 0.25*(gp + np_)
    selected, ranking = legacy_ranker(dual, all_families, max_families)
    if len(selected) != max_families:
        raise RuntimeError('Preselection did not return expected family count')
    return selected, ranking
