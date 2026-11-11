"""GlassDesign ML step 391: aligned GlassNet predictions with lossless audit export.

Caches ONLY the extra GlassNet Littleton Tsoft already returned by the same
prediction call, keyed by complete composition; does not call GlassNet again.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from glass390_glassnet import FrozenGlassNetAligned

class FrozenGlassNetAligned391(FrozenGlassNetAligned):
    def __init__(self,oxides,network=None):
        super().__init__(oxides,network=network)
        self._littleton_by_composition={}

    def _composition_keys(self,frame):
        data=frame[self.oxides].to_numpy(float)
        return [tuple(np.round(row,8)) for row in data]

    def __call__(self,frame):
        output=super().__call__(frame)
        keys=self._composition_keys(frame)
        for key,value in zip(keys,output['GN_Littleton_Tsoft_C'].to_numpy(float)):
            prior=self._littleton_by_composition.get(key)
            if prior is not None and abs(prior-float(value))>1e-6:
                raise AssertionError('Different Littleton Tsoft for same composition')
            self._littleton_by_composition[key]=float(value)
        return output

    def attach_littleton(self,frame):
        result=frame.copy()
        keys=self._composition_keys(result)
        missing=[key for key in keys if key not in self._littleton_by_composition]
        if missing:
            raise AssertionError('Missing GlassNet audit prediction for %d compositions' % len(missing))
        result['GN_Littleton_Tsoft_C']=[self._littleton_by_composition[key] for key in keys]
        return result
