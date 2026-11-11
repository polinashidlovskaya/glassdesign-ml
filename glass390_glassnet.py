"""GlassDesign ML step 390: unit-checked, property-aligned frozen GlassNet adapter.

Compatibility contract: GD's legacy Tsoft_C is the model trained against
TdilatometricSoftening (degrees C). Therefore GN_Tsoft_C MUST contain the
GlassNet TdilatometricSoftening output converted K -> degrees C.

This is NOT the GlassNet Littleton Tsoft output. Keep it separately for audit.
No fitting, no mutation of any stored source or model.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

ALIGNED_GLASSNET_PROPERTY = "TdilatometricSoftening"
ALIGNED_INTERNAL_KEY = "GN_Tsoft_C"  # historical key used by frozen optimizer
KELVIN_OFFSET = 273.15


class FrozenGlassNetAligned:
    """Drop-in adapter preserving the optimizer's historic GN_* output keys."""

    def __init__(self, oxides, network=None):
        if network is None:
            from glasspy.predict import GlassNet
            network = GlassNet()
        self.network = network
        self.oxides = list(oxides)
        if not self.oxides or len(set(self.oxides)) != len(self.oxides):
            raise ValueError("Oxide list is empty or not unique")
        self.rows = 0
        self.calls = 0

    def __call__(self, frame):
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("GlassNet requires a non-empty DataFrame")
        if "Candidate_ID" not in frame or not set(self.oxides).issubset(frame.columns):
            raise ValueError("Missing Candidate_ID or required oxide columns")
        mol = frame[self.oxides].to_numpy(dtype=float)
        if (not np.isfinite(mol).all() or (mol < -1e-9).any()
                or not np.allclose(mol.sum(axis=1), 100.0, atol=1e-5, rtol=0)):
            raise ValueError("GlassNet inputs must be nonnegative mol.% summing to 100")
        active = [o for o in self.oxides if (mol[:, self.oxides.index(o)] > 0).any()]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            raw = self.network.predict(frame[active].copy())
        if not isinstance(raw, pd.DataFrame):
            raw = pd.DataFrame(raw)
        required = ("Density293K", "RefractiveIndex", "Tg", "Tsoft", ALIGNED_GLASSNET_PROPERTY)
        missing = set(required) - set(raw.columns)
        if missing:
            raise RuntimeError("GlassNet missing required properties: " + ", ".join(sorted(missing)))
        if len(raw) != len(frame):
            raise RuntimeError("GlassNet returned incorrect number of rows")
        vals = raw[list(required)].apply(pd.to_numeric, errors="raise").reset_index(drop=True)
        if not np.isfinite(vals.to_numpy(dtype=float)).all():
            raise RuntimeError("GlassNet returned nonfinite predictions")
        result = pd.DataFrame({"Candidate_ID": frame["Candidate_ID"].to_numpy()})
        result["GN_Density_g_cm3"] = vals["Density293K"].to_numpy(float)
        result["GN_RefractiveIndex"] = vals["RefractiveIndex"].to_numpy(float)
        result["GN_Tg_C"] = vals["Tg"].to_numpy(float) - KELVIN_OFFSET
        dil = vals[ALIGNED_GLASSNET_PROPERTY].to_numpy(float) - KELVIN_OFFSET
        result[ALIGNED_INTERNAL_KEY] = dil
        result["GN_TdilatometricSoftening_C"] = dil
        # Diagnostic-only column, never used as the target for GD's Td model.
        result["GN_Littleton_Tsoft_C"] = vals["Tsoft"].to_numpy(float) - KELVIN_OFFSET
        self.rows += len(frame)
        self.calls += 1
        return result
