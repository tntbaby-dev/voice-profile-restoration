from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def clamp(x, min_val=-6.0, max_val=6.0):
    return max(min(x, max_val), min_val)


def map_to_eq_settings(imbalance: dict) -> dict:
    eq = {}

    def scale(x):
        return clamp(float(x) * 10.0)

    eq["low_shelf"] = scale(imbalance.get("low", 0.0))
    eq["low_mid"] = scale(imbalance.get("low_mid", 0.0))
    eq["mid"] = scale(imbalance.get("mid", 0.0))
    eq["presence"] = scale(imbalance.get("presence", 0.0))
    eq["high_shelf"] = scale(imbalance.get("air", 0.0))

    return eq


def smooth_correction_curve(delta_db, sigma=3):
    delta_db = np.asarray(delta_db, dtype=np.float64)
    return gaussian_filter1d(delta_db, sigma=sigma)


def limit_curve(delta_db, min_db=-6.0, max_db=6.0):
    delta_db = np.asarray(delta_db, dtype=np.float64)
    return np.clip(delta_db, min_db, max_db)