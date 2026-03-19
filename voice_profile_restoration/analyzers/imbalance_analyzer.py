from __future__ import annotations

from typing import Dict


TONAL_KEYS = [
    "spectral_tilt",
    "sub",
    "low",
    "low_mid",
    "mid",
    "presence",
    "air",
]


def compute_imbalance(input_features, reference_profile):
    imbalance = {}

    for key in TONAL_KEYS:
        imbalance[key] = reference_profile[key] - input_features[key]

    return imbalance