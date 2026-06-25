from __future__ import annotations

import numpy as np


TONAL_KEYS = [
    "spectral_tilt",
    "sub",
    "low",
    "low_mid",
    "mid",
    "presence",
    "air",
]


def compute_imbalance(
    input_features: dict,
    reference_profile: dict,
) -> dict:

    imbalance = {}

    for key in TONAL_KEYS:

        if key not in input_features:
            continue

        if key not in reference_profile:
            continue

        imbalance[key] = (
            reference_profile[key]
            - input_features[key]
        )

    return imbalance


def compute_spectral_delta(
    input_spectrum,
    reference_spectrum,
):

    input_arr = np.asarray(
        input_spectrum,
        dtype=np.float64,
    )

    ref_arr = np.asarray(
        reference_spectrum,
        dtype=np.float64,
    )

    if len(input_arr) != len(ref_arr):
        raise ValueError(
            f"Spectrum length mismatch: input={len(input_arr)}, reference={len(ref_arr)}"
        )

    epsilon = 1e-6

    input_arr = np.maximum(input_arr, 1e-4)
    ref_arr = np.maximum(ref_arr, 1e-4)

    delta_db = 20.0 * np.log10(
        (ref_arr + epsilon)
        / (input_arr + epsilon)
    )

    print("\n===== SPECTRAL DELTA DEBUG =====")
    print("Delta Length:", len(delta_db))
    print("Delta Min:", np.min(delta_db))
    print("Delta Max:", np.max(delta_db))
    print("Delta Mean:", np.mean(delta_db))

    return delta_db