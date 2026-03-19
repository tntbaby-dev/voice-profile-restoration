from __future__ import annotations


TONAL_KEYS = [
    "spectral_tilt",
    "sub",
    "low",
    "low_mid",
    "mid",
    "presence",
    "air",
]

WEIGHTS = {
    "spectral_tilt": 3.0,
    "sub": 1.0,
    "low": 2.0,
    "low_mid": 2.5,
    "mid": 2.5,
    "presence": 3.0,
    "air": 3.0,
}


def compute_match_error(features: dict, reference: dict) -> float:
    total_error = 0.0

    for key in TONAL_KEYS:
        diff = abs(reference[key] - features[key])
        total_error += diff * WEIGHTS[key]

    return float(total_error)


def compute_quality_score(features: dict, reference: dict) -> float:
    error = compute_match_error(features, reference)

    # gentler conversion from error -> score
    score = 100.0 / (1.0 + error)

    return float(max(0.0, min(100.0, score)))