from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from .spectral_features import extract_spectral_features


def build_average_profile(file_paths):
    spectra = []
    all_features = []

    for path in file_paths:
        features = extract_spectral_features(path)
        all_features.append(features)

        normalized_spectrum = features["normalized_spectrum"]
        spectrum = np.array(normalized_spectrum, dtype=np.float32)
        spectra.append(spectrum)

    if not spectra:
        raise ValueError("No reference files provided")

    avg_spectrum = np.mean(spectra, axis=0)

    return {
        "average_spectrum": avg_spectrum.tolist(),
        "spectral_tilt": float(np.mean([f["spectral_tilt"] for f in all_features])),
        "sub": float(np.mean([f["sub"] for f in all_features])),
        "low": float(np.mean([f["low"] for f in all_features])),
        "low_mid": float(np.mean([f["low_mid"] for f in all_features])),
        "mid": float(np.mean([f["mid"] for f in all_features])),
        "presence": float(np.mean([f["presence"] for f in all_features])),
        "air": float(np.mean([f["air"] for f in all_features])),
    }


def build_speaker_profile(file_paths):
    return build_average_profile(file_paths)