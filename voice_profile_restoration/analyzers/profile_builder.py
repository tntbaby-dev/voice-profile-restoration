from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np

from voice_profile_restoration.analyzers.spectral_features import extract_spectral_features


def build_speaker_profile(audio_files: List[str | Path]) -> dict:
    all_features = []

    for file_path in audio_files:
        features = extract_spectral_features(file_path)
        all_features.append(features)

    if not all_features:
        raise ValueError("No audio files provided")

    # Collect keys
    keys = all_features[0].keys()

    averaged_profile = {}

    for key in keys:
        values = [f[key] for f in all_features if isinstance(f[key], (int, float))]
        averaged_profile[key] = float(np.mean(values))

    return averaged_profile