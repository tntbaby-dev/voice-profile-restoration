from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def db_to_linear(gain_db: float) -> float:
    return 10 ** (gain_db / 20.0)


def linear_to_db(value: float, floor: float = 1e-12) -> float:
    return 20.0 * np.log10(max(value, floor))


def apply_polish_eq(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Small finishing EQ in the frequency domain.
    Intentionally kept subtle to avoid harshness and fake brightness.
    """
    n = len(samples)
    if n == 0:
        return samples

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    gains = np.ones_like(freqs, dtype=np.float64)

    # Small cleanup in low-mid area
    gains[(freqs >= 200) & (freqs < 500)] *= db_to_linear(-0.4)

    # Small clarity lift
    gains[(freqs >= 2000) & (freqs < 5000)] *= db_to_linear(+0.1)

    # Small air lift
    gains[freqs >= 5000] *= db_to_linear(+0.4)

    polished_spectrum = spectrum * gains
    polished = np.fft.irfft(polished_spectrum, n=n)

    return polished.astype(np.float32)


def normalize_peak(samples: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    peak = np.max(np.abs(samples)) + 1e-12
    if peak == 0:
        return samples.astype(np.float32)

    normalized = samples * (target_peak / peak)
    return normalized.astype(np.float32)


def polish_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    polished = apply_polish_eq(samples, sample_rate)
    polished = normalize_peak(polished, target_peak=0.85)
    return polished.astype(np.float32)


def process_polish_file(input_file: str | Path, output_file: str | Path) -> None:
    samples, sample_rate = sf.read(str(input_file), always_2d=False)

    if samples.ndim > 1:
        channels = []
        for ch in range(samples.shape[1]):
            processed_ch = polish_audio(samples[:, ch].astype(np.float32), sample_rate)
            channels.append(processed_ch)
        processed = np.stack(channels, axis=1)
    else:
        processed = polish_audio(samples.astype(np.float32), sample_rate)

    sf.write(str(output_file), processed, sample_rate)