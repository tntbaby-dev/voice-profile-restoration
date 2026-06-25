from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def db_to_linear(gain_db: float) -> float:
    return 10 ** (gain_db / 20.0)


def apply_fft_eq(samples: np.ndarray, sample_rate: int, eq_settings: dict) -> np.ndarray:
    """
    FFT-domain EQ processor.

    Supports:
    1. Legacy broad-band EQ settings:
       low_shelf, low_mid, mid, presence, high_shelf

    2. New adaptive spectral curve:
       spectral_curve
    """

    n = len(samples)
    if n == 0:
        return samples

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    gains = np.ones_like(freqs, dtype=np.float64)

    if "spectral_curve" in eq_settings:
        delta_db = np.asarray(eq_settings["spectral_curve"], dtype=np.float64)

        gain_curve = 10 ** (delta_db / 20.0)

        if len(gain_curve) != len(freqs):
            gain_curve = np.interp(
                freqs,
                np.linspace(freqs.min(), freqs.max(), len(gain_curve)),
                gain_curve,
            )

        gains *= gain_curve

        print("\n===== EQ APPLY DEBUG =====")
        print("Gain Min:", np.min(gains))
        print("Gain Max:", np.max(gains))
        print("Gain Mean:", np.mean(gains))

    else:
        low_shelf_db = float(eq_settings.get("low_shelf", 0.0))
        gains[freqs < 200] *= db_to_linear(low_shelf_db)

        low_mid_db = float(eq_settings.get("low_mid", 0.0))
        gains[(freqs >= 200) & (freqs < 500)] *= db_to_linear(low_mid_db)

        mid_db = float(eq_settings.get("mid", 0.0))
        gains[(freqs >= 500) & (freqs < 2000)] *= db_to_linear(mid_db)

        presence_db = float(eq_settings.get("presence", 0.0))
        gains[(freqs >= 2000) & (freqs < 5000)] *= db_to_linear(presence_db)

        high_shelf_db = float(eq_settings.get("high_shelf", 0.0))
        gains[freqs >= 5000] *= db_to_linear(high_shelf_db)

    processed_spectrum = spectrum * gains
    processed = np.fft.irfft(processed_spectrum, n=n)

    peak = np.max(np.abs(processed)) + 1e-12
    if peak > 1.0:
        processed = processed / peak

    return processed.astype(np.float32)


def process_file(input_file: str | Path, output_file: str | Path, eq_settings: dict) -> None:
    print("EQ SETTINGS KEYS:", eq_settings.keys())
    
    samples, sample_rate = sf.read(str(input_file), always_2d=False)

    if samples.ndim > 1:
        channels = []

        for ch in range(samples.shape[1]):
            processed_ch = apply_fft_eq(
                samples[:, ch].astype(np.float32),
                sample_rate,
                eq_settings,
            )
            channels.append(processed_ch)

        processed = np.stack(channels, axis=1)

    else:
        processed = apply_fft_eq(
            samples.astype(np.float32),
            sample_rate,
            eq_settings,
        )

    sf.write(str(output_file), processed, sample_rate)