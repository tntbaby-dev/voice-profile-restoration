from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter


def db_to_linear(gain_db: float) -> float:
    return 10 ** (gain_db / 20.0)


def apply_fft_eq(samples: np.ndarray, sample_rate: int, eq_settings: dict) -> np.ndarray:
    """
    Simple frequency-domain EQ for v1.
    This is not a studio-grade parametric EQ yet, but it is enough
    to prove the restoration pipeline end-to-end.
    """

    n = len(samples)
    if n == 0:
        return samples

    spectrum = np.fft.rfft(samples)
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    gains = np.ones_like(freqs, dtype=np.float64)

    # Low shelf-ish
    low_shelf_db = float(eq_settings.get("low_shelf", 0.0))
    gains[freqs < 200] *= db_to_linear(low_shelf_db)

    # Low-mid bell-ish
    low_mid_db = float(eq_settings.get("low_mid", 0.0))
    gains[(freqs >= 200) & (freqs < 500)] *= db_to_linear(low_mid_db)

    # Mid bell-ish
    mid_db = float(eq_settings.get("mid", 0.0))
    gains[(freqs >= 500) & (freqs < 2000)] *= db_to_linear(mid_db)

    # Presence bell-ish
    presence_db = float(eq_settings.get("presence", 0.0))
    gains[(freqs >= 2000) & (freqs < 5000)] *= db_to_linear(presence_db)

    # High shelf-ish
    high_shelf_db = float(eq_settings.get("high_shelf", 0.0))
    gains[freqs >= 5000] *= db_to_linear(high_shelf_db)

    processed_spectrum = spectrum * gains
    processed = np.fft.irfft(processed_spectrum, n=n)

    # Keep safe output range
    peak = np.max(np.abs(processed)) + 1e-12
    if peak > 1.0:
        processed = processed / peak

    return processed.astype(np.float32)


def process_file(input_file: str | Path, output_file: str | Path, eq_settings: dict) -> None:
    samples, sample_rate = sf.read(str(input_file), always_2d=False)

    stereo = False
    if samples.ndim > 1:
        stereo = True
        channels = []
        for ch in range(samples.shape[1]):
            processed_ch = apply_fft_eq(samples[:, ch].astype(np.float32), sample_rate, eq_settings)
            channels.append(processed_ch)
        processed = np.stack(channels, axis=1)
    else:
        processed = apply_fft_eq(samples.astype(np.float32), sample_rate, eq_settings)

    sf.write(str(output_file), processed, sample_rate)