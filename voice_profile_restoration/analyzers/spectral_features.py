from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


BANDS = {
    "sub": (20, 80),
    "low": (80, 250),
    "low_mid": (250, 500),
    "mid": (500, 2000),
    "presence": (2000, 5000),
    "air": (5000, 10000),
}

FFT_SIZE = 16384

def load_audio(file_path: str | Path) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(str(file_path), always_2d=False)

    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)

    samples = samples.astype(np.float32)
    return samples, int(sample_rate)


def compute_magnitude_spectrum(
    samples: np.ndarray,
    sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:

    if len(samples) == 0:
        raise ValueError("Audio file is empty")

    # Pad short audio
    if len(samples) < FFT_SIZE:
        samples = np.pad(samples, (0, FFT_SIZE - len(samples)))

    # Trim long audio
    samples = samples[:FFT_SIZE]

    # Windowing
    window = np.hanning(FFT_SIZE)
    windowed = samples * window

    # FFT
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(FFT_SIZE, d=1.0 / sample_rate)

    return freqs, spectrum


def normalize_spectrum(spectrum: np.ndarray) -> np.ndarray:
    power = spectrum ** 2
    total = np.sum(power) + 1e-12

    return power / total


def compute_band_energies(freqs: np.ndarray, spectrum: np.ndarray) -> dict[str, float]:
    band_energies: dict[str, float] = {}

    power_spectrum = spectrum ** 2
    total_energy = float(np.sum(power_spectrum) + 1e-12)

    for band_name, (low_hz, high_hz) in BANDS.items():
        idx = np.where((freqs >= low_hz) & (freqs < high_hz))[0]

        if len(idx) == 0:
            band_energies[band_name] = 0.0
            continue

        band_energy = float(np.sum(power_spectrum[idx]))
        band_energies[band_name] = band_energy / total_energy

    return band_energies


def compute_spectral_tilt(freqs: np.ndarray, spectrum: np.ndarray) -> float:
    valid = (freqs > 0) & (spectrum > 0)

    freqs_valid = freqs[valid]
    spectrum_valid = spectrum[valid]

    if len(freqs_valid) < 2:
        return 0.0

    log_freqs = np.log10(freqs_valid)
    log_mag = np.log10(spectrum_valid)

    slope, _ = np.polyfit(log_freqs, log_mag, 1)
    return float(slope)


def extract_spectral_features(file_path: str | Path) -> dict[str, float]:
    samples, sample_rate = load_audio(file_path)
    freqs, spectrum = compute_magnitude_spectrum(samples, sample_rate)

    band_energies = compute_band_energies(freqs, spectrum)
    spectral_tilt = compute_spectral_tilt(freqs, spectrum)

    normalized_spectrum = normalize_spectrum(spectrum)

    print("Spectrum Length:", len(normalized_spectrum))
    print("Spectrum Sum:", np.sum(normalized_spectrum))


    features = {
        "sample_rate": float(sample_rate),
        "duration_sec": float(len(samples) / sample_rate),
        "spectral_tilt": spectral_tilt,
        "normalized_spectrum": normalized_spectrum.tolist(),
        **band_energies,
    }

    return features