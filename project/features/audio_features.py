import math
from dataclasses import dataclass
from typing import Literal

import librosa
import numpy as np


NoiseType = Literal["stationary", "non-stationary", "impulsive"]


@dataclass
class FrameFeatures:
    snr_db: float
    energy: float
    spectral_variance: float
    noise_type: NoiseType


def _safe_db(x: float, eps: float = 1e-8) -> float:
    return 10.0 * math.log10(max(x, eps))


def _estimate_noise_power(magnitude: np.ndarray) -> float:
    """Estimate background noise power from STFT magnitude using a low percentile."""
    floor = np.percentile(magnitude, 20, axis=None)
    return float(floor * floor)


def classify_noise(snr_db: float, energy: float, spectral_variance: float) -> NoiseType:
    """Lightweight noise classifier based on simple thresholds."""
    if spectral_variance < 0.3 and snr_db < 8:
        return "stationary"
    if spectral_variance > 1.0 and energy > 0.02:
        return "non-stationary"
    if energy > 0.05 and spectral_variance > 1.5:
        return "impulsive"
    return "stationary"


def extract_features(
    frame: np.ndarray,
    sample_rate: int,
    n_fft: int = 512,
    hop_length: int | None = None,
) -> FrameFeatures:
    """Compute noise-aware features for a single audio frame."""
    n_fft = max(16, min(n_fft, len(frame)))
    hop = max(1, hop_length or n_fft // 4)
    stft = librosa.stft(frame, n_fft=n_fft, hop_length=hop, center=False)
    mag = np.abs(stft)

    energy = float(np.mean(frame ** 2))
    noise_power = _estimate_noise_power(mag)
    signal_power = float(np.mean(mag**2))
    snr_db = _safe_db(signal_power / (noise_power + 1e-8))

    spectral_variance = float(np.var(mag))
    noise_type = classify_noise(snr_db, energy, spectral_variance)

    return FrameFeatures(
        snr_db=snr_db,
        energy=energy,
        spectral_variance=spectral_variance,
        noise_type=noise_type,
    )
