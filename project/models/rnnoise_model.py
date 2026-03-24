from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from project.features.audio_features import FrameFeatures


class RNNoiseModel:
    """Lightweight RNNoise-inspired denoiser for frame-by-frame processing."""

    def __init__(self, sample_rate: int, frame_length: int, profile_alpha: float = 0.1) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.profile_alpha = profile_alpha
        self._noise_profile: np.ndarray | None = None
        self._b, self._a = butter(2, [80 / (sample_rate * 0.5), 7800 / (sample_rate * 0.5)], btype="band")

    def _update_profile(self, magnitude: np.ndarray) -> np.ndarray:
        floor = np.percentile(magnitude, 10, axis=0)
        if self._noise_profile is None:
            self._noise_profile = floor
        else:
            self._noise_profile = (
                self.profile_alpha * floor + (1.0 - self.profile_alpha) * self._noise_profile
            )
        return self._noise_profile

    def process_frame(self, frame: np.ndarray, features: FrameFeatures | None = None) -> np.ndarray:
        stft = np.fft.rfft(frame * np.hanning(len(frame)))
        mag = np.abs(stft)
        phase = np.angle(stft)

        profile = self._update_profile(mag[None, :])[0]
        gain = np.clip((mag - 0.6 * profile) / (mag + 1e-8), 0.0, 1.0)
        den_mag = mag * gain

        den = np.fft.irfft(den_mag * np.exp(1j * phase))
        den = lfilter(self._b, self._a, den).astype(np.float32)
        return np.clip(den[: len(frame)], -1.0, 1.0)
