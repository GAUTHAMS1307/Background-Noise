import numpy as np

from app.services.denoisers.base import BaseDenoiser
from app.utils.spectral import AdaptiveNoiseProfile, spectral_gate


class DemucsDenoiser(BaseDenoiser):
    name = "demucs"

    def __init__(self) -> None:
        self._profile = AdaptiveNoiseProfile(alpha=0.03)

    def process(self, signal: np.ndarray, sample_rate: int, enhance_voice: bool = False) -> np.ndarray:
        # Demucs-quality path can be swapped in here; this keeps API stable.
        # The adaptive profile uses a slower alpha for higher stability.
        den = spectral_gate(signal, sample_rate, strength=0.5, noise_profile=self._profile)
        if enhance_voice:
            den = np.tanh(1.2 * den)
        return den.astype(np.float32)
