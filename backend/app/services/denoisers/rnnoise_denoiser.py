import numpy as np

from app.services.denoisers.base import BaseDenoiser
from app.utils.spectral import AdaptiveNoiseProfile, spectral_gate


class RNNoiseDenoiser(BaseDenoiser):
    name = "rnnoise"

    def __init__(self) -> None:
        self._profile = AdaptiveNoiseProfile(alpha=0.05)

    def process(self, signal: np.ndarray, sample_rate: int, enhance_voice: bool = False) -> np.ndarray:
        # Fallback spectral denoising if native RNNoise binding is unavailable.
        # The adaptive noise profile tracks the environment across streaming frames.
        den = spectral_gate(signal, sample_rate, strength=0.35, noise_profile=self._profile)
        if enhance_voice:
            den = np.tanh(1.1 * den)
        return den.astype(np.float32)
