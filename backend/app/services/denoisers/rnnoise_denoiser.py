import numpy as np

from app.services.denoisers.base import BaseDenoiser
from app.utils.spectral import spectral_gate


class RNNoiseDenoiser(BaseDenoiser):
    name = "rnnoise"

    def process(self, signal: np.ndarray, sample_rate: int, enhance_voice: bool = False) -> np.ndarray:
        # Fallback spectral denoising if native RNNoise binding is unavailable.
        den = spectral_gate(signal, sample_rate, strength=0.35)
        if enhance_voice:
            den = np.tanh(1.1 * den)
        return den.astype(np.float32)
