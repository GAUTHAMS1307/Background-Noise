import numpy as np

from app.services.denoisers.base import BaseDenoiser
from app.utils.spectral import spectral_gate


class DemucsDenoiser(BaseDenoiser):
    name = "demucs"

    def process(self, signal: np.ndarray, sample_rate: int, enhance_voice: bool = False) -> np.ndarray:
        # Demucs-quality path can be swapped in here; this keeps API stable.
        den = spectral_gate(signal, sample_rate, strength=0.5)
        if enhance_voice:
            den = np.tanh(1.2 * den)
        return den.astype(np.float32)
