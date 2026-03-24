from __future__ import annotations

import numpy as np

from project.features.audio_features import FrameFeatures


class AdaptiveController:
    """Compute adaptive fusion weights from frame-level features."""

    def __init__(
        self,
        w_snr: float = 1.2,
        w_var: float = 0.8,
        w_energy: float = -0.6,
        bias: float = 0.0,
        smooth: float = 0.15,
    ) -> None:
        self.w_snr = w_snr
        self.w_var = w_var
        self.w_energy = w_energy
        self.bias = bias
        self.smooth = smooth
        self._prev_alpha = 0.5

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def compute_alpha(self, features: FrameFeatures) -> float:
        raw = (
            self.w_snr * features.snr_db
            + self.w_var * features.spectral_variance
            + self.w_energy * features.energy
            + self.bias
        )
        alpha = self._sigmoid(raw)
        # Temporal smoothing to avoid frame boundary artifacts
        alpha = self.smooth * alpha + (1.0 - self.smooth) * self._prev_alpha
        alpha_clipped = float(np.clip(alpha, 0.0, 1.0))
        self._prev_alpha = alpha_clipped
        return alpha_clipped
