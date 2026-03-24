from __future__ import annotations

import numpy as np


class DynamicFuser:
    """Blend fast and quality outputs using a dynamic alpha with overlap smoothing."""

    def __init__(self, crossfade: float = 0.1) -> None:
        self.crossfade = crossfade
        self._prev_tail: np.ndarray | None = None

    def fuse(self, fast: np.ndarray, quality: np.ndarray, alpha: float) -> np.ndarray:
        assert fast.shape == quality.shape, "Frame alignment mismatch between fast and quality paths"
        fused = alpha * fast + (1.0 - alpha) * quality

        if self._prev_tail is not None and len(self._prev_tail) == len(fused):
            blend = self.crossfade
            fused = blend * self._prev_tail + (1.0 - blend) * fused
        self._prev_tail = fused.copy()
        return fused.astype(np.float32)
