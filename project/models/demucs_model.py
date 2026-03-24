from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - torch may be unavailable in the runtime
    torch = None  # type: ignore
    nn = None  # type: ignore


@dataclass
class _DemucsConfig:
    hidden: int = 16
    kernel_size: int = 7
    depth: int = 2


if nn is not None:

    class _TinyDemucs(nn.Module):  # type: ignore[misc]
        def __init__(self, cfg: _DemucsConfig) -> None:
            super().__init__()
            layers = []
            in_channels = 1
            for _ in range(cfg.depth):
                layers.append(nn.Conv1d(in_channels, cfg.hidden, cfg.kernel_size, padding=cfg.kernel_size // 2))
                layers.append(nn.ReLU())
                layers.append(nn.Conv1d(cfg.hidden, in_channels, cfg.kernel_size, padding=cfg.kernel_size // 2))
                in_channels = 1
            self.net = nn.Sequential(*layers)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            residual = x
            out = self.net(x)
            return torch.tanh(out + residual)
else:  # pragma: no cover - fallback path without torch
    _TinyDemucs = None


class DemucsModel:
    """Simplified Demucs-like model for quality path inference."""

    def __init__(self, sample_rate: int, frame_length: int, config: Optional[_DemucsConfig] = None) -> None:
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.cfg = config or _DemucsConfig()
        self._torch_enabled = torch is not None and nn is not None and _TinyDemucs is not None
        if self._torch_enabled:
            self.device = torch.device("cpu")
            self.model = _TinyDemucs(self.cfg).to(self.device)
            self.model.eval()
        else:
            self.model = None
            self.device = None

    def _fallback(self, frame: np.ndarray) -> np.ndarray:
        # Mild smoothing filter to emulate quality enhancement when torch is unavailable.
        window = np.hanning(len(frame))
        smoothed = np.convolve(frame * window, np.ones(5) / 5, mode="same")
        return np.clip(smoothed.astype(np.float32), -1.0, 1.0)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self._torch_enabled:
            return self._fallback(frame)

        with torch.no_grad():
            x = torch.from_numpy(frame[None, None, :]).to(self.device, dtype=torch.float32)
            y = self.model(x).squeeze(0).squeeze(0)
            out = y.cpu().numpy()
        return np.clip(out.astype(np.float32), -1.0, 1.0)
