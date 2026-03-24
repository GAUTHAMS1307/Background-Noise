from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from project.fusion.pipeline import AdaptiveHybridEnhancer
from project.utils.audio_io import frame_generator, load_audio, save_audio
from project.utils.logger import FrameCSVLogger


class OfflineFileProcessor:
    """Offline enhancer that reuses the adaptive hybrid pipeline frame-by-frame."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,
        hop_size: int | None = None,
        logger: FrameCSVLogger | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size or frame_size
        self.enhancer = AdaptiveHybridEnhancer(sample_rate=sample_rate, frame_size=frame_size, hop_size=self.hop_size)
        self.logger = logger

    def process_file(self, input_path: str | Path, output_path: str | Path) -> None:
        signal, sr = load_audio(input_path, target_sr=self.sample_rate)
        enhanced_frames: list[np.ndarray] = []
        for frame in frame_generator(signal, self.frame_size, self.hop_size):
            out = self.enhancer.process_frame(frame)
            enhanced_frames.append(out.enhanced)
            if self.logger:
                self.logger.log(
                    {
                        "alpha": out.alpha,
                        "snr_db": out.features.snr_db,
                        "energy": out.features.energy,
                        "spectral_variance": out.features.spectral_variance,
                        "noise_type": out.features.noise_type,
                        "processing_ms": out.processing_ms,
                    }
                )
        enhanced = np.concatenate(enhanced_frames)[: len(signal)]
        save_audio(output_path, enhanced, sr)
