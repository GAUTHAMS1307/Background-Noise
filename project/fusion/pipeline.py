from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from project.controller.adaptive_controller import AdaptiveController
from project.features.audio_features import FrameFeatures, extract_features
from project.fusion.dynamic_fusion import DynamicFuser
from project.models.demucs_model import DemucsModel
from project.models.rnnoise_model import RNNoiseModel


@dataclass
class FrameOutput:
    enhanced: np.ndarray
    alpha: float
    features: FrameFeatures
    processing_ms: float


class AdaptiveHybridEnhancer:
    """Core adaptive hybrid pipeline shared by offline and realtime paths."""

    def __init__(self, sample_rate: int = 16000, frame_size: int = 320, hop_size: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size or frame_size
        self.fast_model = RNNoiseModel(sample_rate=sample_rate, frame_length=frame_size)
        self.quality_model = DemucsModel(sample_rate=sample_rate, frame_length=frame_size)
        self.controller = AdaptiveController()
        self.fuser = DynamicFuser()

    def process_frame(self, frame: np.ndarray) -> FrameOutput:
        t0 = time.perf_counter()
        feats = extract_features(frame, self.sample_rate)
        fast = self.fast_model.process_frame(frame, feats)
        quality = self.quality_model.process_frame(frame)
        alpha = self.controller.compute_alpha(feats)
        fused = self.fuser.fuse(fast, quality, alpha)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return FrameOutput(enhanced=fused, alpha=alpha, features=feats, processing_ms=elapsed_ms)
