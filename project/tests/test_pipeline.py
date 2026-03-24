import numpy as np

from project.controller.adaptive_controller import AdaptiveController
from project.features.audio_features import FrameFeatures, extract_features
from project.fusion.dynamic_fusion import DynamicFuser
from project.fusion.pipeline import AdaptiveHybridEnhancer


def test_extract_features_outputs():
    sr = 16000
    t = np.linspace(0, 0.02, int(0.02 * sr), endpoint=False)
    frame = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    feats = extract_features(frame, sr)
    assert isinstance(feats, FrameFeatures)
    assert feats.noise_type in {"stationary", "non-stationary", "impulsive"}


def test_adaptive_controller_alpha_range():
    feats = FrameFeatures(snr_db=5.0, energy=0.01, spectral_variance=0.5, noise_type="stationary")
    controller = AdaptiveController()
    alpha = controller.compute_alpha(feats)
    assert 0.0 <= alpha <= 1.0


def test_dynamic_fuser_alignment():
    fuser = DynamicFuser()
    fast = np.ones(5, dtype=np.float32)
    quality = np.zeros(5, dtype=np.float32)
    fused = fuser.fuse(fast, quality, alpha=0.25)
    assert fused.shape == fast.shape
    assert np.isclose(fused.mean(), 0.25, atol=0.1)


def test_pipeline_process_frame():
    enhancer = AdaptiveHybridEnhancer(sample_rate=16000, frame_size=320)
    frame = (0.03 * np.random.randn(320)).astype(np.float32)
    out = enhancer.process_frame(frame)
    assert out.enhanced.shape == frame.shape
    assert 0.0 <= out.alpha <= 1.0
