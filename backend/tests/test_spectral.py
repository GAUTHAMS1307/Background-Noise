"""Tests for the spectral gating utility and adaptive noise profiling."""
import warnings

import numpy as np
import pytest

from app.utils.spectral import AdaptiveNoiseProfile, spectral_gate


def _sine(freq: int = 440, sr: int = 16000, duration: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * amp).astype(np.float32)


def _noise(sr: int = 16000, duration: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal(int(sr * duration)).astype(np.float32) * 0.1


class TestSpectralGate:
    def test_output_shape_matches_input(self):
        sig = _sine() + _noise()
        out = spectral_gate(sig, 16000)
        assert out.shape == sig.shape

    def test_output_dtype_is_float32(self):
        sig = _sine() + _noise()
        out = spectral_gate(sig, 16000)
        assert out.dtype == np.float32

    def test_output_clipped_to_unity(self):
        sig = _sine(amp=0.9) + _noise()
        out = spectral_gate(sig, 16000)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= -1.0

    def test_silence_in_silence_out(self):
        sig = np.zeros(8000, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = spectral_gate(sig, 16000)
        assert np.allclose(out, 0.0, atol=1e-4)

    def test_handles_empty_signal(self):
        sig = np.array([], dtype=np.float32)
        out = spectral_gate(sig, 16000)
        assert out.shape == sig.shape
        assert out.dtype == np.float32
        assert out.size == 0

    def test_noise_is_attenuated(self):
        """After gating, RMS of the signal should decrease."""
        sig = _noise(duration=1.0)
        sig_rms = float(np.sqrt(np.mean(sig**2)))
        out = spectral_gate(sig, 16000, strength=0.8)
        out_rms = float(np.sqrt(np.mean(out**2)))
        assert out_rms < sig_rms, "Noise should be attenuated by spectral gate"

    def test_with_adaptive_profile(self):
        profile = AdaptiveNoiseProfile(alpha=0.1)
        sig = _sine() + _noise()
        out = spectral_gate(sig, 16000, noise_profile=profile)
        assert out.shape == sig.shape
        assert out.dtype == np.float32

    def test_strength_zero_passes_through_approx(self):
        """Strength=0 means no gating; output should be non-trivially non-zero."""
        sig = _sine(freq=1000, amp=0.5)
        out = spectral_gate(sig, 16000, strength=0.0)
        # Signal should survive (non-zero RMS after bandpass)
        assert float(np.sqrt(np.mean(out**2))) > 0.01


class TestAdaptiveNoiseProfile:
    def test_update_returns_floor_shape(self):
        profile = AdaptiveNoiseProfile()
        mag = np.abs(np.random.default_rng(0).standard_normal((20, 257))).astype(np.float32)
        floor = profile.update(mag)
        assert floor.shape == (257,)

    def test_floor_decreases_monotonically_with_low_alpha(self):
        """With very low alpha the floor should barely move from its initial value."""
        profile = AdaptiveNoiseProfile(alpha=0.01)
        rng = np.random.default_rng(1)
        # Feed high-energy frames
        for _ in range(50):
            mag = rng.uniform(0.5, 1.0, size=(10, 257)).astype(np.float32)
            profile.update(mag)
        floor_high = profile._floor.copy()  # type: ignore[union-attr]

        # Feed low-energy frames – floor should slowly drop
        for _ in range(50):
            mag = rng.uniform(0.0, 0.01, size=(10, 257)).astype(np.float32)
            profile.update(mag)
        floor_low = profile._floor  # type: ignore[union-attr]

        # With alpha=0.01 and 50 updates, floor should have decreased somewhat
        assert float(floor_low.mean()) < float(floor_high.mean())

    def test_reset_clears_state(self):
        profile = AdaptiveNoiseProfile()
        mag = np.ones((10, 257), dtype=np.float32)
        profile.update(mag)
        assert profile._floor is not None
        profile.reset()
        assert profile._floor is None
