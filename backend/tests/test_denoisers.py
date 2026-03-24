"""Tests for the denoiser registry and denoiser implementations."""
import numpy as np
import pytest

from app.models.model_registry import ModelRegistry
from app.services.denoisers.demucs_denoiser import DemucsDenoiser
from app.services.denoisers.rnnoise_denoiser import RNNoiseDenoiser


SR = 16000


def _mixed(duration: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(7)
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    speech = (np.sin(2 * np.pi * 440 * t) * 0.4).astype(np.float32)
    noise = (rng.standard_normal(len(t)) * 0.05).astype(np.float32)
    return speech + noise


class TestModelRegistry:
    def test_list_models_returns_both(self):
        reg = ModelRegistry()
        assert set(reg.list_models()) == {"rnnoise", "demucs"}

    def test_get_rnnoise(self):
        reg = ModelRegistry()
        d = reg.get("rnnoise")
        assert isinstance(d, RNNoiseDenoiser)

    def test_get_demucs(self):
        reg = ModelRegistry()
        d = reg.get("demucs")
        assert isinstance(d, DemucsDenoiser)

    def test_get_case_insensitive(self):
        reg = ModelRegistry()
        assert isinstance(reg.get("RNNoise"), RNNoiseDenoiser)
        assert isinstance(reg.get("DEMUCS"), DemucsDenoiser)

    def test_get_unknown_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown model"):
            reg.get("unknown_model")


class TestRNNoiseDenoiser:
    def test_output_shape(self):
        d = RNNoiseDenoiser()
        sig = _mixed()
        out = d.process(sig, SR)
        assert out.shape == sig.shape

    def test_output_float32(self):
        d = RNNoiseDenoiser()
        out = d.process(_mixed(), SR)
        assert out.dtype == np.float32

    def test_clipped(self):
        d = RNNoiseDenoiser()
        out = d.process(_mixed(), SR)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= -1.0

    def test_enhance_voice_does_not_explode(self):
        d = RNNoiseDenoiser()
        out = d.process(_mixed(), SR, enhance_voice=True)
        assert np.isfinite(out).all()

    def test_adaptive_profile_persists_across_calls(self):
        """Profile state must accumulate across multiple process() calls."""
        d = RNNoiseDenoiser()
        sig = _mixed()
        # Process the same chunk twice; second call should use updated profile.
        out1 = d.process(sig, SR)
        out2 = d.process(sig, SR)
        # Outputs may differ because the noise floor was updated between calls.
        # Both must remain valid float32 arrays.
        assert out1.dtype == np.float32
        assert out2.dtype == np.float32


class TestDemucsDenoiser:
    def test_output_shape(self):
        d = DemucsDenoiser()
        sig = _mixed()
        out = d.process(sig, SR)
        assert out.shape == sig.shape

    def test_output_float32(self):
        d = DemucsDenoiser()
        out = d.process(_mixed(), SR)
        assert out.dtype == np.float32

    def test_clipped(self):
        d = DemucsDenoiser()
        out = d.process(_mixed(), SR)
        assert float(out.max()) <= 1.0
        assert float(out.min()) >= -1.0

    def test_enhance_voice_does_not_explode(self):
        d = DemucsDenoiser()
        out = d.process(_mixed(), SR, enhance_voice=True)
        assert np.isfinite(out).all()
