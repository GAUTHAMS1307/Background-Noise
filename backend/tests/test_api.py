"""Integration tests for the FastAPI endpoints."""
import io
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _make_wav(sr: int = 16000, duration: float = 0.2) -> bytes:
    """Create a minimal WAV file in memory."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_health_ok(self):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


class TestModelsEndpoint:
    def test_list_models(self):
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        data = r.json()
        assert "models" in data
        assert set(data["models"]) == {"rnnoise", "demucs"}


class TestNoiseLevelEndpoint:
    def test_noise_level_default(self):
        r = client.get("/api/v1/metrics/noise-level")
        assert r.status_code == 200
        data = r.json()
        assert "db" in data
        assert isinstance(data["db"], float)


class TestOfflineDenoise:
    def test_denoise_rnnoise(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "rnnoise", "enhance_voice": "false"},
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/wav")

    def test_denoise_demucs(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "demucs", "enhance_voice": "false"},
        )
        assert r.status_code == 200

    def test_denoise_with_enhance_voice(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "rnnoise", "enhance_voice": "true"},
        )
        assert r.status_code == 200

    def test_denoise_unknown_model_returns_400(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "nonexistent"},
        )
        assert r.status_code == 400

    def test_denoise_output_is_valid_wav(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("test.wav", wav_bytes, "audio/wav")},
            data={"model": "rnnoise"},
        )
        assert r.status_code == 200
        buf = io.BytesIO(r.content)
        out, sr = sf.read(buf)
        assert sr == 16000
        assert len(out) > 0

    def test_content_disposition_header(self):
        wav_bytes = _make_wav()
        r = client.post(
            "/api/v1/offline/denoise",
            files={"audio": ("my_file.wav", wav_bytes, "audio/wav")},
            data={"model": "rnnoise"},
        )
        assert "cleaned_my_file.wav" in r.headers.get("content-disposition", "")
