from io import BytesIO

import numpy as np
import soundfile as sf

from app.models.model_registry import registry


def denoise_bytes(raw_audio: bytes, model_name: str, enhance_voice: bool = False) -> bytes:
    signal, sr = sf.read(BytesIO(raw_audio), dtype="float32")
    if signal.ndim == 2:
        signal = signal.mean(axis=1)
    signal = np.asarray(signal, dtype=np.float32)

    denoiser = registry.get(model_name)
    enhanced = denoiser.process(signal=signal, sample_rate=sr, enhance_voice=enhance_voice)

    out = BytesIO()
    sf.write(out, enhanced, sr, format="WAV")
    out.seek(0)
    return out.read()
