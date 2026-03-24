import struct
import time

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.models.model_registry import registry


router = APIRouter()
_noise_levels: list[float] = []


def _pcm16_to_float(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def _float_to_pcm16(data: np.ndarray) -> bytes:
    clipped = np.clip(data, -1.0, 1.0)
    out = (clipped * 32767.0).astype(np.int16)
    return out.tobytes()


@router.get("/metrics/noise-level")
async def noise_level() -> dict[str, float]:
    if not _noise_levels:
        return {"db": -90.0}
    return {"db": float(sum(_noise_levels[-50:]) / min(50, len(_noise_levels)))}


@router.websocket("/realtime/ws")
async def realtime_ws(ws: WebSocket) -> None:
    await ws.accept()
    model = ws.query_params.get("model", "rnnoise")
    sr = int(ws.query_params.get("sr", "16000"))
    enhance_voice = ws.query_params.get("enhance", "false").lower() == "true"
    denoiser = registry.get(model)

    try:
        while True:
            payload = await ws.receive_bytes()
            t0 = time.perf_counter()
            chunk = _pcm16_to_float(payload)
            cleaned = denoiser.process(chunk, sample_rate=sr, enhance_voice=enhance_voice)
            rms = float(np.sqrt(np.mean(np.square(cleaned)) + 1e-9))
            db = 20.0 * np.log10(rms + 1e-9)
            _noise_levels.append(db)

            latency_ms = (time.perf_counter() - t0) * 1000.0
            meta = struct.pack("f", latency_ms)
            await ws.send_bytes(meta + _float_to_pcm16(cleaned))
    except WebSocketDisconnect:
        return
