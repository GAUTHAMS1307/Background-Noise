import os
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.model_registry import registry
from app.services.offline_processor import denoise_bytes

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
SUPPORTED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/flac",
    "audio/x-flac",
    "audio/ogg",
    "audio/vorbis",
    "audio/mp4",
    "audio/aac",
    "audio/webm",
    "audio/x-m4a",
    "application/octet-stream",  # some browsers send this for .wav
}
# Only allow alphanumerics, dots, dashes, underscores in filenames
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-]")

router = APIRouter()


def _sanitize_filename(name: str) -> str:
    """Return a safe filename stripped of path components and dangerous characters."""
    base = os.path.basename(name)
    return _SAFE_FILENAME_RE.sub("_", base) or "output.wav"


@router.get("/models")
async def list_models() -> dict[str, list[str]]:
    return {"models": registry.list_models()}


@router.post("/offline/denoise")
async def offline_denoise(
    audio: UploadFile = File(...),
    model: str = Form("rnnoise"),
    enhance_voice: bool = Form(False),
) -> StreamingResponse:
    # Validate model name
    available = registry.list_models()
    if model not in available:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model '{model}'. Available: {', '.join(available)}",
        )

    # Validate content type (lenient — also accept octet-stream)
    ct = (audio.content_type or "").split(";")[0].strip().lower()
    if ct and ct not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{ct}'. Expected an audio file.",
        )

    # Read with size guard to avoid memory exhaustion
    raw = await audio.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    if len(raw) == 0:
        raise HTTPException(status_code=422, detail="Uploaded file is empty.")

    try:
        denoised = denoise_bytes(raw, model_name=model, enhance_voice=enhance_voice)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    safe_name = _sanitize_filename(audio.filename or "output.wav")
    headers = {"Content-Disposition": f'attachment; filename="cleaned_{safe_name}"'}
    return StreamingResponse(iter([denoised]), media_type="audio/wav", headers=headers)
