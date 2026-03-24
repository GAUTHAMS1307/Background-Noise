from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.models.model_registry import registry
from app.services.offline_processor import denoise_bytes


router = APIRouter()


@router.get("/models")
async def list_models() -> dict[str, list[str]]:
    return {"models": registry.list_models()}


@router.post("/offline/denoise")
async def offline_denoise(
    audio: UploadFile = File(...),
    model: str = Form("rnnoise"),
    enhance_voice: bool = Form(False),
) -> StreamingResponse:
    try:
        raw = await audio.read()
        denoised = denoise_bytes(raw, model_name=model, enhance_voice=enhance_voice)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    headers = {"Content-Disposition": f'attachment; filename="cleaned_{audio.filename or "output.wav"}"'}
    return StreamingResponse(iter([denoised]), media_type="audio/wav", headers=headers)
