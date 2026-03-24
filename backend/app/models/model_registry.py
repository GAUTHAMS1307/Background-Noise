from app.services.denoisers.base import BaseDenoiser
from app.services.denoisers.demucs_denoiser import DemucsDenoiser
from app.services.denoisers.rnnoise_denoiser import RNNoiseDenoiser


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, BaseDenoiser] = {
            "rnnoise": RNNoiseDenoiser(),
            "demucs": DemucsDenoiser(),
        }

    def list_models(self) -> list[str]:
        return sorted(self._models.keys())

    def get(self, name: str) -> BaseDenoiser:
        key = name.lower()
        if key not in self._models:
            raise ValueError(f"Unknown model: {name}")
        return self._models[key]


registry = ModelRegistry()
