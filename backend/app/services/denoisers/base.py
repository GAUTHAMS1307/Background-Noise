from abc import ABC, abstractmethod

import numpy as np


class BaseDenoiser(ABC):
    name: str = "base"

    @abstractmethod
    def process(self, signal: np.ndarray, sample_rate: int, enhance_voice: bool = False) -> np.ndarray:
        raise NotImplementedError
