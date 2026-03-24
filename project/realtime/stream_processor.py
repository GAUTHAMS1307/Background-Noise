from __future__ import annotations

import queue
import threading
from typing import Optional

import numpy as np

from project.fusion.pipeline import AdaptiveHybridEnhancer
from project.utils.logger import FrameCSVLogger


class RealTimeStreamProcessor:
    """Duplex sounddevice stream that performs adaptive hybrid denoising in real time."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_size: int = 320,
        logger: FrameCSVLogger | None = None,
        device: Optional[int | str] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.logger = logger
        self.device = device
        self.enhancer = AdaptiveHybridEnhancer(sample_rate=sample_rate, frame_size=frame_size)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stop_event = threading.Event()

    def _callback(self, indata, outdata, frames, time_info, status) -> None:  # pragma: no cover - realtime path
        frame = indata[:, 0].copy().astype(np.float32)
        out = self.enhancer.process_frame(frame)
        if self.logger:
            self.logger.log(
                {
                    "alpha": out.alpha,
                    "snr_db": out.features.snr_db,
                    "energy": out.features.energy,
                    "spectral_variance": out.features.spectral_variance,
                    "noise_type": out.features.noise_type,
                    "processing_ms": out.processing_ms,
                }
            )
        outdata[:, 0] = out.enhanced

    def start(self, duration_seconds: Optional[float] = None) -> None:  # pragma: no cover - realtime path
        import sounddevice as sd

        stream = sd.Stream(
            device=self.device,
            samplerate=self.sample_rate,
            blocksize=self.frame_size,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )
        stream.start()
        try:
            if duration_seconds is not None:
                sd.sleep(int(duration_seconds * 1000))
            else:
                # Keep running until stop() is called.
                while not self._stop_event.is_set():
                    sd.sleep(200)
        finally:
            stream.stop()
            stream.close()

    def stop(self) -> None:  # pragma: no cover - realtime path
        self._stop_event.set()
