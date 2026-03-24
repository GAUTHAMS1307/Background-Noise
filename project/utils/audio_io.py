from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int) -> Tuple[np.ndarray, int]:
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return audio.astype(np.float32), sr


def save_audio(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(Path(path), audio, sample_rate)


def frame_generator(signal: np.ndarray, frame_size: int, hop_size: int) -> Iterable[np.ndarray]:
    for start in range(0, len(signal), hop_size):
        frame = signal[start : start + frame_size]
        if len(frame) < frame_size:
            frame = np.pad(frame, (0, frame_size - len(frame)), mode="constant")
        yield frame.astype(np.float32)
