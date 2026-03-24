import numpy as np
from scipy.signal import butter, lfilter, wiener


def _bandpass(signal: np.ndarray, sr: int, low: int = 80, high: int = 7800) -> np.ndarray:
    nyquist = sr * 0.5
    low_n = max(0.001, low / nyquist)
    high_n = min(0.999, high / nyquist)
    b, a = butter(4, [low_n, high_n], btype="band")
    return lfilter(b, a, signal)


def spectral_gate(signal: np.ndarray, sr: int, strength: float = 0.4) -> np.ndarray:
    frame = 512
    hop = 128
    window = np.hanning(frame)
    padded = np.pad(signal, (0, frame), mode="constant")
    frames = []
    for i in range(0, len(padded) - frame, hop):
        frames.append(padded[i : i + frame] * window)
    stft = np.fft.rfft(np.array(frames), axis=1)
    mag = np.abs(stft)
    phase = np.angle(stft)

    noise_floor = np.percentile(mag, 20, axis=0, keepdims=True)
    gate = np.clip((mag - strength * noise_floor) / (mag + 1e-9), 0.0, 1.0)
    den_mag = mag * gate
    den_stft = den_mag * np.exp(1j * phase)

    out = np.zeros(len(padded), dtype=np.float32)
    norm = np.zeros(len(padded), dtype=np.float32)
    for n, spec in enumerate(den_stft):
        chunk = np.fft.irfft(spec).real
        idx = n * hop
        out[idx : idx + frame] += chunk * window
        norm[idx : idx + frame] += window ** 2
    out /= np.maximum(norm, 1e-8)
    out = out[: len(signal)]
    out = wiener(out, mysize=15)
    out = _bandpass(out, sr)
    return np.clip(out.astype(np.float32), -1.0, 1.0)
