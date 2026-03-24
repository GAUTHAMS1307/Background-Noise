import numpy as np
from scipy.signal import butter, lfilter, wiener


SILENCE_RMS_THRESHOLD = 1e-6  # 20*log10(1e-6) ≈ -120 dBFS for normalized audio; avoids Wiener warnings on near-silence


def _bandpass(signal: np.ndarray, sr: int, low: int = 80, high: int = 7800) -> np.ndarray:
    nyquist = sr * 0.5
    low_n = max(0.001, low / nyquist)
    high_n = min(0.999, high / nyquist)
    b, a = butter(4, [low_n, high_n], btype="band")
    return lfilter(b, a, signal)


class AdaptiveNoiseProfile:
    """Exponentially-weighted moving average of the STFT magnitude noise floor.

    Call :meth:`update` with consecutive magnitude frames to adapt the profile
    to the current acoustic environment.  The learned floor is used instead of
    a percentile-based estimate, giving more robust suppression across varying
    background noise conditions.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha  # EMA smoothing factor (lower = slower adaptation)
        self._floor: np.ndarray | None = None

    def update(self, mag: np.ndarray) -> np.ndarray:
        """Update the noise-floor estimate and return the current estimate.

        Parameters
        ----------
        mag:
            2-D magnitude array ``(frames, freq_bins)`` from the STFT.

        Returns
        -------
        np.ndarray
            1-D noise-floor estimate ``(freq_bins,)``.
        """
        # Use a low percentile rather than the bare minimum to avoid
        # underestimation caused by transient signal peaks.
        frame_floor = np.percentile(mag, 5, axis=0)
        if self._floor is None:
            self._floor = frame_floor.copy()
        else:
            self._floor = self.alpha * frame_floor + (1.0 - self.alpha) * self._floor
        return self._floor

    def reset(self) -> None:
        self._floor = None


def spectral_gate(
    signal: np.ndarray,
    sr: int,
    strength: float = 0.4,
    noise_profile: AdaptiveNoiseProfile | None = None,
) -> np.ndarray:
    """Apply spectral gating to suppress noise in *signal*.

    Parameters
    ----------
    signal:
        Mono float32 audio signal in ``[-1, 1]``.
    sr:
        Sample rate in Hz.
    strength:
        Gate strength multiplier applied to the noise floor (0–1).
    noise_profile:
        Optional :class:`AdaptiveNoiseProfile` instance.  When supplied, the
        noise floor is learned from the signal rather than estimated via a
        percentile, enabling adaptive suppression that tracks environment
        changes over time.
    """
    # Standardize to float32 since the downstream DSP path assumes normalized audio.
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return np.zeros_like(signal)
    # Extremely low-energy frames (e.g., dead-silent background capture) can
    # trigger divide-by-zero warnings inside Wiener filtering. Short-circuit
    # to silence for such cases so the backend remains stable for all inputs.
    # Standard RMS computation keeps things vectorized and efficient for long buffers.
    rms = np.sqrt(np.mean(np.square(signal)))
    if rms < SILENCE_RMS_THRESHOLD:
        return np.zeros_like(signal)

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

    if noise_profile is not None:
        noise_floor = noise_profile.update(mag)[np.newaxis, :]
    else:
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
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = _bandpass(out, sr)
    return np.clip(out.astype(np.float32), -1.0, 1.0)
