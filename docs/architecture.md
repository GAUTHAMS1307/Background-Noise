# Architecture and Model Choice

## Pipeline overview

1. Input enters either realtime websocket stream or offline upload endpoint.
2. Audio is normalized to float32 mono.
3. Selected denoiser processes audio.
4. Optional voice enhancement stage boosts clarity.
5. Output is returned as PCM stream (realtime) or WAV file (offline).

## Why RNNoise vs Demucs

- RNNoise path targets low-latency deployment for live calls and lower CPU usage.
- Demucs path targets stronger denoising quality for offline files and high-end hardware.

## Latency targets

- Realtime frame duration should remain around 10-30 ms.
- End-to-end processing target is under 50 ms.

## Online call integration

- Browser call path uses WebRTC + websocket denoiser bridging.
- Desktop path can route processed signal to a virtual microphone driver.
