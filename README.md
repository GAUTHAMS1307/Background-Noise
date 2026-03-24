# NoiseShield: Real-Time and Offline Noise Cancellation

NoiseShield is a modular project for suppressing background noise in voice calls and recorded audio.

## What is included

- Real-time denoising websocket API for streaming microphone chunks
- Offline upload-clean-download API for recordings
- Model switching (`rnnoise`, `demucs`) through a stable registry
- Noise level telemetry endpoint for UI visualization
- React frontend with live toggle, model switch, and upload flow
- Benchmark script for SNR and latency comparison

## Architecture

- `backend/`: FastAPI service, denoiser registry, offline and realtime routes
- `frontend/`: React + Vite UI for control and demos
- `realtime/`: CLI websocket mic streaming client
- `scripts/`: benchmarking and evaluation utilities
- `docs/`: architecture and performance notes

## Quick start

1. Backend setup

```bash
make backend-install
make backend-run
```

2. Frontend setup (Node.js required)

```bash
cd frontend
npm install
npm run dev
```

3. Realtime streaming simulation

```bash
cd backend
. .venv/bin/activate
python ../realtime/mic_stream_client.py --api "ws://localhost:8000/api/v1/realtime/ws?model=rnnoise&sr=16000"
```

4. Offline denoise via API

```bash
curl -X POST "http://localhost:8000/api/v1/offline/denoise" \
  -F "audio=@noisy.wav" \
  -F "model=demucs" \
  -F "enhance_voice=true" \
  --output cleaned.wav
```

## Production notes

- Replace fallback denoisers with native RNNoise and Demucs inference backends.
- Run Demucs on GPU for best quality where available.
- Keep realtime chunk sizes low (10-30 ms) to stay below 50 ms latency.
- Add PESQ/STOI batch evaluation in CI when reference clean targets are available.

## Adaptive hybrid research pipeline

The `/project` directory contains a research-only, UI-free implementation of **Adaptive Hybrid Noise-Aware Real-Time Speech Enhancement with Dynamic Model Fusion**. It is fully modular:

- `features/audio_features.py` – STFT-based SNR, energy, variance + noise-type classification
- `models/rnnoise_model.py` – fast path denoiser for 20 ms frames
- `models/demucs_model.py` – simplified Demucs-style quality path (uses PyTorch when available)
- `controller/adaptive_controller.py` – computes adaptive α from features
- `fusion/dynamic_fusion.py` & `fusion/pipeline.py` – dynamic fusion of fast/quality outputs
- `realtime/stream_processor.py` – microphone streaming path (sounddevice)
- `offline/file_processor.py` – offline WAV processing using the same pipeline
- `utils/audio_io.py`, `utils/logger.py` – helpers for framing, IO, and per-frame CSV telemetry

### Install dependencies (research pipeline)

```bash
pip install -r project/requirements.txt  # add --index-url https://download.pytorch.org/whl/cpu if torch is missing
```

### Offline processing

```bash
python -m project.main offline --input noisy.wav --output enhanced.wav --log frame_log.csv --sample-rate 16000
```

### Real-time processing

```bash
python -m project.main realtime --sample-rate 16000 --frame-size 320 --log frame_log.csv
```

Each frame logs α, SNR, energy, spectral variance, noise type, and processing time to CSV for analysis.
