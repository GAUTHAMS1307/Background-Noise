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
