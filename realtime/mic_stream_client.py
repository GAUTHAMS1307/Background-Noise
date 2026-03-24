import argparse
import asyncio
import struct
from collections import deque

import numpy as np
import sounddevice as sd
import websockets


async def stream(uri: str, sample_rate: int, block_size: int) -> None:
    """Stream microphone input to the denoiser WebSocket and play back cleaned audio."""
    send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)
    # Use a deque as a lock-free ring-buffer for cleaned PCM frames played by the callback.
    playback_buf: deque[np.ndarray] = deque(maxlen=8)

    def callback(indata: np.ndarray, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(status)
        # Enqueue raw mic data for sending.
        pcm = np.clip(indata[:, 0], -1.0, 1.0)
        packet = (pcm * 32767.0).astype(np.int16).tobytes()
        try:
            send_queue.put_nowait(packet)
        except asyncio.QueueFull:
            pass
        # Play back the latest cleaned frame, or silence if none ready yet.
        # Write directly into outdata to avoid per-callback allocation.
        outdata[:, 0] = 0.0
        if playback_buf:
            cleaned = playback_buf.popleft()
            length = min(len(cleaned), frames)
            outdata[:length, 0] = cleaned[:length]

    async with websockets.connect(uri, max_size=2**22) as ws:
        with sd.Stream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block_size,
            callback=callback,
        ):
            print("Live streaming started. Ctrl+C to stop.")
            while True:
                payload = await send_queue.get()
                await ws.send(payload)
                data = await ws.recv()
                if isinstance(data, bytes) and len(data) > 4:
                    latency = struct.unpack("f", data[:4])[0]
                    if latency > 50:
                        print(f"Warning: latency {latency:.2f} ms")
                    pcm_bytes = data[4:]
                    cleaned = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
                    playback_buf.append(cleaned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="ws://localhost:8000/api/v1/realtime/ws?model=rnnoise&sr=16000")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--block", type=int, default=480)
    args = parser.parse_args()
    asyncio.run(stream(args.api, args.sr, args.block))


if __name__ == "__main__":
    main()
