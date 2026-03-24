import argparse
import asyncio
import struct

import numpy as np
import sounddevice as sd
import websockets


async def stream(uri: str, sample_rate: int, block_size: int) -> None:
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=32)

    def callback(indata, outdata, frames, _time, status):
        if status:
            print(status)
        pcm = np.clip(indata[:, 0], -1.0, 1.0)
        packet = (pcm * 32767.0).astype(np.int16).tobytes()
        try:
            queue.put_nowait(packet)
        except asyncio.QueueFull:
            pass
        outdata[:, 0] = indata[:, 0]

    async with websockets.connect(uri, max_size=2**22) as ws:
        with sd.Stream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=block_size, callback=callback):
            print("Live streaming started. Ctrl+C to stop.")
            while True:
                payload = await queue.get()
                await ws.send(payload)
                data = await ws.recv()
                if isinstance(data, bytes) and len(data) > 4:
                    latency = struct.unpack("f", data[:4])[0]
                    if latency > 50:
                        print(f"Warning: latency {latency:.2f} ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="ws://localhost:8000/api/v1/realtime/ws?model=rnnoise&sr=16000")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--block", type=int, default=480)
    args = parser.parse_args()
    asyncio.run(stream(args.api, args.sr, args.block))


if __name__ == "__main__":
    main()
