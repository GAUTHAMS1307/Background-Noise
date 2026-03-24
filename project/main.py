from __future__ import annotations

import argparse
from pathlib import Path

from project.offline.file_processor import OfflineFileProcessor
from project.realtime.stream_processor import RealTimeStreamProcessor
from project.utils.logger import FrameCSVLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Hybrid Noise-Aware Speech Enhancement")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    rt = subparsers.add_parser("realtime", help="Run real-time microphone enhancement")
    rt.add_argument("--duration", type=float, default=None, help="Duration in seconds (omit to run until Ctrl+C)")
    rt.add_argument("--sample-rate", type=int, default=16000)
    rt.add_argument("--frame-size", type=int, default=320)
    rt.add_argument("--device", type=str, default=None, help="sounddevice input/output device")
    rt.add_argument("--log", type=Path, default=None, help="CSV log path")

    offline = subparsers.add_parser("offline", help="Process a WAV file with the adaptive hybrid pipeline")
    offline.add_argument("--input", type=Path, required=True)
    offline.add_argument("--output", type=Path, required=True)
    offline.add_argument("--log", type=Path, default=None, help="CSV log path")
    offline.add_argument("--sample-rate", type=int, default=16000)
    offline.add_argument("--frame-size", type=int, default=320)

    return parser.parse_args()


def run_realtime(args: argparse.Namespace) -> None:
    logger = FrameCSVLogger(args.log) if args.log else None
    processor = RealTimeStreamProcessor(
        sample_rate=args.sample_rate,
        frame_size=args.frame_size,
        device=args.device,
        logger=logger,
    )
    try:
        processor.start(duration_seconds=args.duration)
    finally:
        if logger:
            logger.close()


def run_offline(args: argparse.Namespace) -> None:
    logger = FrameCSVLogger(args.log) if args.log else None
    processor = OfflineFileProcessor(
        sample_rate=args.sample_rate, frame_size=args.frame_size, hop_size=args.frame_size, logger=logger
    )
    try:
        processor.process_file(args.input, args.output)
    finally:
        if logger:
            logger.close()


def main() -> None:
    args = parse_args()
    if args.mode == "realtime":
        run_realtime(args)
    else:
        run_offline(args)


if __name__ == "__main__":
    main()
