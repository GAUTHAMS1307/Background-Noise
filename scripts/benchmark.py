import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models.model_registry import registry


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    noise = clean - test
    signal_power = np.mean(clean**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9
    return float(10 * np.log10(signal_power / noise_power))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="docs/benchmarks")
    args = parser.parse_args()

    wav, sr = sf.read(args.input, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for model in registry.list_models():
        denoiser = registry.get(model)
        t0 = time.perf_counter()
        cleaned = denoiser.process(wav, sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        score = snr_db(wav, cleaned)
        sf.write(out_dir / f"{model}_cleaned.wav", cleaned, sr)
        report.append((model, score, elapsed_ms))

    lines = ["# Benchmark Report", "", "| Model | SNR(dB) | Latency(ms) |", "|---|---:|---:|"]
    for model, score, latency in report:
        lines.append(f"| {model} | {score:.2f} | {latency:.2f} |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("Benchmark report written to", out_dir / "report.md")


if __name__ == "__main__":
    main()
