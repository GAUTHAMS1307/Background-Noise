import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from app.models.model_registry import registry


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB (higher is better)."""
    noise = clean - test
    signal_power = np.mean(clean**2) + 1e-9
    noise_power = np.mean(noise**2) + 1e-9
    return float(10 * np.log10(signal_power / noise_power))


def _try_pesq(ref: np.ndarray, deg: np.ndarray, sr: int) -> float | None:
    """Compute PESQ score (wideband) if the library and sample-rate are available."""
    try:
        from pesq import pesq  # type: ignore[import-untyped]

        if sr not in (8000, 16000):
            return None
        return float(pesq(sr, ref, deg, "wb"))
    except Exception:
        return None


def _try_stoi(ref: np.ndarray, deg: np.ndarray, sr: int) -> float | None:
    """Compute STOI score if the library is available."""
    try:
        from pystoi import stoi  # type: ignore[import-untyped]

        return float(stoi(ref, deg, sr, extended=False))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark denoiser models with SNR, PESQ and STOI metrics.")
    parser.add_argument("--input", required=True, help="Path to the noisy input WAV file")
    parser.add_argument("--reference", default=None, help="Optional path to clean reference WAV for PESQ/STOI")
    parser.add_argument("--output-dir", default="docs/benchmarks", help="Directory to write cleaned files and report")
    args = parser.parse_args()

    wav, sr = sf.read(args.input, dtype="float32")
    if wav.ndim == 2:
        wav = wav.mean(axis=1)

    ref: np.ndarray | None = None
    if args.reference:
        ref_raw, ref_sr = sf.read(args.reference, dtype="float32")
        if ref_raw.ndim == 2:
            ref_raw = ref_raw.mean(axis=1)
        # Align lengths for PESQ/STOI
        min_len = min(len(wav), len(ref_raw))
        ref = ref_raw[:min_len]
        wav = wav[:min_len]
        if ref_sr != sr:
            print(f"Warning: reference sample-rate {ref_sr} != input {sr}; PESQ/STOI may be inaccurate.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = []
    for model_name in registry.list_models():
        denoiser = registry.get(model_name)
        t0 = time.perf_counter()
        cleaned = denoiser.process(wav, sr)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        score = snr_db(wav, cleaned)
        sf.write(out_dir / f"{model_name}_cleaned.wav", cleaned, sr)

        compare = ref if ref is not None else wav
        pesq_score = _try_pesq(compare, cleaned, sr)
        stoi_score = _try_stoi(compare, cleaned, sr)

        report.append((model_name, score, elapsed_ms, pesq_score, stoi_score))

    lines = [
        "# Benchmark Report",
        "",
        "| Model | SNR (dB) | Latency (ms) | PESQ | STOI |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, score, latency, pesq_score, stoi_score in report:
        pesq_str = f"{pesq_score:.3f}" if pesq_score is not None else "n/a"
        stoi_str = f"{stoi_score:.3f}" if stoi_score is not None else "n/a"
        lines.append(f"| {model_name} | {score:.2f} | {latency:.2f} | {pesq_str} | {stoi_str} |")
        print(f"{model_name}: SNR={score:.2f}dB  latency={latency:.2f}ms  PESQ={pesq_str}  STOI={stoi_str}")

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("Benchmark report written to", report_path)


if __name__ == "__main__":
    main()
