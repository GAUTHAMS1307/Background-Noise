import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { denoiseUpload, fetchModels, fetchNoiseLevel } from "./lib/api";

/* ── Constants ─────────────────────────────────────────────────────── */
const WS_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1")
  .replace(/^http/, "ws")
  .replace(/^https/, "wss");

const SAMPLE_RATE = 16000;
const BLOCK_SAMPLES = 480; // 30 ms frames
const MAX_FILE_BYTES = 50 * 1024 * 1024; // 50 MB — must match backend MAX_UPLOAD_BYTES

// Upload progress simulation constants
const MAX_SIMULATED_PROGRESS = 88;
const PROGRESS_INCREMENT = 7;
const PROGRESS_TICK_MS = 350;

/* ── Toast types ───────────────────────────────────────────────────── */
type ToastKind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

let toastSeq = 0;

/* ── Helpers ───────────────────────────────────────────────────────── */
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* ── ToastStack ────────────────────────────────────────────────────── */
function ToastStack({
  toasts,
  onDismiss,
}: {
  toasts: Toast[];
  onDismiss: (id: number) => void;
}) {
  const icons: Record<ToastKind, string> = { success: "✓", error: "✕", info: "ℹ" };
  return (
    <div className="toast-stack" role="region" aria-label="Notifications">
      {toasts.map((t) => (
        <div key={t.id} className={`toast toast--${t.kind}`} role="alert">
          <span className="toast__icon">{icons[t.kind]}</span>
          <span>{t.message}</span>
          <button
            className="toast__close"
            aria-label="Dismiss"
            onClick={() => onDismiss(t.id)}
          >
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

/* ── Main App ──────────────────────────────────────────────────────── */
export default function App() {
  /* ── State ── */
  const [models, setModels] = useState<string[]>(["rnnoise", "demucs"]);
  const [model, setModel] = useState<string>("rnnoise");
  const [enhanceVoice, setEnhanceVoice] = useState<boolean>(false);
  const [liveEnabled, setLiveEnabled] = useState<boolean>(false);
  const [noiseDb, setNoiseDb] = useState<number>(-90);
  const [uploading, setUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [downloadUrl, setDownloadUrl] = useState<string>("");
  const [liveError, setLiveError] = useState<string>("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState<boolean>(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [latencyMs, setLatencyMs] = useState<number>(0);
  const [processedSize, setProcessedSize] = useState<number>(0);
  const [liveFrameCount, setLiveFrameCount] = useState<number>(0);

  /* ── Refs ── */
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animRef = useRef<number>(0);
  const waveDataRef = useRef<Float32Array>(new Float32Array(128));

  /* ── Toasts ── */
  const addToast = useCallback((kind: ToastKind, message: string) => {
    const id = ++toastSeq;
    setToasts((prev) => [...prev, { id, kind, message }]);
    setTimeout(() => setToasts((prev) => prev.filter((t) => t.id !== id)), 4500);
  }, []);

  const dismissToast = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  /* ── Boot: load models ── */
  useEffect(() => {
    fetchModels()
      .then(setModels)
      .catch(() => setModels(["rnnoise", "demucs"]));
  }, []);

  /* ── Noise level polling ── */
  useEffect(() => {
    const timer = setInterval(() => {
      fetchNoiseLevel()
        .then(setNoiseDb)
        .catch(() => setNoiseDb(-90));
    }, 800);
    return () => clearInterval(timer);
  }, []);

  /* ── Noise meter percentage (0–100) ── */
  const noisePercent = useMemo(
    () => Math.min(100, Math.max(0, (noiseDb + 90) * 1.2)),
    [noiseDb],
  );

  /* ── Waveform animation ── */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const draw = () => {
      const W = canvas.width;
      const H = canvas.height;
      ctx.clearRect(0, 0, W, H);

      const data = waveDataRef.current;
      const barW = Math.max(2, Math.floor(W / data.length) - 1);
      const gap = 1;

      data.forEach((v, i) => {
        const h = Math.max(3, Math.abs(v) * H * 0.85);
        const x = i * (barW + gap);
        const y = (H - h) / 2;
        const alpha = liveEnabled ? 0.9 : 0.25;
        ctx.fillStyle = `rgba(73,242,184,${alpha})`;
        ctx.beginPath();
        ctx.roundRect(x, y, barW, h, 2);
        ctx.fill();
      });

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [liveEnabled]);

  /* ── Stop live pipeline ── */
  const stopLive = useCallback(() => {
    processorRef.current?.disconnect();
    sourceNodeRef.current?.disconnect();
    audioCtxRef.current?.close().catch(() => {});
    wsRef.current?.close();
    streamRef.current?.getTracks().forEach((t) => t.stop());
    processorRef.current = null;
    sourceNodeRef.current = null;
    audioCtxRef.current = null;
    wsRef.current = null;
    streamRef.current = null;
    waveDataRef.current = new Float32Array(128);
  }, []);

  /* ── Start live pipeline ── */
  const startLive = useCallback(async () => {
    setLiveError("");
    setLiveFrameCount(0);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: false, sampleRate: SAMPLE_RATE },
        video: false,
      });
      streamRef.current = mediaStream;

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE });
      audioCtxRef.current = ctx;

      const wsUrl = `${WS_BASE}/realtime/ws?model=${model}&sr=${SAMPLE_RATE}&enhance=${enhanceVoice}`;
      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve();
        ws.onerror = () => reject(new Error("WebSocket connection failed"));
        setTimeout(() => reject(new Error("Connection timed out")), 5000);
      });

      addToast("success", "Live noise cancellation started");

      const source = ctx.createMediaStreamSource(mediaStream);
      sourceNodeRef.current = source;

      const proc = ctx.createScriptProcessor(BLOCK_SAMPLES, 1, 1);
      processorRef.current = proc;

      const pendingFrames: Float32Array[] = [];

      ws.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
        const buf = ev.data;
        if (buf.byteLength <= 4) return;
        const latency = new DataView(buf).getFloat32(0, true);
        setLatencyMs(latency);
        const pcm16 = new Int16Array(buf, 4);
        const f32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32767;
        pendingFrames.push(f32);
      };

      proc.onaudioprocess = (ev: AudioProcessingEvent) => {
        const input = ev.inputBuffer.getChannelData(0);

        // Update waveform visualizer: compute RMS per segment for accurate visualization
        const bucketCount = waveDataRef.current.length;
        const samplesPerBucket = Math.max(1, Math.floor(input.length / bucketCount));
        for (let i = 0; i < bucketCount; i++) {
          let sumSq = 0;
          const start = i * samplesPerBucket;
          const end = Math.min(start + samplesPerBucket, input.length);
          for (let j = start; j < end; j++) sumSq += input[j] * input[j];
          waveDataRef.current[i] = Math.sqrt(sumSq / (end - start));
        }

        const pcm16 = Int16Array.from(input, (x) =>
          Math.max(-32768, Math.min(32767, (x * 32767) | 0)),
        );
        if (ws.readyState === WebSocket.OPEN) ws.send(pcm16.buffer);

        setLiveFrameCount((n) => n + 1);

        const out = ev.outputBuffer.getChannelData(0);
        if (pendingFrames.length > 0) {
          const frame = pendingFrames.shift()!;
          const len = Math.min(frame.length, out.length);
          out.set(frame.subarray(0, len));
          if (len < out.length) out.fill(0, len);
        } else {
          out.fill(0);
        }
      };

      source.connect(proc);
      proc.connect(ctx.destination);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Microphone access denied";
      setLiveError(msg);
      addToast("error", msg);
      stopLive();
    }
  }, [model, enhanceVoice, stopLive, addToast]);

  const toggleLive = useCallback(() => {
    if (liveEnabled) {
      stopLive();
      setLiveEnabled(false);
      addToast("info", "Live session stopped");
    } else {
      setLiveEnabled(true);
      void startLive();
    }
  }, [liveEnabled, startLive, stopLive, addToast]);

  useEffect(() => () => stopLive(), [stopLive]);

  /* ── File selection ── */
  const handleFileSelect = useCallback(
    (file: File) => {
      if (file.size > MAX_FILE_BYTES) {
        addToast("error", `File too large — max ${formatBytes(MAX_FILE_BYTES)}`);
        return;
      }
      if (!file.type.startsWith("audio/") && !/\.(wav|mp3|flac|ogg|m4a|aac|webm)$/i.test(file.name)) {
        addToast("error", "Please select an audio file (WAV, MP3, FLAC, OGG, M4A…)");
        return;
      }
      setSelectedFile(file);
      setDownloadUrl("");
    },
    [addToast],
  );

  /* ── Offline upload ── */
  const onUpload = useCallback(async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadProgress(10);
    try {
      // Animate progress while waiting
      const tick = setInterval(() => {
        setUploadProgress((p) => Math.min(MAX_SIMULATED_PROGRESS, p + PROGRESS_INCREMENT));
      }, PROGRESS_TICK_MS);
      const blob = await denoiseUpload(selectedFile, model, enhanceVoice);
      clearInterval(tick);
      setUploadProgress(100);
      if (downloadUrl) URL.revokeObjectURL(downloadUrl);
      const url = URL.createObjectURL(blob);
      setDownloadUrl(url);
      setProcessedSize(blob.size);
      addToast("success", "Audio cleaned — ready to download!");
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Processing failed";
      addToast("error", msg);
      setUploadProgress(0);
    } finally {
      setUploading(false);
      setTimeout(() => setUploadProgress(0), 1200);
    }
  }, [selectedFile, model, enhanceVoice, downloadUrl, addToast]);

  /* ── Drag & drop handlers ── */
  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(true);
  };
  const onDragLeave = () => setDragOver(false);
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  };

  /* ── Computed stats ── */
  const liveDurationSec = Math.round((liveFrameCount * BLOCK_SAMPLES) / SAMPLE_RATE);

  /* ── Render ── */
  return (
    <>
      <ToastStack toasts={toasts} onDismiss={dismissToast} />

      <div className="app-shell">
        {/* Hero */}
        <header className="hero">
          <div className="hero__badge">
            <span className="hero__badge-dot" />
            AI Powered
          </div>
          <h1 className="hero__title">NoiseShield Studio</h1>
          <p className="hero__subtitle">
            Real-time and offline background noise cancellation — cleaner voice, effortlessly.
          </p>
        </header>

        {/* ── Settings Card ── */}
        <section className="card" aria-labelledby="settings-title">
          <div className="card__header">
            <div className="card__icon">⚙</div>
            <div>
              <div className="card__title" id="settings-title">Processing Settings</div>
              <div className="card__desc">Choose model and enhancement options</div>
            </div>
          </div>

          <div className="field-row">
            <div className="field">
              <label htmlFor="model-select">Denoising Model</label>
              <select
                id="model-select"
                value={model}
                onChange={(e) => {
                  if (liveEnabled) {
                    stopLive();
                    setLiveEnabled(false);
                  }
                  setModel(e.target.value);
                }}
              >
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m === "rnnoise" ? "RNNoise (fast)" : m === "demucs" ? "Demucs (deep)" : m.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>Voice Enhancement</label>
              <label className="toggle-field" htmlFor="enhance-toggle">
                <div className="toggle-field__label">
                  <span className="toggle-field__name">Boost voice clarity</span>
                  <span className="toggle-field__hint">Apply harmonic enhancement</span>
                </div>
                <span className="toggle-switch">
                  <input
                    id="enhance-toggle"
                    type="checkbox"
                    checked={enhanceVoice}
                    onChange={(e) => setEnhanceVoice(e.target.checked)}
                  />
                  <span className="toggle-switch__track" />
                </span>
              </label>
            </div>
          </div>
        </section>

        {/* ── Live Mic Card ── */}
        <section className="card" aria-labelledby="live-title">
          <div className="card__header">
            <div className="card__icon">🎙</div>
            <div>
              <div className="card__title" id="live-title">Live Microphone Filter</div>
              <div className="card__desc">Real-time noise cancellation via WebSocket</div>
            </div>
          </div>

          <div className="live-controls">
            <button
              className={`btn ${liveEnabled ? "btn--danger" : "btn--primary"}`}
              onClick={toggleLive}
              aria-pressed={liveEnabled}
            >
              {liveEnabled ? "⏹ Stop Session" : "▶ Start Session"}
            </button>

            <span className={`live-status ${liveEnabled ? "live-status--on" : "live-status--off"}`}>
              <span className="live-status__dot" />
              {liveEnabled ? "Live" : "Idle"}
            </span>

            {liveEnabled && (
              <span style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginLeft: "auto" }}>
                Latency: <strong style={{ color: "var(--text-secondary)" }}>{latencyMs.toFixed(1)} ms</strong>
              </span>
            )}
          </div>

          {liveError && (
            <div className="inline-error" role="alert">
              <span>⚠</span>
              {liveError}
            </div>
          )}

          {/* Waveform */}
          <div className="waveform-wrap">
            <canvas
              ref={canvasRef}
              className="waveform-canvas"
              width={800}
              height={80}
              aria-label="Audio waveform visualizer"
            />
            {!liveEnabled && (
              <div className="waveform-placeholder">
                Waveform appears during live session
              </div>
            )}
          </div>

          {/* Noise Meter */}
          <div className="meter-section" style={{ marginTop: 18 }}>
            <div className="meter-header">
              <span className="meter-label">Noise Level</span>
              <span className="meter-value">{noiseDb.toFixed(1)} dB</span>
            </div>
            <div className="meter-track" role="meter" aria-valuenow={noisePercent} aria-valuemin={0} aria-valuemax={100}>
              <div className="meter-fill" style={{ width: `${noisePercent}%` }} />
            </div>
            <div className="meter-ticks">
              <span className="meter-tick">–90 dB</span>
              <span className="meter-tick">–60 dB</span>
              <span className="meter-tick">–30 dB</span>
              <span className="meter-tick">0 dB</span>
            </div>
          </div>

          {/* Live Stats */}
          {liveEnabled && (
            <div className="stats-row">
              <div className="stat-chip">
                <div className="stat-chip__value">{latencyMs.toFixed(0)}</div>
                <div className="stat-chip__label">Latency (ms)</div>
              </div>
              <div className="stat-chip">
                <div className="stat-chip__value">{liveDurationSec}</div>
                <div className="stat-chip__label">Duration (s)</div>
              </div>
              <div className="stat-chip">
                <div className="stat-chip__value">{model.toUpperCase()}</div>
                <div className="stat-chip__label">Model</div>
              </div>
            </div>
          )}
        </section>

        {/* ── Offline Upload Card ── */}
        <section className="card" aria-labelledby="offline-title">
          <div className="card__header">
            <div className="card__icon">🎧</div>
            <div>
              <div className="card__title" id="offline-title">Offline Audio Cleanup</div>
              <div className="card__desc">Upload a file and download the cleaned version</div>
            </div>
          </div>

          {/* Drop Zone */}
          <div
            className={`drop-zone${dragOver ? " drop-zone--active" : ""}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={() => document.getElementById("audio-file-input")?.click()}
            role="button"
            tabIndex={0}
            aria-label="Upload audio file"
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                document.getElementById("audio-file-input")?.click();
              }
            }}
          >
            <div className="drop-zone__icon">{selectedFile ? "🎵" : "📂"}</div>
            <div className="drop-zone__title">
              {selectedFile ? "Change file" : "Drop audio file here"}
            </div>
            <div className="drop-zone__hint">
              {selectedFile ? "Click to choose a different file" : "or click to browse — WAV, MP3, FLAC, OGG, M4A supported (max 50 MB)"}
            </div>
            <input
              id="audio-file-input"
              type="file"
              accept="audio/*,.wav,.mp3,.flac,.ogg,.m4a,.aac,.webm"
              className="drop-zone__input"
              tabIndex={-1}
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleFileSelect(f);
                e.target.value = "";
              }}
            />
          </div>

          {/* File Info */}
          {selectedFile && (
            <div className="file-info">
              <span className="file-info__icon">🎵</span>
              <div className="file-info__meta">
                <div className="file-info__name" title={selectedFile.name}>
                  {selectedFile.name}
                </div>
                <div className="file-info__size">{formatBytes(selectedFile.size)}</div>
              </div>
              <button
                className="btn btn--icon btn--ghost file-info__clear"
                aria-label="Remove file"
                onClick={() => {
                  setSelectedFile(null);
                  setDownloadUrl("");
                }}
              >
                ✕
              </button>
            </div>
          )}

          {/* Upload Button */}
          {selectedFile && !downloadUrl && (
            <div style={{ marginTop: 14 }}>
              <button
                className="btn btn--primary btn--full"
                onClick={() => void onUpload()}
                disabled={uploading}
              >
                {uploading ? (
                  <>
                    <span className="btn__spinner" />
                    Processing…
                  </>
                ) : (
                  "✦ Remove Noise"
                )}
              </button>
            </div>
          )}

          {/* Progress */}
          {uploading && (
            <div className="progress-wrap">
              <div className="progress-label">
                <span>Processing audio with {model.toUpperCase()}…</span>
                <span>{uploadProgress}%</span>
              </div>
              <div className="progress-track">
                <div
                  className={`progress-fill${uploadProgress < 95 ? " progress-fill--indeterminate" : ""}`}
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          {/* Result */}
          {downloadUrl && !uploading && (
            <div className="result-section">
              <div className="divider" />
              <div className="result-header">
                <span className="result-check">✓</span>
                Noise removal complete
              </div>
              <audio className="result-player" controls src={downloadUrl} />
              <div className="result-actions" style={{ marginTop: 12 }}>
                <a
                  className="btn btn--primary"
                  href={downloadUrl}
                  download={`cleaned_${selectedFile?.name ?? "output.wav"}`}
                >
                  ↓ Download Cleaned Audio
                </a>
                <button
                  className="btn btn--ghost"
                  onClick={() => {
                    setDownloadUrl("");
                    setSelectedFile(null);
                    setProcessedSize(0);
                  }}
                >
                  Process another file
                </button>
              </div>
              {processedSize > 0 && (
                <div className="stats-row" style={{ marginTop: 14 }}>
                  <div className="stat-chip">
                    <div className="stat-chip__value">{formatBytes(selectedFile?.size ?? 0)}</div>
                    <div className="stat-chip__label">Input size</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-chip__value">{formatBytes(processedSize)}</div>
                    <div className="stat-chip__label">Output size</div>
                  </div>
                  <div className="stat-chip">
                    <div className="stat-chip__value">{model.toUpperCase()}</div>
                    <div className="stat-chip__label">Model used</div>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>

        {/* Footer */}
        <footer className="app-footer">
          NoiseShield Studio — AI noise cancellation powered by RNNoise &amp; Demucs
        </footer>
      </div>
    </>
  );
}
