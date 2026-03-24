import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { denoiseUpload, fetchModels, fetchNoiseLevel } from "./lib/api";

const WS_BASE = (import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1")
  .replace(/^http/, "ws")
  .replace(/^https/, "wss");

const SAMPLE_RATE = 16000;
const BLOCK_SAMPLES = 480; // 30 ms frames

export default function App() {
  const [models, setModels] = useState<string[]>(["rnnoise", "demucs"]);
  const [model, setModel] = useState<string>("rnnoise");
  const [enhanceVoice, setEnhanceVoice] = useState<boolean>(false);
  const [liveEnabled, setLiveEnabled] = useState<boolean>(false);
  const [noiseDb, setNoiseDb] = useState<number>(-90);
  const [uploading, setUploading] = useState<boolean>(false);
  const [downloadUrl, setDownloadUrl] = useState<string>("");
  const [liveError, setLiveError] = useState<string>("");

  // Refs for WebSocket + Web Audio live pipeline
  const wsRef = useRef<WebSocket | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    fetchModels().then(setModels).catch(() => setModels(["rnnoise", "demucs"]));
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      fetchNoiseLevel().then(setNoiseDb).catch(() => setNoiseDb(-90));
    }, 800);
    return () => clearInterval(timer);
  }, []);

  const noisePercent = useMemo(() => Math.min(100, Math.max(0, (noiseDb + 90) * 1.2)), [noiseDb]);

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
  }, []);

  const startLive = useCallback(async () => {
    setLiveError("");
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
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
      });

      const source = ctx.createMediaStreamSource(mediaStream);
      sourceNodeRef.current = source;

      // ScriptProcessorNode is deprecated but has the widest browser support for
      // raw PCM access without AudioWorklet setup overhead.
      const proc = ctx.createScriptProcessor(BLOCK_SAMPLES, 1, 1);
      processorRef.current = proc;

      const pendingFrames: Float32Array[] = [];

      ws.onmessage = (ev: MessageEvent<ArrayBuffer>) => {
        const buf = ev.data;
        if (buf.byteLength <= 4) return;
        const pcm16 = new Int16Array(buf, 4);
        const f32 = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) f32[i] = pcm16[i] / 32767;
        pendingFrames.push(f32);
      };

      proc.onaudioprocess = (ev: AudioProcessingEvent) => {
        const input = ev.inputBuffer.getChannelData(0);
        // Send raw PCM int16 to the server
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) pcm16[i] = Math.max(-32768, Math.min(32767, input[i] * 32767));
        if (ws.readyState === WebSocket.OPEN) ws.send(pcm16.buffer);
        // Write cleaned audio to output (or silence if nothing yet)
        const out = ev.outputBuffer.getChannelData(0);
        if (pendingFrames.length > 0) {
          const frame = pendingFrames.shift()!;
          const len = Math.min(frame.length, out.length);
          out.set(frame.subarray(0, len));
        } else {
          out.fill(0);
        }
      };

      source.connect(proc);
      proc.connect(ctx.destination);
    } catch (err) {
      setLiveError(err instanceof Error ? err.message : "Microphone access denied");
      stopLive();
    }
  }, [model, enhanceVoice, stopLive]);

  const toggleLive = useCallback(() => {
    if (liveEnabled) {
      stopLive();
      setLiveEnabled(false);
    } else {
      setLiveEnabled(true);
      void startLive();
    }
  }, [liveEnabled, startLive, stopLive]);

  // Stop live session when component unmounts or model changes mid-session
  useEffect(() => {
    return () => {
      stopLive();
    };
  }, [stopLive]);

  async function onUpload(file: File) {
    setUploading(true);
    try {
      const blob = await denoiseUpload(file, model, enhanceVoice);
      if (downloadUrl) {
        URL.revokeObjectURL(downloadUrl);
      }
      setDownloadUrl(URL.createObjectURL(blob));
    } finally {
      setUploading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="hero">
        <h1>NoiseShield Studio</h1>
        <p>Real-time and offline background noise cancellation for voice calls.</p>
      </section>

      <section className="card">
        <h2>Model Selection</h2>
        <div className="row">
          <label htmlFor="model">Model</label>
          <select id="model" value={model} onChange={(e) => setModel(e.target.value)}>
            {models.map((m) => (
              <option key={m} value={m}>
                {m.toUpperCase()}
              </option>
            ))}
          </select>
        </div>
        <div className="row">
          <label htmlFor="enhance">Voice Enhancement</label>
          <input
            id="enhance"
            type="checkbox"
            checked={enhanceVoice}
            onChange={(e) => setEnhanceVoice(e.target.checked)}
          />
        </div>
      </section>

      <section className="card">
        <h2>Live Mic Suppression</h2>
        <div className="row">
          <button onClick={toggleLive}>{liveEnabled ? "Disable" : "Enable"} Live Filter</button>
          <span className={liveEnabled ? "status-on" : "status-off"}>{liveEnabled ? "● Running" : "○ Stopped"}</span>
        </div>
        {liveError ? <p className="error">{liveError}</p> : null}
        <div className="meter-wrap">
          <div className="meter" style={{ width: `${noisePercent}%` }} />
        </div>
        <small>Detected noise level: {noiseDb.toFixed(1)} dB</small>
      </section>

      <section className="card">
        <h2>Offline Audio Cleanup</h2>
        <input
          type="file"
          accept="audio/*"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void onUpload(f);
          }}
        />
        {uploading ? <p>Processing audio…</p> : null}
        {downloadUrl ? (
          <div className="result-row">
            <audio controls src={downloadUrl} />
            <a className="download" href={downloadUrl} download="cleaned.wav">
              Download Cleaned Audio
            </a>
          </div>
        ) : null}
      </section>
    </main>
  );
}
