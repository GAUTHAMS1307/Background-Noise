import { useEffect, useMemo, useState } from "react";

import { denoiseUpload, fetchModels, fetchNoiseLevel } from "./lib/api";

export default function App() {
  const [models, setModels] = useState<string[]>(["rnnoise", "demucs"]);
  const [model, setModel] = useState<string>("rnnoise");
  const [enhanceVoice, setEnhanceVoice] = useState<boolean>(false);
  const [liveEnabled, setLiveEnabled] = useState<boolean>(false);
  const [noiseDb, setNoiseDb] = useState<number>(-90);
  const [uploading, setUploading] = useState<boolean>(false);
  const [downloadUrl, setDownloadUrl] = useState<string>("");

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
          <button onClick={() => setLiveEnabled((x) => !x)}>{liveEnabled ? "Disable" : "Enable"} Live Filter</button>
          <span>{liveEnabled ? "Running" : "Stopped"}</span>
        </div>
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
        {uploading ? <p>Processing audio...</p> : null}
        {downloadUrl ? (
          <a className="download" href={downloadUrl} download="cleaned.wav">
            Download Cleaned Audio
          </a>
        ) : null}
      </section>
    </main>
  );
}
