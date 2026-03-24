const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

export async function fetchModels(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/models`);
  if (!res.ok) throw new Error("Failed to load models");
  const data = (await res.json()) as { models: string[] };
  return data.models;
}

export async function fetchNoiseLevel(): Promise<number> {
  const res = await fetch(`${API_BASE}/metrics/noise-level`);
  if (!res.ok) return -90;
  const data = (await res.json()) as { db: number };
  return data.db;
}

export async function denoiseUpload(file: File, model: string, enhanceVoice: boolean): Promise<Blob> {
  const fd = new FormData();
  fd.append("audio", file);
  fd.append("model", model);
  fd.append("enhance_voice", String(enhanceVoice));
  const res = await fetch(`${API_BASE}/offline/denoise`, { method: "POST", body: fd });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Offline denoise failed");
  }
  return await res.blob();
}
