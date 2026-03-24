const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000/api/v1";

const DEFAULT_REQUEST_TIMEOUT_MS = 10_000; // 10 seconds
const UPLOAD_TIMEOUT_MS = 300_000; // 5 minutes — allow time for large file processing

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), DEFAULT_REQUEST_TIMEOUT_MS);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}: ${res.statusText}`);
    }
    return (await res.json()) as T;
  } finally {
    clearTimeout(timeout);
  }
}

export async function fetchModels(): Promise<string[]> {
  const data = await fetchJSON<{ models: string[] }>(`${API_BASE}/models`);
  return data.models;
}

export async function fetchNoiseLevel(): Promise<number> {
  try {
    const data = await fetchJSON<{ db: number }>(`${API_BASE}/metrics/noise-level`);
    return data.db;
  } catch {
    return -90;
  }
}

export async function denoiseUpload(
  file: File,
  model: string,
  enhanceVoice: boolean,
): Promise<Blob> {
  const fd = new FormData();
  fd.append("audio", file);
  fd.append("model", model);
  fd.append("enhance_voice", String(enhanceVoice));

  const controller = new AbortController();
  // Allow up to 5 minutes for large files
  const timeout = setTimeout(() => controller.abort(), UPLOAD_TIMEOUT_MS);
  try {
    const res = await fetch(`${API_BASE}/offline/denoise`, {
      method: "POST",
      body: fd,
      signal: controller.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `Processing failed (HTTP ${res.status})`);
    }
    return await res.blob();
  } finally {
    clearTimeout(timeout);
  }
}
