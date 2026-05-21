import type {
  ProposeMaskResponse,
  RestoreRequest,
  RestoreResponse,
  VersionResponse,
} from "../types/api";

// Backend URL resolution (in priority order):
//   1. window.__RESTORECOINS_BACKEND__  — runtime override for cached builds
//   2. VITE_BACKEND_URL at build time   — set in CI for prod
//   3. /api/...                         — dev proxy via vite.config.ts
function backendBase(): string {
  const w = (globalThis as { __RESTORECOINS_BACKEND__?: string }).__RESTORECOINS_BACKEND__;
  if (w) return w.replace(/\/$/, "");
  const env = (import.meta as { env?: Record<string, string> }).env?.VITE_BACKEND_URL;
  if (env) return env.replace(/\/$/, "");
  return "/api";
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${backendBase()}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`POST ${path} failed: ${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${backendBase()}${path}`);
  if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
  return (await res.json()) as T;
}

export const api = {
  version: () => getJson<VersionResponse>("/version"),
  proposeMask: (image: string) => postJson<ProposeMaskResponse>(
    "/propose_mask", { image },
  ),
  restore: (req: RestoreRequest) => postJson<RestoreResponse>("/restore", req),
};

// --- base64 helpers (browser-side) ----------------------------------------- //

export async function fileToBase64Png(file: File): Promise<string> {
  // Re-encode to PNG so the backend gets a known format regardless of input.
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
  const blob = await new Promise<Blob>((res) =>
    canvas.toBlob((b) => res(b!), "image/png"),
  );
  return blobToBase64(blob);
}

export async function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => {
      const s = r.result as string;
      // strip "data:image/png;base64," prefix
      resolve(s.slice(s.indexOf(",") + 1));
    };
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

export function base64ToDataUrl(b64: string, mime = "image/png"): string {
  return `data:${mime};base64,${b64}`;
}

export async function downloadBase64(b64: string, filename: string): Promise<void> {
  const url = base64ToDataUrl(b64);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
}

export function downloadJson(data: unknown, filename: string): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
