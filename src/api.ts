// ─── API utility ───────────────────────────────────────────────────────────────

export async function api<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `API error ${response.status}: ${response.statusText}`);
  }
  const ct = response.headers.get("content-type") ?? "";
  if (ct.includes("application/json")) return response.json() as Promise<T>;
  return response.blob() as unknown as Promise<T>;
}

export function fmt(v: number, decimals = 4): string {
  return Number.isFinite(v) ? v.toFixed(decimals) : "—";
}

export function fmtGHz(hz: number): string {
  return `${(hz / 1e9).toFixed(4)} GHz`;
}

export function fmtMHz(hz: number): string {
  return `${(hz / 1e6).toFixed(3)} MHz`;
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
