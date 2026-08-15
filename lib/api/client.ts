import type { AssetKind, AssetRecord, JobDiagnosticsRecord, JobRecord, StyleRecord, TransferSettings } from "./types";

export const API_BASE_URL = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api/v1").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code = "REQUEST_FAILED",
    readonly status = 500,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

let responseCsrfToken: string | undefined;

function csrfToken(): string | undefined {
  if (responseCsrfToken) return responseCsrfToken;
  if (typeof document === "undefined") return undefined;
  const value = document.cookie
    .split(";")
    .map((part) => part.trim())
    .find((part) => part.startsWith("pst_csrf="))
    ?.slice("pst_csrf=".length);
  return value ? decodeURIComponent(value) : undefined;
}

async function request<T>(path: string, init?: RequestInit, retriedCsrf = false): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const token = !["GET", "HEAD", "OPTIONS"].includes(method) ? csrfToken() : undefined;
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    credentials: "include",
    headers: {
      Accept: "application/json",
      ...(!(init?.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(token ? { "X-CSRF-Token": token } : {}),
      ...init?.headers,
    },
  });
  responseCsrfToken = response.headers.get("X-CSRF-Token") ?? responseCsrfToken;

  if (!response.ok) {
    const body = await response.json().catch(() => null) as { error?: { code?: string; message?: string; details?: unknown } } | null;
    if (response.status === 403 && body?.error?.code === "CSRF_FAILED" && responseCsrfToken && !retriedCsrf) {
      return request<T>(path, init, true);
    }
    throw new ApiError(
      body?.error?.message ?? "The request could not be completed.",
      body?.error?.code ?? "REQUEST_FAILED",
      response.status,
      body?.error?.details,
    );
  }

  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function uploadAsset(file: File, kind: AssetKind): Promise<AssetRecord> {
  const body = new FormData();
  body.append("file", file);
  body.append("kind", kind);
  return request<AssetRecord>("/assets/upload", { method: "POST", body });
}

export function createJob(payload: {
  input_asset_id: string;
  reference_asset_id?: string;
  style_id?: string;
  settings: TransferSettings;
}): Promise<JobRecord> {
  return request<JobRecord>("/jobs", { method: "POST", body: JSON.stringify(payload) });
}

export function getJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/jobs/${jobId}`);
}

export function getJobDiagnostics(jobId: string): Promise<JobDiagnosticsRecord> {
  return request<JobDiagnosticsRecord>(`/jobs/${jobId}/diagnostics`);
}

export function cancelJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/jobs/${jobId}/cancel`, { method: "POST", body: "{}" });
}

export function deleteJob(jobId: string): Promise<void> {
  return request<void>(`/jobs/${jobId}`, { method: "DELETE" });
}

export function getJobDownloadUrl(jobId: string): Promise<{ url: string; expires_at: string }> {
  return request<{ url: string; expires_at: string }>(`/jobs/${jobId}/download-url`, { method: "POST", body: "{}" });
}

export function rerunJob(jobId: string): Promise<JobRecord> {
  return request<JobRecord>(`/jobs/${jobId}/rerun`, { method: "POST", body: "{}" });
}

export function listStyles(): Promise<StyleRecord[]> {
  return request<StyleRecord[]>("/styles");
}

export function createStyle(payload: { name: string; description: string; rights_confirmed: boolean }): Promise<StyleRecord> {
  return request<StyleRecord>("/styles", { method: "POST", body: JSON.stringify(payload) });
}

export function deleteStyle(styleId: string): Promise<void> {
  return request<void>(`/styles/${styleId}`, { method: "DELETE" });
}

export async function addStyleExample(styleId: string, file: File): Promise<unknown> {
  const asset = await uploadAsset(file, "STYLE_EXAMPLE");
  return request(`/styles/${styleId}/examples`, {
    method: "POST",
    body: JSON.stringify({ asset_id: asset.id }),
  });
}

export function subscribeToJob(jobId: string, onUpdate: (job: JobRecord) => void, onError: () => void): () => void {
  const source = new EventSource(`${API_BASE_URL}/jobs/${jobId}/events`, { withCredentials: true });
  source.onmessage = (event) => {
    try { onUpdate(JSON.parse(event.data) as JobRecord); } catch { onError(); }
  };
  source.onerror = onError;
  return () => source.close();
}
