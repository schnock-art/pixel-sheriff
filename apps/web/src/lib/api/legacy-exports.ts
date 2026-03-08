import { apiGet, apiPost } from "./client";
import type { ExportCreatePayload, ExportVersion } from "./types";

export function createExport(projectId: string, payload: ExportCreatePayload = {}): Promise<ExportVersion> {
  return apiPost<ExportVersion, ExportCreatePayload>(`/projects/${projectId}/exports`, payload);
}

export function listExports(projectId: string): Promise<ExportVersion[]> {
  return apiGet<ExportVersion[]>(`/projects/${projectId}/exports`);
}
