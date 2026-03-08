import { apiGet, apiPost, requestJson } from "./client";
import type {
  ProjectModelCreatePayload,
  ProjectModelCreateResponse,
  ProjectModelRecord,
  ProjectModelSummary,
  ProjectModelUpdatePayload,
} from "./types";

export function listProjectModels(projectId: string): Promise<ProjectModelSummary[]> {
  return apiGet<ProjectModelSummary[]>(`/projects/${projectId}/models`);
}

export function createProjectModel(
  projectId: string,
  payload: ProjectModelCreatePayload = {},
): Promise<ProjectModelCreateResponse> {
  return apiPost<ProjectModelCreateResponse, ProjectModelCreatePayload>(`/projects/${projectId}/models`, payload);
}

export function getProjectModel(projectId: string, modelId: string): Promise<ProjectModelRecord> {
  return apiGet<ProjectModelRecord>(`/projects/${projectId}/models/${modelId}`);
}

export function updateProjectModel(
  projectId: string,
  modelId: string,
  payload: ProjectModelUpdatePayload,
): Promise<ProjectModelRecord> {
  return requestJson<ProjectModelRecord>(`/projects/${projectId}/models/${modelId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
