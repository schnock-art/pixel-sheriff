import { apiGet, apiPost, requestJson } from "./client";
import type {
  CreateDeploymentPayload,
  DeploymentItem,
  DeploymentListResponse,
  PatchDeploymentPayload,
  PredictBatchPayload,
  PredictBatchResponse,
  PredictPayload,
  PredictResponse,
} from "./types";

export function listDeployments(projectId: string): Promise<DeploymentListResponse> {
  return apiGet<DeploymentListResponse>(`/projects/${projectId}/deployments`);
}

export function createDeployment(projectId: string, payload: CreateDeploymentPayload): Promise<{ deployment: DeploymentItem }> {
  return apiPost<{ deployment: DeploymentItem }, CreateDeploymentPayload>(`/projects/${projectId}/deployments`, payload);
}

export function patchDeployment(
  projectId: string,
  deploymentId: string,
  payload: PatchDeploymentPayload,
): Promise<{ deployment: DeploymentItem }> {
  return requestJson<{ deployment: DeploymentItem }>(`/projects/${projectId}/deployments/${deploymentId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function predict(projectId: string, payload: PredictPayload): Promise<PredictResponse> {
  return apiPost<PredictResponse, PredictPayload>(`/projects/${projectId}/predict`, payload);
}

export function predictBatch(projectId: string, payload: PredictBatchPayload): Promise<PredictBatchResponse> {
  return apiPost<PredictBatchResponse, PredictBatchPayload>(`/projects/${projectId}/predict/batch`, payload);
}

export function warmupDeployment(projectId: string, deploymentId: string): Promise<{ ok: boolean; device_selected: "cuda" | "cpu" }> {
  return apiPost<{ ok: boolean; device_selected: "cuda" | "cpu" }, Record<string, never>>(
    `/projects/${projectId}/deployments/${deploymentId}/warmup`,
    {},
  );
}
