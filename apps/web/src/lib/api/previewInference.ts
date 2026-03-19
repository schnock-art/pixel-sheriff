import { requestJson } from "./client";
import type { PrelabelConfig, PreviewInferenceResponse, PreviewInferenceTaskKind } from "./types";

interface PreviewInferenceRequest {
  file: Blob;
  filename?: string;
  taskKind: PreviewInferenceTaskKind;
  prelabelConfig?: PrelabelConfig | null;
  deploymentId?: string | null;
  topK?: number | null;
  signal?: AbortSignal;
}

export function previewInference(
  projectId: string,
  taskId: string,
  { file, filename = "preview.jpg", taskKind, prelabelConfig = null, deploymentId = null, topK = null, signal }: PreviewInferenceRequest,
): Promise<PreviewInferenceResponse> {
  const formData = new FormData();
  formData.append("file", file, filename);
  formData.append("task_kind", taskKind);
  if (prelabelConfig) formData.append("prelabel_config", JSON.stringify(prelabelConfig));
  if (deploymentId) formData.append("deployment_id", deploymentId);
  if (typeof topK === "number" && Number.isFinite(topK)) formData.append("top_k", String(topK));
  return requestJson<PreviewInferenceResponse>(`/projects/${projectId}/tasks/${taskId}/preview-inference`, {
    method: "POST",
    body: formData,
    signal,
  });
}
