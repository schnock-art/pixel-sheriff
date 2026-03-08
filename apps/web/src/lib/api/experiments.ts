import { apiGet, apiPost, getApiBase, requestJson } from "./client";
import {
  buildExperimentAnalyticsPath,
  buildExperimentEventsUrl,
  buildExperimentListPath,
  buildExperimentLogsPath,
  buildExperimentSamplesPath,
} from "./paths";
import type {
  ExperimentActionResponse,
  ExperimentEvaluationPayload,
  ExperimentEvent,
  ExperimentEventEnvelope,
  ExperimentLogsChunk,
  ExperimentOnnxPayload,
  ExperimentRuntimePayload,
  ExperimentSamplesResponse,
  ProjectExperimentAnalyticsResponse,
  ProjectExperimentCreatePayload,
  ProjectExperimentListResponse,
  ProjectExperimentRecord,
  ProjectExperimentUpdatePayload,
  StreamExperimentHandlers,
} from "./types";

export function listExperiments(projectId: string, options: { modelId?: string } = {}): Promise<ProjectExperimentListResponse> {
  return apiGet<ProjectExperimentListResponse>(buildExperimentListPath(projectId, options));
}

export function getExperimentAnalytics(
  projectId: string,
  options: { maxPoints?: number } = {},
): Promise<ProjectExperimentAnalyticsResponse> {
  return apiGet<ProjectExperimentAnalyticsResponse>(buildExperimentAnalyticsPath(projectId, options));
}

export function createExperiment(
  projectId: string,
  payload: ProjectExperimentCreatePayload,
): Promise<ProjectExperimentRecord> {
  return apiPost<ProjectExperimentRecord, ProjectExperimentCreatePayload>(`/projects/${projectId}/experiments`, payload);
}

export function getExperiment(projectId: string, experimentId: string): Promise<ProjectExperimentRecord> {
  return apiGet<ProjectExperimentRecord>(`/projects/${projectId}/experiments/${experimentId}`);
}

export function getExperimentEvaluation(projectId: string, experimentId: string): Promise<ExperimentEvaluationPayload> {
  return apiGet<ExperimentEvaluationPayload>(`/projects/${projectId}/experiments/${experimentId}/evaluation`);
}

export function getExperimentRuntime(projectId: string, experimentId: string): Promise<ExperimentRuntimePayload> {
  return apiGet<ExperimentRuntimePayload>(`/projects/${projectId}/experiments/${experimentId}/runtime`);
}

export function getExperimentOnnx(projectId: string, experimentId: string): Promise<ExperimentOnnxPayload> {
  return apiGet<ExperimentOnnxPayload>(`/projects/${projectId}/experiments/${experimentId}/onnx`);
}

export function getExperimentLogs(
  projectId: string,
  experimentId: string,
  options: { attempt?: number; fromByte?: number; maxBytes?: number } = {},
): Promise<ExperimentLogsChunk> {
  return apiGet<ExperimentLogsChunk>(buildExperimentLogsPath(projectId, experimentId, options));
}

export function listExperimentSamples(
  projectId: string,
  experimentId: string,
  options: {
    mode: "misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong";
    trueClassIndex?: number;
    predClassIndex?: number;
    limit?: number;
  },
): Promise<ExperimentSamplesResponse> {
  return apiGet<ExperimentSamplesResponse>(buildExperimentSamplesPath(projectId, experimentId, options));
}

export function updateExperiment(
  projectId: string,
  experimentId: string,
  payload: ProjectExperimentUpdatePayload,
): Promise<ProjectExperimentRecord> {
  return requestJson<ProjectExperimentRecord>(`/projects/${projectId}/experiments/${experimentId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function startExperiment(projectId: string, experimentId: string): Promise<ExperimentActionResponse> {
  return apiPost<ExperimentActionResponse, Record<string, never>>(`/projects/${projectId}/experiments/${experimentId}/start`, {});
}

export function cancelExperiment(projectId: string, experimentId: string): Promise<ExperimentActionResponse> {
  return apiPost<ExperimentActionResponse, Record<string, never>>(`/projects/${projectId}/experiments/${experimentId}/cancel`, {});
}

export function streamExperimentEvents(
  projectId: string,
  experimentId: string,
  options: { fromLine?: number; attempt?: number } = {},
  handlers: StreamExperimentHandlers,
): () => void {
  const url = buildExperimentEventsUrl(getApiBase(), projectId, experimentId, options);
  const source = new EventSource(url);

  source.onmessage = (event) => {
    try {
      const parsed = JSON.parse(event.data) as ExperimentEvent | ExperimentEventEnvelope;
      if (parsed && typeof parsed === "object" && "event" in parsed && "line" in parsed) {
        const envelope = parsed as ExperimentEventEnvelope;
        handlers.onEnvelope?.(envelope);
        handlers.onEvent?.(envelope.event);
      } else {
        handlers.onEvent?.(parsed as ExperimentEvent);
      }
    } catch {
      return;
    }
  };
  source.onerror = (event) => {
    handlers.onError?.(event);
  };

  return () => source.close();
}
