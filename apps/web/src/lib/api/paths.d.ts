import type { AnnotationStatus } from "./types";

export function buildDatasetVersionsPath(projectId: string, taskId?: string): string;

export function buildDatasetVersionAssetsPath(
  projectId: string,
  datasetVersionId: string,
  options?: {
    page?: number;
    page_size?: number;
    split?: "train" | "val" | "test";
    status?: AnnotationStatus;
    class_id?: string;
    search?: string;
  },
): string;

export function buildExperimentListPath(projectId: string, options?: { modelId?: string }): string;

export function buildExperimentAnalyticsPath(projectId: string, options?: { maxPoints?: number }): string;

export function buildExperimentLogsPath(
  projectId: string,
  experimentId: string,
  options?: { fromByte?: number; maxBytes?: number },
): string;

export function buildExperimentSamplesPath(
  projectId: string,
  experimentId: string,
  options: {
    mode: "misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong";
    trueClassIndex?: number;
    predClassIndex?: number;
    limit?: number;
  },
): string;

export function buildExperimentEventsUrl(
  apiBase: string,
  projectId: string,
  experimentId: string,
  options?: { fromLine?: number; attempt?: number },
): string;
