import type { ProjectExperimentSummary } from "../api";

export type ModelListStatus = "draft" | "ready" | "training" | "completed" | "failed";

export interface ModelDatasetVersionView {
  datasetVersionId: string | null;
  datasetVersionName: string;
  hasSourceDataset: boolean;
}

export function deriveModelDatasetVersion(
  config: Record<string, unknown> | null | undefined,
  datasetVersionNameById: Record<string, string>,
): ModelDatasetVersionView;

export function deriveModelStatus(
  experiments: ProjectExperimentSummary[],
  modelId: string,
  hasSourceDataset: boolean,
): ModelListStatus;
