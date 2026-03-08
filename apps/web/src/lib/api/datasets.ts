import { apiGet, apiPost, requestJson } from "./client";
import { buildDatasetVersionAssetsPath, buildDatasetVersionsPath } from "./paths";
import type {
  AnnotationStatus,
  DatasetPreviewPayload,
  DatasetSelectionFilters,
  DatasetSplitConfig,
  DatasetVersionAssetsPayload,
  DatasetVersionExportPayload,
  DatasetVersionListPayload,
  DatasetVersionSummaryEnvelope,
} from "./types";

export function listDatasetVersions(projectId: string, taskId?: string): Promise<DatasetVersionListPayload> {
  return apiGet<DatasetVersionListPayload>(buildDatasetVersionsPath(projectId, taskId));
}

export function previewDatasetVersion(
  projectId: string,
  payload: {
    task_id: string;
    selection: { mode: "filter_snapshot" | "explicit_asset_ids"; filters?: DatasetSelectionFilters; explicit_asset_ids?: string[] };
    split?: DatasetSplitConfig;
    strict_preview_cap?: boolean;
    preview_cap?: number;
  },
): Promise<DatasetPreviewPayload> {
  return apiPost<DatasetPreviewPayload, typeof payload>(`/projects/${projectId}/datasets/versions/preview`, payload);
}

export function createDatasetVersion(
  projectId: string,
  payload: {
    name: string;
    description?: string;
    task_id: string;
    created_by?: string;
    selection: { mode: "filter_snapshot" | "explicit_asset_ids"; filters?: DatasetSelectionFilters; explicit_asset_ids?: string[] };
    split?: DatasetSplitConfig;
    set_active?: boolean;
  },
): Promise<DatasetVersionSummaryEnvelope> {
  return apiPost<DatasetVersionSummaryEnvelope, typeof payload>(`/projects/${projectId}/datasets/versions`, payload);
}

export function setActiveDatasetVersion(
  projectId: string,
  activeDatasetVersionId: string | null,
): Promise<{ ok: boolean; active_dataset_version_id: string | null }> {
  return requestJson<{ ok: boolean; active_dataset_version_id: string | null }>(`/projects/${projectId}/datasets/active`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ active_dataset_version_id: activeDatasetVersionId }),
  });
}

export function getDatasetVersion(projectId: string, datasetVersionId: string): Promise<DatasetVersionSummaryEnvelope> {
  return apiGet<DatasetVersionSummaryEnvelope>(`/projects/${projectId}/datasets/versions/${datasetVersionId}`);
}

export function listDatasetVersionAssets(
  projectId: string,
  datasetVersionId: string,
  options: {
    page?: number;
    page_size?: number;
    split?: "train" | "val" | "test";
    status?: AnnotationStatus;
    class_id?: string;
    search?: string;
  } = {},
): Promise<DatasetVersionAssetsPayload> {
  return apiGet<DatasetVersionAssetsPayload>(buildDatasetVersionAssetsPath(projectId, datasetVersionId, options));
}

export function exportDatasetVersion(projectId: string, datasetVersionId: string): Promise<DatasetVersionExportPayload> {
  return apiPost<DatasetVersionExportPayload, Record<string, never>>(
    `/projects/${projectId}/datasets/versions/${datasetVersionId}/export`,
    {},
  );
}
