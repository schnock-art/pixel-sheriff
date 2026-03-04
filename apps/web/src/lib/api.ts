const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

export class ApiError extends Error {
  readonly url: string;
  readonly method: string;
  readonly status?: number;
  readonly responseBody?: string;

  constructor(params: { message: string; url: string; method: string; status?: number; responseBody?: string }) {
    super(params.message);
    this.name = "ApiError";
    this.url = params.url;
    this.method = params.method;
    this.status = params.status;
    this.responseBody = params.responseBody;
  }
}

async function requestJson<T>(path: string, init: RequestInit): Promise<T> {
  const url = `${API_BASE}/api/v1${path}`;
  const method = (init.method ?? "GET").toUpperCase();

  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new ApiError({
      message: `NetworkError on ${method} ${url}`,
      url,
      method,
      responseBody: error instanceof Error ? error.message : String(error),
    });
  }

  if (!response.ok) {
    const responseBody = await response.text();
    throw new ApiError({
      message: `Request failed (${response.status}) on ${method} ${url}`,
      url,
      method,
      status: response.status,
      responseBody,
    });
  }

  return response.json() as Promise<T>;
}

async function requestNoContent(path: string, init: RequestInit): Promise<void> {
  const url = `${API_BASE}/api/v1${path}`;
  const method = (init.method ?? "GET").toUpperCase();

  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    throw new ApiError({
      message: `NetworkError on ${method} ${url}`,
      url,
      method,
      responseBody: error instanceof Error ? error.message : String(error),
    });
  }

  if (!response.ok) {
    const responseBody = await response.text();
    throw new ApiError({
      message: `Request failed (${response.status}) on ${method} ${url}`,
      url,
      method,
      status: response.status,
      responseBody,
    });
  }
}

export async function apiGet<T>(path: string): Promise<T> {
  return requestJson<T>(path, { cache: "no-store" });
}

export async function apiPost<T, TBody = unknown>(path: string, body: TBody): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export async function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  return requestJson<T>(path, {
    method: "POST",
    body: formData,
  });
}

export type AnnotationStatus = "unlabeled" | "labeled" | "skipped" | "needs_review" | "approved";
export type ProjectTaskType = "classification" | "classification_single" | "bbox" | "segmentation";

export interface Project {
  id: string;
  name: string;
  task_type: ProjectTaskType;
  schema_version: string;
}

export interface ProjectCreatePayload {
  name: string;
  task_type?: ProjectTaskType;
}

export interface Category {
  id: string;
  project_id: string;
  name: string;
  display_order: number;
  is_active: boolean;
}

export interface CategoryCreatePayload {
  name: string;
  display_order?: number;
}

export interface CategoryUpdatePayload {
  name?: string;
  display_order?: number;
  is_active?: boolean;
}

export interface Asset {
  id: string;
  project_id: string;
  type: string;
  uri: string;
  mime_type: string;
  width: number | null;
  height: number | null;
  checksum: string;
  metadata_json: Record<string, unknown>;
}

export interface AssetCreatePayload {
  type?: "image" | "video" | "frame";
  uri: string;
  mime_type: string;
  width?: number | null;
  height?: number | null;
  checksum: string;
  metadata_json?: Record<string, unknown>;
}

export interface Annotation {
  id: string;
  asset_id: string;
  project_id: string;
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  annotated_by: string | null;
}

export interface AnnotationUpsert {
  asset_id: string;
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
  annotated_by?: string;
}

export interface ExportCreatePayload {
  selection_criteria_json?: Record<string, unknown>;
}

export interface ExportVersion {
  id: string;
  project_id: string;
  selection_criteria_json: Record<string, unknown>;
  manifest_json: Record<string, unknown>;
  export_uri: string;
  hash: string;
}

export interface DatasetSelectionFilters {
  include_labeled_only?: boolean;
  include_statuses?: AnnotationStatus[];
  exclude_statuses?: AnnotationStatus[];
  include_category_ids?: string[];
  exclude_category_ids?: string[];
  include_folder_paths?: string[];
  exclude_folder_paths?: string[];
  include_negative_images?: boolean;
}

export interface DatasetSplitConfig {
  seed?: number;
  ratios?: { train: number; val: number; test: number };
  stratify?: {
    enabled?: boolean;
    by?: "label_primary" | "label_multi_hot" | "embedding_cluster";
    strict_stratify?: boolean;
  };
}

export interface DatasetVersionSummaryEnvelope {
  version: Record<string, unknown>;
  is_archived: boolean;
  is_active: boolean;
}

export interface DatasetVersionListPayload {
  active_dataset_version_id: string | null;
  items: DatasetVersionSummaryEnvelope[];
}

export interface DatasetPreviewPayload {
  asset_ids: string[];
  sample_asset_ids: string[];
  counts: {
    total: number;
    class_counts: Record<string, number>;
    split_counts: { train: number; val: number; test: number };
  };
  warnings: string[];
}

export interface DatasetVersionAssetsPayload {
  items: Array<{
    asset_id: string;
    filename: string;
    relative_path: string;
    status: AnnotationStatus;
    split?: "train" | "val" | "test" | null;
    label_summary: Record<string, unknown>;
  }>;
  page: number;
  page_size: number;
  total: number;
}

export interface DatasetVersionExportPayload {
  dataset_version_id: string;
  hash: string;
  export_uri: string;
}

export interface ProjectModelSummary {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  task: string;
  backbone_name: string;
  num_classes: number;
}

export interface ProjectModelCreatePayload {
  name?: string;
}

export interface ProjectModelCreateResponse {
  id: string;
  name: string;
  config: Record<string, unknown>;
}

export interface ProjectModelRecord {
  id: string;
  project_id: string;
  name: string;
  config_json: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ProjectModelUpdatePayload {
  config_json: Record<string, unknown>;
}

export type ExperimentStatus = "draft" | "queued" | "running" | "completed" | "failed" | "canceled";
export type ExperimentTask = "classification" | "detection" | "segmentation";

export interface ExperimentSummaryJson {
  best_metric_name: string | null;
  best_metric_value: number | null;
  best_epoch: number | null;
  last_epoch: number | null;
}

export interface ExperimentCheckpoint {
  kind: "best_loss" | "best_metric" | "latest";
  epoch: number | null;
  metric_name: string | null;
  value: number | null;
  uri?: string | null;
  updated_at: string | null;
  status?: "pending" | "ok" | "error" | null;
  error?: string | null;
}

export interface ExperimentMetricPoint {
  attempt?: number | null;
  epoch: number;
  train_loss?: number;
  train_accuracy?: number;
  val_loss?: number;
  val_accuracy?: number;
  val_macro_f1?: number;
  val_macro_precision?: number;
  val_macro_recall?: number;
  val_map?: number;
  val_iou?: number;
  epoch_seconds?: number;
  eta_seconds?: number;
  created_at?: string;
}

export interface ProjectExperimentSummary {
  id: string;
  project_id: string;
  model_id: string;
  name: string;
  created_at: string;
  updated_at: string;
  status: ExperimentStatus;
  summary_json: ExperimentSummaryJson;
  current_run_attempt?: number | null;
  last_completed_attempt?: number | null;
  active_job_id?: string | null;
  error?: string | null;
}

export interface ProjectExperimentRecord extends ProjectExperimentSummary {
  config_json: Record<string, unknown>;
  artifacts_json: Record<string, unknown>;
  checkpoints: ExperimentCheckpoint[];
  metrics: ExperimentMetricPoint[];
}

export interface ProjectExperimentListResponse {
  items: ProjectExperimentSummary[];
}

export interface ProjectExperimentCreatePayload {
  model_id: string;
  dataset_version_id?: string;
  name?: string;
  config_overrides?: Record<string, unknown>;
}

export interface ProjectExperimentUpdatePayload {
  name?: string;
  config_json?: Record<string, unknown>;
  selected_checkpoint_kind?: "best_loss" | "best_metric" | "latest";
}

export interface ExperimentActionResponse {
  ok: boolean;
  status?: ExperimentStatus | null;
  attempt?: number | null;
  job_id?: string | null;
}

export interface ExperimentAnalyticsBest {
  metric_name?: string | null;
  metric_value?: number | null;
  epoch?: number | null;
}

export interface ExperimentAnalyticsConfig {
  optimizer?: { type?: string | null; lr?: number | null };
  batch_size?: number | null;
  epochs?: number | null;
  augmentation?: string | null;
}

export interface ExperimentAnalyticsItem {
  experiment_id: string;
  name: string;
  model_id: string;
  model_name: string;
  status: ExperimentStatus;
  updated_at: string;
  config: ExperimentAnalyticsConfig;
  best: ExperimentAnalyticsBest;
  final: Record<string, number | null>;
  series: Record<string, unknown>;
  runtime?: { device_selected?: string } | null;
}

export interface ProjectExperimentAnalyticsResponse {
  items: ExperimentAnalyticsItem[];
  available_series: string[];
}

export interface ExperimentEvaluationOverall {
  accuracy?: number;
  macro_f1?: number;
  macro_precision?: number;
  macro_recall?: number;
}

export interface ExperimentEvaluationPerClassRow {
  class_index: number;
  class_id: string;
  name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ExperimentEvaluationSampleRow {
  asset_id: string;
  relative_path?: string;
  true_class_index: number;
  pred_class_index: number;
  confidence: number;
  margin?: number | null;
}

export interface ExperimentEvaluationPayload {
  attempt: number;
  schema_version?: string;
  task?: string;
  computed_at?: string;
  split?: string;
  num_samples?: number;
  classes?: {
    class_order?: string[];
    class_names?: string[];
    id_to_index?: Record<string, number>;
  };
  overall?: ExperimentEvaluationOverall;
  per_class?: ExperimentEvaluationPerClassRow[];
  confusion_matrix?: {
    matrix?: number[][];
    normalized_by?: string;
    labels?: Record<string, string>;
  };
  samples?: {
    misclassified?: ExperimentEvaluationSampleRow[];
    lowest_confidence_correct?: ExperimentEvaluationSampleRow[];
    highest_confidence_wrong?: ExperimentEvaluationSampleRow[];
  };
}

export interface ExperimentSamplesResponse {
  attempt: number;
  mode: "misclassified" | "lowest_confidence_correct" | "highest_confidence_wrong";
  items: ExperimentEvaluationSampleRow[];
  message?: string | null;
}

export interface ExperimentRuntimePayload {
  attempt: number;
  device_selected: string;
  cuda_available: boolean;
  mps_available: boolean;
  amp_enabled: boolean;
  torch_version: string;
  torchvision_version: string;
  num_workers: number;
  pin_memory: boolean;
  persistent_workers: boolean;
  prefetch_factor?: number;
  cache_resized_images?: boolean;
  max_cached_images?: number;
}

export interface ExperimentOnnxPayload {
  attempt: number;
  status: "exported" | "failed";
  model_onnx_url?: string | null;
  metadata_url: string;
  input_shape?: number[];
  class_names?: string[];
  class_order?: string[];
  preprocess?: Record<string, unknown>;
  validation?: Record<string, unknown> | null;
  error?: string | null;
}

export interface ExperimentLogsChunk {
  from_byte: number;
  to_byte: number;
  content: string;
}

export type DeploymentDevicePreference = "auto" | "cuda" | "cpu";
export type DeploymentStatus = "available" | "archived";

export interface DeploymentSource {
  experiment_id: string;
  attempt: number;
  checkpoint_kind: "best_metric" | "best_loss" | "latest";
  onnx_relpath: string;
  metadata_relpath: string;
}

export interface DeploymentItem {
  deployment_id: string;
  name: string;
  task: "classification";
  provider: "onnxruntime";
  device_preference: DeploymentDevicePreference;
  model_key: string;
  source: DeploymentSource;
  status: DeploymentStatus;
  created_at: string;
  updated_at: string;
}

export interface DeploymentListResponse {
  active_deployment_id: string | null;
  items: DeploymentItem[];
}

export interface CreateDeploymentPayload {
  name: string;
  task?: "classification";
  device_preference?: DeploymentDevicePreference;
  source: {
    experiment_id: string;
    attempt: number;
    checkpoint_kind?: "best_metric" | "best_loss" | "latest";
  };
  is_active?: boolean;
}

export interface PatchDeploymentPayload {
  is_active?: boolean;
  name?: string;
  device_preference?: DeploymentDevicePreference;
  status?: DeploymentStatus;
}

export interface PredictPayload {
  asset_id: string;
  deployment_id?: string | null;
  top_k?: number;
}

export interface PredictPrediction {
  class_index: number;
  class_id: string;
  class_name: string;
  score: number;
}

export interface PredictResponse {
  asset_id: string;
  deployment_id: string;
  task: "classification";
  device_selected: "cuda" | "cpu";
  predictions: PredictPrediction[];
  deployment_name?: string | null;
  device_preference?: DeploymentDevicePreference | null;
}

export type ExperimentEvent =
  | { type: "status"; status: ExperimentStatus; attempt?: number; job_id?: string; ts?: string; message?: string }
  | ({ type: "metric"; attempt?: number; ts?: string } & ExperimentMetricPoint)
  | ({ type: "checkpoint"; attempt?: number; ts?: string } & ExperimentCheckpoint)
  | { type: "onnx_export"; status: "exported" | "failed"; attempt?: number; model_uri?: string; metadata_uri?: string; error?: string; ts?: string }
  | { type: "done"; status: ExperimentStatus; attempt?: number; job_id?: string; ts?: string; message?: string; error_code?: string };

export interface ExperimentEventEnvelope {
  line: number;
  attempt: number | null;
  event: ExperimentEvent;
}

export interface StreamExperimentHandlers {
  onEvent?: (event: ExperimentEvent) => void;
  onEnvelope?: (payload: ExperimentEventEnvelope) => void;
  onError?: (event: Event) => void;
}

function inferMimeType(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  if (ext === "jpg" || ext === "jpeg") return "image/jpeg";
  if (ext === "png") return "image/png";
  if (ext === "gif") return "image/gif";
  if (ext === "webp") return "image/webp";
  if (ext === "bmp") return "image/bmp";
  if (ext === "tif" || ext === "tiff") return "image/tiff";
  return "application/octet-stream";
}

function readFileWithFallback(file: File): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (reader.result instanceof ArrayBuffer) {
        resolve(reader.result);
      } else {
        reject(new Error("FileReader did not return an ArrayBuffer"));
      }
    };
    reader.onerror = () => reject(reader.error ?? new Error("FileReader failed"));
    reader.readAsArrayBuffer(file);
  });
}

export function listProjects(): Promise<Project[]> {
  return apiGet<Project[]>("/projects");
}

export function listCategories(projectId: string): Promise<Category[]> {
  return apiGet<Category[]>(`/projects/${projectId}/categories`);
}

export function createCategory(projectId: string, payload: CategoryCreatePayload): Promise<Category> {
  return apiPost<Category, CategoryCreatePayload>(`/projects/${projectId}/categories`, payload);
}

export function patchCategory(categoryId: string, payload: CategoryUpdatePayload): Promise<Category> {
  return requestJson<Category>(`/categories/${categoryId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function createProject(payload: ProjectCreatePayload): Promise<Project> {
  return apiPost<Project, ProjectCreatePayload>("/projects", payload);
}

export function deleteProject(projectId: string): Promise<void> {
  return requestNoContent(`/projects/${projectId}`, { method: "DELETE" });
}

export function listAssets(projectId: string): Promise<Asset[]> {
  return apiGet<Asset[]>(`/projects/${projectId}/assets`);
}

export function createAsset(projectId: string, payload: AssetCreatePayload): Promise<Asset> {
  return apiPost<Asset, AssetCreatePayload>(`/projects/${projectId}/assets`, payload);
}

export function deleteAsset(projectId: string, assetId: string): Promise<void> {
  return requestNoContent(`/projects/${projectId}/assets/${assetId}`, { method: "DELETE" });
}

export function listAnnotations(projectId: string): Promise<Annotation[]> {
  return apiGet<Annotation[]>(`/projects/${projectId}/annotations`);
}

export function upsertAnnotation(projectId: string, payload: AnnotationUpsert): Promise<Annotation> {
  return apiPost<Annotation, AnnotationUpsert>(`/projects/${projectId}/annotations`, payload);
}

export function createExport(projectId: string, payload: ExportCreatePayload = {}): Promise<ExportVersion> {
  return apiPost<ExportVersion, ExportCreatePayload>(`/projects/${projectId}/exports`, payload);
}

export function listExports(projectId: string): Promise<ExportVersion[]> {
  return apiGet<ExportVersion[]>(`/projects/${projectId}/exports`);
}

export function listDatasetVersions(projectId: string): Promise<DatasetVersionListPayload> {
  return apiGet<DatasetVersionListPayload>(`/projects/${projectId}/datasets/versions`);
}

export function previewDatasetVersion(
  projectId: string,
  payload: {
    task: "classification" | "bbox" | "segmentation";
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
    task: "classification" | "bbox" | "segmentation";
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
  const params = new URLSearchParams();
  if (typeof options.page === "number" && Number.isFinite(options.page) && options.page >= 1) {
    params.set("page", String(Math.floor(options.page)));
  }
  if (typeof options.page_size === "number" && Number.isFinite(options.page_size) && options.page_size >= 1) {
    params.set("page_size", String(Math.floor(options.page_size)));
  }
  if (options.split) params.set("split", options.split);
  if (options.status) params.set("status", options.status);
  if (options.class_id) params.set("class_id", options.class_id);
  if (options.search) params.set("search", options.search);
  const query = params.toString();
  return apiGet<DatasetVersionAssetsPayload>(
    `/projects/${projectId}/datasets/versions/${datasetVersionId}/assets${query ? `?${query}` : ""}`,
  );
}

export function exportDatasetVersion(projectId: string, datasetVersionId: string): Promise<DatasetVersionExportPayload> {
  return apiPost<DatasetVersionExportPayload, Record<string, never>>(
    `/projects/${projectId}/datasets/versions/${datasetVersionId}/export`,
    {},
  );
}

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

export function listExperiments(projectId: string, options: { modelId?: string } = {}): Promise<ProjectExperimentListResponse> {
  const params = new URLSearchParams();
  if (options.modelId) params.set("model_id", options.modelId);
  const query = params.toString();
  const suffix = query ? `?${query}` : "";
  return apiGet<ProjectExperimentListResponse>(`/projects/${projectId}/experiments${suffix}`);
}

export function getExperimentAnalytics(
  projectId: string,
  options: { maxPoints?: number } = {},
): Promise<ProjectExperimentAnalyticsResponse> {
  const params = new URLSearchParams();
  if (typeof options.maxPoints === "number" && Number.isFinite(options.maxPoints) && options.maxPoints >= 1) {
    params.set("max_points", String(Math.floor(options.maxPoints)));
  }
  const query = params.toString();
  return apiGet<ProjectExperimentAnalyticsResponse>(
    `/projects/${projectId}/experiments/analytics${query ? `?${query}` : ""}`,
  );
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
  options: { fromByte?: number; maxBytes?: number } = {},
): Promise<ExperimentLogsChunk> {
  const params = new URLSearchParams();
  if (typeof options.fromByte === "number" && Number.isFinite(options.fromByte) && options.fromByte >= 0) {
    params.set("from_byte", String(Math.floor(options.fromByte)));
  }
  if (typeof options.maxBytes === "number" && Number.isFinite(options.maxBytes) && options.maxBytes >= 1) {
    params.set("max_bytes", String(Math.floor(options.maxBytes)));
  }
  const suffix = params.toString();
  return apiGet<ExperimentLogsChunk>(
    `/projects/${projectId}/experiments/${experimentId}/logs${suffix ? `?${suffix}` : ""}`,
  );
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
  const params = new URLSearchParams();
  params.set("mode", options.mode);
  if (typeof options.trueClassIndex === "number" && Number.isFinite(options.trueClassIndex) && options.trueClassIndex >= 0) {
    params.set("true_class_index", String(Math.floor(options.trueClassIndex)));
  }
  if (typeof options.predClassIndex === "number" && Number.isFinite(options.predClassIndex) && options.predClassIndex >= 0) {
    params.set("pred_class_index", String(Math.floor(options.predClassIndex)));
  }
  if (typeof options.limit === "number" && Number.isFinite(options.limit) && options.limit >= 1) {
    params.set("limit", String(Math.floor(options.limit)));
  }
  return apiGet<ExperimentSamplesResponse>(
    `/projects/${projectId}/experiments/${experimentId}/samples?${params.toString()}`,
  );
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
  return apiPost<ExperimentActionResponse, Record<string, never>>(
    `/projects/${projectId}/experiments/${experimentId}/start`,
    {},
  );
}

export function cancelExperiment(projectId: string, experimentId: string): Promise<ExperimentActionResponse> {
  return apiPost<ExperimentActionResponse, Record<string, never>>(
    `/projects/${projectId}/experiments/${experimentId}/cancel`,
    {},
  );
}

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

export function warmupDeployment(projectId: string, deploymentId: string): Promise<{ ok: boolean; device_selected: "cuda" | "cpu" }> {
  return apiPost<{ ok: boolean; device_selected: "cuda" | "cpu" }, Record<string, never>>(
    `/projects/${projectId}/deployments/${deploymentId}/warmup`,
    {},
  );
}

export function streamExperimentEvents(
  projectId: string,
  experimentId: string,
  options: { fromLine?: number; attempt?: number } = {},
  handlers: StreamExperimentHandlers,
): () => void {
  const urlBase = API_BASE.endsWith("/") ? API_BASE.slice(0, -1) : API_BASE;
  const params = new URLSearchParams();
  if (typeof options.fromLine === "number" && Number.isFinite(options.fromLine) && options.fromLine >= 0) {
    params.set("from_line", String(Math.floor(options.fromLine)));
  }
  if (typeof options.attempt === "number" && Number.isFinite(options.attempt) && options.attempt >= 1) {
    params.set("attempt", String(Math.floor(options.attempt)));
  }
  const query = params.toString();
  const url = `${urlBase}/api/v1/projects/${encodeURIComponent(projectId)}/experiments/${encodeURIComponent(experimentId)}/events${query ? `?${query}` : ""}`;
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

export function uploadAsset(projectId: string, file: File, relativePath?: string): Promise<Asset> {
  return (async () => {
    let bytes: ArrayBuffer;
    try {
      bytes = await file.arrayBuffer();
    } catch (error) {
      try {
        bytes = await readFileWithFallback(file);
      } catch (fallbackError) {
        const primaryDetail = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
        const fallbackDetail = fallbackError instanceof Error ? `${fallbackError.name}: ${fallbackError.message}` : String(fallbackError);
        throw new ApiError({
          message: `Local file read failed for "${file.name}"`,
          method: "READ",
          url: file.name,
          responseBody: `primary=${primaryDetail}; fallback=${fallbackDetail}; size=${file.size}; type=${file.type || "unknown"}; lastModified=${file.lastModified}`,
        });
      }
    }

    const formData = new FormData();
    const mime = file.type || inferMimeType(file.name);
    const blob = new Blob([bytes], { type: mime });
    formData.append("file", blob, file.name);
    if (relativePath) formData.append("relative_path", relativePath);
    return apiPostForm<Asset>(`/projects/${projectId}/assets/upload`, formData);
  })();
}

export function resolveAssetUri(uri: string): string {
  if (uri.startsWith("http://") || uri.startsWith("https://") || uri.startsWith("blob:") || uri.startsWith("data:")) {
    return uri;
  }
  if (uri.startsWith("/")) {
    return `${API_BASE}${uri}`;
  }
  return `${API_BASE}/${uri}`;
}
