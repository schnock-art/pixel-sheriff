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
  id: number;
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
}

export interface ExperimentMetricPoint {
  attempt?: number | null;
  epoch: number;
  train_loss?: number;
  val_loss?: number;
  val_accuracy?: number;
  val_map?: number;
  val_iou?: number;
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

export type ExperimentEvent =
  | { type: "status"; status: ExperimentStatus; attempt?: number; job_id?: string; ts?: string; message?: string }
  | ({ type: "metric"; attempt?: number; ts?: string } & ExperimentMetricPoint)
  | ({ type: "checkpoint"; attempt?: number; ts?: string } & ExperimentCheckpoint)
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

export function patchCategory(categoryId: number, payload: CategoryUpdatePayload): Promise<Category> {
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

export function createExperiment(
  projectId: string,
  payload: ProjectExperimentCreatePayload,
): Promise<ProjectExperimentRecord> {
  return apiPost<ProjectExperimentRecord, ProjectExperimentCreatePayload>(`/projects/${projectId}/experiments`, payload);
}

export function getExperiment(projectId: string, experimentId: string): Promise<ProjectExperimentRecord> {
  return apiGet<ProjectExperimentRecord>(`/projects/${projectId}/experiments/${experimentId}`);
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
