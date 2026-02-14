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

export interface Project {
  id: string;
  name: string;
  task_type: string;
  schema_version: string;
}

export interface ProjectCreatePayload {
  name: string;
  task_type?: "classification_single";
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
