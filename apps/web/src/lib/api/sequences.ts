import { apiGet, apiPost, apiPostForm } from "./client";
import type {
  Asset,
  AssetSequence,
  SequenceStatus,
  VideoImportPayload,
  VideoImportResponse,
  WebcamSessionCreatePayload,
  WebcamSessionCreateResponse,
} from "./types";

export function listSequences(projectId: string, params?: { task_id?: string | null; folder_id?: string | null }): Promise<AssetSequence[]> {
  const query = new URLSearchParams();
  if (params?.task_id) query.set("task_id", params.task_id);
  if (params?.folder_id) query.set("folder_id", params.folder_id);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiGet<AssetSequence[]>(`/projects/${projectId}/sequences${suffix}`);
}

export function getSequence(projectId: string, sequenceId: string, taskId?: string | null): Promise<AssetSequence> {
  const query = new URLSearchParams();
  if (taskId) query.set("task_id", taskId);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiGet<AssetSequence>(`/projects/${projectId}/sequences/${sequenceId}${suffix}`);
}

export function getSequenceStatus(projectId: string, sequenceId: string): Promise<SequenceStatus> {
  return apiGet<SequenceStatus>(`/projects/${projectId}/sequences/${sequenceId}/status`);
}

export function createWebcamSession(projectId: string, payload: WebcamSessionCreatePayload): Promise<WebcamSessionCreateResponse> {
  return apiPost<WebcamSessionCreateResponse, WebcamSessionCreatePayload>(`/projects/${projectId}/webcam-sessions`, payload);
}

export function uploadSequenceFrame(
  projectId: string,
  sequenceId: string,
  file: Blob,
  filename: string,
  frameIndex: number,
  timestampSeconds?: number | null,
): Promise<Asset> {
  const formData = new FormData();
  formData.append("file", file, filename);
  formData.append("frame_index", String(frameIndex));
  if (typeof timestampSeconds === "number") formData.append("timestamp_seconds", String(timestampSeconds));
  return apiPostForm<Asset>(`/projects/${projectId}/sequences/${sequenceId}/frames`, formData);
}

export function importVideo(projectId: string, file: File, payload: VideoImportPayload): Promise<VideoImportResponse> {
  const formData = new FormData();
  formData.append("file", file, file.name);
  if (payload.task_id) formData.append("task_id", payload.task_id);
  if (payload.folder_id) formData.append("folder_id", payload.folder_id);
  if (payload.name) formData.append("name", payload.name);
  formData.append("fps", String(payload.fps));
  formData.append("max_frames", String(payload.max_frames));
  formData.append("resize_mode", payload.resize_mode);
  if (typeof payload.resize_width === "number") formData.append("resize_width", String(payload.resize_width));
  if (typeof payload.resize_height === "number") formData.append("resize_height", String(payload.resize_height));
   if (payload.prelabel_config) formData.append("prelabel_config", JSON.stringify(payload.prelabel_config));
  return apiPostForm<VideoImportResponse>(`/projects/${projectId}/video-imports`, formData);
}
