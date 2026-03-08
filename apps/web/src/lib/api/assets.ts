import { ApiError, apiGet, apiPost, apiPostForm, requestNoContent } from "./client";
import type { Asset, AssetCreatePayload } from "./types";

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

export function listAssets(projectId: string): Promise<Asset[]> {
  return apiGet<Asset[]>(`/projects/${projectId}/assets`);
}

export function createAsset(projectId: string, payload: AssetCreatePayload): Promise<Asset> {
  return apiPost<Asset, AssetCreatePayload>(`/projects/${projectId}/assets`, payload);
}

export function deleteAsset(projectId: string, assetId: string): Promise<void> {
  return requestNoContent(`/projects/${projectId}/assets/${assetId}`, { method: "DELETE" });
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
