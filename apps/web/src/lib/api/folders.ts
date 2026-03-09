import { apiGet, requestNoContent } from "./client";
import type { Folder } from "./types";

export function listFolders(projectId: string): Promise<Folder[]> {
  return apiGet<Folder[]>(`/projects/${projectId}/folders`);
}

export function deleteFolder(projectId: string, folderId: string): Promise<void> {
  return requestNoContent(`/projects/${projectId}/folders/${folderId}`, { method: "DELETE" });
}
