import { apiGet, apiPost, requestNoContent } from "./client";
import type { Project, ProjectCreatePayload } from "./types";

export function listProjects(): Promise<Project[]> {
  return apiGet<Project[]>("/projects");
}

export function createProject(payload: ProjectCreatePayload): Promise<Project> {
  return apiPost<Project, ProjectCreatePayload>("/projects", payload);
}

export function deleteProject(projectId: string): Promise<void> {
  return requestNoContent(`/projects/${projectId}`, { method: "DELETE" });
}
