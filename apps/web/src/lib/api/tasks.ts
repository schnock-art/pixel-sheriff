import { apiGet, apiPost, requestJson } from "./client";
import type { Task, TaskCreatePayload } from "./types";

export function listTasks(projectId: string): Promise<Task[]> {
  return apiGet<Task[]>(`/projects/${projectId}/tasks`);
}

export function createTask(projectId: string, payload: TaskCreatePayload): Promise<Task> {
  return apiPost<Task, TaskCreatePayload>(`/projects/${projectId}/tasks`, payload);
}

export function getTask(projectId: string, taskId: string): Promise<Task> {
  return apiGet<Task>(`/projects/${projectId}/tasks/${taskId}`);
}

export function deleteTask(projectId: string, taskId: string): Promise<{ ok: boolean; task_id: string }> {
  return requestJson<{ ok: boolean; task_id: string }>(`/projects/${projectId}/tasks/${taskId}`, { method: "DELETE" });
}
