import { apiGet, apiPost, requestJson } from "./client";
import type { Category, CategoryCreatePayload, CategoryUpdatePayload } from "./types";

export function listCategories(projectId: string, taskId: string): Promise<Category[]> {
  const params = new URLSearchParams({ task_id: taskId });
  return apiGet<Category[]>(`/projects/${projectId}/categories?${params.toString()}`);
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

export function deleteCategory(categoryId: string): Promise<{ ok: boolean; category_id: string }> {
  return requestJson<{ ok: boolean; category_id: string }>(`/categories/${categoryId}`, { method: "DELETE" });
}
