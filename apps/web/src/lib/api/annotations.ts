import { apiGet, apiPost } from "./client";
import type { Annotation, AnnotationUpsert } from "./types";

export function listAnnotations(projectId: string, taskId: string): Promise<Annotation[]> {
  const params = new URLSearchParams({ task_id: taskId });
  return apiGet<Annotation[]>(`/projects/${projectId}/annotations?${params.toString()}`);
}

export function upsertAnnotation(projectId: string, payload: AnnotationUpsert): Promise<Annotation> {
  return apiPost<Annotation, AnnotationUpsert>(`/projects/${projectId}/annotations`, payload);
}
