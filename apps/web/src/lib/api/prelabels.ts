import { apiGet, apiPost } from "./client";
import type { PrelabelProposal, PrelabelSession } from "./types";


export function listPrelabelSessions(projectId: string, taskId: string, sequenceId: string): Promise<{ items: PrelabelSession[] }> {
  const params = new URLSearchParams({ sequence_id: sequenceId });
  return apiGet<{ items: PrelabelSession[] }>(`/projects/${projectId}/tasks/${taskId}/prelabels?${params.toString()}`);
}

export function getPrelabelSession(projectId: string, taskId: string, sessionId: string): Promise<{ session: PrelabelSession }> {
  return apiGet<{ session: PrelabelSession }>(`/projects/${projectId}/tasks/${taskId}/prelabels/${sessionId}`);
}

export function listPrelabelProposals(
  projectId: string,
  taskId: string,
  sessionId: string,
  params?: { asset_id?: string | null; status_filter?: string | null },
): Promise<{ items: PrelabelProposal[] }> {
  const query = new URLSearchParams();
  if (params?.asset_id) query.set("asset_id", params.asset_id);
  if (params?.status_filter) query.set("status_filter", params.status_filter);
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return apiGet<{ items: PrelabelProposal[] }>(`/projects/${projectId}/tasks/${taskId}/prelabels/${sessionId}/proposals${suffix}`);
}

export function acceptPrelabelProposals(
  projectId: string,
  taskId: string,
  sessionId: string,
  payload?: { asset_id?: string | null; proposal_ids?: string[] },
): Promise<{ session: PrelabelSession; updated: number; annotation_ids: string[] }> {
  return apiPost<{ session: PrelabelSession; updated: number; annotation_ids: string[] }, { asset_id?: string | null; proposal_ids?: string[] }>(
    `/projects/${projectId}/tasks/${taskId}/prelabels/${sessionId}/accept`,
    payload ?? {},
  );
}

export function rejectPrelabelProposals(
  projectId: string,
  taskId: string,
  sessionId: string,
  payload?: { asset_id?: string | null; proposal_ids?: string[] },
): Promise<{ session: PrelabelSession; updated: number; annotation_ids: string[] }> {
  return apiPost<{ session: PrelabelSession; updated: number; annotation_ids: string[] }, { asset_id?: string | null; proposal_ids?: string[] }>(
    `/projects/${projectId}/tasks/${taskId}/prelabels/${sessionId}/reject`,
    payload ?? {},
  );
}

export function closePrelabelInput(projectId: string, taskId: string, sessionId: string): Promise<{ session: PrelabelSession }> {
  return apiPost<{ session: PrelabelSession }, Record<string, never>>(
    `/projects/${projectId}/tasks/${taskId}/prelabels/${sessionId}/close-input`,
    {},
  );
}

