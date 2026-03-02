import type { PendingAnnotation } from "../hooks/useAnnotationWorkflow";

export function isAnnotationSubmitNotFoundError(error: unknown, projectId: string | null): boolean;

export function prunePendingAnnotationsForKnownAssets(
  pendingAnnotations: Record<string, PendingAnnotation>,
  knownAssetIds: string[],
): { nextPendingAnnotations: Record<string, PendingAnnotation>; removedAssetIds: string[] };

