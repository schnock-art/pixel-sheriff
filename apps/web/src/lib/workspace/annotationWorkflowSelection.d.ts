import type { AnnotationStatus } from "../api";

export interface PendingAnnotationLike {
  labelIds: number[];
  status: AnnotationStatus;
}

export interface AnnotationLike {
  status: AnnotationStatus;
  payload_json: Record<string, unknown>;
}

export function resolveSelectionForAsset(params: {
  currentAssetId: string | null;
  pendingAnnotations: Record<string, PendingAnnotationLike>;
  annotationByAssetId: Map<string, AnnotationLike>;
}): {
  labelIds: number[];
  status: AnnotationStatus;
  source: "pending" | "committed" | "empty";
};
