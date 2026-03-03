import type { AnnotationStatus } from "../api";
import type { GeometryObject, ImageBasis } from "./annotationState";

export interface PendingAnnotationLike {
  labelIds: string[];
  status: AnnotationStatus;
  objects?: GeometryObject[];
  imageBasis?: ImageBasis | null;
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
  labelIds: string[];
  status: AnnotationStatus;
  source: "pending" | "committed" | "empty";
};
