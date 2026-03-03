import type { Annotation } from "../api";
import type { TreeEntry } from "./tree";

export interface PendingAnnotationLike {
  status: string;
  objects: unknown[];
}

export interface AssetReviewState {
  status: "labeled" | "unlabeled";
  isDirty: boolean;
}

export function buildVisibleTreeEntries(treeEntries: TreeEntry[], collapsedFolders: Record<string, boolean>): TreeEntry[];

export function buildAssetReviewStateById(params: {
  orderedAssetRows: Array<{ id: string }>;
  pendingAnnotations: Record<string, PendingAnnotationLike>;
  annotationByAssetId: Map<string, Annotation>;
}): Map<string, AssetReviewState>;

export function buildFolderReviewStatusByPath(params: {
  folderAssetIds: Record<string, string[]>;
  assetReviewStateById: Map<string, AssetReviewState>;
}): Record<string, "all_labeled" | "has_unlabeled" | "empty">;

export function buildFolderDirtyByPath(params: {
  folderAssetIds: Record<string, string[]>;
  assetReviewStateById: Map<string, AssetReviewState>;
}): Record<string, boolean>;

export function deriveMessageTone(message: string | null): "info" | "error" | "success";
