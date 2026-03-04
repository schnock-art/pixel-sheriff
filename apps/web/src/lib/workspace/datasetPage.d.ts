export type AnnotationStatus = "unlabeled" | "labeled" | "skipped" | "needs_review" | "approved";

export interface FolderTreeNode {
  name: string;
  path: string;
  children: FolderTreeNode[];
}

export const ALL_STATUSES: AnnotationStatus[];

export function normalizeStatusList(values: unknown): AnnotationStatus[];

export function toggleStatusSelection(params: {
  selected: AnnotationStatus[];
  otherSelected: AnnotationStatus[];
  status: AnnotationStatus;
  checked: boolean;
}): {
  selected: AnnotationStatus[];
  otherSelected: AnnotationStatus[];
};

export function buildFolderTree(folderPaths: string[]): FolderTreeNode[];

export function buildDescendantsByPath(folderPaths: string[]): Record<string, string[]>;

export function folderCheckState(path: string, selectedPaths: string[], descendantsByPath: Record<string, string[]>): "checked" | "unchecked" | "indeterminate";

export function toggleFolderPathSelection(params: {
  selectedPaths: string[];
  opposingSelectedPaths: string[];
  path: string;
  checked: boolean;
  descendantsByPath: Record<string, string[]>;
}): {
  selectedPaths: string[];
  opposingSelectedPaths: string[];
};

export function contentUrlForAsset(assetId: string): string;
