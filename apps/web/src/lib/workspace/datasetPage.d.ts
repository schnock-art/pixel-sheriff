import type { DatasetVersionSummaryEnvelope } from "../api";

export type AnnotationStatus = "unlabeled" | "labeled" | "skipped" | "needs_review" | "approved";

export interface FolderTreeNode {
  name: string;
  path: string;
  children: FolderTreeNode[];
}

export interface DatasetSummaryPayload {
  total: number;
  class_counts: Record<string, number>;
  split_counts: { train: number; val: number; test: number };
  warnings: string[];
}

export const ALL_STATUSES: AnnotationStatus[];

export function asRecord(value: unknown): Record<string, unknown>;

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

export function datasetVersionIdOf(item: DatasetVersionSummaryEnvelope): string;

export function summaryFromVersion(versionEnvelope: DatasetVersionSummaryEnvelope | null): DatasetSummaryPayload | null;

export function selectedVersionName(item: DatasetVersionSummaryEnvelope | null): string;

export function classNamesFromVersion(versionEnvelope: DatasetVersionSummaryEnvelope | null): Record<string, string>;

export function fallbackClassName(classId: string): string;

export function classDisplayName(
  classId: string,
  sources: {
    summaryClassNames: Record<string, string>;
    versionClassNames: Record<string, string>;
    categoryNameById: Record<string, string>;
  },
): string;

export function previewAssetPrimaryCategoryId(item: {
  label_summary?: { primary_category_id?: string | null } | null;
} | null | undefined): string | null;

export function filterPreviewAssets(
  items: Array<{
    asset_id: string;
    filename: string;
    relative_path: string;
    status: AnnotationStatus;
    split?: "train" | "val" | "test" | null;
    label_summary?: { primary_category_id?: string | null } | null;
  }>,
  filters: {
    splitFilter?: "all" | "train" | "val" | "test";
    statusFilter?: "all" | AnnotationStatus;
    classFilter?: string;
    searchText?: string;
  },
): Array<{
  asset_id: string;
  filename: string;
  relative_path: string;
  status: AnnotationStatus;
  split?: "train" | "val" | "test" | null;
  label_summary?: { primary_category_id?: string | null } | null;
}>;
