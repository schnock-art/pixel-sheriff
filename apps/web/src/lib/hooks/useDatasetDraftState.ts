import { useState } from "react";

import type { AnnotationStatus } from "../api";
import { normalizeStatusList } from "../workspace/datasetPage";

export interface DraftPreviewSummary {
  total: number;
  class_counts: Record<string, number>;
  split_counts: { train: number; val: number; test: number };
  warnings: string[];
  sample_asset_ids: string[];
}

export interface DraftStateFromVersion {
  name?: string;
  include_labeled_only?: boolean;
  include_statuses?: AnnotationStatus[];
  exclude_statuses?: AnnotationStatus[];
  include_folder_paths?: string[];
  exclude_folder_paths?: string[];
  seed?: number;
  ratios?: { train?: number; val?: number; test?: number };
  stratify?: { enabled?: boolean };
}

const DEFAULT_INCLUDE_STATUSES: AnnotationStatus[] = ["labeled", "approved", "needs_review"];

export function useDatasetDraftState() {
  const [draftName, setDraftName] = useState("Dataset v1");
  const [includeLabeledOnly, setIncludeLabeledOnly] = useState(true);
  const [includeStatuses, setIncludeStatuses] = useState<AnnotationStatus[]>(DEFAULT_INCLUDE_STATUSES);
  const [excludeStatuses, setExcludeStatuses] = useState<AnnotationStatus[]>([]);
  const [includeFolderPaths, setIncludeFolderPaths] = useState<string[]>([]);
  const [excludeFolderPaths, setExcludeFolderPaths] = useState<string[]>([]);
  const [seed, setSeed] = useState(1337);
  const [trainRatio, setTrainRatio] = useState(0.8);
  const [valRatio, setValRatio] = useState(0.1);
  const [testRatio, setTestRatio] = useState(0.1);
  const [stratify, setStratify] = useState(true);
  const [previewSummary, setPreviewSummary] = useState<DraftPreviewSummary | null>(null);

  function resetDraft() {
    setDraftName("Dataset v1");
    setIncludeLabeledOnly(true);
    setIncludeStatuses(DEFAULT_INCLUDE_STATUSES);
    setExcludeStatuses([]);
    setIncludeFolderPaths([]);
    setExcludeFolderPaths([]);
    setSeed(1337);
    setTrainRatio(0.8);
    setValRatio(0.1);
    setTestRatio(0.1);
    setStratify(true);
    setPreviewSummary(null);
  }

  function initDraftFromVersion(value: DraftStateFromVersion) {
    setDraftName(typeof value.name === "string" && value.name.trim() ? `${value.name} copy` : "Dataset version copy");
    setIncludeLabeledOnly(Boolean(value.include_labeled_only));
    setIncludeStatuses(normalizeStatusList(value.include_statuses ?? []) as AnnotationStatus[]);
    setExcludeStatuses(normalizeStatusList(value.exclude_statuses ?? []) as AnnotationStatus[]);
    setIncludeFolderPaths(Array.isArray(value.include_folder_paths) ? value.include_folder_paths.filter((item) => typeof item === "string") : []);
    setExcludeFolderPaths(Array.isArray(value.exclude_folder_paths) ? value.exclude_folder_paths.filter((item) => typeof item === "string") : []);
    setSeed(typeof value.seed === "number" && Number.isFinite(value.seed) ? value.seed : 1337);
    setTrainRatio(typeof value.ratios?.train === "number" ? value.ratios.train : 0.8);
    setValRatio(typeof value.ratios?.val === "number" ? value.ratios.val : 0.1);
    setTestRatio(typeof value.ratios?.test === "number" ? value.ratios.test : 0.1);
    setStratify(Boolean(value.stratify?.enabled));
    setPreviewSummary(null);
  }

  return {
    draftName,
    setDraftName,
    includeLabeledOnly,
    setIncludeLabeledOnly,
    includeStatuses,
    setIncludeStatuses,
    excludeStatuses,
    setExcludeStatuses,
    includeFolderPaths,
    setIncludeFolderPaths,
    excludeFolderPaths,
    setExcludeFolderPaths,
    seed,
    setSeed,
    trainRatio,
    setTrainRatio,
    valRatio,
    setValRatio,
    testRatio,
    setTestRatio,
    stratify,
    setStratify,
    previewSummary,
    setPreviewSummary,
    resetDraft,
    initDraftFromVersion,
  };
}
