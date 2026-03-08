import { useEffect, useMemo, useState } from "react";

import {
  ApiError,
  createDatasetVersion,
  exportDatasetVersion,
  listAssets,
  listCategories,
  listDatasetVersionAssets,
  listDatasetVersions,
  listTasks,
  previewDatasetVersion,
  setActiveDatasetVersion,
  type AnnotationStatus,
  type DatasetVersionAssetsPayload,
  type DatasetVersionSummaryEnvelope,
  type Task,
} from "../api";
import { collectFolderPaths } from "../workspace/tree";
import { useDatasetBrowserState } from "./useDatasetBrowserState";
import { useDatasetDraftState } from "./useDatasetDraftState";
import {
  asRecord,
  classDisplayName,
  classNamesFromVersion,
  datasetVersionIdOf,
  filterPreviewAssets,
  summaryFromVersion,
} from "../workspace/datasetPage";

function parseApiErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof ApiError && error.responseBody) {
    try {
      const parsed = JSON.parse(error.responseBody) as {
        error?: { message?: string; details?: { issues?: Array<{ path?: string; message?: string }> } };
      };
      const issue = parsed.error?.details?.issues?.[0];
      if (issue?.path && issue?.message) {
        return `${parsed.error?.message ?? fallback} (${issue.path}: ${issue.message})`;
      }
      if (parsed.error?.message) return parsed.error.message;
      return error.responseBody;
    } catch {
      return error.responseBody;
    }
  }
  if (error instanceof Error) return error.message;
  return fallback;
}

export function useDatasetPageState({
  projectId,
  requestedTaskId,
}: {
  projectId: string;
  requestedTaskId: string | null;
}) {
  const browser = useDatasetBrowserState();
  const draft = useDatasetDraftState();

  const [mode, setMode] = useState<"browse" | "draft">("browse");
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [versions, setVersions] = useState<DatasetVersionSummaryEnvelope[]>([]);
  const [activeDatasetVersionId, setActiveDatasetVersionIdState] = useState<string | null>(null);
  const [assetsPayload, setAssetsPayload] = useState<DatasetVersionAssetsPayload | null>(null);
  const [isLoadingVersions, setIsLoadingVersions] = useState(true);
  const [isLoadingAssets, setIsLoadingAssets] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [categoryNameById, setCategoryNameById] = useState<Record<string, string>>({});
  const [folderPaths, setFolderPaths] = useState<string[]>([]);
  const [classFilter, setClassFilter] = useState<string>("all");

  const selectedVersion = useMemo(
    () => versions.find((item) => datasetVersionIdOf(item) === browser.selectedDatasetVersionId) ?? null,
    [browser.selectedDatasetVersionId, versions],
  );
  const selectedTask = useMemo(() => tasks.find((task) => task.id === selectedTaskId) ?? null, [tasks, selectedTaskId]);
  const savedSummary = useMemo(() => summaryFromVersion(selectedVersion), [selectedVersion]);
  const versionClassNames = useMemo(() => classNamesFromVersion(selectedVersion), [selectedVersion]);
  const summarySource: "draft" | "saved" = mode === "draft" ? "draft" : "saved";
  const summaryData = summarySource === "saved" ? savedSummary : draft.previewSummary;
  const summaryClassNames = summarySource === "saved" ? versionClassNames : (draft.previewSummary?.class_names ?? {});

  const classFilterOptions = useMemo(() => {
    const sourceIds =
      mode === "draft" && draft.previewSummary
        ? Object.keys(draft.previewSummary.class_counts).filter((classId) => classId !== "__missing__")
        : savedSummary && savedSummary.class_counts
          ? Object.keys(savedSummary.class_counts).filter((classId) => classId !== "__missing__")
          : Object.keys({ ...categoryNameById, ...(draft.previewSummary?.class_names ?? {}) });
    const deduped = Array.from(new Set(sourceIds));
    return deduped
      .map((classId) => ({
        id: classId,
        name: classDisplayName(classId, {
          summaryClassNames: draft.previewSummary?.class_names ?? {},
          versionClassNames,
          categoryNameById,
        }),
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [categoryNameById, draft.previewSummary, mode, savedSummary, versionClassNames]);

  const filteredPreviewAssets = useMemo(
    () =>
      filterPreviewAssets(draft.previewSummary?.sample_assets ?? [], {
        splitFilter: browser.splitFilter,
        statusFilter: browser.statusFilter,
        classFilter,
        searchText: browser.searchText,
      }),
    [browser.searchText, browser.splitFilter, browser.statusFilter, classFilter, draft.previewSummary?.sample_assets],
  );

  useEffect(() => {
    let isMounted = true;
    async function loadTasks() {
      try {
        const rows = await listTasks(projectId);
        if (!isMounted) return;
        setTasks(rows);
      } catch {
        if (!isMounted) return;
        setTasks([]);
      }
    }
    void loadTasks();
    return () => {
      isMounted = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (tasks.length === 0) {
      setSelectedTaskId(null);
      return;
    }
    const validIds = new Set(tasks.map((task) => task.id));
    const storageKey = `pixel-sheriff:project-active-task:v1:${projectId}`;
    const storedTaskId = typeof window !== "undefined" ? window.localStorage.getItem(storageKey) : null;
    const defaultTaskId = tasks.find((task) => task.is_default)?.id ?? tasks[0]?.id ?? null;
    const nextTaskId =
      [requestedTaskId, storedTaskId, defaultTaskId].find(
        (value): value is string => Boolean(value && validIds.has(value)),
      ) ?? null;
    setSelectedTaskId((previous) => (previous === nextTaskId ? previous : nextTaskId));
    if (nextTaskId && typeof window !== "undefined") {
      window.localStorage.setItem(storageKey, nextTaskId);
    }
  }, [projectId, requestedTaskId, tasks]);

  useEffect(() => {
    if (!selectedTaskId || typeof window === "undefined") return;
    const storageKey = `pixel-sheriff:project-active-task:v1:${projectId}`;
    window.localStorage.setItem(storageKey, selectedTaskId);
    const url = new URL(window.location.href);
    if (url.searchParams.get("taskId") === selectedTaskId) return;
    url.searchParams.set("taskId", selectedTaskId);
    window.history.replaceState({}, "", url.toString());
  }, [projectId, selectedTaskId]);

  async function loadVersions() {
    setIsLoadingVersions(true);
    try {
      const payload = await listDatasetVersions(projectId, selectedTaskId ?? undefined);
      const nextVersions = payload.items ?? [];
      setVersions(nextVersions);
      setActiveDatasetVersionIdState(payload.active_dataset_version_id ?? null);
      const firstId =
        payload.active_dataset_version_id ?? datasetVersionIdOf(nextVersions[0] ?? { version: {}, is_active: false, is_archived: false });
      browser.setSelectedDatasetVersionId(firstId || null);
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset versions"));
      setVersions([]);
      browser.setSelectedDatasetVersionId(null);
    } finally {
      setIsLoadingVersions(false);
    }
  }

  useEffect(() => {
    void loadVersions();
  }, [projectId, selectedTaskId]);

  useEffect(() => {
    let isMounted = true;
    async function loadCategoryNames() {
      if (!selectedTaskId) {
        if (!isMounted) return;
        setCategoryNameById({});
        return;
      }
      try {
        const categories = await listCategories(projectId, selectedTaskId);
        if (!isMounted) return;
        const mapping: Record<string, string> = {};
        for (const category of categories) {
          if (typeof category.id === "string" && category.id.trim() && typeof category.name === "string" && category.name.trim()) {
            mapping[category.id] = category.name;
          }
        }
        setCategoryNameById(mapping);
      } catch {
        if (!isMounted) return;
        setCategoryNameById({});
      }
    }
    void loadCategoryNames();
    return () => {
      isMounted = false;
    };
  }, [projectId, selectedTaskId]);

  useEffect(() => {
    let isMounted = true;
    async function loadFolderPaths() {
      try {
        const assets = await listAssets(projectId);
        if (!isMounted) return;
        setFolderPaths(collectFolderPaths(assets));
      } catch {
        if (!isMounted) return;
        setFolderPaths([]);
      }
    }
    void loadFolderPaths();
    return () => {
      isMounted = false;
    };
  }, [projectId]);

  useEffect(() => {
    if (!browser.selectedDatasetVersionId) {
      setAssetsPayload(null);
      return;
    }
    let isMounted = true;
    async function loadAssets() {
      setIsLoadingAssets(true);
      try {
        const payload = await listDatasetVersionAssets(projectId, browser.selectedDatasetVersionId as string, {
          page: browser.page,
          page_size: 50,
          split: browser.splitFilter === "all" ? undefined : browser.splitFilter,
          status: browser.statusFilter === "all" ? undefined : browser.statusFilter,
          class_id: classFilter === "all" ? undefined : classFilter,
          search: browser.searchText.trim() || undefined,
        });
        if (!isMounted) return;
        setAssetsPayload(payload);
      } catch (error) {
        if (!isMounted) return;
        setErrorMessage(parseApiErrorMessage(error, "Failed to load dataset assets"));
      } finally {
        if (isMounted) setIsLoadingAssets(false);
      }
    }
    void loadAssets();
    return () => {
      isMounted = false;
    };
  }, [
    browser.page,
    browser.searchText,
    browser.selectedDatasetVersionId,
    browser.splitFilter,
    browser.statusFilter,
    classFilter,
    projectId,
  ]);

  useEffect(() => {
    setClassFilter("all");
  }, [browser.selectedDatasetVersionId]);

  function selectionPayload() {
    return {
      mode: "filter_snapshot" as const,
      filters: {
        include_labeled_only: draft.includeLabeledOnly,
        include_negative_images: selectedTask?.kind === "bbox" || selectedTask?.kind === "segmentation" ? draft.includeNegativeImages : undefined,
        include_statuses: draft.includeStatuses,
        exclude_statuses: draft.excludeStatuses,
        include_folder_paths: draft.includeFolderPaths,
        exclude_folder_paths: draft.excludeFolderPaths,
      },
    };
  }

  function splitPayload() {
    return {
      seed: draft.seed,
      ratios: { train: draft.trainRatio, val: draft.valRatio, test: draft.testRatio },
      stratify: { enabled: draft.stratify, by: "label_primary" as const, strict_stratify: false },
    };
  }

  async function handlePreview() {
    if (!selectedTaskId) {
      setErrorMessage("Select a task before previewing a dataset version.");
      return;
    }
    setIsPreviewing(true);
    try {
      const payload = await previewDatasetVersion(projectId, {
        task_id: selectedTaskId,
        selection: selectionPayload(),
        split: splitPayload(),
      });
      draft.setPreviewSummary({
        total: payload.counts.total,
        class_counts: payload.counts.class_counts,
        class_names: payload.class_names ?? {},
        split_counts: payload.counts.split_counts,
        warnings: payload.warnings ?? [],
        sample_asset_ids: payload.sample_asset_ids ?? [],
        sample_assets: payload.sample_assets ?? [],
      });
      setMode("draft");
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to preview dataset"));
    } finally {
      setIsPreviewing(false);
    }
  }

  async function handleCreate() {
    if (!selectedTaskId) {
      setErrorMessage("Select a task before creating a dataset version.");
      return;
    }
    setIsCreating(true);
    try {
      const created = await createDatasetVersion(projectId, {
        name: draft.draftName.trim() || "Dataset version",
        task_id: selectedTaskId,
        selection: selectionPayload(),
        split: splitPayload(),
        set_active: true,
      });
      const createdId = datasetVersionIdOf(created);
      await loadVersions();
      if (createdId) browser.setSelectedDatasetVersionId(createdId);
      setMode("browse");
      draft.resetDraft();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to create dataset version"));
    } finally {
      setIsCreating(false);
    }
  }

  async function handleSetActive(datasetVersionId: string) {
    try {
      const response = await setActiveDatasetVersion(projectId, datasetVersionId);
      setActiveDatasetVersionIdState(response.active_dataset_version_id);
      await loadVersions();
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to set active dataset version"));
    }
  }

  async function handleExport() {
    if (!browser.selectedDatasetVersionId) return null;
    setIsExporting(true);
    try {
      const exported = await exportDatasetVersion(projectId, browser.selectedDatasetVersionId);
      setErrorMessage(null);
      return exported;
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to export dataset version"));
      return null;
    } finally {
      setIsExporting(false);
    }
  }

  function handleDuplicateAndEdit() {
    if (!selectedVersion) return;
    const version = asRecord(selectedVersion.version);
    const selection = asRecord(version.selection);
    const filters = asRecord(selection.filters);
    const splits = asRecord(version.splits);
    const ratios = asRecord(splits.ratios);
    const stratify = asRecord(splits.stratify);
    draft.initDraftFromVersion({
      name: typeof version.name === "string" ? version.name : "Dataset version",
      include_labeled_only: Boolean(filters.include_labeled_only),
      include_negative_images: filters.include_negative_images !== false,
      include_statuses: Array.isArray(filters.include_statuses) ? (filters.include_statuses as AnnotationStatus[]) : [],
      exclude_statuses: Array.isArray(filters.exclude_statuses) ? (filters.exclude_statuses as AnnotationStatus[]) : [],
      include_folder_paths: Array.isArray(filters.include_folder_paths) ? (filters.include_folder_paths as string[]) : [],
      exclude_folder_paths: Array.isArray(filters.exclude_folder_paths) ? (filters.exclude_folder_paths as string[]) : [],
      seed: typeof splits.seed === "number" ? splits.seed : 1337,
      ratios: {
        train: typeof ratios.train === "number" ? ratios.train : 0.8,
        val: typeof ratios.val === "number" ? ratios.val : 0.1,
        test: typeof ratios.test === "number" ? ratios.test : 0.1,
      },
      stratify: {
        enabled: Boolean(stratify.enabled),
      },
    });
    setMode("draft");
  }

  function handleDiscardDraft() {
    draft.resetDraft();
    setMode("browse");
  }

  return {
    browser,
    draft,
    mode,
    setMode,
    tasks,
    selectedTaskId,
    setSelectedTaskId,
    selectedTask,
    versions,
    selectedVersion,
    activeDatasetVersionId,
    assetsPayload,
    isLoadingVersions,
    isLoadingAssets,
    isPreviewing,
    isCreating,
    isExporting,
    errorMessage,
    categoryNameById,
    folderPaths,
    classFilter,
    setClassFilter,
    summarySource,
    summaryData,
    summaryClassNames,
    versionClassNames,
    classFilterOptions,
    filteredPreviewAssets,
    handlePreview,
    handleCreate,
    handleSetActive,
    handleExport,
    handleDuplicateAndEdit,
    handleDiscardDraft,
  };
}
