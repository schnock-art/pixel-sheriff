"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

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
  resolveAssetUri,
  setActiveDatasetVersion,
  type AnnotationStatus,
  type DatasetVersionAssetsPayload,
  type DatasetVersionSummaryEnvelope,
  type Task,
} from "../../../../lib/api";
import { useDatasetBrowserState } from "../../../../lib/hooks/useDatasetBrowserState";
import { useDatasetDraftState } from "../../../../lib/hooks/useDatasetDraftState";
import { collectFolderPaths } from "../../../../lib/workspace/tree";
import {
  ALL_STATUSES,
  buildDescendantsByPath,
  buildFolderTree,
  contentUrlForAsset,
  folderCheckState,
  toggleFolderPathSelection,
  toggleStatusSelection,
} from "../../../../lib/workspace/datasetPage";

interface DatasetPageProps {
  params: {
    projectId: string;
  };
}

type DatasetMode = "browse" | "draft";
type SummaryPayload = {
  total: number;
  class_counts: Record<string, number>;
  split_counts: { train: number; val: number; test: number };
  warnings: string[];
};

type FolderTreeNode = {
  name: string;
  path: string;
  children: FolderTreeNode[];
};

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

function versionIdOf(item: DatasetVersionSummaryEnvelope): string {
  const version = item.version;
  const value = version?.dataset_version_id;
  return typeof value === "string" ? value : "";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function summaryFromVersion(versionEnvelope: DatasetVersionSummaryEnvelope | null): SummaryPayload | null {
  if (!versionEnvelope) return null;
  const stats = asRecord(asRecord(versionEnvelope.version).stats);
  const splitCounts = asRecord(stats.split_counts);
  const warnings = Array.isArray(stats.warnings) ? stats.warnings.filter((item): item is string => typeof item === "string") : [];
  const classCountsRaw = asRecord(stats.class_counts);
  const classCounts: Record<string, number> = {};
  for (const [key, value] of Object.entries(classCountsRaw)) {
    if (typeof value === "number" && Number.isFinite(value)) classCounts[key] = value;
  }

  const total =
    typeof stats.asset_count === "number"
      ? stats.asset_count
      : Object.values(splitCounts).reduce<number>(
          (sum, value) => sum + (typeof value === "number" ? value : 0),
          0,
        );
  return {
    total,
    class_counts: classCounts,
    split_counts: {
      train: typeof splitCounts.train === "number" ? splitCounts.train : 0,
      val: typeof splitCounts.val === "number" ? splitCounts.val : 0,
      test: typeof splitCounts.test === "number" ? splitCounts.test : 0,
    },
    warnings,
  };
}

function selectedVersionName(item: DatasetVersionSummaryEnvelope | null): string {
  if (!item) return "(none)";
  const name = asRecord(item.version).name;
  const id = asRecord(item.version).dataset_version_id;
  if (typeof name === "string" && name.trim()) return name;
  if (typeof id === "string" && id.trim()) return id;
  return "(unnamed version)";
}

function FolderTreeRow({
  node,
  depth,
  selectedPaths,
  descendantsByPath,
  collapsed,
  onToggleCollapsed,
  onToggleChecked,
}: {
  node: FolderTreeNode;
  depth: number;
  selectedPaths: string[];
  descendantsByPath: Record<string, string[]>;
  collapsed: Record<string, boolean>;
  onToggleCollapsed: (path: string) => void;
  onToggleChecked: (path: string, checked: boolean) => void;
}) {
  const checkState = folderCheckState(node.path, selectedPaths, descendantsByPath);
  const isCollapsed = Boolean(collapsed[node.path]);
  const checkboxRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!checkboxRef.current) return;
    checkboxRef.current.indeterminate = checkState === "indeterminate";
  }, [checkState]);

  return (
    <div>
      <div style={{ display: "flex", alignItems: "center", gap: 6, paddingLeft: depth * 12 }}>
        <button
          type="button"
          className="ghost-button"
          style={{ width: 22, height: 22, padding: 0 }}
          onClick={() => onToggleCollapsed(node.path)}
          aria-label={isCollapsed ? "Expand folder" : "Collapse folder"}
        >
          {isCollapsed ? ">" : "v"}
        </button>
        <input
          ref={checkboxRef}
          type="checkbox"
          checked={checkState === "checked"}
          onChange={(event) => onToggleChecked(node.path, event.target.checked)}
        />
        <span>{node.name}</span>
      </div>
      {!isCollapsed && node.children.length > 0
        ? node.children.map((child) => (
            <FolderTreeRow
              key={child.path}
              node={child}
              depth={depth + 1}
              selectedPaths={selectedPaths}
              descendantsByPath={descendantsByPath}
              collapsed={collapsed}
              onToggleCollapsed={onToggleCollapsed}
              onToggleChecked={onToggleChecked}
            />
          ))
        : null}
    </div>
  );
}

function FolderMultiSelectDropdown({
  label,
  folderPaths,
  selectedPaths,
  opposingSelectedPaths,
  onSelectedChange,
  onOpposingChange,
}: {
  label: string;
  folderPaths: string[];
  selectedPaths: string[];
  opposingSelectedPaths: string[];
  onSelectedChange: (value: string[]) => void;
  onOpposingChange: (value: string[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const tree = useMemo(() => buildFolderTree(folderPaths), [folderPaths]);
  const descendantsByPath = useMemo(() => buildDescendantsByPath(folderPaths), [folderPaths]);

  function toggleChecked(path: string, checked: boolean) {
    const next = toggleFolderPathSelection({
      selectedPaths,
      opposingSelectedPaths,
      path,
      checked,
      descendantsByPath,
    });
    onSelectedChange(next.selectedPaths);
    onOpposingChange(next.opposingSelectedPaths);
  }

  return (
    <div style={{ position: "relative" }}>
      <button type="button" className="ghost-button" onClick={() => setOpen((value) => !value)}>
        {label}: {selectedPaths.length} selected
      </button>
      {open ? (
        <div
          style={{
            position: "absolute",
            zIndex: 20,
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            maxHeight: 260,
            overflow: "auto",
            border: "1px solid var(--line, #d8dce6)",
            borderRadius: 8,
            background: "var(--frame, #f8f9fc)",
            padding: 8,
            boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
          }}
        >
          {tree.length === 0 ? <p style={{ margin: 0 }}>No folders found.</p> : null}
          {tree.map((node) => (
            <FolderTreeRow
              key={node.path}
              node={node}
              depth={0}
              selectedPaths={selectedPaths}
              descendantsByPath={descendantsByPath}
              collapsed={collapsed}
              onToggleCollapsed={(path) => setCollapsed((previous) => ({ ...previous, [path]: !previous[path] }))}
              onToggleChecked={toggleChecked}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function StatusMultiSelectDropdown({
  label,
  selected,
  otherSelected,
  onSelectedChange,
  onOtherSelectedChange,
}: {
  label: string;
  selected: AnnotationStatus[];
  otherSelected: AnnotationStatus[];
  onSelectedChange: (value: AnnotationStatus[]) => void;
  onOtherSelectedChange: (value: AnnotationStatus[]) => void;
}) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ position: "relative" }}>
      <button type="button" className="ghost-button" onClick={() => setOpen((value) => !value)}>
        {label}: {selected.length === 0 ? "none" : selected.join(", ")}
      </button>
      {open ? (
        <div
          style={{
            position: "absolute",
            zIndex: 20,
            top: "calc(100% + 6px)",
            left: 0,
            right: 0,
            border: "1px solid var(--line, #d8dce6)",
            borderRadius: 8,
            background: "var(--frame, #f8f9fc)",
            padding: 8,
            boxShadow: "0 6px 16px rgba(0,0,0,0.12)",
          }}
        >
          <div style={{ display: "grid", gap: 6 }}>
            {ALL_STATUSES.map((status) => (
              <label key={status} style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <input
                  type="checkbox"
                  checked={selected.includes(status as AnnotationStatus)}
                  onChange={(event) => {
                    const next = toggleStatusSelection({
                      selected,
                      otherSelected,
                      status: status as AnnotationStatus,
                      checked: event.target.checked,
                    });
                    onSelectedChange(next.selected as AnnotationStatus[]);
                    onOtherSelectedChange(next.otherSelected as AnnotationStatus[]);
                  }}
                />
                <span>{status}</span>
              </label>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

export default function DatasetPage({ params }: DatasetPageProps) {
  const projectId = useMemo(() => decodeURIComponent(params.projectId), [params.projectId]);
  const searchParams = useSearchParams();
  const browser = useDatasetBrowserState();
  const draft = useDatasetDraftState();

  const [mode, setMode] = useState<DatasetMode>("browse");
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
  const requestedTaskId = searchParams.get("taskId");

  const selectedVersion = useMemo(
    () => versions.find((item) => versionIdOf(item) === browser.selectedDatasetVersionId) ?? null,
    [browser.selectedDatasetVersionId, versions],
  );
  const selectedTask = useMemo(() => tasks.find((task) => task.id === selectedTaskId) ?? null, [tasks, selectedTaskId]);

  const savedSummary = useMemo(() => summaryFromVersion(selectedVersion), [selectedVersion]);
  const summarySource = mode === "draft" ? "draft" : "saved";
  const summaryData = summarySource === "saved" ? savedSummary : draft.previewSummary;
  const classFilterOptions = useMemo(() => {
    const sourceIds =
      savedSummary && savedSummary.class_counts
        ? Object.keys(savedSummary.class_counts).filter((classId) => classId !== "__missing__")
        : Object.keys(categoryNameById);
    const deduped = Array.from(new Set(sourceIds));
    return deduped
      .map((classId) => ({
        id: classId,
        name: categoryNameById[classId] ?? classId,
      }))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [categoryNameById, savedSummary]);

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
        payload.active_dataset_version_id ?? versionIdOf(nextVersions[0] ?? { version: {}, is_active: false, is_archived: false });
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

  function classDisplayName(classId: string): string {
    if (classId === "__missing__") return "Unlabeled / missing primary";
    return categoryNameById[classId] ?? classId;
  }

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
        split_counts: payload.counts.split_counts,
        warnings: payload.warnings ?? [],
        sample_asset_ids: payload.sample_asset_ids ?? [],
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
      const createdId = versionIdOf(created);
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
    if (!browser.selectedDatasetVersionId) return;
    setIsExporting(true);
    try {
      const exported = await exportDatasetVersion(projectId, browser.selectedDatasetVersionId);
      const url = resolveAssetUri(exported.export_uri);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${browser.selectedDatasetVersionId}-${exported.hash.slice(0, 8)}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      setErrorMessage(null);
    } catch (error) {
      setErrorMessage(parseApiErrorMessage(error, "Failed to export dataset version"));
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

  return (
    <main className="workspace-shell project-page-shell">
      <section className="workspace-frame project-content-frame placeholder-page">
        <header className="project-section-header">
          <h2>Dataset</h2>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <label htmlFor="dataset-task-select">Task</label>
            <select
              id="dataset-task-select"
              value={selectedTaskId ?? ""}
              onChange={(event) => setSelectedTaskId(event.target.value || null)}
              disabled={tasks.length === 0}
            >
              {tasks.length === 0 ? <option value="">No tasks</option> : null}
              {tasks.map((task) => (
                <option key={task.id} value={task.id}>
                  {task.name} [{task.kind}]
                </option>
              ))}
            </select>
            {selectedTask ? <span style={{ fontSize: 12, opacity: 0.8 }}>{selectedTask.kind}</span> : null}
          </div>
        </header>

        {errorMessage ? <p className="project-field-error">{errorMessage}</p> : null}

        <div style={{ display: "grid", gap: 12, gridTemplateColumns: "280px 1fr 320px" }}>
          <section className="placeholder-card">
            <h3>Versions</h3>
            <div style={{ display: "grid", gap: 8 }}>
              {isLoadingVersions ? <p>Loading versions...</p> : null}
              {!isLoadingVersions && versions.length === 0 ? <p>No dataset versions yet.</p> : null}
              {versions.map((item) => {
                const id = versionIdOf(item);
                const isActive = activeDatasetVersionId === id;
                const isSelected = browser.selectedDatasetVersionId === id;
                const name = typeof item.version?.name === "string" ? item.version.name : id;
                const versionTaskId = typeof item.version?.task_id === "string" ? item.version.task_id : "";
                const versionTaskName = tasks.find((task) => task.id === versionTaskId)?.name;
                const versionTaskKind = typeof item.version?.task === "string" ? item.version.task : "";
                return (
                  <button
                    key={id}
                    type="button"
                    className={isSelected ? "ghost-button active-toggle" : "ghost-button"}
                    onClick={() => {
                      browser.setSelectedDatasetVersionId(id);
                      setMode("browse");
                    }}
                    style={{ justifyContent: "space-between", display: "flex", alignItems: "center" }}
                  >
                    <span>
                      {name}
                      {versionTaskKind ? ` [${versionTaskName ?? versionTaskKind}]` : ""}
                    </span>
                    <span>{isActive ? "active" : item.is_archived ? "archived" : ""}</span>
                  </button>
                );
              })}
            </div>
            {browser.selectedDatasetVersionId ? (
              <div style={{ marginTop: 12, display: "grid", gap: 8 }}>
                <button type="button" className="ghost-button" onClick={() => void handleSetActive(browser.selectedDatasetVersionId as string)}>
                  Set Active
                </button>
                <button type="button" className="primary-button" disabled={isExporting} onClick={() => void handleExport()}>
                  {isExporting ? "Exporting..." : "Export Dataset Zip"}
                </button>
              </div>
            ) : null}
          </section>

          <section className="placeholder-card">
            {mode === "browse" ? (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
                <strong>Browsing saved dataset version: {selectedVersionName(selectedVersion)}</strong>
                <button type="button" className="ghost-button" disabled={!selectedVersion} onClick={handleDuplicateAndEdit}>
                  Duplicate &amp; Edit
                </button>
              </div>
            ) : (
              <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 8 }}>
                <strong>Draft preview - not saved</strong>
                <div style={{ display: "flex", gap: 8 }}>
                  <button type="button" className="primary-button" disabled={isCreating} onClick={() => void handleCreate()}>
                    {isCreating ? "Creating..." : "Create Version"}
                  </button>
                  <button type="button" className="ghost-button" onClick={handleDiscardDraft}>
                    Discard Draft
                  </button>
                </div>
              </div>
            )}

            <h3>Create Version</h3>
            <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
              <label className="project-field">
                <span>Name</span>
                <input value={draft.draftName} onChange={(event) => draft.setDraftName(event.target.value)} />
              </label>
              <label className="project-field">
                <span>Seed</span>
                <input type="number" value={draft.seed} onChange={(event) => draft.setSeed(Number(event.target.value) || 1337)} />
              </label>
              <label className="project-field">
                <span>Train Ratio</span>
                <input type="number" step="0.01" value={draft.trainRatio} onChange={(event) => draft.setTrainRatio(Number(event.target.value) || 0)} />
              </label>
              <label className="project-field">
                <span>Val Ratio</span>
                <input type="number" step="0.01" value={draft.valRatio} onChange={(event) => draft.setValRatio(Number(event.target.value) || 0)} />
              </label>
              <label className="project-field">
                <span>Test Ratio</span>
                <input type="number" step="0.01" value={draft.testRatio} onChange={(event) => draft.setTestRatio(Number(event.target.value) || 0)} />
              </label>
            </div>

            <div style={{ display: "grid", gap: 8, marginTop: 8, gridTemplateColumns: "repeat(2, minmax(0, 1fr))" }}>
              <StatusMultiSelectDropdown
                label="Include statuses"
                selected={draft.includeStatuses}
                otherSelected={draft.excludeStatuses}
                onSelectedChange={draft.setIncludeStatuses}
                onOtherSelectedChange={draft.setExcludeStatuses}
              />
              <StatusMultiSelectDropdown
                label="Exclude statuses"
                selected={draft.excludeStatuses}
                otherSelected={draft.includeStatuses}
                onSelectedChange={draft.setExcludeStatuses}
                onOtherSelectedChange={draft.setIncludeStatuses}
              />
              <FolderMultiSelectDropdown
                label="Include folders"
                folderPaths={folderPaths}
                selectedPaths={draft.includeFolderPaths}
                opposingSelectedPaths={draft.excludeFolderPaths}
                onSelectedChange={draft.setIncludeFolderPaths}
                onOpposingChange={draft.setExcludeFolderPaths}
              />
              <FolderMultiSelectDropdown
                label="Exclude folders"
                folderPaths={folderPaths}
                selectedPaths={draft.excludeFolderPaths}
                opposingSelectedPaths={draft.includeFolderPaths}
                onSelectedChange={draft.setExcludeFolderPaths}
                onOpposingChange={draft.setIncludeFolderPaths}
              />
            </div>

            <p style={{ marginTop: 6, marginBottom: 0, fontSize: 12, color: "var(--muted, #6f7b8a)" }}>
              Exclude statuses: {draft.excludeStatuses.length === 0 ? "none" : draft.excludeStatuses.join(", ")}
            </p>

            <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
              <label className="model-builder-checkbox">
                <input type="checkbox" checked={draft.includeLabeledOnly} onChange={(event) => draft.setIncludeLabeledOnly(event.target.checked)} />
                <span>Labeled only</span>
              </label>
              <label className="model-builder-checkbox">
                <input type="checkbox" checked={draft.stratify} onChange={(event) => draft.setStratify(event.target.checked)} />
                <span>Stratify by primary label</span>
              </label>
            </div>
            <div style={{ display: "flex", gap: 8, marginTop: 10 }}>
              <button type="button" className="ghost-button" disabled={isPreviewing} onClick={() => void handlePreview()}>
                {isPreviewing ? "Previewing..." : "Preview"}
              </button>
            </div>

            <h3 style={{ marginTop: 16 }}>Preview Assets</h3>
            <div style={{ display: "flex", gap: 8, marginBottom: 10, flexWrap: "wrap" }}>
              <select value={browser.splitFilter} onChange={(event) => browser.setSplitFilter(event.target.value as "all" | "train" | "val" | "test")}>
                <option value="all">All splits</option>
                <option value="train">Train</option>
                <option value="val">Val</option>
                <option value="test">Test</option>
              </select>
              <select value={browser.statusFilter} onChange={(event) => browser.setStatusFilter(event.target.value as "all" | AnnotationStatus)}>
                <option value="all">All statuses</option>
                <option value="unlabeled">unlabeled</option>
                <option value="labeled">labeled</option>
                <option value="skipped">skipped</option>
                <option value="needs_review">needs_review</option>
                <option value="approved">approved</option>
              </select>
              <select
                value={classFilter}
                onChange={(event) => {
                  setClassFilter(event.target.value);
                  browser.setPage(1);
                }}
              >
                <option value="all">All classes</option>
                {classFilterOptions.map((row) => (
                  <option key={row.id} value={row.id}>
                    {row.name}
                  </option>
                ))}
              </select>
              <input placeholder="Search assets..." value={browser.searchText} onChange={(event) => browser.setSearchText(event.target.value)} />
              <div style={{ display: "flex", gap: 6 }}>
                <button
                  type="button"
                  className={browser.viewMode === "list" ? "ghost-button active-toggle" : "ghost-button"}
                  onClick={() => browser.setViewMode("list")}
                >
                  List
                </button>
                <button
                  type="button"
                  className={browser.viewMode === "grid" ? "ghost-button active-toggle" : "ghost-button"}
                  onClick={() => browser.setViewMode("grid")}
                >
                  Grid
                </button>
              </div>
            </div>
            {isLoadingAssets ? <p>Loading assets...</p> : null}
            {!isLoadingAssets && browser.viewMode === "list" ? (
              <div style={{ display: "grid", gap: 6 }}>
                {(assetsPayload?.items ?? []).map((item) => (
                  <div key={item.asset_id} style={{ display: "grid", gridTemplateColumns: "60px 1fr auto auto", gap: 8, alignItems: "center" }}>
                    <img
                      src={resolveAssetUri(contentUrlForAsset(item.asset_id))}
                      alt={item.filename}
                      style={{ width: 56, height: 42, objectFit: "cover", borderRadius: 6, border: "1px solid var(--line, #d8dce6)" }}
                    />
                    <span>{item.relative_path || item.filename}</span>
                    <span>{item.status}</span>
                    <span>{item.split ?? "-"}</span>
                  </div>
                ))}
              </div>
            ) : null}
            {!isLoadingAssets && browser.viewMode === "grid" ? (
              <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fill, minmax(140px, 1fr))" }}>
                {(assetsPayload?.items ?? []).map((item) => (
                  <div key={item.asset_id} style={{ border: "1px solid var(--line, #d8dce6)", borderRadius: 8, padding: 8 }}>
                    <img
                      src={resolveAssetUri(contentUrlForAsset(item.asset_id))}
                      alt={item.filename}
                      style={{ width: "100%", height: 96, objectFit: "cover", borderRadius: 6, border: "1px solid var(--line, #d8dce6)" }}
                    />
                    <div style={{ marginTop: 6, display: "grid", gap: 4 }}>
                      <span style={{ fontSize: 12, wordBreak: "break-word" }}>{item.relative_path || item.filename}</span>
                      <span style={{ fontSize: 12, color: "var(--muted, #6f7b8a)" }}>
                        {item.status} | {item.split ?? "-"}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            ) : null}
            {assetsPayload ? (
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
                <button type="button" className="ghost-button" disabled={browser.page <= 1} onClick={() => browser.setPage((value) => Math.max(1, value - 1))}>
                  Prev
                </button>
                <span>
                  Page {assetsPayload.page} | {assetsPayload.total} assets
                </span>
                <button
                  type="button"
                  className="ghost-button"
                  disabled={assetsPayload.page * assetsPayload.page_size >= assetsPayload.total}
                  onClick={() => browser.setPage((value) => value + 1)}
                >
                  Next
                </button>
              </div>
            ) : null}
          </section>

          <section className="placeholder-card">
            <h3>{summarySource === "saved" ? "Summary (Saved Version)" : "Summary (Draft Preview)"}</h3>
            {!summaryData ? <p>{summarySource === "saved" ? "Select a dataset version." : "Run preview to compute counts."}</p> : null}
            {summaryData ? (
              <div style={{ display: "grid", gap: 10 }}>
                <p>Total: {summaryData.total}</p>
                <p>
                  Splits: train {summaryData.split_counts.train} | val {summaryData.split_counts.val} | test {summaryData.split_counts.test}
                </p>
                <div>
                  <h4>Class Distribution</h4>
                  <div style={{ display: "grid", gap: 6 }}>
                    {Object.entries(summaryData.class_counts).map(([classId, count]) => (
                      <div key={classId} style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 8 }}>
                        <span>{classDisplayName(classId)}</span>
                        <span>{count}</span>
                      </div>
                    ))}
                  </div>
                </div>
                {summaryData.warnings.length > 0 ? (
                  <div>
                    <h4>Warnings</h4>
                    <ul>
                      {summaryData.warnings.map((warning) => (
                        <li key={warning}>{warning}</li>
                      ))}
                    </ul>
                  </div>
                ) : null}
              </div>
            ) : null}
          </section>
        </div>
      </section>
    </main>
  );
}
