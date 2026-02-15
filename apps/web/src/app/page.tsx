"use client";

import { useEffect, useMemo, useState } from "react";

import { AssetGrid } from "../components/AssetGrid";
import { Filters } from "../components/Filters";
import { LabelPanel } from "../components/LabelPanel";
import { Viewer } from "../components/Viewer";
import {
  ApiError,
  createExport,
  createCategory,
  createProject,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  type Annotation,
} from "../lib/api";
import { useAnnotationWorkflow } from "../lib/hooks/useAnnotationWorkflow";
import { useAssets } from "../lib/hooks/useAssets";
import { useDeleteWorkflow } from "../lib/hooks/useDeleteWorkflow";
import { buildTargetRelativePath, isImageCandidate, useImportWorkflow } from "../lib/hooks/useImportWorkflow";
import { useLabels } from "../lib/hooks/useLabels";
import { useProject } from "../lib/hooks/useProject";
import { resolveWorkspaceHotkeyAction } from "../lib/workspace/hotkeys";
import { asRelativePath, buildTreeEntries, collectFolderPathsFromRelativePaths, folderChain, type TreeEntry } from "../lib/workspace/tree";

const PROJECT_MULTILABEL_STORAGE_KEY = "pixel-sheriff:project-multilabel:v1";
const LAST_PROJECT_STORAGE_KEY = "pixel-sheriff:last-project-id:v1";

type FolderReviewStatus = "all_labeled" | "has_unlabeled" | "empty";

export default function HomePage() {
  const { data: projects, refetch: refetchProjects } = useProject();
  const [selectedProjectId, setSelectedProjectId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [assetIndex, setAssetIndex] = useState(0);
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [projectMultiLabelSettings, setProjectMultiLabelSettings] = useState<Record<string, boolean>>({});
  const [message, setMessage] = useState<string | null>(null);
  const [selectedTreeFolderPath, setSelectedTreeFolderPath] = useState<string | null>(null);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});
  const [preferredProjectId, setPreferredProjectId] = useState<string | null>(null);
  const [hasLoadedPreferredProject, setHasLoadedPreferredProject] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(LAST_PROJECT_STORAGE_KEY);
      setPreferredProjectId(raw && raw.trim() !== "" ? raw : null);
    } finally {
      setHasLoadedPreferredProject(true);
    }
  }, []);

  useEffect(() => {
    if (!hasLoadedPreferredProject) return;

    if (projects.length === 0) {
      if (selectedProjectId !== null) setSelectedProjectId(null);
      return;
    }

    if (selectedProjectId && projects.some((project) => project.id === selectedProjectId)) {
      return;
    }

    if (preferredProjectId && projects.some((project) => project.id === preferredProjectId)) {
      setSelectedProjectId(preferredProjectId);
      return;
    }

    setSelectedProjectId(projects[0].id);
  }, [hasLoadedPreferredProject, preferredProjectId, projects, selectedProjectId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (selectedProjectId) {
      window.localStorage.setItem(LAST_PROJECT_STORAGE_KEY, selectedProjectId);
    } else {
      window.localStorage.removeItem(LAST_PROJECT_STORAGE_KEY);
    }
  }, [selectedProjectId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    try {
      const raw = window.localStorage.getItem(PROJECT_MULTILABEL_STORAGE_KEY);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return;

      const normalized: Record<string, boolean> = {};
      for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
        normalized[key] = Boolean(value);
      }
      setProjectMultiLabelSettings(normalized);
    } catch {
      setProjectMultiLabelSettings({});
    }
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(PROJECT_MULTILABEL_STORAGE_KEY, JSON.stringify(projectMultiLabelSettings));
  }, [projectMultiLabelSettings]);

  const datasets = projects.map((project) => ({ id: project.id, name: project.name }));
  const filteredDatasets = datasets.filter((dataset) => dataset.name.toLowerCase().includes(query.trim().toLowerCase()));
  const activeDatasetId = selectedProjectId;
  const multiLabelEnabled = selectedProjectId ? Boolean(projectMultiLabelSettings[selectedProjectId]) : false;

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(selectedProjectId);
  const { data: labels, refetch: refetchLabels } = useLabels(selectedProjectId);
  const selectedDatasetName = datasets.find((dataset) => dataset.id === activeDatasetId)?.name ?? "No dataset selected";
  const importWorkflow = useImportWorkflow({
    assets,
    selectedProjectId,
    isAssetsLoading,
    fetchProjectAssets: listAssets,
  });
  const {
    isImporting,
    setIsImporting,
    importFailures,
    setImportFailures,
    importDialog,
    importMode,
    setImportMode,
    importExistingProjectId,
    setImportExistingProjectId,
    importNewProjectName,
    setImportNewProjectName,
    importFolderName,
    setImportFolderName,
    setImportFolderOptionsByProject,
    selectedImportExistingFolder,
    setSelectedImportExistingFolder,
    setImportProgress,
    importExistingFolderOptions,
    importProgressView,
    openImportDialog,
    closeImportDialog,
  } = importWorkflow;
  const assetById = useMemo(() => {
    const map = new Map<string, (typeof assets)[number]>();
    for (const asset of assets) map.set(asset.id, asset);
    return map;
  }, [assets]);

  const treeBuild = useMemo(() => buildTreeEntries(assets), [assets]);
  const treeEntries = treeBuild.entries;
  const treeFolderPaths = useMemo(() => Object.keys(treeBuild.folderAssetIds), [treeBuild.folderAssetIds]);
  const visibleTreeEntries = useMemo(() => {
    function isHiddenByCollapsedAncestor(entry: TreeEntry): boolean {
      const parentPath = entry.kind === "folder" ? entry.path.split("/").slice(0, -1).join("/") : entry.folderPath ?? "";
      if (!parentPath) return false;
      for (const ancestor of folderChain(parentPath)) {
        if (collapsedFolders[ancestor]) return true;
      }
      return false;
    }

    return treeEntries.filter((entry) => !isHiddenByCollapsedAncestor(entry));
  }, [collapsedFolders, treeEntries]);
  const orderedAssetRows = useMemo(
    () =>
      treeBuild.orderedAssetIds
        .map((assetId) => assetById.get(assetId))
        .filter((asset): asset is (typeof assets)[number] => asset !== undefined),
    [assetById, treeBuild.orderedAssetIds],
  );

  const filteredAssetRows = useMemo(() => {
    if (!selectedTreeFolderPath) return orderedAssetRows;
    const prefix = `${selectedTreeFolderPath}/`;
    return orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(prefix));
  }, [orderedAssetRows, selectedTreeFolderPath]);

  const assetRows = filteredAssetRows;
  const allLabelRows = labels.map((label) => ({
    id: label.id,
    name: label.name,
    isActive: label.is_active,
    displayOrder: label.display_order,
  }));
  const activeLabelRows = allLabelRows.filter((label) => label.isActive).sort((a, b) => a.displayOrder - b.displayOrder);

  const safeAssetIndex = Math.min(assetIndex, Math.max(assetRows.length - 1, 0));
  const currentAsset = assetRows[safeAssetIndex] ?? null;
  const viewerAsset = currentAsset ? { id: currentAsset.id, uri: resolveAssetUri(currentAsset.uri) } : null;

  const annotationByAssetId = useMemo(() => {
    const map = new Map<string, Annotation>();
    for (const annotation of annotations) {
      map.set(annotation.asset_id, annotation);
    }
    return map;
  }, [annotations]);
  const annotationWorkflow = useAnnotationWorkflow({
    selectedProjectId,
    currentAsset,
    annotationByAssetId,
    activeLabelRows,
    multiLabelEnabled,
    editMode,
    setEditMode,
    setAnnotations,
    setMessage,
  });
  const {
    selectedLabelIds,
    setSelectedLabelIds,
    pendingAnnotations,
    pendingCount,
    canSubmit,
    isSaving,
    handleToggleLabel,
    handleSubmit,
    resetAnnotationWorkflow,
  } = annotationWorkflow;
  const deleteWorkflow = useDeleteWorkflow({
    selectedProjectId,
    selectedDatasetName,
    selectedTreeFolderPath,
    setSelectedTreeFolderPath,
    currentAsset,
    assetRows,
    assets,
    treeFolderAssetIds: treeBuild.folderAssetIds,
    assetById,
    annotations,
    setAnnotations,
    pendingAnnotations,
    setPendingAnnotations: annotationWorkflow.setPendingAnnotations,
    setAssetIndex,
    setCollapsedFolders,
    setSelectedProjectId,
    setProjectMultiLabelSettings,
    setImportExistingProjectId,
    setSelectedImportExistingFolder,
    setImportFolderOptionsByProject,
    setMessage,
    resetAnnotationWorkflow,
    setEditMode,
    refetchAssets,
    refetchProjects,
  });
  const {
    isDeletingAssets,
    isDeletingProject,
    bulkDeleteMode,
    selectedDeleteAssets,
    selectedDeleteAssetIds,
    handleToggleBulkDeleteMode,
    handleToggleDeleteSelection,
    handleSelectAllDeleteScope,
    handleClearDeleteSelection,
    handleDeleteCurrentAsset,
    handleDeleteSelectedAssets,
    handleDeleteSelectedFolder,
    handleDeleteFolderPath,
    handleDeleteCurrentProject,
    resetDeleteWorkflow,
  } = deleteWorkflow;

  const assetReviewStatusById = useMemo(() => {
    const map = new Map<string, "labeled" | "unlabeled">();
    for (const asset of orderedAssetRows) {
      const pending = pendingAnnotations[asset.id];
      if (pending) {
        const isLabeled = pending.status !== "unlabeled" && pending.labelIds.length > 0;
        map.set(asset.id, isLabeled ? "labeled" : "unlabeled");
        continue;
      }
      const annotation = annotationByAssetId.get(asset.id);
      const isLabeled = Boolean(annotation && annotation.status !== "unlabeled");
      map.set(asset.id, isLabeled ? "labeled" : "unlabeled");
    }
    return map;
  }, [annotationByAssetId, orderedAssetRows, pendingAnnotations]);

  const pageStatuses = useMemo(
    () => assetRows.map((asset) => assetReviewStatusById.get(asset.id) ?? "unlabeled"),
    [assetReviewStatusById, assetRows],
  );
  const selectedFolderAssetCount = useMemo(() => {
    if (!selectedTreeFolderPath) return 0;
    return treeBuild.folderAssetIds[selectedTreeFolderPath]?.length ?? 0;
  }, [selectedTreeFolderPath, treeBuild.folderAssetIds]);
  const messageTone = useMemo(() => {
    if (!message) return "info";
    const lower = message.toLowerCase();
    if (lower.includes("failed") || lower.includes("error")) return "error";
    return "success";
  }, [message]);

  const folderReviewStatusByPath = useMemo(() => {
    const status: Record<string, FolderReviewStatus> = {};
    for (const [folderPath, assetIds] of Object.entries(treeBuild.folderAssetIds)) {
      if (assetIds.length === 0) {
        status[folderPath] = "empty";
        continue;
      }
      const hasUnlabeled = assetIds.some((assetId) => (assetReviewStatusById.get(assetId) ?? "unlabeled") === "unlabeled");
      status[folderPath] = hasUnlabeled ? "has_unlabeled" : "all_labeled";
    }
    return status;
  }, [assetReviewStatusById, treeBuild.folderAssetIds]);

  useEffect(() => {
    setAssetIndex((previous) => Math.min(previous, Math.max(assetRows.length - 1, 0)));
  }, [assetRows.length]);

  useEffect(() => {
    if (!message) return;
    const timeout = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timeout);
  }, [message]);

  useEffect(() => {
    function onKeyDown(event: KeyboardEvent) {
      const action = resolveWorkspaceHotkeyAction(event, { activeLabelCount: activeLabelRows.length });
      if (!action) return;

      if (action.type === "navigate_prev") {
        event.preventDefault();
        setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
        return;
      }
      if (action.type === "navigate_next") {
        event.preventDefault();
        setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
        return;
      }

      const label = activeLabelRows[action.labelIndex];
      if (!label) return;

      event.preventDefault();
      handleToggleLabel(label.id);
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [activeLabelRows, assetRows.length, handleToggleLabel]);

  function handleSelectDataset(id: string) {
    setSelectedProjectId(id);
    setSelectedTreeFolderPath(null);
    setCollapsedFolders({});
    resetDeleteWorkflow();
    setAssetIndex(0);
    resetAnnotationWorkflow();
    setEditMode(false);
    setMessage(null);
  }

  function handlePrevAsset() {
    setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
  }

  function handleNextAsset() {
    setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
  }

  function handleToggleProjectMultiLabel() {
    if (!selectedProjectId) return;
    setProjectMultiLabelSettings((previous) => ({
      ...previous,
      [selectedProjectId]: !Boolean(previous[selectedProjectId]),
    }));
  }

  async function handleCreateLabel(name: string) {
    if (!selectedProjectId) {
      setMessage("Select a dataset before creating labels.");
      return;
    }

    try {
      setIsCreatingLabel(true);
      setMessage(null);
      const created = await createCategory(selectedProjectId, { name, display_order: allLabelRows.length });
      await refetchLabels();
      setSelectedLabelIds([created.id]);
      setMessage(`Created label "${created.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to create label: ${error.message}` : "Failed to create label.");
    } finally {
      setIsCreatingLabel(false);
    }
  }

  async function handleSaveLabelChanges(
    changes: Array<{ id: number; name: string; isActive: boolean; displayOrder: number }>,
  ) {
    try {
      setIsSavingLabelChanges(true);
      setMessage(null);
      for (const change of changes) {
        await patchCategory(change.id, {
          name: change.name.trim(),
          is_active: change.isActive,
          display_order: change.displayOrder,
        });
      }
      await refetchLabels();
      setMessage("Saved label configuration.");
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to save labels: ${error.message}` : "Failed to save labels.");
    } finally {
      setIsSavingLabelChanges(false);
    }
  }

  async function confirmImportFromDialog() {
    const files = importDialog.files;
    const sourceFolderName = importDialog.sourceFolderName;
    const folderName = importFolderName.trim();
    if (files.length === 0) {
      setMessage("Import cancelled: no files selected.");
      closeImportDialog();
      return;
    }
    if (!folderName) {
      setMessage("Folder name is required.");
      return;
    }

    try {
      setIsImporting(true);
      setMessage("Importing images...");
      setImportFailures([]);
      const totalBytes = files.reduce((sum, file) => sum + file.size, 0);
      setImportProgress({
        totalFiles: files.length,
        completedFiles: 0,
        uploadedFiles: 0,
        failedFiles: 0,
        totalBytes,
        processedBytes: 0,
        startedAtMs: Date.now(),
        activeFileName: null,
      });

      let targetProjectId = "";
      let targetProjectName = "";

      if (importMode === "new") {
        const projectName = importNewProjectName.trim();
        if (!projectName) {
          setMessage("Project name is required for new project imports.");
          return;
        }
        const project = await createProject({ name: projectName, task_type: "classification_single" });
        targetProjectId = project.id;
        targetProjectName = project.name;
      } else {
        const project = projects.find((item) => item.id === importExistingProjectId);
        if (!project) {
          setMessage("Please select an existing project.");
          return;
        }
        targetProjectId = project.id;
        targetProjectName = project.name;
      }

      let uploadedCount = 0;
      const failures: string[] = [];

      for (const file of files) {
        setImportProgress((previous) => (previous ? { ...previous, activeFileName: file.name } : previous));
        try {
          const targetRelativePath = buildTargetRelativePath(file, folderName);
          await uploadAsset(targetProjectId, file, targetRelativePath);
          uploadedCount += 1;
          setImportProgress((previous) =>
            previous
              ? {
                  ...previous,
                  completedFiles: previous.completedFiles + 1,
                  uploadedFiles: previous.uploadedFiles + 1,
                  processedBytes: previous.processedBytes + file.size,
                  activeFileName: null,
                }
              : previous,
          );
        } catch (error) {
          if (error instanceof ApiError) {
            const reason = error.responseBody ? ` (${error.responseBody})` : "";
            failures.push(`${file.name}: ${error.message}${reason}`);
          } else {
            failures.push(`${file.name}: ${error instanceof Error ? error.message : "unknown upload error"}`);
          }
          setImportProgress((previous) =>
            previous
              ? {
                  ...previous,
                  completedFiles: previous.completedFiles + 1,
                  failedFiles: previous.failedFiles + 1,
                  processedBytes: previous.processedBytes + file.size,
                  activeFileName: null,
                }
              : previous,
          );
        }
      }

      await refetchProjects();
      await refetchAssets(targetProjectId);
      setSelectedProjectId(targetProjectId);
      setSelectedTreeFolderPath(null);
      setCollapsedFolders({});
      setAssetIndex(0);
      resetAnnotationWorkflow();
      setEditMode(false);
      setSelectedImportExistingFolder("");
      setImportFolderOptionsByProject((previous) => {
        const importedRelativePaths = files.map((file) => buildTargetRelativePath(file, folderName));
        const importedFolders = collectFolderPathsFromRelativePaths(importedRelativePaths);
        const merged = new Set([...(previous[targetProjectId] ?? []), ...importedFolders]);
        return {
          ...previous,
          [targetProjectId]: Array.from(merged).sort((a, b) => a.localeCompare(b)),
        };
      });
      setImportFailures(failures);
      setSelectedImportExistingFolder("");
      closeImportDialog();

      if (uploadedCount === 0) setMessage(`Import failed: no files uploaded to "${folderName}".`);
      else if (failures.length > 0)
        setMessage(`Imported ${uploadedCount}/${files.length} images into "${targetProjectName}/${folderName}".`);
      else setMessage(`Imported ${uploadedCount} images into "${targetProjectName}/${folderName}".`);
    } catch (error) {
      setImportFailures([]);
      setImportProgress(null);
      setMessage(error instanceof Error ? `Import failed: ${error.message}` : "Import failed.");
    } finally {
      setIsImporting(false);
    }
  }

  async function handleImport() {
    const picker = document.createElement("input");
    picker.type = "file";
    picker.accept = "image/*";
    picker.multiple = true;
    (picker as HTMLInputElement & { webkitdirectory?: boolean }).webkitdirectory = true;

    picker.onchange = async () => {
      const files = Array.from(picker.files ?? []).filter(isImageCandidate);
      if (files.length === 0) {
        setImportFailures([]);
        setMessage("No image files were selected (supported by MIME or extension).");
        return;
      }
      const rootName = files[0].webkitRelativePath.split("/")[0] || `Dataset ${new Date().toLocaleString()}`;
      const defaultProject = projects.find((project) => project.id === selectedProjectId) ?? projects[0];
      openImportDialog(files, rootName, defaultProject?.id ?? "");
    };

    picker.click();
  }

  async function handleExport() {
    if (!selectedProjectId) {
      setMessage("Select a dataset before exporting.");
      return;
    }

    try {
      setIsExporting(true);
      setMessage("Building export...");
      const created = await createExport(selectedProjectId, {
        selection_criteria_json: { statuses: ["labeled", "approved", "needs_review", "skipped"] },
      });

      const url = resolveAssetUri(created.export_uri);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `${selectedDatasetName.replace(/[^a-zA-Z0-9-_]+/g, "_") || "dataset"}-${created.hash.slice(0, 8)}.zip`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();

      const counts = created.manifest_json.counts as Record<string, number> | undefined;
      if (counts && typeof counts.assets === "number" && typeof counts.annotations === "number") {
        setMessage(`Export ready. ${counts.assets} assets, ${counts.annotations} annotations.`);
      } else {
        setMessage("Export ready.");
      }
    } catch (error) {
      setMessage(error instanceof Error ? `Export failed: ${error.message}` : "Export failed.");
    } finally {
      setIsExporting(false);
    }
  }

  function handleSelectFolderScope(folderPath: string | null) {
    if (folderPath) {
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const path of folderChain(folderPath)) next[path] = false;
        return next;
      });
    }
    setSelectedTreeFolderPath(folderPath);
    setAssetIndex(0);
  }

  function handleToggleFolderCollapsed(folderPath: string) {
    setCollapsedFolders((previous) => ({
      ...previous,
      [folderPath]: !Boolean(previous[folderPath]),
    }));
  }

  function handleCollapseAllFolders() {
    const next: Record<string, boolean> = {};
    for (const folderPath of treeFolderPaths) {
      next[folderPath] = true;
    }
    setCollapsedFolders(next);
  }

  function handleExpandAllFolders() {
    setCollapsedFolders({});
  }

  function handleSelectTreeAsset(assetId: string, folderPath?: string) {
    let scopedRows = assetRows;

    if (folderPath && folderPath !== selectedTreeFolderPath) {
      const prefix = `${folderPath}/`;
      scopedRows = orderedAssetRows.filter((asset) => asRelativePath(asset).replaceAll("\\", "/").startsWith(prefix));
      setSelectedTreeFolderPath(folderPath);
      setCollapsedFolders((previous) => {
        const next = { ...previous };
        for (const path of folderChain(folderPath)) next[path] = false;
        return next;
      });
    }

    let index = scopedRows.findIndex((item) => item.id === assetId);
    if (index < 0) {
      setSelectedTreeFolderPath(null);
      scopedRows = orderedAssetRows;
      index = scopedRows.findIndex((item) => item.id === assetId);
    }

    if (index >= 0) setAssetIndex(index);
  }

  const headerTitle = selectedTreeFolderPath ? `${selectedDatasetName} / ${selectedTreeFolderPath}` : selectedDatasetName;

  return (
    <main className="workspace-shell">
      <section className="workspace-frame">
        <header className="workspace-header">
          <div className="workspace-header-cell">Datasets</div>
          <div className="workspace-header-cell workspace-header-title">{headerTitle}</div>
          <div className="workspace-header-cell workspace-header-actions" aria-label="Toolbar">
            <span />
            <span />
            <span />
            <span />
          </div>
        </header>

        <div className="workspace-body">
          <aside className="workspace-sidebar">
            <Filters query={query} onQueryChange={setQuery} />
            <AssetGrid datasets={filteredDatasets} selectedDatasetId={activeDatasetId} onSelectDataset={handleSelectDataset} />
            <section className="project-tree">
              <div className="project-tree-head">
                <h3>Files</h3>
                <div className="tree-head-actions">
                  <button type="button" className="tree-scope-button" onClick={handleCollapseAllFolders}>
                    Collapse all
                  </button>
                  <button type="button" className="tree-scope-button" onClick={handleExpandAllFolders}>
                    Expand all
                  </button>
                  <button
                    type="button"
                    className={selectedTreeFolderPath === null ? "tree-scope-button active" : "tree-scope-button"}
                    onClick={() => handleSelectFolderScope(null)}
                  >
                    All files
                  </button>
                </div>
              </div>
              <div className="tree-delete-toolbar">
                <button
                  type="button"
                  className={bulkDeleteMode ? "tree-scope-button danger active" : "tree-scope-button danger"}
                  onClick={handleToggleBulkDeleteMode}
                  disabled={!selectedProjectId || isDeletingAssets}
                >
                  {bulkDeleteMode ? "Exit multi-delete" : "Multi-delete"}
                </button>
                {bulkDeleteMode ? (
                  <>
                    <button type="button" className="tree-scope-button" onClick={handleSelectAllDeleteScope} disabled={isDeletingAssets}>
                      Select scope
                    </button>
                    <button type="button" className="tree-scope-button" onClick={handleClearDeleteSelection} disabled={isDeletingAssets}>
                      Clear
                    </button>
                    <button
                      type="button"
                      className="tree-scope-button danger"
                      onClick={handleDeleteSelectedAssets}
                      disabled={isDeletingAssets || selectedDeleteAssetIds.length === 0}
                    >
                      Delete selected ({selectedDeleteAssetIds.length})
                    </button>
                  </>
                ) : null}
                {selectedTreeFolderPath ? (
                  <button
                    type="button"
                    className="tree-scope-button danger"
                    onClick={handleDeleteSelectedFolder}
                    disabled={isDeletingAssets || selectedFolderAssetCount === 0}
                  >
                    Delete folder ({selectedFolderAssetCount})
                  </button>
                ) : null}
              </div>
              {selectedTreeFolderPath ? <p className="tree-scope-caption">Scope: {selectedTreeFolderPath}</p> : null}
              <ul>
                {visibleTreeEntries.map((entry) => (
                  <li key={entry.key}>
                    {entry.kind === "folder" ? (
                      <div className="tree-folder-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                        <button
                          type="button"
                          className="tree-folder-toggle"
                          aria-label={collapsedFolders[entry.path] ? "Expand folder" : "Collapse folder"}
                          onClick={() => handleToggleFolderCollapsed(entry.path)}
                        >
                          {collapsedFolders[entry.path] ? ">" : "v"}
                        </button>
                        <button
                          type="button"
                          className={`tree-folder-button${selectedTreeFolderPath === entry.path ? " active" : ""} ${
                            folderReviewStatusByPath[entry.path] === "all_labeled"
                              ? "is-labeled"
                              : folderReviewStatusByPath[entry.path] === "has_unlabeled"
                                ? "has-unlabeled"
                                : "is-empty"
                          }`}
                          onClick={() => handleSelectFolderScope(entry.path)}
                        >
                          {entry.name}
                        </button>
                        <button
                          type="button"
                          className="tree-row-delete"
                          onClick={() => void handleDeleteFolderPath(entry.path)}
                          disabled={isDeletingAssets}
                          title={`Delete "${entry.path}"`}
                        >
                          x
                        </button>
                      </div>
                    ) : (
                      <div className="tree-file-row" style={{ paddingLeft: `${entry.depth * 14 + 8}px` }}>
                        {bulkDeleteMode && entry.assetId ? (
                          <input
                            className="tree-file-checkbox"
                            type="checkbox"
                            checked={Boolean(selectedDeleteAssets[entry.assetId])}
                            onChange={() => {
                              if (entry.assetId) handleToggleDeleteSelection(entry.assetId);
                            }}
                            disabled={isDeletingAssets}
                            aria-label={`Select ${entry.name} for delete`}
                          />
                        ) : null}
                        <button
                          type="button"
                          className={`tree-file${entry.assetId === currentAsset?.id ? " active" : ""} ${
                            entry.assetId && assetReviewStatusById.get(entry.assetId) === "labeled" ? "is-labeled" : "is-unlabeled"
                          }${entry.assetId && selectedDeleteAssets[entry.assetId] ? " delete-selected" : ""}`}
                          onClick={() =>
                            entry.assetId &&
                            (bulkDeleteMode ? handleToggleDeleteSelection(entry.assetId) : handleSelectTreeAsset(entry.assetId, entry.folderPath))
                          }
                        >
                          {entry.name}
                        </button>
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          </aside>

          <Viewer
            currentAsset={viewerAsset}
            totalAssets={assetRows.length}
            currentIndex={safeAssetIndex}
            pageStatuses={pageStatuses}
            onSelectIndex={setAssetIndex}
            onPrev={handlePrevAsset}
            onNext={handleNextAsset}
          />
          <LabelPanel
            labels={activeLabelRows}
            allLabels={allLabelRows}
            selectedLabelIds={selectedLabelIds}
            onToggleLabel={handleToggleLabel}
            onSubmit={handleSubmit}
            isSaving={isSaving}
            onCreateLabel={handleCreateLabel}
            isCreatingLabel={isCreatingLabel}
            editMode={editMode}
            onToggleEditMode={() => setEditMode((value) => !value)}
            pendingCount={pendingCount}
            onSaveLabelChanges={handleSaveLabelChanges}
            isSavingLabelChanges={isSavingLabelChanges}
            canSubmit={canSubmit}
            multiLabelEnabled={multiLabelEnabled}
            onToggleMultiLabel={handleToggleProjectMultiLabel}
          />
        </div>

        <footer className="workspace-footer">
          <div className="footer-left">
            <button type="button" className="ghost-button" onClick={handleImport} disabled={isImporting}>
              {isImporting ? "Importing..." : "Import"}
            </button>
            <button type="button" className="ghost-button" onClick={handleExport} disabled={isExporting || !selectedProjectId}>
              {isExporting ? "Exporting..." : "Export Dataset"}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteCurrentAsset}
              disabled={isDeletingAssets || !selectedProjectId || !currentAsset}
            >
              {isDeletingAssets ? "Removing..." : "Remove Image"}
            </button>
            <button
              type="button"
              className={bulkDeleteMode ? "ghost-button active-toggle" : "ghost-button"}
              onClick={handleToggleBulkDeleteMode}
              disabled={!selectedProjectId || isDeletingAssets}
            >
              {bulkDeleteMode ? "Exit Multi-delete" : "Multi-delete"}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteSelectedAssets}
              disabled={isDeletingAssets || selectedDeleteAssetIds.length === 0}
            >
              {isDeletingAssets ? "Removing..." : `Delete Selected (${selectedDeleteAssetIds.length})`}
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteSelectedFolder}
              disabled={isDeletingAssets || !selectedTreeFolderPath || selectedFolderAssetCount === 0}
            >
              Delete Folder
            </button>
            <button
              type="button"
              className="ghost-button danger-button"
              onClick={handleDeleteCurrentProject}
              disabled={isDeletingProject || !selectedProjectId}
            >
              {isDeletingProject ? "Deleting..." : "Delete Project"}
            </button>
          </div>
        </footer>
      </section>
      {message ? (
        <div className={`status-toast ${messageTone === "error" ? "is-error" : "is-success"}`} role="status" aria-live="polite">
          <span>{message}</span>
          <button type="button" aria-label="Dismiss message" onClick={() => setMessage(null)}>
            x
          </button>
        </div>
      ) : null}
      {importFailures.length > 0 ? (
        <ul className="status-errors">
          {importFailures.map((failure) => (
            <li key={failure}>{failure}</li>
          ))}
        </ul>
      ) : null}
      {importDialog.open ? (
        <div className="import-modal-backdrop">
          <div className="import-modal">
            <h3>Import Images</h3>
            <div className="import-mode-row">
              <label>
                <input
                  type="radio"
                  checked={importMode === "existing"}
                  onChange={() => {
                    setImportMode("existing");
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                  disabled={projects.length === 0}
                />
                Existing Project
              </label>
              <label>
                <input
                  type="radio"
                  checked={importMode === "new"}
                  onChange={() => {
                    setImportMode("new");
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                />
                New Project
              </label>
            </div>
            <label className="import-field">
              <span>Project</span>
              {importMode === "new" ? (
                <input value={importNewProjectName} onChange={(event) => setImportNewProjectName(event.target.value)} placeholder="Project name" />
              ) : (
                <select
                  value={importExistingProjectId}
                  onChange={(event) => {
                    setImportExistingProjectId(event.target.value);
                    setSelectedImportExistingFolder("");
                    setImportFolderName(importDialog.sourceFolderName);
                  }}
                >
                  <option value="">Select project</option>
                  {projects.map((project) => (
                    <option key={project.id} value={project.id}>
                      {project.name}
                    </option>
                  ))}
                </select>
              )}
            </label>
            {importMode === "existing" ? (
              <label className="import-field">
                <span>Existing Folder/Subfolder (optional)</span>
                <select
                  value={selectedImportExistingFolder}
                  onChange={(event) => {
                    const value = event.target.value;
                    setSelectedImportExistingFolder(value);
                    if (value) setImportFolderName(value);
                  }}
                >
                  <option value="">Create new / custom</option>
                  {importExistingFolderOptions.map((folderPath) => (
                    <option key={folderPath} value={folderPath}>
                      {folderPath}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="import-field">
              <span>Folder Name</span>
              <input value={importFolderName} onChange={(event) => setImportFolderName(event.target.value)} placeholder={importDialog.sourceFolderName} />
            </label>
            {isImporting && importProgressView ? (
              <section className="import-progress" aria-live="polite">
                <div className="import-progress-head">
                  <strong>Importing {importProgressView.progressText}</strong>
                  <span>{importProgressView.percent}%</span>
                </div>
                <div className="import-progress-bar">
                  <span style={{ width: `${importProgressView.percent}%` }} />
                </div>
                <div className="import-progress-metrics">
                  <span>{importProgressView.bytesText}</span>
                  <span>{importProgressView.speedText}</span>
                  <span>{importProgressView.fileRateText}</span>
                </div>
                <div className="import-progress-metrics">
                  <span>{importProgressView.uploadedFilesText}</span>
                  <span>{importProgressView.failedFilesText}</span>
                  <span>{importProgressView.remainingFilesText}</span>
                </div>
                <div className="import-progress-metrics">
                  <span>Elapsed: {importProgressView.elapsedText}</span>
                  <span>ETA: {importProgressView.etaText}</span>
                </div>
                {importProgressView.activeFileName ? <p className="import-progress-file">Uploading: {importProgressView.activeFileName}</p> : null}
              </section>
            ) : null}
            <div className="import-modal-actions">
                <button
                  type="button"
                  className="ghost-button"
                  onClick={closeImportDialog}
                  disabled={isImporting}
                >
                Cancel
              </button>
              <button type="button" className="primary-button" onClick={confirmImportFromDialog} disabled={isImporting}>
                {isImporting ? "Importing..." : "Import"}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}
