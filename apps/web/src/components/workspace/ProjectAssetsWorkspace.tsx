"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { LabelPanel } from "../LabelPanel";
import { Viewer } from "../Viewer";
import {
  ApiError,
  createProjectModel,
  createExport,
  createCategory,
  createProject,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  type Annotation,
  type ProjectTaskType,
} from "../../lib/api";
import { useAnnotationWorkflow } from "../../lib/hooks/useAnnotationWorkflow";
import { useAssets } from "../../lib/hooks/useAssets";
import { useDeleteWorkflow } from "../../lib/hooks/useDeleteWorkflow";
import { buildTargetRelativePath, isImageCandidate, useImportWorkflow } from "../../lib/hooks/useImportWorkflow";
import { useLabels } from "../../lib/hooks/useLabels";
import { useProjectMultiLabelSettings } from "../../lib/hooks/useProjectMultiLabelSettings";
import { useProject } from "../../lib/hooks/useProject";
import { useWorkspaceHotkeys } from "../../lib/hooks/useWorkspaceHotkeys";
import {
  buildAssetReviewStateById,
  buildFolderDirtyByPath,
  buildFolderReviewStatusByPath,
  buildVisibleTreeEntries,
  deriveMessageTone,
} from "../../lib/workspace/projectAssetsDerived";
import { buildModelBuilderHref } from "../../lib/workspace/projectRouting";
import { asRelativePath, buildTreeEntries, collectFolderPathsFromRelativePaths, folderChain } from "../../lib/workspace/tree";
import { ProjectAssetsFooterActions } from "./project-assets/ProjectAssetsFooterActions";
import { ProjectAssetsImportModal } from "./project-assets/ProjectAssetsImportModal";
import { ProjectAssetsStatusOverlay } from "./project-assets/ProjectAssetsStatusOverlay";
import { ProjectAssetsTreeSidebar } from "./project-assets/ProjectAssetsTreeSidebar";
import { useProjectNavigationGuard } from "./ProjectNavigationContext";

const PROJECT_MULTILABEL_STORAGE_KEY = "pixel-sheriff:project-multilabel:v1";
const PROJECT_STATUS_REFRESH_EVENT = "pixel-sheriff:project-status-refresh";

type FolderReviewStatus = "all_labeled" | "has_unlabeled" | "empty";
type WorkspaceAnnotationMode = "labels" | "bbox" | "segmentation";
type NewProjectTaskType = "classification_single" | "bbox" | "segmentation";

function projectTaskTypeToAnnotationMode(taskType: ProjectTaskType | null | undefined): WorkspaceAnnotationMode {
  if (taskType === "bbox") return "bbox";
  if (taskType === "segmentation") return "segmentation";
  return "labels";
}

function annotationModeToNewProjectTaskType(mode: WorkspaceAnnotationMode): NewProjectTaskType {
  if (mode === "bbox") return "bbox";
  if (mode === "segmentation") return "segmentation";
  return "classification_single";
}

export default function ProjectAssetsWorkspace() {
  const router = useRouter();
  const params = useParams<{ projectId: string }>();
  const routeProjectId = decodeURIComponent(params?.projectId ?? "");
  const { guardedNavigate, setHasUnsavedDrafts } = useProjectNavigationGuard();
  const { data: projects, refetch: refetchProjects } = useProject();
  const selectedProjectId = routeProjectId.trim() ? routeProjectId : null;
  const [assetIndex, setAssetIndex] = useState(0);
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isCreatingModel, setIsCreatingModel] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const { projectMultiLabelSettings, setProjectMultiLabelSettings } = useProjectMultiLabelSettings(PROJECT_MULTILABEL_STORAGE_KEY);
  const [message, setMessage] = useState<string | null>(null);
  const [annotationMode, setAnnotationMode] = useState<WorkspaceAnnotationMode>("labels");
  const [importNewProjectTaskType, setImportNewProjectTaskType] = useState<NewProjectTaskType>("classification_single");
  const [geometryCategoryId, setGeometryCategoryId] = useState<number | null>(null);
  const [hoveredGeometryObjectId, setHoveredGeometryObjectId] = useState<string | null>(null);
  const [selectedTreeFolderPath, setSelectedTreeFolderPath] = useState<string | null>(null);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );
  const projectAnnotationMode = useMemo(
    () => projectTaskTypeToAnnotationMode(selectedProject?.task_type),
    [selectedProject?.task_type],
  );
  useEffect(() => {
    if (annotationMode === projectAnnotationMode) return;
    setAnnotationMode(projectAnnotationMode);
  }, [annotationMode, projectAnnotationMode]);
  const multiLabelEnabled = selectedProjectId ? Boolean(projectMultiLabelSettings[selectedProjectId]) : false;

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(selectedProjectId);
  const { data: labels, refetch: refetchLabels } = useLabels(selectedProjectId);
  const availableAssetIds = useMemo(() => assets.map((asset) => asset.id), [assets]);
  const selectedProjectName = selectedProject?.name ?? "No project selected";
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
    importExistingProjectId,
    setImportExistingProjectId,
    importNewProjectName,
    setImportNewProjectName,
    importFolderName,
    setImportFolderName,
    setImportModeWithDefaults,
    setImportExistingProjectWithDefaults,
    setImportExistingFolderWithDefaults,
    setImportFolderOptionsByProject,
    selectedImportExistingFolder,
    setSelectedImportExistingFolder,
    setImportProgress,
    importExistingFolderOptions,
    importValidation,
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
  const visibleTreeEntries = useMemo(() => buildVisibleTreeEntries(treeEntries, collapsedFolders), [collapsedFolders, treeEntries]);
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
  const labelNameById = useMemo(() => {
    const map = new Map<number, string>();
    for (const label of allLabelRows) map.set(label.id, label.name);
    return map;
  }, [allLabelRows]);

  const safeAssetIndex = Math.min(assetIndex, Math.max(assetRows.length - 1, 0));
  const currentAsset = assetRows[safeAssetIndex] ?? null;
  const viewerAsset = currentAsset
    ? {
        id: currentAsset.id,
        uri: resolveAssetUri(currentAsset.uri),
        width: currentAsset.width,
        height: currentAsset.height,
      }
    : null;

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
    availableAssetIds,
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
    currentObjects,
    selectedObjectId,
    setSelectedObjectId,
    setCurrentImageBasis,
    handleToggleLabel,
    clearSelectedLabels,
    assignSelectedGeometryCategory,
    upsertGeometryObject,
    deleteSelectedGeometryObject,
    handleSubmit,
    resetAnnotationWorkflow,
  } = annotationWorkflow;
  const geometryObjectRows = useMemo(
    () =>
      currentObjects.map((object) => ({
        id: object.id,
        kind: object.kind,
        categoryId: object.category_id,
        categoryName: labelNameById.get(object.category_id) ?? `#${object.category_id}`,
      })),
    [currentObjects, labelNameById],
  );
  const deleteWorkflow = useDeleteWorkflow({
    selectedProjectId,
    selectedDatasetName: selectedProjectName,
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
    setProjectMultiLabelSettings,
    setImportExistingProjectId,
    setSelectedImportExistingFolder,
    setImportFolderOptionsByProject,
    setMessage,
    resetAnnotationWorkflow,
    setEditMode,
    refetchAssets,
    refetchProjects,
    onProjectDeleted: () => router.replace("/projects"),
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

  const assetReviewStateById = useMemo(
    () =>
      buildAssetReviewStateById({
        orderedAssetRows,
        pendingAnnotations,
        annotationByAssetId,
      }),
    [annotationByAssetId, orderedAssetRows, pendingAnnotations],
  );

  const pageStatuses = useMemo(
    () => assetRows.map((asset) => assetReviewStateById.get(asset.id)?.status ?? "unlabeled"),
    [assetReviewStateById, assetRows],
  );
  const pageDirtyFlags = useMemo(
    () => assetRows.map((asset) => Boolean(assetReviewStateById.get(asset.id)?.isDirty)),
    [assetReviewStateById, assetRows],
  );
  const selectedFolderAssetCount = useMemo(() => {
    if (!selectedTreeFolderPath) return 0;
    return treeBuild.folderAssetIds[selectedTreeFolderPath]?.length ?? 0;
  }, [selectedTreeFolderPath, treeBuild.folderAssetIds]);
  const messageTone = useMemo(() => deriveMessageTone(message), [message]);
  const labeledImageCount = useMemo(() => {
    const labeledAssetIds = new Set<string>();
    for (const annotation of annotations) {
      if (annotation.status !== "unlabeled") labeledAssetIds.add(annotation.asset_id);
    }
    return labeledAssetIds.size;
  }, [annotations]);

  useEffect(() => {
    setHasUnsavedDrafts(pendingCount > 0);
  }, [pendingCount, setHasUnsavedDrafts]);

  useEffect(() => () => setHasUnsavedDrafts(false), [setHasUnsavedDrafts]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!selectedProjectId) return;
    window.dispatchEvent(
      new CustomEvent(PROJECT_STATUS_REFRESH_EVENT, {
        detail: {
          projectId: selectedProjectId,
          labeledImageCount,
          classCount: labels.length,
        },
      }),
    );
  }, [labeledImageCount, labels.length, selectedProjectId]);

  const folderReviewStatusByPath = useMemo(
    () => buildFolderReviewStatusByPath({ folderAssetIds: treeBuild.folderAssetIds, assetReviewStateById }) as Record<string, FolderReviewStatus>,
    [assetReviewStateById, treeBuild.folderAssetIds],
  );
  const folderDirtyByPath = useMemo(
    () => buildFolderDirtyByPath({ folderAssetIds: treeBuild.folderAssetIds, assetReviewStateById }),
    [assetReviewStateById, treeBuild.folderAssetIds],
  );

  useEffect(() => {
    setAssetIndex((previous) => Math.min(previous, Math.max(assetRows.length - 1, 0)));
  }, [assetRows.length]);

  useEffect(() => {
    if (!message) return;
    const timeout = window.setTimeout(() => setMessage(null), 6000);
    return () => window.clearTimeout(timeout);
  }, [message]);

  useEffect(() => {
    if (annotationMode === "labels") return;
    if (selectedObjectId) {
      const selectedObject = currentObjects.find((item) => item.id === selectedObjectId);
      if (selectedObject) {
        setGeometryCategoryId(selectedObject.category_id);
        return;
      }
    }
    if (geometryCategoryId && activeLabelRows.some((label) => label.id === geometryCategoryId)) {
      return;
    }
    setGeometryCategoryId(activeLabelRows[0]?.id ?? null);
  }, [activeLabelRows, annotationMode, currentObjects, geometryCategoryId, selectedObjectId]);

  useEffect(() => {
    setHoveredGeometryObjectId(null);
  }, [currentAsset?.id, annotationMode]);

  useEffect(() => {
    setAnnotationMode(projectTaskTypeToAnnotationMode(selectedProject?.task_type));
    setGeometryCategoryId(null);
    setHoveredGeometryObjectId(null);
    setSelectedTreeFolderPath(null);
    setCollapsedFolders({});
    resetDeleteWorkflow();
    setAssetIndex(0);
    resetAnnotationWorkflow();
    setEditMode(false);
    setMessage(null);
  }, [routeProjectId, selectedProject?.task_type]);

  function handlePrevAsset() {
    setAssetIndex((previous) => (previous <= 0 ? 0 : previous - 1));
  }

  function handleNextAsset() {
    setAssetIndex((previous) => (previous >= assetRows.length - 1 ? previous : previous + 1));
  }

  useWorkspaceHotkeys({
    activeLabelRows,
    annotationMode,
    assetRowsLength: assetRows.length,
    selectedObjectId,
    onPrev: handlePrevAsset,
    onNext: handleNextAsset,
    onLabelHotkey: (labelId, selectedHotkeyObjectId) => {
      if (annotationMode === "labels") {
        handleToggleLabel(labelId);
      } else {
        setGeometryCategoryId(labelId);
        if (selectedHotkeyObjectId) {
          assignSelectedGeometryCategory(labelId);
        }
      }
    },
  });

  function handleToggleProjectMultiLabel() {
    if (!selectedProjectId) return;
    if (projectAnnotationMode !== "labels") return;
    setProjectMultiLabelSettings((previous) => ({
      ...previous,
      [selectedProjectId]: !Boolean(previous[selectedProjectId]),
    }));
  }

  function handleChangeAnnotationMode(nextMode: WorkspaceAnnotationMode) {
    if (nextMode !== projectAnnotationMode) return;
    setAnnotationMode(nextMode);
  }

  const defaultGeometryCategoryId = geometryCategoryId ?? activeLabelRows[0]?.id ?? null;
  const effectiveSelectedLabelIds =
    annotationMode === "labels" ? selectedLabelIds : defaultGeometryCategoryId ? [defaultGeometryCategoryId] : [];

  function handleToggleLabelForCurrentMode(labelId: number) {
    if (annotationMode === "labels") {
      handleToggleLabel(labelId);
      return;
    }
    setGeometryCategoryId(labelId);
    if (selectedObjectId) {
      assignSelectedGeometryCategory(labelId);
      return;
    }
  }

  function handleSelectGeometryObject(objectId: string | null) {
    setSelectedObjectId(objectId);
    if (!objectId) return;
    const selectedObject = currentObjects.find((item) => item.id === objectId);
    if (!selectedObject) return;
    setGeometryCategoryId(selectedObject.category_id);
    setSelectedLabelIds([selectedObject.category_id]);
  }

  async function handleCreateLabel(name: string) {
    if (!selectedProjectId) {
      setMessage("Select a project before creating labels.");
      return;
    }

    try {
      setIsCreatingLabel(true);
      setMessage(null);
      const created = await createCategory(selectedProjectId, { name, display_order: allLabelRows.length });
      await refetchLabels();
      if (annotationMode !== "labels") setGeometryCategoryId(created.id);
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
    const folderName = importFolderName.trim();
    if (!importValidation.canSubmit) {
      setMessage(importValidation.filesError ?? importValidation.projectError ?? importValidation.folderError ?? "Import is not ready.");
      if (importValidation.filesError) closeImportDialog();
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
        const project = await createProject({ name: projectName, task_type: importNewProjectTaskType });
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
      const targetProject = importMode === "new" ? null : projects.find((item) => item.id === targetProjectId) ?? null;
      setAnnotationMode(projectTaskTypeToAnnotationMode(targetProject?.task_type ?? importNewProjectTaskType));
      setSelectedTreeFolderPath(null);
      setCollapsedFolders({});
      setAssetIndex(0);
      resetAnnotationWorkflow();
      setEditMode(false);
      setHasUnsavedDrafts(false);
      if (targetProjectId !== selectedProjectId) {
        router.push(`/projects/${encodeURIComponent(targetProjectId)}/datasets`);
      }
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
      setImportNewProjectTaskType(annotationModeToNewProjectTaskType(projectTaskTypeToAnnotationMode(defaultProject?.task_type)));
      openImportDialog(files, rootName, defaultProject?.id ?? "");
    };

    picker.click();
  }

  async function handleExport() {
    if (!selectedProjectId) {
      setMessage("Select a project before exporting.");
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
      anchor.download = `${selectedProjectName.replace(/[^a-zA-Z0-9-_]+/g, "_") || "dataset"}-${created.hash.slice(0, 8)}.zip`;
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

  function handleBuildModel() {
    if (!selectedProjectId) {
      setMessage("Select a project before building a model.");
      return;
    }
    guardedNavigate(() => {
      void (async () => {
        try {
          setIsCreatingModel(true);
          setMessage("Creating model draft...");
          const created = await createProjectModel(selectedProjectId, {});
          router.push(buildModelBuilderHref(selectedProjectId, created.id));
        } catch (error) {
          if (error instanceof ApiError) {
            setMessage(`Build model failed: ${error.responseBody ?? error.message}`);
          } else {
            setMessage(error instanceof Error ? `Build model failed: ${error.message}` : "Build model failed.");
          }
        } finally {
          setIsCreatingModel(false);
        }
      })();
    });
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

  const headerTitle = selectedTreeFolderPath ? `${selectedProjectName} / ${selectedTreeFolderPath}` : selectedProjectName;

  return (
    <main className="workspace-shell">
      <section className="workspace-frame">
        <header className="workspace-header">
          <div className="workspace-header-cell">Project Assets</div>
          <div className="workspace-header-cell workspace-header-title">{headerTitle}</div>
          <div className="workspace-header-cell workspace-header-actions" aria-label="Toolbar">
            <span />
            <span />
            <span />
            <span />
          </div>
        </header>

        <div className="workspace-body">
          <ProjectAssetsTreeSidebar
            selectedTreeFolderPath={selectedTreeFolderPath}
            bulkDeleteMode={bulkDeleteMode}
            isDeletingAssets={isDeletingAssets}
            selectedProjectId={selectedProjectId}
            selectedDeleteAssetIdsLength={selectedDeleteAssetIds.length}
            selectedFolderAssetCount={selectedFolderAssetCount}
            visibleTreeEntries={visibleTreeEntries}
            collapsedFolders={collapsedFolders}
            folderReviewStatusByPath={folderReviewStatusByPath}
            folderDirtyByPath={folderDirtyByPath}
            selectedDeleteAssets={selectedDeleteAssets}
            currentAssetId={currentAsset?.id ?? null}
            assetReviewStateById={assetReviewStateById}
            onCollapseAllFolders={handleCollapseAllFolders}
            onExpandAllFolders={handleExpandAllFolders}
            onSelectFolderScope={handleSelectFolderScope}
            onToggleBulkDeleteMode={handleToggleBulkDeleteMode}
            onSelectAllDeleteScope={handleSelectAllDeleteScope}
            onClearDeleteSelection={handleClearDeleteSelection}
            onDeleteSelectedAssets={handleDeleteSelectedAssets}
            onDeleteSelectedFolder={handleDeleteSelectedFolder}
            onToggleFolderCollapsed={handleToggleFolderCollapsed}
            onDeleteFolderPath={handleDeleteFolderPath}
            onToggleDeleteSelection={handleToggleDeleteSelection}
            onSelectTreeAsset={handleSelectTreeAsset}
          />

          <Viewer
            currentAsset={viewerAsset}
            totalAssets={assetRows.length}
            currentIndex={safeAssetIndex}
            pageStatuses={pageStatuses}
            pageDirtyFlags={pageDirtyFlags}
            annotationMode={annotationMode}
            geometryObjects={currentObjects}
            selectedObjectId={selectedObjectId}
            hoveredObjectId={hoveredGeometryObjectId}
            defaultCategoryId={defaultGeometryCategoryId}
            onSelectObject={handleSelectGeometryObject}
            onHoverObject={setHoveredGeometryObjectId}
            onUpsertObject={upsertGeometryObject}
            onDeleteSelectedObject={deleteSelectedGeometryObject}
            onImageBasisChange={setCurrentImageBasis}
            onSelectIndex={setAssetIndex}
            onPrev={handlePrevAsset}
            onNext={handleNextAsset}
          />
          <LabelPanel
            labels={activeLabelRows}
            allLabels={allLabelRows}
            selectedLabelIds={effectiveSelectedLabelIds}
            onToggleLabel={handleToggleLabelForCurrentMode}
            onClearLabels={clearSelectedLabels}
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
            annotationMode={annotationMode}
            projectMode={projectAnnotationMode}
            onChangeAnnotationMode={handleChangeAnnotationMode}
            selectedObjectId={selectedObjectId}
            geometryObjectCount={currentObjects.length}
            geometryObjects={geometryObjectRows}
            hoveredObjectId={hoveredGeometryObjectId}
            onHoverObject={setHoveredGeometryObjectId}
            onSelectObject={handleSelectGeometryObject}
            onDeleteSelectedObject={deleteSelectedGeometryObject}
          />
        </div>

        <ProjectAssetsFooterActions
          isImporting={isImporting}
          isExporting={isExporting}
          selectedProjectId={selectedProjectId}
          isDeletingAssets={isDeletingAssets}
          hasCurrentAsset={Boolean(currentAsset)}
          bulkDeleteMode={bulkDeleteMode}
          selectedDeleteAssetIdsLength={selectedDeleteAssetIds.length}
          selectedTreeFolderPath={selectedTreeFolderPath}
          selectedFolderAssetCount={selectedFolderAssetCount}
          isDeletingProject={isDeletingProject}
          isCreatingModel={isCreatingModel}
          onImport={handleImport}
          onExport={handleExport}
          onDeleteCurrentAsset={handleDeleteCurrentAsset}
          onToggleBulkDeleteMode={handleToggleBulkDeleteMode}
          onDeleteSelectedAssets={handleDeleteSelectedAssets}
          onDeleteSelectedFolder={handleDeleteSelectedFolder}
          onDeleteCurrentProject={handleDeleteCurrentProject}
          onBuildModel={handleBuildModel}
        />
      </section>
      <ProjectAssetsStatusOverlay
        message={message}
        messageTone={messageTone}
        importFailures={importFailures}
        onDismissMessage={() => setMessage(null)}
      />
      <ProjectAssetsImportModal
        open={importDialog.open}
        filesCount={importDialog.files.length}
        isImporting={isImporting}
        projects={projects}
        importMode={importMode}
        importNewProjectName={importNewProjectName}
        importExistingProjectId={importExistingProjectId}
        importExistingFolderOptions={importExistingFolderOptions}
        selectedImportExistingFolder={selectedImportExistingFolder}
        importFolderName={importFolderName}
        importSourceFolderName={importDialog.sourceFolderName}
        importNewProjectTaskType={importNewProjectTaskType}
        importValidation={importValidation}
        importProgressView={importProgressView}
        onSetImportModeWithDefaults={setImportModeWithDefaults}
        onSetImportNewProjectName={setImportNewProjectName}
        onSetImportExistingProjectWithDefaults={setImportExistingProjectWithDefaults}
        onSetImportNewProjectTaskType={setImportNewProjectTaskType}
        onSetImportExistingFolderWithDefaults={setImportExistingFolderWithDefaults}
        onSetImportFolderName={setImportFolderName}
        onClose={closeImportDialog}
        onConfirm={confirmImportFromDialog}
      />
    </main>
  );
}
