"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";

import { LabelPanel } from "../LabelPanel";
import { Viewer } from "../Viewer";
import {
  ApiError,
  listDatasetVersions,
  createProjectModel,
  createCategory,
  deleteCategory,
  createProject,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  type Annotation,
  type AnnotationStatus,
  type ProjectTaskType,
  type TaskKind,
} from "../../lib/api";
import { useAnnotationWorkflow } from "../../lib/hooks/useAnnotationWorkflow";
import { useAssets } from "../../lib/hooks/useAssets";
import { useDeleteWorkflow } from "../../lib/hooks/useDeleteWorkflow";
import { buildTargetRelativePath, isImageCandidate, useImportWorkflow } from "../../lib/hooks/useImportWorkflow";
import { useLabels } from "../../lib/hooks/useLabels";
import { useProjectAssetsTreeState } from "../../lib/hooks/useProjectAssetsTreeState";
import { useProject } from "../../lib/hooks/useProject";
import { useWorkspaceSuggestions } from "../../lib/hooks/useWorkspaceSuggestions";
import { useWorkspaceTaskState } from "../../lib/hooks/useWorkspaceTaskState";
import { useTasks } from "../../lib/hooks/useTasks";
import { useWorkspaceHotkeys } from "../../lib/hooks/useWorkspaceHotkeys";
import {
  buildAssetReviewStateById,
  buildFolderDirtyByPath,
  buildFolderReviewStatusByPath,
  deriveMessageTone,
} from "../../lib/workspace/projectAssetsDerived";
import { buildModelBuilderHref } from "../../lib/workspace/projectRouting";
import { collectFolderPathsFromRelativePaths } from "../../lib/workspace/tree";
import { ProjectAssetsFooterActions } from "./project-assets/ProjectAssetsFooterActions";
import { ProjectAssetsImportModal } from "./project-assets/ProjectAssetsImportModal";
import { ProjectAssetsStatusOverlay } from "./project-assets/ProjectAssetsStatusOverlay";
import { ProjectAssetsTaskModal } from "./project-assets/ProjectAssetsTaskModal";
import { ProjectAssetsTreeSidebar } from "./project-assets/ProjectAssetsTreeSidebar";
import { useProjectNavigationGuard } from "./ProjectNavigationContext";

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

function taskKindToAnnotationMode(taskKind: TaskKind | null | undefined): WorkspaceAnnotationMode {
  if (taskKind === "bbox") return "bbox";
  if (taskKind === "segmentation") return "segmentation";
  return "labels";
}

export default function ProjectAssetsWorkspace() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const params = useParams<{ projectId: string }>();
  const routeProjectId = decodeURIComponent(params?.projectId ?? "");
  const { guardedNavigate, setHasUnsavedDrafts } = useProjectNavigationGuard();
  const { data: projects, refetch: refetchProjects } = useProject();
  const selectedProjectId = routeProjectId.trim() ? routeProjectId : null;
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [deletingLabelId, setDeletingLabelId] = useState<string | null>(null);
  const [isCreatingModel, setIsCreatingModel] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [annotationMode, setAnnotationMode] = useState<WorkspaceAnnotationMode>("labels");
  const [importNewProjectTaskType, setImportNewProjectTaskType] = useState<NewProjectTaskType>("classification_single");
  const [geometryCategoryId, setGeometryCategoryId] = useState<string | null>(null);
  const [hoveredGeometryObjectId, setHoveredGeometryObjectId] = useState<string | null>(null);

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );
  const { data: tasks, refetch: refetchTasks } = useTasks(selectedProjectId);
  const requestedTaskId = searchParams.get("taskId");
  function updateTaskInUrl(taskId: string) {
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.set("taskId", taskId);
    router.replace(`/projects/${encodeURIComponent(routeProjectId)}/datasets?${nextParams.toString()}`);
  }
  const taskState = useWorkspaceTaskState({
    selectedProjectId,
    selectedProjectDefaultTaskId: selectedProject?.default_task_id,
    tasks,
    requestedTaskId,
    syncTaskInUrl: updateTaskInUrl,
    refetchTasks,
    setMessage,
  });
  const selectedTask = taskState.selectedTask;
  const selectedTaskId = taskState.selectedTaskId;
  const projectAnnotationMode = useMemo(() => taskKindToAnnotationMode(selectedTask?.kind), [selectedTask?.kind]);

  useEffect(() => {
    if (annotationMode === projectAnnotationMode) return;
    setAnnotationMode(projectAnnotationMode);
  }, [annotationMode, projectAnnotationMode]);
  const multiLabelEnabled = selectedTask?.label_mode === "multi_label";

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(
    selectedProjectId,
    selectedTaskId,
  );
  const { data: labels, refetch: refetchLabels } = useLabels(selectedProjectId, selectedTaskId);
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

  const treeState = useProjectAssetsTreeState({
    assets,
    annotations,
    assetById,
    projectAnnotationMode,
  });
  const allLabelRows = labels.map((label) => ({
    id: label.id,
    name: label.name,
    isActive: label.is_active,
    displayOrder: label.display_order,
  }));
  const activeLabelRows = allLabelRows.filter((label) => label.isActive).sort((a, b) => a.displayOrder - b.displayOrder);
  const labelNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const label of allLabelRows) map.set(label.id, label.name);
    return map;
  }, [allLabelRows]);
  const suggestionState = useWorkspaceSuggestions({
    selectedProjectId,
    currentAssetId: treeState.currentAsset?.id ?? null,
    setMessage,
  });

  const viewerAsset = useMemo(
    () =>
      treeState.currentAsset
        ? {
            id: treeState.currentAsset.id,
            uri: resolveAssetUri(treeState.currentAsset.uri),
            width: treeState.currentAsset.width,
            height: treeState.currentAsset.height,
          }
        : null,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [treeState.currentAsset?.id, treeState.currentAsset?.uri, treeState.currentAsset?.width, treeState.currentAsset?.height],
  );

  const annotationWorkflow = useAnnotationWorkflow({
    selectedProjectId,
    selectedTaskId,
    currentAsset: treeState.currentAsset,
    availableAssetIds,
    annotationByAssetId: treeState.annotationByAssetId,
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
    selectedTreeFolderPath: treeState.selectedTreeFolderPath,
    setSelectedTreeFolderPath: treeState.setSelectedTreeFolderPath,
    currentAsset: treeState.currentAsset,
    assetRows: treeState.assetRows,
    assets,
    treeFolderAssetIds: treeState.treeBuild.folderAssetIds,
    assetById,
    annotations,
    setAnnotations,
    pendingAnnotations,
    setPendingAnnotations: annotationWorkflow.setPendingAnnotations,
    setAssetIndex: treeState.setAssetIndex,
    setCollapsedFolders: treeState.setCollapsedFolders,
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
        orderedAssetRows: treeState.orderedAssetRows,
        pendingAnnotations,
        annotationByAssetId: treeState.annotationByAssetId,
      }),
    [pendingAnnotations, treeState.annotationByAssetId, treeState.orderedAssetRows],
  );

  const pageStatuses = useMemo(
    () => treeState.assetRows.map((asset) => assetReviewStateById.get(asset.id)?.status ?? "unlabeled"),
    [assetReviewStateById, treeState.assetRows],
  );
  const pageDirtyFlags = useMemo(
    () => treeState.assetRows.map((asset) => Boolean(assetReviewStateById.get(asset.id)?.isDirty)),
    [assetReviewStateById, treeState.assetRows],
  );
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
    () => buildFolderReviewStatusByPath({ folderAssetIds: treeState.treeBuild.folderAssetIds, assetReviewStateById }) as Record<string, FolderReviewStatus>,
    [assetReviewStateById, treeState.treeBuild.folderAssetIds],
  );
  const folderDirtyByPath = useMemo(
    () => buildFolderDirtyByPath({ folderAssetIds: treeState.treeBuild.folderAssetIds, assetReviewStateById }),
    [assetReviewStateById, treeState.treeBuild.folderAssetIds],
  );

  useEffect(() => {
    treeState.setAssetIndex((previous) => Math.min(previous, Math.max(treeState.assetRows.length - 1, 0)));
  }, [treeState.assetRows.length, treeState.setAssetIndex]);

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
  }, [treeState.currentAsset?.id, annotationMode]);

  useEffect(() => {
    setAnnotationMode(taskKindToAnnotationMode(selectedTask?.kind));
    setGeometryCategoryId(null);
    setHoveredGeometryObjectId(null);
    resetDeleteWorkflow();
    treeState.resetTreeState();
    resetAnnotationWorkflow();
    setEditMode(false);
    setMessage(null);
  }, [routeProjectId, selectedTask?.kind]);

  useWorkspaceHotkeys({
    activeLabelRows,
    annotationMode,
    assetRowsLength: treeState.assetRows.length,
    selectedObjectId,
    onPrev: treeState.handlePrevAsset,
    onNext: treeState.handleNextAsset,
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

  function handleChangeAnnotationMode(nextMode: WorkspaceAnnotationMode) {
    if (nextMode !== projectAnnotationMode) return;
    setAnnotationMode(nextMode);
  }

  const defaultGeometryCategoryId = geometryCategoryId ?? activeLabelRows[0]?.id ?? null;
  const effectiveSelectedLabelIds =
    annotationMode === "labels" ? selectedLabelIds : defaultGeometryCategoryId ? [defaultGeometryCategoryId] : [];

  function handleToggleLabelForCurrentMode(labelId: string) {
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
    if (!selectedProjectId || !selectedTaskId) {
      setMessage("Select a project before creating labels.");
      return;
    }
    if (taskState.isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }

    try {
      setIsCreatingLabel(true);
      setMessage(null);
      const created = await createCategory(selectedProjectId, { task_id: selectedTaskId, name, display_order: allLabelRows.length });
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
    changes: Array<{ id: string; name: string; isActive: boolean; displayOrder: number }>,
  ) {
    if (taskState.isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }
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

  async function handleDeleteLabel(labelId: string, labelName: string) {
    if (taskState.isTaskLabelsLocked) {
      setMessage("This task is locked because dataset versions already exist. Create a new task to change labels.");
      return;
    }
    if (!window.confirm(`Delete label "${labelName}"? This cannot be undone.`)) return;
    try {
      setDeletingLabelId(labelId);
      await deleteCategory(labelId);
      await refetchLabels();
      setSelectedLabelIds((previous) => previous.filter((id) => id !== labelId));
      if (geometryCategoryId === labelId) {
        setGeometryCategoryId(null);
      }
      setMessage(`Deleted label "${labelName}".`);
    } catch (error) {
      if (error instanceof ApiError && error.responseBody) {
        try {
          const body = JSON.parse(error.responseBody) as { error?: { code?: string; message?: string; details?: { annotation_references?: number } } };
          const code = body.error?.code;
          const msg = body.error?.message;
          const refs = body.error?.details?.annotation_references;
          if (code === "category_in_use" && typeof refs === "number") {
            setMessage(`Cannot delete "${labelName}": ${refs} annotation${refs === 1 ? "" : "s"} still reference this class. Clear those annotations and submit before deleting.`);
          } else if (msg) {
            setMessage(`Failed to delete label: ${msg}`);
          } else {
            setMessage(`Failed to delete label: ${error.message}`);
          }
        } catch {
          setMessage(`Failed to delete label: ${error.message}`);
        }
      } else {
        setMessage(error instanceof Error ? `Failed to delete label: ${error.message}` : "Failed to delete label.");
      }
    } finally {
      setDeletingLabelId(null);
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
      treeState.resetTreeState();
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

  function handleApplySuggestedLabel(categoryId: string) {
    clearSelectedLabels();
    handleToggleLabelForCurrentMode(categoryId);
  }

  const headerTitle = treeState.selectedTreeFolderPath ? `${selectedProjectName} / ${treeState.selectedTreeFolderPath}` : selectedProjectName;

  return (
    <main className="workspace-shell">
      <section className="workspace-frame">
        <header className="workspace-header">
          <div className="workspace-header-cell">Project Assets</div>
          <div className="workspace-header-cell workspace-header-title">{headerTitle}</div>
          <div className="workspace-header-cell workspace-header-actions" aria-label="Toolbar">
            <label htmlFor="task-selector" className="sr-only">
              Task
            </label>
            <select
              id="task-selector"
              value={selectedTaskId ?? ""}
              onChange={(event) => taskState.handleSelectTask(event.target.value)}
              disabled={!selectedProjectId || tasks.length === 0}
              className="workspace-task-select"
              title={selectedTask?.kind ? `${selectedTask.kind}${selectedTask.label_mode ? ` (${selectedTask.label_mode})` : ""}` : "Task"}
            >
              {tasks.length === 0 ? <option value="">No tasks</option> : null}
              {tasks.map((task) => (
                <option key={task.id} value={task.id}>
                  {task.name} [{task.kind}]
                </option>
              ))}
            </select>
            <button
              type="button"
              className="primary-button workspace-task-create-button"
              onClick={taskState.handleOpenCreateTaskModal}
              disabled={!selectedProjectId}
            >
              + New Task
            </button>
          </div>
        </header>

        <div className="workspace-body">
          <ProjectAssetsTreeSidebar
            selectedTreeFolderPath={treeState.selectedTreeFolderPath}
            bulkDeleteMode={bulkDeleteMode}
            isDeletingAssets={isDeletingAssets}
            selectedProjectId={selectedProjectId}
            selectedDeleteAssetIdsLength={selectedDeleteAssetIds.length}
            selectedFolderAssetCount={treeState.selectedFolderAssetCount}
            visibleTreeEntries={treeState.visibleTreeEntries}
            collapsedFolders={treeState.collapsedFolders}
            folderReviewStatusByPath={folderReviewStatusByPath}
            folderDirtyByPath={folderDirtyByPath}
            selectedDeleteAssets={selectedDeleteAssets}
            currentAssetId={treeState.currentAsset?.id ?? null}
            assetReviewStateById={assetReviewStateById}
            onCollapseAllFolders={treeState.handleCollapseAllFolders}
            onExpandAllFolders={treeState.handleExpandAllFolders}
            onSelectFolderScope={treeState.handleSelectFolderScope}
            onToggleBulkDeleteMode={handleToggleBulkDeleteMode}
            onSelectAllDeleteScope={handleSelectAllDeleteScope}
            onClearDeleteSelection={handleClearDeleteSelection}
            onDeleteSelectedAssets={handleDeleteSelectedAssets}
            onDeleteSelectedFolder={handleDeleteSelectedFolder}
            onToggleFolderCollapsed={treeState.handleToggleFolderCollapsed}
            onDeleteFolderPath={handleDeleteFolderPath}
            onToggleDeleteSelection={handleToggleDeleteSelection}
            onSelectTreeAsset={treeState.handleSelectTreeAsset}
            filterStatus={treeState.filterStatus}
            filterCategoryId={treeState.filterCategoryId}
            filterLabelRows={activeLabelRows}
            onChangeFilterStatus={treeState.setFilterStatus}
            onChangeFilterCategoryId={treeState.setFilterCategoryId}
          />

          <Viewer
            currentAsset={viewerAsset}
            totalAssets={treeState.assetRows.length}
            currentIndex={treeState.safeAssetIndex}
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
            onSelectIndex={treeState.setAssetIndex}
            onPrev={treeState.handlePrevAsset}
            onNext={treeState.handleNextAsset}
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
            onDeleteLabel={handleDeleteLabel}
            deletingLabelId={deletingLabelId}
            labelsLocked={taskState.isTaskLabelsLocked}
            canSubmit={canSubmit}
            multiLabelEnabled={multiLabelEnabled}
            onToggleMultiLabel={() => setMessage("Classification label mode is controlled by the selected task.")}
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
            activeDeploymentName={suggestionState.activeDeployment?.name ?? null}
            activeDeploymentDevicePreference={suggestionState.activeDeployment?.device_preference ?? null}
            lastInferenceDeviceSelected={suggestionState.lastInferenceDeviceSelected}
            suggestionPredictions={suggestionState.suggestionPredictions}
            isSuggesting={suggestionState.isSuggesting}
            hasActiveDeployment={Boolean(suggestionState.activeDeployment)}
            onSuggest={suggestionState.handleSuggest}
            onApplySuggestedLabel={handleApplySuggestedLabel}
          />
        </div>

        <ProjectAssetsFooterActions
          isImporting={isImporting}
          selectedProjectId={selectedProjectId}
          isDeletingAssets={isDeletingAssets}
          hasCurrentAsset={Boolean(treeState.currentAsset)}
          bulkDeleteMode={bulkDeleteMode}
          selectedDeleteAssetIdsLength={selectedDeleteAssetIds.length}
          selectedTreeFolderPath={treeState.selectedTreeFolderPath}
          selectedFolderAssetCount={treeState.selectedFolderAssetCount}
          isDeletingProject={isDeletingProject}
          isCreatingModel={isCreatingModel}
          onImport={handleImport}
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
      <ProjectAssetsTaskModal
        open={taskState.isTaskModalOpen}
        newTaskName={taskState.newTaskName}
        newTaskKind={taskState.newTaskKind}
        newTaskLabelMode={taskState.newTaskLabelMode}
        isCreatingTask={taskState.isCreatingTask}
        onSetNewTaskName={taskState.setNewTaskName}
        onSetNewTaskKind={taskState.setNewTaskKind}
        onSetNewTaskLabelMode={taskState.setNewTaskLabelMode}
        onClose={() => taskState.setIsTaskModalOpen(false)}
        onCreate={() => void taskState.handleCreateTask()}
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
