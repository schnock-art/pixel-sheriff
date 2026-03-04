"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter, useSearchParams } from "next/navigation";

import { LabelPanel } from "../LabelPanel";
import { Viewer } from "../Viewer";
import {
  ApiError,
  listDeployments,
  listDatasetVersions,
  predict,
  createProjectModel,
  createCategory,
  deleteCategory,
  createTask,
  createProject,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  type Annotation,
  type TaskKind,
  type ProjectTaskType,
} from "../../lib/api";
import { useAnnotationWorkflow } from "../../lib/hooks/useAnnotationWorkflow";
import { useAssets } from "../../lib/hooks/useAssets";
import { useDeleteWorkflow } from "../../lib/hooks/useDeleteWorkflow";
import { buildTargetRelativePath, isImageCandidate, useImportWorkflow } from "../../lib/hooks/useImportWorkflow";
import { useLabels } from "../../lib/hooks/useLabels";
import { useProject } from "../../lib/hooks/useProject";
import { useTasks } from "../../lib/hooks/useTasks";
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

const PROJECT_STATUS_REFRESH_EVENT = "pixel-sheriff:project-status-refresh";
const PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX = "pixel-sheriff:project-active-task:v1:";

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
  const [assetIndex, setAssetIndex] = useState(0);
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [deletingLabelId, setDeletingLabelId] = useState<string | null>(null);
  const [isTaskLabelsLocked, setIsTaskLabelsLocked] = useState(false);
  const [isCreatingModel, setIsCreatingModel] = useState(false);
  const [isCreatingTask, setIsCreatingTask] = useState(false);
  const [editMode, setEditMode] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [annotationMode, setAnnotationMode] = useState<WorkspaceAnnotationMode>("labels");
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  const [importNewProjectTaskType, setImportNewProjectTaskType] = useState<NewProjectTaskType>("classification_single");
  const [isTaskModalOpen, setIsTaskModalOpen] = useState(false);
  const [newTaskName, setNewTaskName] = useState("");
  const [newTaskKind, setNewTaskKind] = useState<TaskKind>("classification");
  const [newTaskLabelMode, setNewTaskLabelMode] = useState<"single_label" | "multi_label">("single_label");
  const [geometryCategoryId, setGeometryCategoryId] = useState<string | null>(null);
  const [hoveredGeometryObjectId, setHoveredGeometryObjectId] = useState<string | null>(null);
  const [selectedTreeFolderPath, setSelectedTreeFolderPath] = useState<string | null>(null);
  const [collapsedFolders, setCollapsedFolders] = useState<Record<string, boolean>>({});

  const selectedProject = useMemo(
    () => projects.find((project) => project.id === selectedProjectId) ?? null,
    [projects, selectedProjectId],
  );
  const { data: tasks, refetch: refetchTasks } = useTasks(selectedProjectId);
  const selectedTask = useMemo(() => tasks.find((task) => task.id === selectedTaskId) ?? null, [tasks, selectedTaskId]);
  const projectAnnotationMode = useMemo(() => taskKindToAnnotationMode(selectedTask?.kind), [selectedTask?.kind]);
  const requestedTaskId = searchParams.get("taskId");
  useEffect(() => {
    if (!selectedProjectId) {
      setSelectedTaskId(null);
      return;
    }
    if (tasks.length === 0) {
      setSelectedTaskId(null);
      return;
    }
    const validIds = new Set(tasks.map((task) => task.id));
    const storageKey = `${PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX}${selectedProjectId}`;
    const storedTaskId =
      typeof window !== "undefined" ? window.localStorage.getItem(storageKey) : null;
    const defaultTaskId = selectedProject?.default_task_id ?? tasks.find((task) => task.is_default)?.id ?? null;
    const nextTaskId =
      [requestedTaskId, storedTaskId, defaultTaskId, tasks[0]?.id].find(
        (value): value is string => Boolean(value && validIds.has(value)),
      ) ?? null;
    if (!nextTaskId) return;
    setSelectedTaskId((previous) => (previous === nextTaskId ? previous : nextTaskId));
    if (typeof window !== "undefined") {
      window.localStorage.setItem(storageKey, nextTaskId);
    }
    if (requestedTaskId !== nextTaskId) {
      const nextParams = new URLSearchParams(searchParams.toString());
      nextParams.set("taskId", nextTaskId);
      router.replace(`/projects/${encodeURIComponent(routeProjectId)}/datasets?${nextParams.toString()}`);
    }
  }, [
    requestedTaskId,
    routeProjectId,
    router,
    searchParams,
    selectedProject?.default_task_id,
    selectedProjectId,
    tasks,
  ]);

  useEffect(() => {
    if (annotationMode === projectAnnotationMode) return;
    setAnnotationMode(projectAnnotationMode);
  }, [annotationMode, projectAnnotationMode]);
  const multiLabelEnabled = selectedTask?.label_mode === "multi_label";

  useEffect(() => {
    let active = true;
    async function loadTaskLockState() {
      if (!selectedProjectId || !selectedTaskId) {
        if (!active) return;
        setIsTaskLabelsLocked(false);
        return;
      }
      try {
        const listed = await listDatasetVersions(selectedProjectId, selectedTaskId);
        if (!active) return;
        const items = Array.isArray(listed.items) ? listed.items : [];
        setIsTaskLabelsLocked(items.length > 0);
      } catch {
        if (!active) return;
        setIsTaskLabelsLocked(false);
      }
    }
    void loadTaskLockState();
    return () => {
      active = false;
    };
  }, [selectedProjectId, selectedTaskId]);

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(
    selectedProjectId,
    selectedTaskId,
  );
  const { data: labels, refetch: refetchLabels } = useLabels(selectedProjectId, selectedTaskId);
  const [deploymentsState, setDeploymentsState] = useState<{
    active_deployment_id: string | null;
    items: Array<{ deployment_id: string; name: string; device_preference: string; status: string }>;
  }>({ active_deployment_id: null, items: [] });
  const [suggestionPredictions, setSuggestionPredictions] = useState<
    Array<{ class_id: string; class_name: string; score: number }>
  >([]);
  const [lastInferenceDeviceSelected, setLastInferenceDeviceSelected] = useState<string | null>(null);
  const [isSuggesting, setIsSuggesting] = useState(false);
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
    const map = new Map<string, string>();
    for (const label of allLabelRows) map.set(label.id, label.name);
    return map;
  }, [allLabelRows]);
  const activeDeployment = useMemo(
    () => deploymentsState.items.find((item) => item.deployment_id === deploymentsState.active_deployment_id) ?? null,
    [deploymentsState.active_deployment_id, deploymentsState.items],
  );

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
    selectedTaskId,
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

  useEffect(() => {
    let mounted = true;
    async function loadDeployments() {
      if (!selectedProjectId) {
        setDeploymentsState({ active_deployment_id: null, items: [] });
        return;
      }
      try {
        const response = await listDeployments(selectedProjectId);
        if (!mounted) return;
        setDeploymentsState(response);
      } catch {
        if (!mounted) return;
        setDeploymentsState({ active_deployment_id: null, items: [] });
      }
    }
    void loadDeployments();
    return () => {
      mounted = false;
    };
  }, [selectedProjectId]);

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
    setAnnotationMode(taskKindToAnnotationMode(selectedTask?.kind));
    setGeometryCategoryId(null);
    setHoveredGeometryObjectId(null);
    setSelectedTreeFolderPath(null);
    setCollapsedFolders({});
    resetDeleteWorkflow();
    setAssetIndex(0);
    resetAnnotationWorkflow();
    setEditMode(false);
    setMessage(null);
  }, [routeProjectId, selectedTask?.kind]);

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

  function handleChangeAnnotationMode(nextMode: WorkspaceAnnotationMode) {
    if (nextMode !== projectAnnotationMode) return;
    setAnnotationMode(nextMode);
  }

  function updateTaskInUrl(taskId: string) {
    const nextParams = new URLSearchParams(searchParams.toString());
    nextParams.set("taskId", taskId);
    router.replace(`/projects/${encodeURIComponent(routeProjectId)}/datasets?${nextParams.toString()}`);
  }

  function handleSelectTask(nextTaskId: string) {
    if (!selectedProjectId || !nextTaskId) return;
    if (!tasks.some((task) => task.id === nextTaskId)) return;
    setSelectedTaskId(nextTaskId);
    if (typeof window !== "undefined") {
      window.localStorage.setItem(`${PROJECT_ACTIVE_TASK_STORAGE_KEY_PREFIX}${selectedProjectId}`, nextTaskId);
    }
    updateTaskInUrl(nextTaskId);
  }

  function handleOpenCreateTaskModal() {
    if (!selectedProjectId) {
      setMessage("Select a project before creating tasks.");
      return;
    }
    setNewTaskName("");
    setNewTaskKind("classification");
    setNewTaskLabelMode("single_label");
    setIsTaskModalOpen(true);
  }

  async function handleCreateTask() {
    if (!selectedProjectId) {
      setMessage("Select a project before creating tasks.");
      return;
    }
    const taskName = newTaskName.trim();
    if (!taskName) {
      setMessage("Task name is required.");
      return;
    }
    const taskKind = newTaskKind;
    const labelMode = taskKind === "classification" ? newTaskLabelMode : undefined;
    try {
      setIsCreatingTask(true);
      const created = await createTask(selectedProjectId, {
        name: taskName,
        kind: taskKind,
        label_mode: labelMode,
      });
      await refetchTasks();
      handleSelectTask(created.id);
      setIsTaskModalOpen(false);
      setMessage(`Created task "${created.name}".`);
    } catch (error) {
      setMessage(error instanceof Error ? `Failed to create task: ${error.message}` : "Failed to create task.");
    } finally {
      setIsCreatingTask(false);
    }
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
    if (isTaskLabelsLocked) {
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
    if (isTaskLabelsLocked) {
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
    if (isTaskLabelsLocked) {
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
      setMessage(error instanceof Error ? `Failed to delete label: ${error.message}` : "Failed to delete label.");
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

  async function handleSuggest() {
    if (!selectedProjectId || !currentAsset) {
      setMessage("Select an image before requesting suggestions.");
      return;
    }
    try {
      setIsSuggesting(true);
      const response = await predict(selectedProjectId, {
        asset_id: currentAsset.id,
        deployment_id: null,
        top_k: 5,
      });
      setSuggestionPredictions(
        (response.predictions ?? []).map((row) => ({
          class_id: row.class_id,
          class_name: row.class_name,
          score: row.score,
        })),
      );
      setLastInferenceDeviceSelected(response.device_selected ?? null);
    } catch (error) {
      if (error instanceof ApiError && error.responseBody) {
        setMessage(`Suggest failed: ${error.responseBody}`);
      } else {
        setMessage(error instanceof Error ? `Suggest failed: ${error.message}` : "Suggest failed.");
      }
    } finally {
      setIsSuggesting(false);
    }
  }

  function handleApplySuggestedLabel(categoryId: string) {
    clearSelectedLabels();
    handleToggleLabelForCurrentMode(categoryId);
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
            <label htmlFor="task-selector" className="sr-only">
              Task
            </label>
            <select
              id="task-selector"
              value={selectedTaskId ?? ""}
              onChange={(event) => handleSelectTask(event.target.value)}
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
              onClick={handleOpenCreateTaskModal}
              disabled={!selectedProjectId}
            >
              + New Task
            </button>
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
            onDeleteLabel={handleDeleteLabel}
            deletingLabelId={deletingLabelId}
            labelsLocked={isTaskLabelsLocked}
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
            activeDeploymentName={activeDeployment?.name ?? null}
            activeDeploymentDevicePreference={activeDeployment?.device_preference ?? null}
            lastInferenceDeviceSelected={lastInferenceDeviceSelected}
            suggestionPredictions={suggestionPredictions}
            isSuggesting={isSuggesting}
            hasActiveDeployment={Boolean(activeDeployment)}
            onSuggest={handleSuggest}
            onApplySuggestedLabel={handleApplySuggestedLabel}
          />
        </div>

        <ProjectAssetsFooterActions
          isImporting={isImporting}
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
      {isTaskModalOpen ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Create Task">
          <section className="placeholder-card" style={{ maxWidth: 520, margin: "10vh auto 0", display: "grid", gap: 12 }}>
            <h3 style={{ margin: 0 }}>Create Task</h3>
            <label className="project-field">
              <span>Name</span>
              <input value={newTaskName} onChange={(event) => setNewTaskName(event.target.value)} maxLength={120} />
            </label>
            <label className="project-field">
              <span>Kind</span>
              <select value={newTaskKind} onChange={(event) => setNewTaskKind(event.target.value as TaskKind)}>
                <option value="classification">Classification</option>
                <option value="bbox">Bounding boxes</option>
                <option value="segmentation">Segmentation</option>
              </select>
            </label>
            {newTaskKind === "classification" ? (
              <label className="project-field">
                <span>Label mode</span>
                <select
                  value={newTaskLabelMode}
                  onChange={(event) => setNewTaskLabelMode(event.target.value as "single_label" | "multi_label")}
                >
                  <option value="single_label">Single label</option>
                  <option value="multi_label">Multi label</option>
                </select>
              </label>
            ) : null}
            <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
              <button type="button" className="ghost-button" onClick={() => setIsTaskModalOpen(false)} disabled={isCreatingTask}>
                Cancel
              </button>
              <button
                type="button"
                className="primary-button"
                onClick={() => void handleCreateTask()}
                disabled={isCreatingTask || !newTaskName.trim()}
              >
                {isCreatingTask ? "Creating..." : "Create Task"}
              </button>
            </div>
          </section>
        </div>
      ) : null}
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
