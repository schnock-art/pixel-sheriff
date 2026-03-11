"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";

import { LabelPanel } from "../LabelPanel";
import { Viewer } from "../Viewer";
import {
  ApiError,
  createCategory,
  deleteCategory,
  createProject,
  importVideo,
  listAssets,
  patchCategory,
  resolveAssetUri,
  uploadAsset,
  type ProjectTaskType,
  type TaskKind,
  type VideoImportPayload,
} from "../../lib/api";
import { useAnnotationWorkflow } from "../../lib/hooks/useAnnotationWorkflow";
import { useAssets } from "../../lib/hooks/useAssets";
import { useDeleteWorkflow } from "../../lib/hooks/useDeleteWorkflow";
import { useFolders } from "../../lib/hooks/useFolders";
import { buildTargetRelativePath, isImageCandidate, useImportWorkflow } from "../../lib/hooks/useImportWorkflow";
import { useLabels } from "../../lib/hooks/useLabels";
import { useProjectAssetsTreeState } from "../../lib/hooks/useProjectAssetsTreeState";
import { usePrelabels } from "../../lib/hooks/usePrelabels";
import { useSequence } from "../../lib/hooks/useSequence";
import { useSequenceNavigation } from "../../lib/hooks/useSequenceNavigation";
import { useWorkspaceSuggestions } from "../../lib/hooks/useWorkspaceSuggestions";
import { useWorkspaceHotkeys } from "../../lib/hooks/useWorkspaceHotkeys";
import {
  buildAssetReviewStateById,
  buildFolderDirtyByPath,
  buildFolderReviewStatusByPath,
  deriveMessageTone,
} from "../../lib/workspace/projectAssetsDerived";
import { collectFolderPathsFromRelativePaths } from "../../lib/workspace/tree";
import { ProjectSectionLayout } from "./project-shell/ProjectSectionLayout";
import { useProjectShell } from "./project-shell/ProjectShellContext";
import { AssetBrowser } from "./project-assets/AssetBrowser";
import { AssetFilmstrip } from "./project-assets/AssetFilmstrip";
import { AiPrelabelsPanel } from "./project-assets/AiPrelabelsPanel";
import { CanvasToolbar, type CanvasTool } from "./project-assets/CanvasToolbar";
import { ProjectAssetsImportModal } from "./project-assets/ProjectAssetsImportModal";
import { ProjectAssetsStatusOverlay } from "./project-assets/ProjectAssetsStatusOverlay";
import { SequenceThumbnailStrip } from "./project-assets/SequenceThumbnailStrip";
import { SequenceTimeline } from "./project-assets/SequenceTimeline";
import { SequenceToolbar } from "./project-assets/SequenceToolbar";
import { VideoImportModal } from "./project-assets/VideoImportModal";
import { WebcamCaptureModal } from "./project-assets/WebcamCaptureModal";
import { useProjectNavigationGuard } from "./ProjectNavigationContext";

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

function formatSequenceTimestamp(seconds: number | null | undefined): string {
  if (typeof seconds !== "number" || !Number.isFinite(seconds) || seconds < 0) return "--:--.-";
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainingSeconds.toFixed(1).padStart(4, "0")}`;
}

export default function ProjectAssetsWorkspace() {
  const router = useRouter();
  const params = useParams<{ projectId: string }>();
  const routeProjectId = decodeURIComponent(params?.projectId ?? "");
  const { guardedNavigate, setHasUnsavedDrafts } = useProjectNavigationGuard();
  const {
    project: selectedProject,
    projects,
    selectedTask,
    selectedTaskId,
    isTaskLabelsLocked,
    refetchProjects,
  } = useProjectShell();
  const selectedProjectId = routeProjectId.trim() ? routeProjectId : null;
  const [isCreatingLabel, setIsCreatingLabel] = useState(false);
  const [isSavingLabelChanges, setIsSavingLabelChanges] = useState(false);
  const [deletingLabelId, setDeletingLabelId] = useState<string | null>(null);
  const [editMode, setEditMode] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [annotationMode, setAnnotationMode] = useState<WorkspaceAnnotationMode>("labels");
  const [canvasTool, setCanvasTool] = useState<CanvasTool>("select");
  const [viewerResetToken, setViewerResetToken] = useState(0);
  const [importNewProjectTaskType, setImportNewProjectTaskType] = useState<NewProjectTaskType>("classification_single");
  const [geometryCategoryId, setGeometryCategoryId] = useState<string | null>(null);
  const [hoveredGeometryObjectId, setHoveredGeometryObjectId] = useState<string | null>(null);
  const [isVideoImportModalOpen, setIsVideoImportModalOpen] = useState(false);
  const [isVideoImporting, setIsVideoImporting] = useState(false);
  const [videoImportError, setVideoImportError] = useState<string | null>(null);
  const [isWebcamModalOpen, setIsWebcamModalOpen] = useState(false);
  const [sequencePauseSignal, setSequencePauseSignal] = useState(0);
  const projectAnnotationMode = useMemo(() => taskKindToAnnotationMode(selectedTask?.kind), [selectedTask?.kind]);

  useEffect(() => {
    if (annotationMode === projectAnnotationMode) return;
    setAnnotationMode(projectAnnotationMode);
  }, [annotationMode, projectAnnotationMode]);

  useEffect(() => {
    if (annotationMode === "bbox") {
      setCanvasTool("bbox");
      return;
    }
    if (annotationMode === "segmentation") {
      setCanvasTool("polygon");
      return;
    }
    setCanvasTool("select");
  }, [annotationMode]);

  const multiLabelEnabled = selectedTask?.label_mode === "multi_label";

  const { data: assets, annotations, setAnnotations, refetch: refetchAssets, isLoading: isAssetsLoading } = useAssets(
    selectedProjectId,
    selectedTaskId,
  );
  const { data: folders, refetch: refetchFolders, hasProcessingSequences } = useFolders(selectedProjectId);
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
    folders,
    annotations,
    assetById,
    projectAnnotationMode,
  });
  const currentSequenceId = treeState.currentAsset?.sequence_id ?? null;
  const {
    data: currentSequence,
    isLoading: isSequenceLoading,
    error: currentSequenceError,
    refetch: refetchSequence,
  } = useSequence(selectedProjectId, currentSequenceId, selectedTaskId);
  const selectSequenceAsset = useCallback(
    (assetId: string) => {
      treeState.handleSelectTreeAsset(assetId, currentSequence?.folder_path ?? treeState.currentAsset?.folder_path ?? undefined);
    },
    [currentSequence?.folder_path, treeState.currentAsset?.folder_path, treeState.handleSelectTreeAsset],
  );
  const sequenceNavigation = useSequenceNavigation({
    sequence: currentSequence,
    currentAssetId: treeState.currentAsset?.id ?? null,
    onSelectAsset: selectSequenceAsset,
    pauseSignal: sequencePauseSignal,
  });
  const isSequenceMode = Boolean(treeState.currentAsset?.sequence_id);
  const currentSequenceFrameNumber = useMemo(() => {
    if (typeof sequenceNavigation.currentFrame?.frame_index === "number") return sequenceNavigation.currentFrame.frame_index + 1;
    if (sequenceNavigation.currentIndex >= 0) return sequenceNavigation.currentIndex + 1;
    return 0;
  }, [sequenceNavigation.currentFrame?.frame_index, sequenceNavigation.currentIndex]);
  const currentSequenceFrameLabel = useMemo(
    () => `Frame ${currentSequenceFrameNumber} / ${currentSequence?.frame_count ?? sequenceNavigation.totalFrames ?? 0}`,
    [currentSequence?.frame_count, currentSequenceFrameNumber, sequenceNavigation.totalFrames],
  );
  const currentSequenceTimestampLabel = useMemo(
    () => `Time ${formatSequenceTimestamp(sequenceNavigation.currentFrame?.timestamp_seconds ?? null)}`,
    [sequenceNavigation.currentFrame?.timestamp_seconds],
  );
  const folderIdByPath = useMemo(() => new Map(folders.map((folder) => [folder.path, folder.id])), [folders]);
  const videoImportDefaultName = useMemo(
    () => `video_${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}`,
    [isVideoImportModalOpen],
  );
  const webcamDefaultName = useMemo(
    () => `webcam_${new Date().toISOString().slice(0, 19).replaceAll(":", "-")}`,
    [isWebcamModalOpen],
  );
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
    replaceGeometryObjects,
    deleteSelectedGeometryObject,
    handleSubmit,
    resetAnnotationWorkflow,
  } = annotationWorkflow;
  const suggestionState = useWorkspaceSuggestions({
    selectedProjectId,
    selectedTaskId: selectedTask?.id ?? null,
    currentAssetId: treeState.currentAsset?.id ?? null,
    selectedTaskKind: selectedTask?.kind ?? null,
    onApplyDetectionSuggestions: replaceGeometryObjects,
    setMessage,
  });
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
  const refreshPrelabelWorkspace = useCallback(async () => {
    await Promise.all([refetchAssets(selectedProjectId), currentSequenceId ? refetchSequence() : Promise.resolve()]);
  }, [currentSequenceId, refetchAssets, refetchSequence, selectedProjectId]);
  const prelabelState = usePrelabels({
    projectId: selectedProjectId,
    taskId: selectedTaskId,
    sessionId: currentSequence?.latest_prelabel_session_id ?? null,
    sessionStatus: currentSequence?.latest_prelabel_session_status ?? null,
    currentAssetId: treeState.currentAsset?.id ?? null,
    currentObjects,
    onLoadProposalIntoDraft: replaceGeometryObjects,
    onRefresh: refreshPrelabelWorkspace,
    setMessage,
  });
  const pendingPrelabelObjects = useMemo(
    () =>
      prelabelState.proposals.map((proposal) => ({
        id: proposal.id,
        category_id: proposal.category_id,
        bbox: proposal.bbox,
        label_text: proposal.label_text,
        confidence: proposal.confidence,
      })),
    [prelabelState.proposals],
  );
  const deleteWorkflow = useDeleteWorkflow({
    selectedProjectId,
    selectedDatasetName: selectedProjectName,
    selectedTreeFolderPath: treeState.selectedTreeFolderPath,
    setSelectedTreeFolderPath: treeState.setSelectedTreeFolderPath,
    currentAsset: treeState.currentAsset,
    assetRows: treeState.assetRows,
    assets,
    folderIdByPath,
    treeFolderAssetIds: treeState.treeBuild.folderAssetIds,
    assetById,
    annotations,
    setAnnotations,
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
    refetchFolders,
    refetchProjects,
    onProjectDeleted: () => router.replace("/projects"),
  });
  const {
    isDeletingAssets,
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
  useEffect(() => {
    setHasUnsavedDrafts(pendingCount > 0);
  }, [pendingCount, setHasUnsavedDrafts]);

  useEffect(() => () => setHasUnsavedDrafts(false), [setHasUnsavedDrafts]);

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
    if (!selectedProjectId) return;
    const folderPaths = folders.map((folder) => folder.path).sort((left, right) => left.localeCompare(right));
    setImportFolderOptionsByProject((previous) => {
      const existing = previous[selectedProjectId] ?? [];
      if (existing.length === folderPaths.length && existing.every((value, index) => value === folderPaths[index])) {
        return previous;
      }
      return {
        ...previous,
        [selectedProjectId]: folderPaths,
      };
    });
  }, [folders, selectedProjectId, setImportFolderOptionsByProject]);

  useEffect(() => {
    if (!selectedProjectId || !hasProcessingSequences) return;
    const intervalId = window.setInterval(() => {
      void refetchFolders(selectedProjectId);
      void refetchAssets(selectedProjectId);
      if (currentSequenceId) void refetchSequence();
    }, 2000);
    return () => window.clearInterval(intervalId);
  }, [currentSequenceId, hasProcessingSequences, refetchAssets, refetchFolders, refetchSequence, selectedProjectId]);

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
    onPrev: isSequenceMode ? sequenceNavigation.goToPrev : treeState.handlePrevAsset,
    onNext: isSequenceMode ? sequenceNavigation.goToNext : treeState.handleNextAsset,
    onJumpPrev: isSequenceMode ? () => sequenceNavigation.jumpBy(-10) : undefined,
    onJumpNext: isSequenceMode ? () => sequenceNavigation.jumpBy(10) : undefined,
    onTogglePlayback: isSequenceMode ? sequenceNavigation.togglePlayback : undefined,
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

  async function handleImportVideoSubmit(file: File, payload: VideoImportPayload) {
    if (!selectedProjectId) {
      setVideoImportError("Select a project before importing a video.");
      return;
    }

    try {
      setIsVideoImporting(true);
      setVideoImportError(null);
      setMessage(null);
      const response = await importVideo(selectedProjectId, file, {
        ...payload,
        task_id: selectedTaskId,
      });
      await Promise.all([refetchFolders(selectedProjectId), refetchAssets(selectedProjectId)]);
      if (response.sequence.folder_path) treeState.handleSelectFolderScope(response.sequence.folder_path);
      setIsVideoImportModalOpen(false);
      setMessage(`Processing video "${response.sequence.name}"...`);
    } catch (error) {
      setVideoImportError(error instanceof Error ? error.message : "Failed to import video.");
    } finally {
      setIsVideoImporting(false);
    }
  }

  function handleCreateDataset() {
    if (!selectedProjectId) {
      setMessage("Select a project before creating a dataset.");
      return;
    }
    guardedNavigate(() => {
      const nextParams = new URLSearchParams();
      if (selectedTaskId) nextParams.set("taskId", selectedTaskId);
      const query = nextParams.toString();
      router.push(`/projects/${encodeURIComponent(selectedProjectId)}/dataset${query ? `?${query}` : ""}`);
    });
  }

  function handleApplySuggestedLabel(categoryId: string) {
    clearSelectedLabels();
    handleToggleLabelForCurrentMode(categoryId);
  }

  const headerTitle = treeState.selectedTreeFolderPath ? `${selectedProjectName} / ${treeState.selectedTreeFolderPath}` : selectedProjectName;

  return (
    <>
      <ProjectSectionLayout
        title="Labeling"
        description={`Manage imported assets, annotate images, and move cleanly into dataset creation. Current scope: ${headerTitle}`}
        actions={
          <button
            type="button"
            className="primary-button"
            onClick={handleCreateDataset}
            disabled={!selectedProjectId}
            data-testid="create-dataset-button"
          >
            Create Dataset
          </button>
        }
      >
        <div className="labeling-workspace-shell" data-testid="labeling-workspace">
          <div className="labeling-workspace-grid" data-testid="labeling-workspace-grid">
          <AssetBrowser
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
            onImportImages={handleImport}
            onImportVideo={() => {
              setVideoImportError(null);
              setIsVideoImportModalOpen(true);
            }}
            onOpenWebcam={() => setIsWebcamModalOpen(true)}
            onCollapseAllFolders={treeState.handleCollapseAllFolders}
            onExpandAllFolders={treeState.handleExpandAllFolders}
            onSelectFolderScope={treeState.handleSelectFolderScope}
            onToggleBulkDeleteMode={handleToggleBulkDeleteMode}
            onSelectAllDeleteScope={handleSelectAllDeleteScope}
            onClearDeleteSelection={handleClearDeleteSelection}
            onDeleteCurrentAsset={handleDeleteCurrentAsset}
            onDeleteSelectedAssets={handleDeleteSelectedAssets}
            onDeleteSelectedFolder={handleDeleteSelectedFolder}
            onDeleteCurrentProject={handleDeleteCurrentProject}
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

          <div className="labeling-canvas-column" data-testid="labeling-canvas-column">
            <CanvasToolbar
              annotationMode={annotationMode}
              activeTool={canvasTool}
              onSelectTool={setCanvasTool}
              onResetView={() => {
                setViewerResetToken((value) => value + 1);
                handleSelectGeometryObject(null);
              }}
            />
            <Viewer
              currentAsset={viewerAsset}
              annotationMode={annotationMode}
              canvasTool={canvasTool}
              resetToken={viewerResetToken}
              geometryObjects={currentObjects}
              pendingPrelabelObjects={pendingPrelabelObjects}
              selectedPendingPrelabelId={prelabelState.selectedProposalId}
              selectedObjectId={selectedObjectId}
              hoveredObjectId={hoveredGeometryObjectId}
              defaultCategoryId={defaultGeometryCategoryId}
              onSelectObject={handleSelectGeometryObject}
              onHoverObject={setHoveredGeometryObjectId}
              onUpsertObject={upsertGeometryObject}
              onDeleteSelectedObject={deleteSelectedGeometryObject}
              onImageBasisChange={setCurrentImageBasis}
              onCanvasInteraction={() => setSequencePauseSignal((value) => value + 1)}
            />
            {isSequenceMode ? (
              <section className="sequence-filmstrip" aria-label="Frame navigator" data-testid="sequence-filmstrip">
                {currentSequence ? (
                  <>
                    <SequenceToolbar
                      currentFrameLabel={currentSequenceFrameLabel}
                      currentTimestampLabel={currentSequenceTimestampLabel}
                      isPlaying={sequenceNavigation.isPlaying}
                      pendingFrameCount={sequenceNavigation.pendingFrameCount}
                      pendingProposalCount={currentSequence.pending_prelabel_count}
                      onFirst={sequenceNavigation.goToFirst}
                      onPrev={sequenceNavigation.goToPrev}
                      onTogglePlayback={sequenceNavigation.togglePlayback}
                      onNext={sequenceNavigation.goToNext}
                      onLast={sequenceNavigation.goToLast}
                      onNextPending={sequenceNavigation.goToNextPending}
                    />
                    <SequenceTimeline
                      assets={currentSequence.assets}
                      currentAssetId={treeState.currentAsset?.id ?? null}
                      onSelectAsset={selectSequenceAsset}
                    />
                    <SequenceThumbnailStrip
                      assets={sequenceNavigation.thumbnailAssets}
                      currentAssetId={treeState.currentAsset?.id ?? null}
                      onSelectAsset={selectSequenceAsset}
                    />
                  </>
                ) : (
                  <div className="sequence-filmstrip-empty">
                    {isSequenceLoading ? "Loading sequence…" : currentSequenceError?.message ?? "Sequence unavailable."}
                  </div>
                )}
              </section>
            ) : (
              <AssetFilmstrip
                assetRows={treeState.assetRows}
                currentIndex={treeState.safeAssetIndex}
                pageStatuses={pageStatuses}
                pageDirtyFlags={pageDirtyFlags}
                onSelectIndex={treeState.setAssetIndex}
                onPrev={treeState.handlePrevAsset}
                onNext={treeState.handleNextAsset}
              />
            )}
          </div>

          <div className="label-sidebar-column">
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
              availableDeployments={suggestionState.availableDeployments}
              selectedDeploymentId={suggestionState.selectedDeploymentId}
              selectedDeploymentName={suggestionState.selectedDeployment?.name ?? null}
              selectedDeploymentDevicePreference={suggestionState.selectedDeployment?.device_preference ?? null}
              lastInferenceDeviceSelected={suggestionState.lastInferenceDeviceSelected}
              suggestionPredictions={suggestionState.suggestionPredictions}
              suggestionBoxes={suggestionState.suggestionBoxes}
              suggestionScoreThreshold={suggestionState.suggestionScoreThreshold}
              onChangeSuggestionScoreThreshold={suggestionState.setSuggestionScoreThreshold}
              isSuggesting={suggestionState.isSuggesting}
              hasCompatibleDeployment={suggestionState.availableDeployments.length > 0}
              onChangeSelectedDeploymentId={suggestionState.setSelectedDeploymentId}
              onSuggest={suggestionState.handleSuggest}
              onApplySuggestedLabel={handleApplySuggestedLabel}
            />
            <AiPrelabelsPanel
              session={prelabelState.session}
              proposals={prelabelState.proposals}
              selectedProposalId={prelabelState.selectedProposalId}
              onSelectProposal={prelabelState.setSelectedProposalId}
              onAcceptSelected={prelabelState.acceptSelectedProposal}
              onRejectSelected={prelabelState.rejectSelectedProposal}
              onAcceptCurrentFrame={prelabelState.acceptCurrentFrame}
              onRejectCurrentFrame={prelabelState.rejectCurrentFrame}
              onAcceptFullSession={prelabelState.acceptFullSession}
              onEditSelected={prelabelState.editSelectedProposal}
              isLoading={prelabelState.isLoading}
              isApplying={prelabelState.isApplying}
              errorMessage={prelabelState.error?.message ?? null}
            />
          </div>
          </div>
        </div>
      </ProjectSectionLayout>
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
      <VideoImportModal
        open={isVideoImportModalOpen}
        defaultName={videoImportDefaultName}
        isImporting={isVideoImporting}
        errorMessage={videoImportError}
        enablePrelabels={selectedTask?.kind === "bbox"}
        defaultPrompts={activeLabelRows.map((label) => label.name)}
        onClose={() => {
          if (isVideoImporting) return;
          setIsVideoImportModalOpen(false);
          setVideoImportError(null);
        }}
        onSubmit={(file, payload) => void handleImportVideoSubmit(file, payload)}
      />
      <WebcamCaptureModal
        open={isWebcamModalOpen}
        projectId={selectedProjectId}
        taskId={selectedTaskId}
        defaultName={webcamDefaultName}
        folderOptions={folders.map((folder) => folder.path)}
        enablePrelabels={selectedTask?.kind === "bbox"}
        defaultPrompts={activeLabelRows.map((label) => label.name)}
        onClose={() => setIsWebcamModalOpen(false)}
        onSequenceCreated={(sequence) => {
          if (sequence.folder_path) treeState.handleSelectFolderScope(sequence.folder_path);
          void refetchFolders(selectedProjectId);
        }}
        onFrameUploaded={(asset, sequence) => {
          void refetchAssets(selectedProjectId);
          void refetchFolders(selectedProjectId);
          if (currentSequenceId === sequence.id) void refetchSequence();
          if (!treeState.currentAsset && sequence.folder_path) treeState.handleSelectFolderScope(sequence.folder_path);
          if (asset.id && currentSequenceId === sequence.id) {
            treeState.handleSelectTreeAsset(asset.id, sequence.folder_path ?? undefined);
          }
        }}
        onFinished={(sequences) => {
          void Promise.all([refetchAssets(selectedProjectId), refetchFolders(selectedProjectId)]).then(() => {
            const firstSequence = sequences[0] ?? null;
            if (firstSequence?.folder_path) treeState.handleSelectFolderScope(firstSequence.folder_path);
            if (sequences.some((sequence) => sequence.id === currentSequenceId)) void refetchSequence();
          });
        }}
      />
    </>
  );
}
