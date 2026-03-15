import { useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { createProject, importVideo, uploadAsset, type ProjectTaskType, type VideoImportPayload } from "../api";
import { isImageCandidate, type ImportDialogState, type ImportProgressState } from "./useImportWorkflow";
import {
  advanceImportProgress,
  buildImportResultMessage,
  createImportProgress,
  formatImportFailure,
  makeTimestampedAssetName,
  mergeImportedFolderOptions,
  resolveImportRootName,
  setActiveImportFile,
} from "../workspace/projectAssetsImport";
import { buildTargetRelativePath } from "./useImportWorkflow";

type WorkspaceAnnotationMode = "labels" | "bbox" | "segmentation";
type NewProjectTaskType = "classification_single" | "bbox" | "segmentation";

interface ProjectSummary {
  id: string;
  name: string;
  task_type: ProjectTaskType;
}

interface ImportValidationState {
  canSubmit: boolean;
  filesError: string | null;
  projectError: string | null;
  folderError: string | null;
}

interface UseProjectAssetsImportActionsParams {
  selectedProjectId: string | null;
  selectedTaskId: string | null;
  projects: ProjectSummary[];
  importMode: "existing" | "new";
  importDialog: ImportDialogState;
  importExistingProjectId: string;
  importNewProjectName: string;
  importFolderName: string;
  importNewProjectTaskType: NewProjectTaskType;
  importValidation: ImportValidationState;
  currentSequenceId: string | null;
  currentAssetId: string | null;
  openImportDialog: (files: File[], sourceFolderName: string, fallbackProjectId: string) => void;
  closeImportDialog: () => void;
  setIsImporting: Dispatch<SetStateAction<boolean>>;
  setImportFailures: Dispatch<SetStateAction<string[]>>;
  setImportProgress: Dispatch<SetStateAction<ImportProgressState | null>>;
  setImportNewProjectTaskType: Dispatch<SetStateAction<NewProjectTaskType>>;
  setImportFolderOptionsByProject: Dispatch<SetStateAction<Record<string, string[]>>>;
  setSelectedImportExistingFolder: Dispatch<SetStateAction<string>>;
  setAnnotationMode: Dispatch<SetStateAction<WorkspaceAnnotationMode>>;
  setEditMode: Dispatch<SetStateAction<boolean>>;
  setHasUnsavedDrafts: (hasUnsavedDrafts: boolean) => void;
  setMessage: Dispatch<SetStateAction<string | null>>;
  refetchProjects: () => Promise<unknown>;
  refetchAssets: (projectIdOverride?: string | null) => Promise<unknown>;
  refetchFolders: (projectIdOverride?: string | null) => Promise<unknown>;
  refetchSequence: () => Promise<unknown>;
  resetTreeState: () => void;
  resetAnnotationWorkflow: () => void;
  handleSelectFolderScope: (folderPath: string) => void;
  handleSelectTreeAsset: (assetId: string, folderPath?: string) => void;
  pushToProject: (projectId: string) => void;
  resolveAnnotationModeForProjectType: (taskType: ProjectTaskType | null | undefined, fallback: NewProjectTaskType) => WorkspaceAnnotationMode;
  resolveNewProjectTaskTypeForProjectType: (taskType: ProjectTaskType | null | undefined) => NewProjectTaskType;
}

export function useProjectAssetsImportActions({
  selectedProjectId,
  selectedTaskId,
  projects,
  importMode,
  importDialog,
  importExistingProjectId,
  importNewProjectName,
  importFolderName,
  importNewProjectTaskType,
  importValidation,
  currentSequenceId,
  currentAssetId,
  openImportDialog,
  closeImportDialog,
  setIsImporting,
  setImportFailures,
  setImportProgress,
  setImportNewProjectTaskType,
  setImportFolderOptionsByProject,
  setSelectedImportExistingFolder,
  setAnnotationMode,
  setEditMode,
  setHasUnsavedDrafts,
  setMessage,
  refetchProjects,
  refetchAssets,
  refetchFolders,
  refetchSequence,
  resetTreeState,
  resetAnnotationWorkflow,
  handleSelectFolderScope,
  handleSelectTreeAsset,
  pushToProject,
  resolveAnnotationModeForProjectType,
  resolveNewProjectTaskTypeForProjectType,
}: UseProjectAssetsImportActionsParams) {
  const [isVideoImportModalOpen, setIsVideoImportModalOpen] = useState(false);
  const [isVideoImporting, setIsVideoImporting] = useState(false);
  const [videoImportError, setVideoImportError] = useState<string | null>(null);
  const [isWebcamModalOpen, setIsWebcamModalOpen] = useState(false);

  const videoImportDefaultName = useMemo(
    () => makeTimestampedAssetName("video"),
    [isVideoImportModalOpen],
  );
  const webcamDefaultName = useMemo(
    () => makeTimestampedAssetName("webcam"),
    [isWebcamModalOpen],
  );

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
      setImportProgress(createImportProgress(files));

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
        setImportProgress((previous) => setActiveImportFile(previous, file.name));
        try {
          const targetRelativePath = buildTargetRelativePath(file, folderName);
          await uploadAsset(targetProjectId, file, targetRelativePath);
          uploadedCount += 1;
          setImportProgress((previous) => advanceImportProgress(previous, file.size, "uploaded"));
        } catch (error) {
          failures.push(formatImportFailure(file.name, error));
          setImportProgress((previous) => advanceImportProgress(previous, file.size, "failed"));
        }
      }

      await refetchProjects();
      await refetchAssets(targetProjectId);
      const targetProject = importMode === "new" ? null : projects.find((item) => item.id === targetProjectId) ?? null;
      setAnnotationMode(resolveAnnotationModeForProjectType(targetProject?.task_type, importNewProjectTaskType));
      resetTreeState();
      resetAnnotationWorkflow();
      setEditMode(false);
      setHasUnsavedDrafts(false);
      if (targetProjectId !== selectedProjectId) {
        pushToProject(targetProjectId);
      }
      setSelectedImportExistingFolder("");
      setImportFolderOptionsByProject((previous) => {
        const importedRelativePaths = files.map((file) => buildTargetRelativePath(file, folderName));
        const existingFolders = previous[targetProjectId];
        return {
          ...previous,
          [targetProjectId]: mergeImportedFolderOptions(existingFolders, importedRelativePaths),
        };
      });
      setImportFailures(failures);
      setSelectedImportExistingFolder("");
      closeImportDialog();
      setMessage(
        buildImportResultMessage({
          uploadedCount,
          totalFiles: files.length,
          targetProjectName,
          folderName,
          failuresCount: failures.length,
        }),
      );
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
      const rootName = resolveImportRootName(files, `Dataset ${new Date().toLocaleString()}`);
      const defaultProject = projects.find((project) => project.id === selectedProjectId) ?? projects[0];
      setImportNewProjectTaskType(resolveNewProjectTaskTypeForProjectType(defaultProject?.task_type));
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
      if (response.sequence.folder_path) handleSelectFolderScope(response.sequence.folder_path);
      setIsVideoImportModalOpen(false);
      setMessage(`Processing video "${response.sequence.name}"...`);
    } catch (error) {
      setVideoImportError(error instanceof Error ? error.message : "Failed to import video.");
    } finally {
      setIsVideoImporting(false);
    }
  }

  function openVideoImportModal() {
    setVideoImportError(null);
    setIsVideoImportModalOpen(true);
  }

  function closeVideoImportModal() {
    if (isVideoImporting) return;
    setIsVideoImportModalOpen(false);
    setVideoImportError(null);
  }

  function openWebcamModal() {
    setIsWebcamModalOpen(true);
  }

  function closeWebcamModal() {
    setIsWebcamModalOpen(false);
  }

  function handleWebcamSequenceCreated(sequence: { folder_path?: string | null }) {
    if (sequence.folder_path) handleSelectFolderScope(sequence.folder_path);
    void refetchFolders(selectedProjectId);
  }

  function handleWebcamFrameUploaded(asset: { id?: string | null }, sequence: { id: string; folder_path?: string | null }) {
    void refetchAssets(selectedProjectId);
    void refetchFolders(selectedProjectId);
    if (currentSequenceId === sequence.id) void refetchSequence();
    if (!currentAssetId && sequence.folder_path) handleSelectFolderScope(sequence.folder_path);
    if (asset.id && currentSequenceId === sequence.id) {
      handleSelectTreeAsset(asset.id, sequence.folder_path ?? undefined);
    }
  }

  function handleWebcamFinished(sequences: Array<{ id: string; folder_path?: string | null }>) {
    void Promise.all([refetchAssets(selectedProjectId), refetchFolders(selectedProjectId)]).then(() => {
      const firstSequence = sequences[0] ?? null;
      if (firstSequence?.folder_path) handleSelectFolderScope(firstSequence.folder_path);
      if (sequences.some((sequence) => sequence.id === currentSequenceId)) void refetchSequence();
    });
  }

  return {
    isVideoImportModalOpen,
    isVideoImporting,
    videoImportError,
    isWebcamModalOpen,
    videoImportDefaultName,
    webcamDefaultName,
    confirmImportFromDialog,
    handleImport,
    handleImportVideoSubmit,
    openVideoImportModal,
    closeVideoImportModal,
    openWebcamModal,
    closeWebcamModal,
    handleWebcamSequenceCreated,
    handleWebcamFrameUploaded,
    handleWebcamFinished,
  };
}
