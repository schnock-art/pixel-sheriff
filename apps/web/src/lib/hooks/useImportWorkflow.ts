import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { type Asset } from "../api";
import {
  getImportValidation,
  resolveExistingProjectSelection,
  resolveImportDialogDefaults,
  resolveModeSelection,
  type ImportDefaults,
} from "../workspace/importDialog";
import { buildTargetRelativePath as buildTargetRelativePathHelper, isImageCandidate as isImageCandidateHelper } from "../workspace/importFiles";
import { collectFolderPaths } from "../workspace/tree";

const IMPORT_DEFAULTS_STORAGE_KEY = "pixel-sheriff:import-defaults:v1";

const DEFAULT_IMPORT_DEFAULTS: ImportDefaults = {
  mode: "existing",
  existingProjectId: "",
  existingFolderByProject: {},
};

export interface ImportDialogState {
  open: boolean;
  sourceFolderName: string;
  files: File[];
}

export interface ImportProgressState {
  totalFiles: number;
  completedFiles: number;
  uploadedFiles: number;
  failedFiles: number;
  totalBytes: number;
  processedBytes: number;
  startedAtMs: number;
  activeFileName: string | null;
}

interface UseImportWorkflowParams {
  assets: Asset[];
  selectedProjectId: string | null;
  isAssetsLoading: boolean;
  fetchProjectAssets: (projectId: string) => Promise<Asset[]>;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatDuration(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  const rounded = Math.round(seconds);
  const mins = Math.floor(rounded / 60);
  const secs = rounded % 60;
  if (mins === 0) return `${secs}s`;
  return `${mins}m ${secs}s`;
}

export function isImageCandidate(file: File): boolean {
  return isImageCandidateHelper(file);
}

export function buildTargetRelativePath(file: File, targetFolder: string): string {
  return buildTargetRelativePathHelper(file, targetFolder);
}

export function useImportWorkflow({ assets, selectedProjectId, isAssetsLoading, fetchProjectAssets }: UseImportWorkflowParams) {
  const [isImporting, setIsImporting] = useState(false);
  const [importFailures, setImportFailures] = useState<string[]>([]);
  const [importDialog, setImportDialog] = useState<ImportDialogState>({ open: false, sourceFolderName: "", files: [] });
  const [importMode, setImportMode] = useState<"existing" | "new">("existing");
  const [importExistingProjectId, setImportExistingProjectId] = useState<string>("");
  const [importNewProjectName, setImportNewProjectName] = useState("");
  const [importFolderName, setImportFolderName] = useState("");
  const [importFolderOptionsByProject, setImportFolderOptionsByProject] = useState<Record<string, string[]>>({});
  const [selectedImportExistingFolder, setSelectedImportExistingFolder] = useState<string>("");
  const [importProgress, setImportProgress] = useState<ImportProgressState | null>(null);
  const [importDefaults, setImportDefaults] = useState<ImportDefaults>(DEFAULT_IMPORT_DEFAULTS);
  const [hasLoadedImportDefaults, setHasLoadedImportDefaults] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    try {
      const raw = window.localStorage.getItem(IMPORT_DEFAULTS_STORAGE_KEY);
      if (!raw) return;
      const parsed: unknown = JSON.parse(raw);
      if (!parsed || typeof parsed !== "object") return;
      const source = parsed as Partial<ImportDefaults> & { existingFolderByProject?: Record<string, unknown> };

      const mode = source.mode === "new" ? "new" : "existing";
      const existingProjectId = typeof source.existingProjectId === "string" ? source.existingProjectId : "";
      const existingFolderByProject: Record<string, string> = {};
      const sourceFolders = source.existingFolderByProject;
      if (sourceFolders && typeof sourceFolders === "object") {
        for (const [key, value] of Object.entries(sourceFolders)) {
          if (typeof value === "string") existingFolderByProject[key] = value;
        }
      }
      setImportDefaults({ mode, existingProjectId, existingFolderByProject });
    } finally {
      setHasLoadedImportDefaults(true);
    }
  }, []);

  useEffect(() => {
    if (!hasLoadedImportDefaults) return;
    if (typeof window === "undefined") return;
    window.localStorage.setItem(IMPORT_DEFAULTS_STORAGE_KEY, JSON.stringify(importDefaults));
  }, [hasLoadedImportDefaults, importDefaults]);

  useEffect(() => {
    if (!hasLoadedImportDefaults) return;
    if (importDefaults.mode === importMode) return;
    setImportDefaults((previous) => ({ ...previous, mode: importMode }));
  }, [hasLoadedImportDefaults, importDefaults.mode, importMode]);

  useEffect(() => {
    if (!hasLoadedImportDefaults) return;
    if (importDefaults.existingProjectId === importExistingProjectId) return;
    setImportDefaults((previous) => ({ ...previous, existingProjectId: importExistingProjectId }));
  }, [hasLoadedImportDefaults, importDefaults.existingProjectId, importExistingProjectId]);

  useEffect(() => {
    if (!hasLoadedImportDefaults) return;
    if (!importExistingProjectId) return;
    setImportDefaults((previous) => {
      const nextFolders = { ...previous.existingFolderByProject };
      if (selectedImportExistingFolder) nextFolders[importExistingProjectId] = selectedImportExistingFolder;
      else delete nextFolders[importExistingProjectId];
      const existing = previous.existingFolderByProject[importExistingProjectId] ?? "";
      const next = nextFolders[importExistingProjectId] ?? "";
      if (existing === next) return previous;
      return { ...previous, existingFolderByProject: nextFolders };
    });
  }, [hasLoadedImportDefaults, importExistingProjectId, selectedImportExistingFolder]);

  useEffect(() => {
    if (!importDialog.open) return;
    if (importMode !== "existing") return;
    if (!importExistingProjectId) return;
    if (importFolderOptionsByProject[importExistingProjectId]) return;

    let isActive = true;

    async function loadFolderOptions() {
      try {
        if (importExistingProjectId === selectedProjectId) {
          if (isAssetsLoading) return;
          if (!isActive) return;
          setImportFolderOptionsByProject((previous) => ({
            ...previous,
            [importExistingProjectId]: collectFolderPaths(assets),
          }));
          return;
        }

        const projectAssets = await fetchProjectAssets(importExistingProjectId);
        if (!isActive) return;
        setImportFolderOptionsByProject((previous) => ({
          ...previous,
          [importExistingProjectId]: collectFolderPaths(projectAssets),
        }));
      } catch {
        if (!isActive) return;
        setImportFolderOptionsByProject((previous) => ({
          ...previous,
          [importExistingProjectId]: [],
        }));
      }
    }

    void loadFolderOptions();
    return () => {
      isActive = false;
    };
  }, [
    assets,
    fetchProjectAssets,
    importDialog.open,
    importExistingProjectId,
    importFolderOptionsByProject,
    importMode,
    isAssetsLoading,
    selectedProjectId,
  ]);

  const importExistingFolderOptions = importFolderOptionsByProject[importExistingProjectId] ?? [];
  const importValidation = useMemo(
    () =>
      getImportValidation({
        filesCount: importDialog.files.length,
        importMode,
        importExistingProjectId,
        importNewProjectName,
        importFolderName,
      }),
    [importDialog.files.length, importExistingProjectId, importFolderName, importMode, importNewProjectName],
  );
  const importProgressView = useMemo(() => {
    if (!importProgress) return null;

    const elapsedSeconds = Math.max((Date.now() - importProgress.startedAtMs) / 1000, 0.001);
    const fileRate = importProgress.completedFiles / elapsedSeconds;
    const byteRate = importProgress.processedBytes / elapsedSeconds;
    const remainingBytes = Math.max(importProgress.totalBytes - importProgress.processedBytes, 0);
    const etaSeconds = byteRate > 0 ? remainingBytes / byteRate : Number.POSITIVE_INFINITY;

    return {
      percent: importProgress.totalFiles > 0 ? Math.round((importProgress.completedFiles / importProgress.totalFiles) * 100) : 0,
      elapsedText: formatDuration(elapsedSeconds),
      etaText: Number.isFinite(etaSeconds) ? formatDuration(etaSeconds) : "--",
      fileRateText: `${fileRate.toFixed(fileRate >= 10 ? 0 : 1)} files/s`,
      speedText: `${formatBytes(byteRate)}/s`,
      progressText: `${importProgress.completedFiles}/${importProgress.totalFiles}`,
      bytesText: `${formatBytes(importProgress.processedBytes)} / ${formatBytes(importProgress.totalBytes)}`,
      remainingFilesText: `${Math.max(importProgress.totalFiles - importProgress.completedFiles, 0)} remaining`,
      uploadedFilesText: `${importProgress.uploadedFiles} uploaded`,
      failedFilesText: `${importProgress.failedFiles} failed`,
      activeFileName: importProgress.activeFileName,
    };
  }, [importProgress]);

  function openImportDialog(files: File[], sourceFolderName: string, fallbackProjectId: string) {
    const defaults = resolveImportDialogDefaults({
      sourceFolderName,
      fallbackProjectId,
      defaults: hasLoadedImportDefaults ? importDefaults : DEFAULT_IMPORT_DEFAULTS,
    });
    setImportMode(defaults.mode);
    setImportExistingProjectId(defaults.existingProjectId);
    setSelectedImportExistingFolder(defaults.selectedExistingFolder);
    setImportNewProjectName(defaults.newProjectName);
    setImportFolderName(defaults.folderName);
    setImportProgress(null);
    setImportDialog({ open: true, sourceFolderName, files });
  }

  function setImportModeWithDefaults(mode: "existing" | "new") {
    const resolved = resolveModeSelection({
      mode,
      currentProjectId: importExistingProjectId,
      fallbackProjectId: selectedProjectId ?? "",
      sourceFolderName: importDialog.sourceFolderName,
      existingFolderByProject: importDefaults.existingFolderByProject,
    });
    setImportMode(resolved.mode);
    setImportExistingProjectId(resolved.existingProjectId);
    setSelectedImportExistingFolder(resolved.selectedExistingFolder);
    setImportFolderName(resolved.folderName);
  }

  function setImportExistingProjectWithDefaults(projectId: string) {
    setImportExistingProjectId(projectId);
    const resolved = resolveExistingProjectSelection({
      projectId,
      sourceFolderName: importDialog.sourceFolderName,
      existingFolderByProject: importDefaults.existingFolderByProject,
    });
    setSelectedImportExistingFolder(resolved.selectedExistingFolder);
    setImportFolderName(resolved.folderName);
  }

  function setImportExistingFolderWithDefaults(folderPath: string) {
    setSelectedImportExistingFolder(folderPath);
    if (folderPath) setImportFolderName(folderPath);
  }

  function closeImportDialog() {
    setSelectedImportExistingFolder("");
    setImportProgress(null);
    setImportDialog({ open: false, sourceFolderName: "", files: [] });
  }

  function resetImportWorkflow() {
    setImportFailures([]);
    closeImportDialog();
    setIsImporting(false);
  }

  function clearProjectImportCache(projectId: string) {
    setImportFolderOptionsByProject((previous) => {
      const next = { ...previous };
      delete next[projectId];
      return next;
    });
  }

  return {
    isImporting,
    setIsImporting,
    importFailures,
    setImportFailures,
    importDialog,
    setImportDialog,
    importMode,
    setImportMode,
    importExistingProjectId,
    setImportExistingProjectId,
    importNewProjectName,
    setImportNewProjectName,
    importFolderName,
    setImportFolderName,
    setImportModeWithDefaults,
    setImportExistingProjectWithDefaults,
    setImportExistingFolderWithDefaults,
    importFolderOptionsByProject,
    setImportFolderOptionsByProject,
    selectedImportExistingFolder,
    setSelectedImportExistingFolder,
    importProgress,
    setImportProgress,
    importExistingFolderOptions,
    importValidation,
    importProgressView,
    openImportDialog,
    closeImportDialog,
    resetImportWorkflow,
    clearProjectImportCache,
  };
}
