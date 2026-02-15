import { useEffect, useMemo, useState, type Dispatch, type SetStateAction } from "react";

import { type Asset } from "../api";
import { collectFolderPaths } from "../workspace/tree";

const IMAGE_EXTENSIONS = new Set(["jpg", "jpeg", "png", "gif", "bmp", "webp", "tif", "tiff"]);

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
  if (file.type.toLowerCase().startsWith("image/")) return true;
  const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
  return IMAGE_EXTENSIONS.has(extension);
}

export function buildTargetRelativePath(file: File, targetFolder: string): string {
  const normalizedFolder = targetFolder.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
  const relative = (file.webkitRelativePath || file.name).replaceAll("\\", "/");
  const parts = relative.split("/").filter(Boolean);
  const remainder = file.webkitRelativePath ? parts.slice(1).join("/") : file.name;
  return `${normalizedFolder}/${remainder || file.name}`;
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
    setImportMode(fallbackProjectId ? "existing" : "new");
    setImportExistingProjectId(fallbackProjectId);
    setSelectedImportExistingFolder("");
    setImportNewProjectName(sourceFolderName);
    setImportFolderName(sourceFolderName);
    setImportProgress(null);
    setImportDialog({ open: true, sourceFolderName, files });
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
    importFolderOptionsByProject,
    setImportFolderOptionsByProject,
    selectedImportExistingFolder,
    setSelectedImportExistingFolder,
    importProgress,
    setImportProgress,
    importExistingFolderOptions,
    importProgressView,
    openImportDialog,
    closeImportDialog,
    resetImportWorkflow,
    clearProjectImportCache,
  };
}
