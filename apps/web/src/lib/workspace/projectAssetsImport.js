import { collectFolderPathsFromRelativePaths } from "./tree.js";

function readErrorMessage(error) {
  if (error instanceof Error && error.message) return error.message;
  if (error && typeof error === "object" && typeof error.message === "string" && error.message) return error.message;
  return "unknown upload error";
}

function readResponseBody(error) {
  if (!error || typeof error !== "object") return null;
  return typeof error.responseBody === "string" ? error.responseBody : null;
}

export function makeTimestampedAssetName(prefix, now = new Date()) {
  return `${prefix}_${now.toISOString().slice(0, 19).replaceAll(":", "-")}`;
}

export function resolveImportRootName(files, fallbackLabel) {
  const firstPath = files[0]?.webkitRelativePath ?? "";
  const rootName = firstPath.split("/")[0];
  return rootName || fallbackLabel;
}

export function createImportProgress(files, startedAtMs = Date.now()) {
  return {
    totalFiles: files.length,
    completedFiles: 0,
    uploadedFiles: 0,
    failedFiles: 0,
    totalBytes: files.reduce((sum, file) => sum + file.size, 0),
    processedBytes: 0,
    startedAtMs,
    activeFileName: null,
  };
}

export function setActiveImportFile(progress, fileName) {
  return progress ? { ...progress, activeFileName: fileName } : progress;
}

export function advanceImportProgress(progress, fileSize, outcome) {
  if (!progress) return progress;
  return {
    ...progress,
    completedFiles: progress.completedFiles + 1,
    uploadedFiles: progress.uploadedFiles + (outcome === "uploaded" ? 1 : 0),
    failedFiles: progress.failedFiles + (outcome === "failed" ? 1 : 0),
    processedBytes: progress.processedBytes + fileSize,
    activeFileName: null,
  };
}

export function formatImportFailure(fileName, error) {
  const responseBody = readResponseBody(error);
  const reason = responseBody ? ` (${responseBody})` : "";
  return `${fileName}: ${readErrorMessage(error)}${reason}`;
}

export function mergeImportedFolderOptions(existingFolders, importedRelativePaths) {
  const merged = new Set([...(existingFolders ?? []), ...collectFolderPathsFromRelativePaths(importedRelativePaths)]);
  return Array.from(merged).sort((left, right) => left.localeCompare(right));
}

export function buildImportResultMessage({ uploadedCount, totalFiles, targetProjectName, folderName, failuresCount }) {
  if (uploadedCount === 0) return `Import failed: no files uploaded to "${folderName}".`;
  if (failuresCount > 0) return `Imported ${uploadedCount}/${totalFiles} images into "${targetProjectName}/${folderName}".`;
  return `Imported ${uploadedCount} images into "${targetProjectName}/${folderName}".`;
}
