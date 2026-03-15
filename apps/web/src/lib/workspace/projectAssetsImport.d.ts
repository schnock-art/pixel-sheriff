export interface ImportProgressShape {
  totalFiles: number;
  completedFiles: number;
  uploadedFiles: number;
  failedFiles: number;
  totalBytes: number;
  processedBytes: number;
  startedAtMs: number;
  activeFileName: string | null;
}

export function makeTimestampedAssetName(prefix: string, now?: Date): string;
export function resolveImportRootName(files: Array<{ webkitRelativePath?: string }>, fallbackLabel: string): string;
export function createImportProgress(
  files: Array<{ size: number }>,
  startedAtMs?: number,
): ImportProgressShape;
export function setActiveImportFile(progress: ImportProgressShape | null, fileName: string | null): ImportProgressShape | null;
export function advanceImportProgress(
  progress: ImportProgressShape | null,
  fileSize: number,
  outcome: "uploaded" | "failed",
): ImportProgressShape | null;
export function formatImportFailure(fileName: string, error: unknown): string;
export function mergeImportedFolderOptions(existingFolders: string[] | undefined, importedRelativePaths: string[]): string[];
export function buildImportResultMessage(args: {
  uploadedCount: number;
  totalFiles: number;
  targetProjectName: string;
  folderName: string;
  failuresCount: number;
}): string;
