export interface ImportDefaults {
  mode: "existing" | "new";
  existingProjectId: string;
  existingFolderByProject: Record<string, string>;
}

export function normalizeFolderName(rawFolderName: string): string;
export function validateImportFolderName(folderName: string): string | null;
export function getImportValidation(params: {
  filesCount: number;
  importMode: "existing" | "new";
  importExistingProjectId: string;
  importNewProjectName: string;
  importFolderName: string;
}): {
  filesError: string | null;
  projectError: string | null;
  folderError: string | null;
  canSubmit: boolean;
};
export function resolveImportDialogDefaults(params: {
  sourceFolderName: string;
  fallbackProjectId: string;
  defaults: ImportDefaults;
}): {
  mode: "existing" | "new";
  existingProjectId: string;
  selectedExistingFolder: string;
  folderName: string;
  newProjectName: string;
};
export function resolveExistingProjectSelection(params: {
  projectId: string;
  sourceFolderName: string;
  existingFolderByProject: Record<string, string>;
}): {
  selectedExistingFolder: string;
  folderName: string;
};
export function resolveModeSelection(params: {
  mode: "existing" | "new";
  currentProjectId: string;
  fallbackProjectId: string;
  sourceFolderName: string;
  existingFolderByProject: Record<string, string>;
}): {
  mode: "existing" | "new";
  existingProjectId: string;
  selectedExistingFolder: string;
  folderName: string;
};
