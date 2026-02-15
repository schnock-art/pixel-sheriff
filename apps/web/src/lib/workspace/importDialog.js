function normalizeFolderName(rawFolderName) {
  return rawFolderName.replaceAll("\\", "/").replace(/^\/+|\/+$/g, "");
}

function validateImportFolderName(folderName) {
  const trimmed = folderName.trim();
  if (!trimmed) return "Folder name is required.";

  const normalized = normalizeFolderName(trimmed);
  if (!normalized) return "Folder name is required.";

  const segments = normalized.split("/").filter(Boolean);
  if (segments.some((segment) => segment === "." || segment === "..")) {
    return "Folder name cannot contain '.' or '..' path segments.";
  }
  return null;
}

function getImportValidation(params) {
  const { filesCount, importMode, importExistingProjectId, importNewProjectName, importFolderName } = params;

  const filesError = filesCount <= 0 ? "No files selected." : null;
  const projectError =
    importMode === "new"
      ? importNewProjectName.trim()
        ? null
        : "Project name is required for new project imports."
      : importExistingProjectId
        ? null
        : "Please select an existing project.";
  const folderError = validateImportFolderName(importFolderName);

  return {
    filesError,
    projectError,
    folderError,
    canSubmit: !filesError && !projectError && !folderError,
  };
}

function resolveImportDialogDefaults(params) {
  const { sourceFolderName, fallbackProjectId, defaults } = params;
  const preferredMode = defaults.mode ?? (fallbackProjectId ? "existing" : "new");
  const preferredExistingProjectId = defaults.existingProjectId || fallbackProjectId;
  const mode = preferredMode === "existing" && !preferredExistingProjectId ? "new" : preferredMode;
  const existingProjectId = mode === "existing" ? preferredExistingProjectId : "";
  const rememberedFolder = existingProjectId ? defaults.existingFolderByProject?.[existingProjectId] ?? "" : "";

  return {
    mode,
    existingProjectId,
    selectedExistingFolder: mode === "existing" ? rememberedFolder : "",
    folderName: mode === "existing" && rememberedFolder ? rememberedFolder : sourceFolderName,
    newProjectName: sourceFolderName,
  };
}

function resolveExistingProjectSelection(params) {
  const { projectId, sourceFolderName, existingFolderByProject } = params;
  const rememberedFolder = projectId ? existingFolderByProject[projectId] ?? "" : "";
  return {
    selectedExistingFolder: rememberedFolder,
    folderName: rememberedFolder || sourceFolderName,
  };
}

function resolveModeSelection(params) {
  const { mode, currentProjectId, fallbackProjectId, sourceFolderName, existingFolderByProject } = params;
  if (mode === "new") {
    return {
      mode,
      existingProjectId: currentProjectId || fallbackProjectId || "",
      selectedExistingFolder: "",
      folderName: sourceFolderName,
    };
  }

  const existingProjectId = currentProjectId || fallbackProjectId || "";
  const rememberedFolder = existingProjectId ? existingFolderByProject[existingProjectId] ?? "" : "";
  return {
    mode,
    existingProjectId,
    selectedExistingFolder: rememberedFolder,
    folderName: rememberedFolder || sourceFolderName,
  };
}

module.exports = {
  normalizeFolderName,
  validateImportFolderName,
  getImportValidation,
  resolveImportDialogDefaults,
  resolveExistingProjectSelection,
  resolveModeSelection,
};
