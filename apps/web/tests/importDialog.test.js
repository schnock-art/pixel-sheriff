const test = require("node:test");
const assert = require("node:assert/strict");

const {
  getImportValidation,
  resolveExistingProjectSelection,
  resolveImportDialogDefaults,
  resolveModeSelection,
  validateImportFolderName,
} = require("../src/lib/workspace/importDialog.js");

test("getImportValidation requires project selection/name and valid folder", () => {
  assert.equal(
    getImportValidation({
      filesCount: 2,
      importMode: "existing",
      importExistingProjectId: "",
      importNewProjectName: "",
      importFolderName: "train",
    }).projectError,
    "Please select an existing project.",
  );

  assert.equal(
    getImportValidation({
      filesCount: 2,
      importMode: "new",
      importExistingProjectId: "abc",
      importNewProjectName: "",
      importFolderName: "train",
    }).projectError,
    "Project name is required for new project imports.",
  );

  assert.equal(validateImportFolderName("../train"), "Folder name cannot contain '.' or '..' path segments.");
});

test("resolveImportDialogDefaults restores remembered existing-project defaults", () => {
  const resolved = resolveImportDialogDefaults({
    sourceFolderName: "incoming",
    fallbackProjectId: "p-fallback",
    defaults: {
      mode: "existing",
      existingProjectId: "p-saved",
      existingFolderByProject: { "p-saved": "train/cats" },
    },
  });

  assert.deepEqual(resolved, {
    mode: "existing",
    existingProjectId: "p-saved",
    selectedExistingFolder: "train/cats",
    folderName: "train/cats",
    newProjectName: "incoming",
  });
});

test("resolveModeSelection and resolveExistingProjectSelection keep folder defaults project-scoped", () => {
  const byProject = { p1: "train/dogs", p2: "val/cats" };

  const modeResolved = resolveModeSelection({
    mode: "existing",
    currentProjectId: "p2",
    fallbackProjectId: "p1",
    sourceFolderName: "incoming",
    existingFolderByProject: byProject,
  });
  assert.equal(modeResolved.selectedExistingFolder, "val/cats");
  assert.equal(modeResolved.folderName, "val/cats");

  const projectResolved = resolveExistingProjectSelection({
    projectId: "p1",
    sourceFolderName: "incoming",
    existingFolderByProject: byProject,
  });
  assert.deepEqual(projectResolved, {
    selectedExistingFolder: "train/dogs",
    folderName: "train/dogs",
  });
});
