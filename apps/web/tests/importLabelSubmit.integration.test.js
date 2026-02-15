const test = require("node:test");
const assert = require("node:assert/strict");

const { canSubmitWithStates, getCommittedSelectionState } = require("../src/lib/workspace/annotationState.js");
const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");
const { getImportValidation } = require("../src/lib/workspace/importDialog.js");
const { buildTargetRelativePath, isImageCandidate } = require("../src/lib/workspace/importFiles.js");

test("import -> label -> submit workflow composes valid paths and annotation payloads", () => {
  const importFiles = [
    { name: "a.jpg", type: "image/jpeg", webkitRelativePath: "raw/a.jpg" },
    { name: "b.png", type: "", webkitRelativePath: "raw/nested/b.png" },
    { name: "notes.txt", type: "text/plain", webkitRelativePath: "raw/notes.txt" },
  ];
  const imageFiles = importFiles.filter((file) => isImageCandidate(file));
  assert.equal(imageFiles.length, 2);

  const validation = getImportValidation({
    filesCount: imageFiles.length,
    importMode: "existing",
    importExistingProjectId: "project-1",
    importNewProjectName: "",
    importFolderName: "train/cats",
  });
  assert.equal(validation.canSubmit, true);

  const relativePaths = imageFiles.map((file) => buildTargetRelativePath(file, "train/cats"));
  assert.deepEqual(relativePaths, ["train/cats/a.jpg", "train/cats/nested/b.png"]);

  const activeLabelRows = [{ id: 7, name: "cat" }, { id: 8, name: "dog" }];

  const firstCommitted = getCommittedSelectionState(null);
  const firstDraftUpsert = buildAnnotationUpsertInput({
    assetId: "asset-a",
    currentStatus: firstCommitted.status,
    selectedLabelIds: [7],
    activeLabelRows,
  });
  assert.equal(
    canSubmitWithStates({
      pendingCount: 0,
      editMode: false,
      hasCurrentAsset: true,
      draftState: { labelIds: [7], status: firstDraftUpsert.status },
      committedState: firstCommitted,
    }),
    true,
  );
  assert.equal(firstDraftUpsert.status, "labeled");
  assert.equal(firstDraftUpsert.payload_json.category_id, 7);

  const secondCommitted = getCommittedSelectionState({
    status: "approved",
    payload_json: { category_ids: [7] },
  });
  const secondDraftUpsert = buildAnnotationUpsertInput({
    assetId: "asset-b",
    currentStatus: secondCommitted.status,
    selectedLabelIds: [],
    activeLabelRows,
  });
  assert.equal(
    canSubmitWithStates({
      pendingCount: 0,
      editMode: false,
      hasCurrentAsset: true,
      draftState: { labelIds: [], status: secondDraftUpsert.status },
      committedState: secondCommitted,
    }),
    true,
  );
  assert.equal(secondDraftUpsert.isUnlabeledSelection, true);
  assert.deepEqual(secondDraftUpsert.payload_json.category_ids, []);
});
