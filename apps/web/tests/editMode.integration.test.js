const test = require("node:test");
const assert = require("node:assert/strict");

const {
  areSelectionStatesEqual,
  canSubmitWithStates,
  deriveNextAnnotationStatus,
  getCommittedSelectionState,
  resolvePendingAnnotation,
} = require("../src/lib/workspace/annotationState.js");
const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");

test("edit-mode stage-submit-clear flow preserves explicit unlabeled transitions", () => {
  const activeLabelRows = [{ id: 1, name: "cat" }, { id: 2, name: "dog" }];
  const committedState = getCommittedSelectionState({
    status: "approved",
    payload_json: { category_ids: [1] },
  });
  const draftState = {
    labelIds: [],
    status: deriveNextAnnotationStatus(committedState.status, []),
  };

  const pending = resolvePendingAnnotation(draftState, committedState);
  assert.notEqual(pending, null);

  assert.equal(
    canSubmitWithStates({
      pendingCount: 1,
      editMode: true,
      hasCurrentAsset: true,
      draftState,
      committedState,
    }),
    true,
  );

  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-1",
    currentStatus: pending.status,
    selectedLabelIds: pending.labelIds,
    activeLabelRows,
  });
  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.isUnlabeledSelection, true);
  assert.deepEqual(upsertInput.payload_json.category_ids, []);

  const savedState = getCommittedSelectionState({
    status: upsertInput.status,
    payload_json: upsertInput.payload_json,
  });
  assert.equal(areSelectionStatesEqual(savedState, draftState), true);
});

test("edit-mode labeled submit flow creates resolved classification payload", () => {
  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-2",
    currentStatus: "unlabeled",
    selectedLabelIds: [2],
    activeLabelRows: [{ id: 1, name: "cat" }, { id: 2, name: "dog" }],
  });

  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.isUnlabeledSelection, false);
  assert.equal(upsertInput.status, "labeled");
  assert.equal(upsertInput.payload_json.category_id, 2);
  assert.deepEqual(upsertInput.payload_json.category_ids, [2]);
});
