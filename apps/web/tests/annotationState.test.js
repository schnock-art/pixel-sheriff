const test = require("node:test");
const assert = require("node:assert/strict");

const {
  areSelectionStatesEqual,
  canSubmitWithStates,
  deriveNextAnnotationStatus,
  getCommittedSelectionState,
  readAnnotationLabelIds,
  resolvePendingAnnotation,
} = require("../src/lib/workspace/annotationState.js");

test("deriveNextAnnotationStatus keeps status explicit for selection transitions", () => {
  assert.equal(deriveNextAnnotationStatus("unlabeled", [1]), "labeled");
  assert.equal(deriveNextAnnotationStatus("approved", [1]), "approved");
  assert.equal(deriveNextAnnotationStatus("approved", []), "unlabeled");
});

test("readAnnotationLabelIds resolves category_ids first and falls back to category_id", () => {
  assert.deepEqual(readAnnotationLabelIds({ category_ids: [3, 1, 3] }), [3, 1]);
  assert.deepEqual(readAnnotationLabelIds({ category_id: 7 }), [7]);
  assert.deepEqual(readAnnotationLabelIds({}), []);
});

test("resolvePendingAnnotation drops pending entry when draft and committed states match", () => {
  const draftState = { labelIds: [4, 2], status: "approved" };
  const committedState = { labelIds: [2, 4], status: "approved" };

  assert.equal(areSelectionStatesEqual(draftState, committedState), true);
  assert.equal(resolvePendingAnnotation(draftState, committedState), null);
});

test("resolvePendingAnnotation keeps pending entry when draft and committed states differ", () => {
  const draftState = { labelIds: [], status: "unlabeled" };
  const committedState = { labelIds: [9], status: "labeled" };

  assert.equal(areSelectionStatesEqual(draftState, committedState), false);
  assert.deepEqual(resolvePendingAnnotation(draftState, committedState), draftState);
});

test("canSubmitWithStates allows non-edit clear-label submission", () => {
  const committedState = getCommittedSelectionState({
    status: "approved",
    payload_json: { category_ids: [5] },
  });
  const draftState = { labelIds: [], status: "unlabeled" };

  assert.equal(
    canSubmitWithStates({
      pendingCount: 0,
      editMode: false,
      hasCurrentAsset: true,
      draftState,
      committedState,
    }),
    true,
  );
});

test("canSubmitWithStates blocks non-edit submit when nothing changed", () => {
  const committedState = getCommittedSelectionState({
    status: "unlabeled",
    payload_json: { category_ids: [] },
  });
  const draftState = { labelIds: [], status: "unlabeled" };

  assert.equal(
    canSubmitWithStates({
      pendingCount: 0,
      editMode: false,
      hasCurrentAsset: true,
      draftState,
      committedState,
    }),
    false,
  );
});

test("canSubmitWithStates allows submit whenever staged edits exist", () => {
  assert.equal(
    canSubmitWithStates({
      pendingCount: 2,
      editMode: true,
      hasCurrentAsset: false,
      draftState: { labelIds: [], status: "unlabeled" },
      committedState: { labelIds: [], status: "unlabeled" },
    }),
    true,
  );
});
