const test = require("node:test");
const assert = require("node:assert/strict");

const { canSubmitWithStates, getCommittedSelectionState, resolvePendingAnnotation } = require("../src/lib/workspace/annotationState.js");
const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");
const { resolveSelectionForAsset } = require("../src/lib/workspace/annotationWorkflowSelection.js");

test("edit-mode staged annotation persists across asset switches until submitted", () => {
  const annotationByAssetId = new Map([
    ["asset-a", { status: "approved", payload_json: { category_ids: [1] } }],
    ["asset-b", { status: "labeled", payload_json: { category_ids: [2] } }],
  ]);
  const pendingAnnotations = {};

  const committedA = getCommittedSelectionState(annotationByAssetId.get("asset-a"));
  const draftA = { labelIds: [], status: "unlabeled" };
  pendingAnnotations["asset-a"] = resolvePendingAnnotation(draftA, committedA);

  const currentA = resolveSelectionForAsset({
    currentAssetId: "asset-a",
    pendingAnnotations,
    annotationByAssetId,
  });
  assert.equal(currentA.source, "pending");
  assert.deepEqual(currentA.labelIds, []);
  assert.equal(currentA.status, "unlabeled");

  const currentB = resolveSelectionForAsset({
    currentAssetId: "asset-b",
    pendingAnnotations,
    annotationByAssetId,
  });
  assert.equal(currentB.source, "committed");
  assert.deepEqual(currentB.labelIds, [2]);

  const currentAAgain = resolveSelectionForAsset({
    currentAssetId: "asset-a",
    pendingAnnotations,
    annotationByAssetId,
  });
  assert.equal(currentAAgain.source, "pending");

  assert.equal(
    canSubmitWithStates({
      pendingCount: 1,
      editMode: true,
      hasCurrentAsset: true,
      draftState: { labelIds: currentAAgain.labelIds, status: currentAAgain.status },
      committedState: committedA,
    }),
    true,
  );

  const savedUpsert = buildAnnotationUpsertInput({
    assetId: "asset-a",
    currentStatus: currentAAgain.status,
    selectedLabelIds: currentAAgain.labelIds,
    activeLabelRows: [{ id: 1, name: "cat" }, { id: 2, name: "dog" }],
  });
  annotationByAssetId.set("asset-a", {
    status: savedUpsert.status,
    payload_json: savedUpsert.payload_json,
  });
  delete pendingAnnotations["asset-a"];

  const afterSubmit = resolveSelectionForAsset({
    currentAssetId: "asset-a",
    pendingAnnotations,
    annotationByAssetId,
  });
  assert.equal(afterSubmit.source, "committed");
  assert.equal(afterSubmit.status, "unlabeled");
  assert.deepEqual(afterSubmit.labelIds, []);
});

test("edit mode blocks submit with zero pending after staged state is cleared", () => {
  const committed = getCommittedSelectionState({
    status: "unlabeled",
    payload_json: { category_ids: [] },
  });

  assert.equal(
    canSubmitWithStates({
      pendingCount: 0,
      editMode: true,
      hasCurrentAsset: true,
      draftState: committed,
      committedState: committed,
    }),
    false,
  );
});
