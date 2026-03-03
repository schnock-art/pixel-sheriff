const test = require("node:test");
const assert = require("node:assert/strict");

const {
  canSubmitWithStates,
  deriveNextAnnotationStatus,
  getCommittedSelectionState,
  resolvePendingAnnotation,
} = require("../src/lib/workspace/annotationState.js");
const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");

test("bbox draw -> label -> submit -> persisted reload", () => {
  const committedState = getCommittedSelectionState(null);
  const drawnObjects = [{ id: "bbox-1", kind: "bbox", category_id: "3", bbox: [10, 20, 100, 50] }];
  const draftState = {
    labelIds: ["3"],
    status: deriveNextAnnotationStatus(committedState.status, ["3"], drawnObjects.length),
    objects: drawnObjects,
    imageBasis: { width: 640, height: 480 },
  };

  const pending = resolvePendingAnnotation(draftState, committedState);
  assert.notEqual(pending, null);
  assert.equal(
    canSubmitWithStates({
      pendingCount: 1,
      editMode: true,
      hasCurrentAsset: true,
      draftState: pending,
      committedState,
    }),
    true,
  );

  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-bbox",
    currentStatus: pending.status,
    selectedLabelIds: pending.labelIds,
    activeLabelRows: [{ id: "3", name: "car" }],
    objects: pending.objects,
    imageBasis: pending.imageBasis,
  });
  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.status, "labeled");
  assert.equal(upsertInput.payload_json.objects.length, 1);
  assert.equal(upsertInput.payload_json.objects[0].kind, "bbox");
  assert.equal(upsertInput.payload_json.classification.primary_category_id, "3");

  const reloaded = getCommittedSelectionState({
    status: upsertInput.status,
    payload_json: upsertInput.payload_json,
  });
  assert.deepEqual(reloaded.objects, drawnObjects);
  assert.deepEqual(reloaded.imageBasis, { width: 640, height: 480 });
  assert.deepEqual(resolvePendingAnnotation(reloaded, reloaded), null);
});

test("polygon draw -> label -> submit -> persisted reload", () => {
  const committedState = getCommittedSelectionState(null);
  const drawnObjects = [
    {
      id: "poly-1",
      kind: "polygon",
      category_id: "5",
      segmentation: [[20, 20, 80, 20, 70, 60]],
    },
  ];
  const draftState = {
    labelIds: ["5"],
    status: deriveNextAnnotationStatus(committedState.status, ["5"], drawnObjects.length),
    objects: drawnObjects,
    imageBasis: { width: 320, height: 240 },
  };

  const pending = resolvePendingAnnotation(draftState, committedState);
  assert.notEqual(pending, null);

  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-poly",
    currentStatus: pending.status,
    selectedLabelIds: pending.labelIds,
    activeLabelRows: [{ id: "5", name: "person" }],
    objects: pending.objects,
    imageBasis: pending.imageBasis,
  });
  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.status, "labeled");
  assert.equal(upsertInput.payload_json.objects.length, 1);
  assert.equal(upsertInput.payload_json.objects[0].kind, "polygon");
  assert.equal(upsertInput.payload_json.classification.primary_category_id, "5");

  const reloaded = getCommittedSelectionState({
    status: upsertInput.status,
    payload_json: upsertInput.payload_json,
  });
  assert.deepEqual(reloaded.objects, drawnObjects);
  assert.deepEqual(reloaded.imageBasis, { width: 320, height: 240 });
});

test("geometry submit infers classification from object categories when selected labels are empty", () => {
  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-implicit-class",
    currentStatus: "unlabeled",
    selectedLabelIds: [],
    activeLabelRows: [{ id: "9", name: "mountain" }],
    objects: [{ id: "bbox-1", kind: "bbox", category_id: "9", bbox: [5, 5, 20, 10] }],
    imageBasis: { width: 200, height: 100 },
  });

  assert.notEqual(upsertInput, null);
  assert.equal(upsertInput.isUnlabeledSelection, false);
  assert.equal(upsertInput.status, "labeled");
  assert.equal(upsertInput.payload_json.category_id, "9");
  assert.deepEqual(upsertInput.payload_json.category_ids, ["9"]);
  assert.equal(upsertInput.payload_json.classification.primary_category_id, "9");
});
