const test = require("node:test");
const assert = require("node:assert/strict");

const { buildAnnotationUpsertInput } = require("../src/lib/workspace/annotationSubmission.js");

test("buildAnnotationUpsertInput preserves accepted prediction review metadata", () => {
  const upsertInput = buildAnnotationUpsertInput({
    assetId: "asset-1",
    currentStatus: "approved",
    selectedLabelIds: ["cat"],
    activeLabelRows: [{ id: "cat", name: "Cat" }],
    predictionReview: {
      origin_kind: "deployment_prediction",
      task: "classification",
      deployment_id: "dep-1",
      deployment_name: "cls-v1",
      selected_class_id: "cat",
      selected_class_name: "Cat",
      score: 0.94,
    },
  });

  assert.equal(upsertInput.status, "approved");
  assert.deepEqual(upsertInput.payload_json.prediction_review, {
    origin_kind: "deployment_prediction",
    task: "classification",
    deployment_id: "dep-1",
    deployment_name: "cls-v1",
    selected_class_id: "cat",
    selected_class_name: "Cat",
    score: 0.94,
  });
});
