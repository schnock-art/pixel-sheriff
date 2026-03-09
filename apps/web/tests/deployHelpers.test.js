const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildPredictPayload,
  deploymentTaskForExperiment,
  detectionBoxesToGeometryObjects,
  deviceLabelToPreference,
  suggestionsPanelState,
} = require("../src/lib/workspace/deployHelpers.js");

test("deviceLabelToPreference maps dropdown values to patch payload", () => {
  assert.equal(deviceLabelToPreference("Auto"), "auto");
  assert.equal(deviceLabelToPreference("CUDA"), "cuda");
  assert.equal(deviceLabelToPreference("CPU"), "cpu");
});

test("suggestionsPanelState transitions correctly", () => {
  assert.equal(suggestionsPanelState({ hasActiveDeployment: false, isSuggesting: false, predictions: [] }), "cta");
  assert.equal(suggestionsPanelState({ hasActiveDeployment: true, isSuggesting: true, predictions: [] }), "loading");
  assert.equal(suggestionsPanelState({ hasActiveDeployment: true, isSuggesting: false, predictions: [] }), "empty");
  assert.equal(
    suggestionsPanelState({ hasActiveDeployment: true, isSuggesting: false, predictions: [{ class_id: 1, score: 0.9 }] }),
    "ready",
  );
});

test("deploymentTaskForExperiment maps detection experiments to bbox deployments", () => {
  assert.equal(deploymentTaskForExperiment("classification"), "classification");
  assert.equal(deploymentTaskForExperiment("detection"), "bbox");
  assert.equal(deploymentTaskForExperiment("bbox"), "bbox");
});

test("buildPredictPayload sends score threshold only for bbox suggestions", () => {
  assert.deepEqual(buildPredictPayload({ assetId: "asset-1", deploymentId: "dep-1", task: "classification", scoreThreshold: 0.7 }), {
    asset_id: "asset-1",
    deployment_id: "dep-1",
    top_k: 5,
  });
  assert.deepEqual(buildPredictPayload({ assetId: "asset-1", deploymentId: "dep-1", task: "bbox", scoreThreshold: 0.7 }), {
    asset_id: "asset-1",
    deployment_id: "dep-1",
    score_threshold: 0.7,
  });
});

test("detectionBoxesToGeometryObjects converts predict response boxes into staged bbox objects", () => {
  assert.deepEqual(
    detectionBoxesToGeometryObjects([{ class_id: "cat-1", bbox: [1, 2, 3, 4] }]),
    [{ id: "suggested-bbox-1", kind: "bbox", category_id: "cat-1", bbox: [1, 2, 3, 4] }],
  );
});
