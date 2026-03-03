const test = require("node:test");
const assert = require("node:assert/strict");

const { deviceLabelToPreference, suggestionsPanelState } = require("../src/lib/workspace/deployHelpers.js");

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
