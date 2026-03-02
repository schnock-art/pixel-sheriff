const test = require("node:test");
const assert = require("node:assert/strict");

const { filterPredictionRows, normalizeConfusion } = require("../src/lib/workspace/experimentDashboard.js");

test("normalizeConfusion handles none/by_true/by_pred and zero-sum safely", () => {
  const matrix = [
    [2, 2],
    [0, 0],
  ];
  const none = normalizeConfusion(matrix, "none");
  assert.deepEqual(none, matrix);

  const byTrue = normalizeConfusion(matrix, "by_true");
  assert.equal(byTrue[0][0], 0.5);
  assert.equal(byTrue[0][1], 0.5);
  assert.equal(byTrue[1][0], 0);
  assert.equal(byTrue[1][1], 0);

  const byPred = normalizeConfusion(matrix, "by_pred");
  assert.equal(byPred[0][0], 1);
  assert.equal(byPred[0][1], 1);
  assert.equal(byPred[1][0], 0);
  assert.equal(byPred[1][1], 0);
});

test("filterPredictionRows applies mode/class filters and ordering", () => {
  const rows = [
    { asset_id: "a", true_class_index: 0, pred_class_index: 1, confidence: 0.9 },
    { asset_id: "b", true_class_index: 0, pred_class_index: 0, confidence: 0.2 },
    { asset_id: "c", true_class_index: 0, pred_class_index: 0, confidence: 0.5 },
  ];

  const wrong = filterPredictionRows(rows, { mode: "misclassified", limit: 10 });
  assert.equal(wrong.length, 1);
  assert.equal(wrong[0].asset_id, "a");

  const lowestCorrect = filterPredictionRows(rows, { mode: "lowest_confidence_correct", limit: 10 });
  assert.equal(lowestCorrect.length, 2);
  assert.equal(lowestCorrect[0].asset_id, "b");

  const filtered = filterPredictionRows(rows, { mode: "lowest_confidence_correct", trueClassIndex: 0, predClassIndex: 0, limit: 1 });
  assert.equal(filtered.length, 1);
  assert.equal(filtered[0].asset_id, "b");
});
