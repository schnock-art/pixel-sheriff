const test = require("node:test");
const assert = require("node:assert/strict");

const {
  bestRunByMetric,
  buildAnalyticsSummary,
  defaultSelectedRunIds,
  filterAnalyticsItems,
  seriesPoints,
  scatterPoints,
} = require("../src/lib/workspace/experimentAnalytics.js");

test("defaultSelectedRunIds picks latest completed runs", () => {
  const selected = defaultSelectedRunIds(
    [
      { experiment_id: "a", status: "completed", updated_at: "2025-01-01T00:00:00Z" },
      { experiment_id: "b", status: "failed", updated_at: "2025-01-03T00:00:00Z" },
      { experiment_id: "c", status: "completed", updated_at: "2025-01-02T00:00:00Z" },
      { experiment_id: "d", status: "completed", updated_at: "2025-01-04T00:00:00Z" },
    ],
    2,
  );
  assert.deepEqual(selected, ["d", "c"]);
});

test("buildAnalyticsSummary computes best and lowest values", () => {
  const summary = buildAnalyticsSummary([
    {
      status: "completed",
      final: { val_accuracy: 0.7, val_loss: 0.5 },
      series: { val_accuracy: [0.7, 0.8], val_loss: [0.5, 0.45] },
    },
    {
      status: "failed",
      final: { val_accuracy: 0.2, val_loss: 0.9 },
      series: {},
    },
  ]);
  assert.equal(summary.totalRuns, 2);
  assert.equal(summary.failures, 1);
  assert.equal(summary.bestAccuracy, 0.8);
  assert.equal(summary.lowestValLoss, 0.45);
});

test("seriesPoints and bestRunByMetric work with chart series arrays", () => {
  const item = {
    experiment_id: "run-1",
    series: { epochs: [1, 2, 3], val_accuracy: [0.5, 0.6, 0.75] },
  };
  const points = seriesPoints(item, "val_accuracy");
  assert.equal(points.length, 3);
  assert.equal(points[2].value, 0.75);

  const bestRun = bestRunByMetric(
    [
      item,
      {
        experiment_id: "run-2",
        series: { epochs: [1, 2, 3], val_accuracy: [0.4, 0.5, 0.6] },
      },
    ],
    "val_accuracy",
  );
  assert.equal(bestRun, "run-1");
});

test("filterAnalyticsItems and scatterPoints apply model/failed filters", () => {
  const filtered = filterAnalyticsItems(
    [
      { experiment_id: "a", model_id: "m1", model_name: "M1", status: "completed", updated_at: "2025-01-01T00:00:00Z", config: {}, series: {}, final: {} },
      { experiment_id: "b", model_id: "m2", model_name: "M2", status: "failed", updated_at: "2025-01-02T00:00:00Z", config: {}, series: {}, final: {} },
    ],
    { modelId: "m2", showFailed: false },
  );
  assert.equal(filtered.length, 0);

  const points = scatterPoints(
    [
      {
        experiment_id: "a",
        name: "run-a",
        status: "completed",
        config: { optimizer: { lr: 0.001 }, batch_size: 32, epochs: 10, augmentation: "light" },
        best: { metric_name: "val_accuracy", metric_value: 0.84 },
        final: { val_accuracy: 0.81 },
        series: { epochs: [1, 2], val_accuracy: [0.6, 0.84] },
      },
    ],
    "learning_rate",
    "best_val_accuracy",
  );
  assert.equal(points.length, 1);
  assert.equal(points[0].x, 0.001);
  assert.equal(points[0].y, 0.84);
});

test("scatterPoints maps custom augmentation to a dedicated bucket", () => {
  const points = scatterPoints(
    [
      {
        experiment_id: "custom-run",
        name: "custom-run",
        status: "completed",
        config: {
          augmentation: "custom",
          augmentation_mode: "custom",
          augmentation_summary: "custom: rotate@1.00(8)",
        },
        best: { metric_name: "val_accuracy", metric_value: 0.9 },
        final: { val_accuracy: 0.88 },
        series: { epochs: [1, 2], val_accuracy: [0.7, 0.9] },
      },
    ],
    "augmentation",
    "best_val_accuracy",
  );
  assert.equal(points.length, 1);
  assert.equal(points[0].x, 4);
  assert.equal(points[0].augmentationMode, "custom");
  assert.equal(points[0].augmentationSummary, "custom: rotate@1.00(8)");
});
