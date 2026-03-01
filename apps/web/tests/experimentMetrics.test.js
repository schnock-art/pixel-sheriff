const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildLinePoints,
  indexCheckpointsByKind,
  mergeMetricPoints,
  metricDomain,
  metricKeyForTask,
} = require("../src/lib/workspace/experimentMetrics.js");

test("metricKeyForTask maps task to primary metric", () => {
  assert.equal(metricKeyForTask("classification"), "val_accuracy");
  assert.equal(metricKeyForTask("detection"), "val_map");
  assert.equal(metricKeyForTask("segmentation"), "val_iou");
});

test("mergeMetricPoints dedupes by epoch and keeps latest values", () => {
  const merged = mergeMetricPoints(
    [
      { epoch: 1, val_loss: 0.9, val_accuracy: 0.6 },
      { epoch: 2, val_loss: 0.8, val_accuracy: 0.65 },
    ],
    [
      { epoch: 2, val_loss: 0.75, val_accuracy: 0.67 },
      { epoch: 3, val_loss: 0.7, val_accuracy: 0.71 },
    ],
  );
  assert.equal(merged.length, 3);
  assert.equal(merged[1].epoch, 2);
  assert.equal(merged[1].val_loss, 0.75);
  assert.equal(merged[2].epoch, 3);
});

test("metricDomain computes min/max from selected series", () => {
  const domain = metricDomain(
    [
      { epoch: 1, val_accuracy: 0.5, val_loss: 0.9 },
      { epoch: 2, val_accuracy: 0.7, val_loss: 0.8 },
    ],
    ["val_accuracy", "val_loss"],
  );
  assert.equal(domain.min, 0.5);
  assert.equal(domain.max, 0.9);
});

test("buildLinePoints generates svg points for chosen series", () => {
  const points = buildLinePoints(
    [
      { epoch: 1, val_accuracy: 0.5 },
      { epoch: 2, val_accuracy: 0.6 },
      { epoch: 3, val_accuracy: 0.8 },
    ],
    "val_accuracy",
    { width: 300, height: 120, padding: 10, seriesKeys: ["val_accuracy"] },
  );
  assert.ok(points.includes(","));
  assert.ok(points.split(" ").length >= 3);
});

test("indexCheckpointsByKind indexes known checkpoint kinds", () => {
  const indexed = indexCheckpointsByKind([
    { kind: "best_metric", epoch: 5, metric_name: "val_accuracy", value: 0.8 },
    { kind: "latest", epoch: 6, metric_name: "val_accuracy", value: 0.79 },
  ]);
  assert.equal(indexed.best_metric.epoch, 5);
  assert.equal(indexed.best_loss, null);
  assert.equal(indexed.latest.epoch, 6);
});

