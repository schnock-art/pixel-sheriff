const test = require("node:test");
const assert = require("node:assert/strict");

const {
  buildExperimentMetricChartModel,
  buildExperimentMetricHoverModel,
  buildTicks,
  buildLinePoints,
  computeSeriesDomain,
  findNearestMetricEpoch,
  formatTick,
  indexCheckpointsByKind,
  isBoundedMetricKey,
  isBoundedSeries,
  isLossMetricKey,
  mergeMetricPoints,
  metricDomain,
  metricKeyForTask,
  metricValueByKey,
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

test("bounded/loss metric key helpers identify expected keys", () => {
  assert.equal(isBoundedMetricKey("val_accuracy"), true);
  assert.equal(isBoundedMetricKey("val_macro_f1"), true);
  assert.equal(isBoundedMetricKey("val_map"), true);
  assert.equal(isBoundedMetricKey("val_loss"), false);
  assert.equal(isLossMetricKey("val_loss"), true);
  assert.equal(isLossMetricKey("train_loss"), true);
  assert.equal(isLossMetricKey("val_accuracy"), false);
});

test("isBoundedSeries detects bounded value ranges for unknown keys", () => {
  assert.equal(
    isBoundedSeries(
      [{ score: 0.1 }, { score: 0.8 }, { score: 1.0 }],
      "score",
    ),
    true,
  );
  assert.equal(
    isBoundedSeries(
      [{ score: 0.1 }, { score: 1.4 }],
      "score",
    ),
    false,
  );
});

test("computeSeriesDomain clamps bounded linear domain and keeps log dynamic", () => {
  const boundedLinear = computeSeriesDomain([0.2, 0.8], { useLog: false, clamp01: true });
  assert.equal(boundedLinear.min, 0);
  assert.equal(boundedLinear.max, 1);

  const boundedLog = computeSeriesDomain([0.2, 0.8], { useLog: true, clamp01: true });
  assert.ok(boundedLog.min < boundedLog.max);
  assert.notEqual(boundedLog.min, 0);
});

test("buildTicks creates deterministic 5-tick bounded and general sets", () => {
  const bounded = buildTicks({ min: 0, max: 1 }, { count: 5, clamp01: true });
  assert.deepEqual(bounded, [0, 0.25, 0.5, 0.75, 1]);

  const general = buildTicks({ min: 2, max: 6 }, { count: 5 });
  assert.deepEqual(general, [2, 3, 4, 5, 6]);
});

test("buildTicks supports log domains and formatTick is stable", () => {
  const logTicks = buildTicks({ min: -3, max: 0 }, { useLog: true, count: 5 });
  assert.equal(logTicks.length, 5);
  assert.equal(formatTick(logTicks[0], { useLog: true }).length > 0, true);
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

test("buildLinePoints respects explicit domain and supports log plotting", () => {
  const linear = buildLinePoints(
    [
      { epoch: 1, val_loss: 0.1 },
      { epoch: 2, val_loss: 1.0 },
    ],
    "val_loss",
    { width: 300, height: 120, padding: 10, domain: { min: 0, max: 1 }, useLog: false },
  );
  const log = buildLinePoints(
    [
      { epoch: 1, val_loss: 0.1 },
      { epoch: 2, val_loss: 1.0 },
    ],
    "val_loss",
    { width: 300, height: 120, padding: 10, domain: { min: -1, max: 0 }, useLog: true },
  );
  assert.ok(linear.includes(","));
  assert.ok(log.includes(","));
  assert.notEqual(linear, log);
});

test("metricValueByKey handles aliases and runtime metrics", () => {
  const row = {
    epoch: 2,
    val_accuracy: 0.8,
    val_map_50_95: 0.42,
    mAP50: 0.61,
    epoch_seconds: 12.5,
  };
  assert.equal(metricValueByKey(row, "val_accuracy"), 0.8);
  assert.equal(metricValueByKey({ mAP50: 0.61 }, "val_map"), 0.61);
  assert.equal(metricValueByKey(row, "val_map_50_95"), 0.42);
  assert.equal(metricValueByKey(row, "epoch_seconds"), 12.5);
});

test("findNearestMetricEpoch picks the closest valid epoch", () => {
  const metrics = [{ epoch: 1 }, { epoch: 4 }, { epoch: "9" }, { epoch: 0 }];
  assert.equal(findNearestMetricEpoch(metrics, 3.2), 4);
  assert.equal(findNearestMetricEpoch(metrics, 8.6), 9);
  assert.equal(findNearestMetricEpoch([], 2), null);
});

test("buildExperimentMetricChartModel builds dual-axis detail chart state", () => {
  const chart = buildExperimentMetricChartModel(
    [
      { epoch: 1, val_accuracy: 0.55, val_loss: 1.2 },
      { epoch: 2, val_accuracy: 0.7, val_loss: 0.9 },
      { epoch: 3, val_accuracy: 0.82, val_loss: 0.65 },
    ],
    {
      primaryMetricKey: "val_accuracy",
      primaryMetricLabel: "val accuracy",
      showPrimary: true,
      showValLoss: true,
      chartWidth: 300,
      chartHeight: 120,
      chartPadding: 10,
    },
  );

  assert.equal(chart.chartMaxEpoch, 3);
  assert.equal(chart.primaryMetricIsBounded, true);
  assert.equal(chart.useSecondaryAxis, true);
  assert.equal(chart.rightAxisTicks.length, 5);
  assert.ok(chart.primaryLinePoints.includes(","));
  assert.ok(chart.valLossLinePoints.includes(","));
});

test("buildExperimentMetricHoverModel computes marker positions and tooltip bounds", () => {
  const metrics = [
    { epoch: 1, val_accuracy: 0.55, val_loss: 1.2 },
    { epoch: 2, val_accuracy: 0.7, val_loss: 0.9 },
    { epoch: 3, val_accuracy: 0.82, val_loss: 0.65 },
  ];
  const chart = buildExperimentMetricChartModel(metrics, {
    primaryMetricKey: "val_accuracy",
    primaryMetricLabel: "val accuracy",
    showPrimary: true,
    showValLoss: true,
    chartWidth: 500,
    chartHeight: 120,
    chartPadding: 10,
  });
  const hover = buildExperimentMetricHoverModel(metrics, {
    hoveredEpoch: 2,
    seriesLegend: chart.seriesLegend,
    chartWidth: chart.chartWidth,
    chartHeight: chart.chartHeight,
    chartPadding: chart.chartPadding,
    chartInnerWidth: chart.chartInnerWidth,
    chartInnerHeight: chart.chartInnerHeight,
    chartMaxEpoch: chart.chartMaxEpoch,
    leftAxisDomain: chart.leftAxisDomain,
    lossDomain: chart.rightAxisDomain,
    useSecondaryAxis: chart.useSecondaryAxis,
  });

  assert.equal(hover.hoveredEpochValue, 2);
  assert.equal(hover.hoveredPlotRows.length, 2);
  assert.ok(Number.isFinite(hover.hoveredX));
  assert.ok(hover.hoverTooltip);
  assert.ok(hover.hoverTooltip.x >= 0);
  assert.ok(hover.hoverTooltip.y >= 0);
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
