function metricKeyForTask(task) {
  if (task === "detection") return "val_map_50_95";
  if (task === "segmentation") return "val_iou";
  return "val_accuracy";
}

const BOUNDED_METRIC_KEYS = new Set([
  "val_accuracy",
  "val_macro_f1",
  "val_macro_precision",
  "val_macro_recall",
  "val_map",
  "val_map_50_95",
  "val_precision",
  "val_recall",
  "val_matched_mean_iou",
  "val_iou",
]);

function isLossMetricKey(key) {
  return typeof key === "string" && key.toLowerCase().includes("loss");
}

function isBoundedMetricKey(key) {
  return typeof key === "string" && BOUNDED_METRIC_KEYS.has(key);
}

function _safeNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function _plotValue(value, useLog) {
  if (!useLog) return value;
  return Math.log10(Math.max(1e-9, value));
}

function _collectSeriesValues(rowsOrPoints, key) {
  const values = [];
  for (const row of Array.isArray(rowsOrPoints) ? rowsOrPoints : []) {
    if (!row || typeof row !== "object") continue;
    let candidate = null;
    if (typeof key === "string" && key.length > 0) candidate = _safeNumber(row[key]);
    if (candidate == null && key == null && "value" in row) candidate = _safeNumber(row.value);
    if (candidate != null) values.push(candidate);
  }
  return values;
}

function isBoundedSeries(rowsOrPoints, key) {
  if (typeof key === "string" && isLossMetricKey(key)) return false;
  if (typeof key === "string" && isBoundedMetricKey(key)) return true;
  const values = _collectSeriesValues(rowsOrPoints, key);
  if (values.length === 0) return false;
  return values.every((value) => value >= 0 && value <= 1);
}

function computeSeriesDomain(values, options = {}) {
  const useLog = options.useLog === true;
  const clamp01 = options.clamp01 === true && !useLog;
  const numeric = (Array.isArray(values) ? values : [])
    .map(_safeNumber)
    .filter((value) => value != null);

  if (clamp01) return { min: 0, max: 1 };
  if (numeric.length === 0) return useLog ? { min: -6, max: 0 } : { min: 0, max: 1 };

  if (useLog) {
    const transformed = numeric
      .filter((value) => value > 0)
      .map((value) => _plotValue(value, true));
    if (transformed.length === 0) return { min: -6, max: 0 };
    let min = Math.min(...transformed);
    let max = Math.max(...transformed);
    if (min === max) {
      min -= 0.5;
      max += 0.5;
    }
    return { min, max };
  }

  let min = Math.min(...numeric);
  let max = Math.max(...numeric);
  if (min === max) {
    const pad = Math.max(0.1, Math.abs(min) * 0.1);
    min -= pad;
    max += pad;
  }
  return { min, max };
}

function buildTicks(domain, options = {}) {
  const useLog = options.useLog === true;
  const count = Math.max(2, Number.isFinite(options.count) ? Math.floor(options.count) : 5);
  const clamp01 = options.clamp01 === true && !useLog;
  if (clamp01) {
    return Array.from({ length: count }, (_, index) => index / (count - 1));
  }
  const min = _safeNumber(domain?.min);
  const max = _safeNumber(domain?.max);
  if (min == null || max == null) return [0, 0.25, 0.5, 0.75, 1];
  if (min === max) return Array.from({ length: count }, () => min);
  return Array.from({ length: count }, (_, index) => min + ((max - min) * (index / (count - 1))));
}

function formatTick(value, options = {}) {
  const useLog = options.useLog === true;
  const bounded = options.bounded === true && !useLog;
  const numeric = _safeNumber(value);
  if (numeric == null) return "-";
  if (bounded) return numeric.toFixed(2);
  const abs = Math.abs(numeric);
  if (abs >= 1000) return numeric.toFixed(0);
  if (abs >= 100) return numeric.toFixed(1);
  if (abs >= 10) return numeric.toFixed(2);
  return numeric.toFixed(3);
}

function mergeMetricPoints(existing, incoming) {
  const byEpoch = new Map();
  for (const row of Array.isArray(existing) ? existing : []) {
    if (!row || typeof row !== "object") continue;
    const epoch = Number.parseInt(String(row.epoch), 10);
    if (!Number.isFinite(epoch) || epoch < 1) continue;
    byEpoch.set(epoch, { ...row, epoch });
  }
  for (const row of Array.isArray(incoming) ? incoming : []) {
    if (!row || typeof row !== "object") continue;
    const epoch = Number.parseInt(String(row.epoch), 10);
    if (!Number.isFinite(epoch) || epoch < 1) continue;
    byEpoch.set(epoch, { ...byEpoch.get(epoch), ...row, epoch });
  }
  return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
}

function collectMetricValues(metrics, seriesKeys) {
  const values = [];
  for (const row of Array.isArray(metrics) ? metrics : []) {
    for (const key of seriesKeys) {
      const value = row?.[key];
      if (typeof value === "number" && Number.isFinite(value)) values.push(value);
    }
  }
  return values;
}

function metricDomain(metrics, seriesKeys, options = {}) {
  const values = collectMetricValues(metrics, seriesKeys);
  const clampBounded = options.clampBounded === true;
  const isSingleBoundedSeries =
    clampBounded &&
    Array.isArray(seriesKeys) &&
    seriesKeys.length === 1 &&
    isBoundedMetricKey(seriesKeys[0]);
  return computeSeriesDomain(values, {
    useLog: options.useLog === true,
    clamp01: isSingleBoundedSeries && options.useLog !== true,
  });
}

function buildLinePoints(metrics, seriesKey, options = {}) {
  const width = Number.isFinite(options.width) ? options.width : 640;
  const height = Number.isFinite(options.height) ? options.height : 240;
  const padding = Number.isFinite(options.padding) ? options.padding : 24;
  const rows = Array.isArray(metrics) ? metrics : [];
  if (rows.length === 0) return "";

  const useLog = options.useLog === true;
  const seriesKeys = options.seriesKeys && Array.isArray(options.seriesKeys) ? options.seriesKeys : [seriesKey];
  const domain = options.domain && Number.isFinite(options.domain.min) && Number.isFinite(options.domain.max)
    ? options.domain
    : metricDomain(rows, seriesKeys, { useLog, clampBounded: true });
  const range = Math.max(1e-9, domain.max - domain.min);
  const maxEpoch = Number.isFinite(options.maxEpoch) && options.maxEpoch >= 1
    ? Math.max(1, Math.floor(options.maxEpoch))
    : Math.max(...rows.map((row) => Number.parseInt(String(row.epoch), 10)).filter((epoch) => Number.isFinite(epoch)));
  const chartWidth = Math.max(1, width - (padding * 2));
  const chartHeight = Math.max(1, height - (padding * 2));

  const points = [];
  for (const row of rows) {
    const epoch = Number.parseInt(String(row.epoch), 10);
    const value = row?.[seriesKey];
    if (!Number.isFinite(epoch) || epoch < 1) continue;
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    const plotted = _plotValue(value, useLog);
    const x = padding + ((epoch - 1) / Math.max(1, maxEpoch - 1)) * chartWidth;
    const y = padding + ((domain.max - plotted) / range) * chartHeight;
    points.push(`${x.toFixed(2)},${y.toFixed(2)}`);
  }
  return points.join(" ");
}

function metricValueByKey(row, key) {
  if (!row || typeof row !== "object") return null;
  if (key === "train_loss") return _safeNumber(row.train_loss);
  if (key === "train_accuracy") return _safeNumber(row.train_accuracy);
  if (key === "val_loss") return _safeNumber(row.val_loss);
  if (key === "val_accuracy") return _safeNumber(row.val_accuracy);
  if (key === "val_macro_f1") return _safeNumber(row.val_macro_f1);
  if (key === "val_macro_precision") return _safeNumber(row.val_macro_precision);
  if (key === "val_macro_recall") return _safeNumber(row.val_macro_recall);
  if (key === "val_map") return _safeNumber(row.val_map) ?? _safeNumber(row.mAP50);
  if (key === "val_map_50_95") return _safeNumber(row.val_map_50_95) ?? _safeNumber(row.mAP50_95);
  if (key === "val_precision") return _safeNumber(row.val_precision) ?? _safeNumber(row.precision);
  if (key === "val_recall") return _safeNumber(row.val_recall) ?? _safeNumber(row.recall);
  if (key === "val_matched_mean_iou") return _safeNumber(row.val_matched_mean_iou) ?? _safeNumber(row.matched_mean_iou);
  if (key === "val_tp") return _safeNumber(row.val_tp) ?? _safeNumber(row.tp);
  if (key === "val_fp") return _safeNumber(row.val_fp) ?? _safeNumber(row.fp);
  if (key === "val_fn") return _safeNumber(row.val_fn) ?? _safeNumber(row.fn);
  if (key === "val_duplicate_fp") return _safeNumber(row.val_duplicate_fp) ?? _safeNumber(row.duplicate_fp);
  if (key === "val_iou") return _safeNumber(row.val_iou);
  if (key === "epoch_seconds") return _safeNumber(row.epoch_seconds);
  if (key === "eta_seconds") return _safeNumber(row.eta_seconds);
  return null;
}

function findNearestMetricEpoch(metrics, approximateEpoch) {
  const target = _safeNumber(approximateEpoch);
  if (target == null) return null;
  let nearestEpoch = null;
  let bestDistance = Number.POSITIVE_INFINITY;
  for (const row of Array.isArray(metrics) ? metrics : []) {
    const epoch = Number.parseInt(String(row?.epoch), 10);
    if (!Number.isFinite(epoch) || epoch < 1) continue;
    const distance = Math.abs(epoch - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      nearestEpoch = epoch;
    }
  }
  return nearestEpoch;
}

function buildExperimentMetricChartModel(metrics, options = {}) {
  const rows = Array.isArray(metrics) ? metrics : [];
  const primaryMetricKey = typeof options.primaryMetricKey === "string" ? options.primaryMetricKey : "val_accuracy";
  const primaryMetricLabel = typeof options.primaryMetricLabel === "string" ? options.primaryMetricLabel : primaryMetricKey.replace("val_", "val ");
  const showPrimary = options.showPrimary !== false;
  const showValLoss = options.showValLoss !== false;
  const chartWidth = Number.isFinite(options.chartWidth) ? options.chartWidth : 760;
  const chartHeight = Number.isFinite(options.chartHeight) ? options.chartHeight : 280;
  const chartPadding = Number.isFinite(options.chartPadding) ? options.chartPadding : 44;
  const chartInnerWidth = chartWidth - (chartPadding * 2);
  const chartInnerHeight = chartHeight - (chartPadding * 2);
  const primaryColor = typeof options.primaryColor === "string" ? options.primaryColor : "#2f6fca";
  const lossColor = typeof options.lossColor === "string" ? options.lossColor : "#c96262";

  const chartKeys = [];
  if (showPrimary) chartKeys.push(primaryMetricKey);
  if (showValLoss) chartKeys.push("val_loss");

  const epochs = rows
    .map((row) => Number.parseInt(String(row?.epoch), 10))
    .filter((epoch) => Number.isFinite(epoch) && epoch >= 1);
  const chartMaxEpoch = epochs.length > 0 ? Math.max(...epochs) : 1;
  const primaryMetricIsBounded = isBoundedMetricKey(primaryMetricKey);
  const useSecondaryAxis = showPrimary && showValLoss && primaryMetricIsBounded;

  const primarySeriesValues = rows.map((row) => metricValueByKey(row, primaryMetricKey)).filter((value) => value != null);
  const lossSeriesValues = rows.map((row) => metricValueByKey(row, "val_loss")).filter((value) => value != null);
  const combinedSeriesValues = [...primarySeriesValues, ...lossSeriesValues];

  const primaryDomain = computeSeriesDomain(primarySeriesValues, {
    useLog: false,
    clamp01: primaryMetricIsBounded,
  });
  const lossDomain = computeSeriesDomain(lossSeriesValues, {
    useLog: false,
    clamp01: false,
  });
  const combinedDomain = computeSeriesDomain(combinedSeriesValues, {
    useLog: false,
    clamp01: false,
  });

  let leftAxisDomain = combinedDomain;
  if (useSecondaryAxis) leftAxisDomain = primaryDomain;
  else if (showPrimary && !showValLoss) leftAxisDomain = primaryDomain;
  else if (!showPrimary && showValLoss) leftAxisDomain = lossDomain;

  const rightAxisDomain = useSecondaryAxis ? lossDomain : null;
  const leftAxisTicks = buildTicks(leftAxisDomain, {
    count: 5,
    clamp01: primaryMetricIsBounded && (useSecondaryAxis || (showPrimary && !showValLoss)),
  });
  const rightAxisTicks = rightAxisDomain ? buildTicks(rightAxisDomain, { count: 5 }) : [];
  const xTickValues = Array.from(new Set(
    buildTicks({ min: 1, max: chartMaxEpoch }, { count: 5 }).map((tick) => Math.max(1, Math.round(tick))),
  ));

  const seriesLegend = [
    {
      key: primaryMetricKey,
      label: primaryMetricLabel,
      color: primaryColor,
      enabled: showPrimary,
      axis: "left",
    },
    {
      key: "val_loss",
      label: "val loss",
      color: lossColor,
      enabled: showValLoss,
      axis: useSecondaryAxis ? "right" : "left",
    },
  ];

  const primaryLinePoints = showPrimary
    ? buildLinePoints(rows, primaryMetricKey, {
        width: chartWidth,
        height: chartHeight,
        padding: chartPadding,
        seriesKeys: chartKeys,
        domain: leftAxisDomain,
        useLog: false,
      })
    : "";
  const valLossLinePoints = showValLoss
    ? buildLinePoints(rows, "val_loss", {
        width: chartWidth,
        height: chartHeight,
        padding: chartPadding,
        seriesKeys: chartKeys,
        domain: useSecondaryAxis ? lossDomain : leftAxisDomain,
        useLog: false,
      })
    : "";

  return {
    chartWidth,
    chartHeight,
    chartPadding,
    chartInnerWidth,
    chartInnerHeight,
    chartKeys,
    chartMaxEpoch,
    primaryMetricIsBounded,
    useSecondaryAxis,
    leftAxisDomain,
    rightAxisDomain,
    leftAxisTicks,
    rightAxisTicks,
    xTickValues,
    seriesLegend,
    primaryLinePoints,
    valLossLinePoints,
  };
}

function buildExperimentMetricHoverModel(metrics, options = {}) {
  const hoveredEpoch = Number.isFinite(options.hoveredEpoch) ? options.hoveredEpoch : null;
  const rows = Array.isArray(metrics) ? metrics : [];
  const seriesLegend = Array.isArray(options.seriesLegend) ? options.seriesLegend : [];
  const chartWidth = Number.isFinite(options.chartWidth) ? options.chartWidth : 760;
  const chartHeight = Number.isFinite(options.chartHeight) ? options.chartHeight : 280;
  const chartPadding = Number.isFinite(options.chartPadding) ? options.chartPadding : 44;
  const chartInnerWidth = Number.isFinite(options.chartInnerWidth) ? options.chartInnerWidth : chartWidth - (chartPadding * 2);
  const chartInnerHeight = Number.isFinite(options.chartInnerHeight) ? options.chartInnerHeight : chartHeight - (chartPadding * 2);
  const chartMaxEpoch = Number.isFinite(options.chartMaxEpoch) ? options.chartMaxEpoch : 1;
  const useSecondaryAxis = options.useSecondaryAxis === true;
  const leftAxisDomain = options.leftAxisDomain && Number.isFinite(options.leftAxisDomain.min) && Number.isFinite(options.leftAxisDomain.max)
    ? options.leftAxisDomain
    : { min: 0, max: 1 };
  const lossDomain = options.lossDomain && Number.isFinite(options.lossDomain.min) && Number.isFinite(options.lossDomain.max)
    ? options.lossDomain
    : leftAxisDomain;

  let hoveredMetric = null;
  if (hoveredEpoch != null) {
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const row of rows) {
      const epoch = Number.parseInt(String(row?.epoch), 10);
      if (!Number.isFinite(epoch) || epoch < 1) continue;
      const distance = Math.abs(epoch - hoveredEpoch);
      if (distance < bestDistance) {
        bestDistance = distance;
        hoveredMetric = row;
      }
    }
  }

  const hoveredEpochValue = hoveredMetric ? Number.parseInt(String(hoveredMetric.epoch), 10) : null;
  const hoveredX = hoveredEpochValue == null
    ? null
    : chartPadding + (((hoveredEpochValue - 1) / Math.max(1, chartMaxEpoch - 1)) * chartInnerWidth);
  const hoveredSeriesValues = hoveredMetric
    ? seriesLegend
      .filter((series) => series.enabled)
      .map((series) => ({
        key: series.key,
        label: series.label,
        color: series.color,
        value: metricValueByKey(hoveredMetric, series.key),
      }))
      .filter((row) => row.value != null)
    : [];
  const hoveredPlotRows = hoveredSeriesValues.map((row) => {
    const domain = useSecondaryAxis && row.key === "val_loss" ? lossDomain : leftAxisDomain;
    const range = Math.max(1e-9, domain.max - domain.min);
    const y = chartPadding + (((domain.max - row.value) / range) * chartInnerHeight);
    return { ...row, y };
  });

  let hoverTooltip = null;
  if (hoveredX != null && hoveredEpochValue != null && hoveredPlotRows.length > 0) {
    const tooltipWidth = 196;
    const tooltipLineHeight = 15;
    const tooltipHeight = 28 + (hoveredPlotRows.length * tooltipLineHeight);
    let x = hoveredX + 10;
    if (x + tooltipWidth > chartWidth - 6) x = hoveredX - tooltipWidth - 10;
    let y = chartPadding + 10;
    if (y + tooltipHeight > chartHeight - 8) y = chartHeight - tooltipHeight - 8;
    hoverTooltip = { x, y, width: tooltipWidth, height: tooltipHeight };
  }

  return {
    hoveredMetric,
    hoveredEpochValue,
    hoveredX,
    hoveredSeriesValues,
    hoveredPlotRows,
    hoverTooltip,
  };
}

function indexCheckpointsByKind(checkpoints) {
  const index = {
    best_metric: null,
    best_loss: null,
    latest: null,
  };
  for (const row of Array.isArray(checkpoints) ? checkpoints : []) {
    const kind = row?.kind;
    if (kind === "best_metric" || kind === "best_loss" || kind === "latest") {
      index[kind] = row;
    }
  }
  return index;
}

module.exports = {
  isLossMetricKey,
  isBoundedMetricKey,
  isBoundedSeries,
  computeSeriesDomain,
  buildTicks,
  formatTick,
  metricKeyForTask,
  mergeMetricPoints,
  metricDomain,
  buildLinePoints,
  metricValueByKey,
  findNearestMetricEpoch,
  buildExperimentMetricChartModel,
  buildExperimentMetricHoverModel,
  indexCheckpointsByKind,
};
