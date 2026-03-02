function metricKeyForTask(task) {
  if (task === "detection") return "val_map";
  if (task === "segmentation") return "val_iou";
  return "val_accuracy";
}

const BOUNDED_METRIC_KEYS = new Set([
  "val_accuracy",
  "val_macro_f1",
  "val_macro_precision",
  "val_macro_recall",
  "val_map",
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
  const maxEpoch = Math.max(...rows.map((row) => Number.parseInt(String(row.epoch), 10)).filter((epoch) => Number.isFinite(epoch)));
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
  indexCheckpointsByKind,
};
