function metricKeyForTask(task) {
  if (task === "detection") return "val_map";
  if (task === "segmentation") return "val_iou";
  return "val_accuracy";
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

function metricDomain(metrics, seriesKeys) {
  const values = collectMetricValues(metrics, seriesKeys);
  if (values.length === 0) return { min: 0, max: 1 };
  let min = Math.min(...values);
  let max = Math.max(...values);
  if (min === max) {
    min = Math.max(0, min - 0.1);
    max = Math.min(1, max + 0.1);
  }
  return { min, max };
}

function buildLinePoints(metrics, seriesKey, options = {}) {
  const width = Number.isFinite(options.width) ? options.width : 640;
  const height = Number.isFinite(options.height) ? options.height : 240;
  const padding = Number.isFinite(options.padding) ? options.padding : 24;
  const rows = Array.isArray(metrics) ? metrics : [];
  if (rows.length === 0) return "";

  const seriesKeys = options.seriesKeys && Array.isArray(options.seriesKeys) ? options.seriesKeys : [seriesKey];
  const domain = metricDomain(rows, seriesKeys);
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
    const x = padding + ((epoch - 1) / Math.max(1, maxEpoch - 1)) * chartWidth;
    const y = padding + ((domain.max - value) / range) * chartHeight;
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
  metricKeyForTask,
  mergeMetricPoints,
  metricDomain,
  buildLinePoints,
  indexCheckpointsByKind,
};

