function asNumber(value) {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function metricDirection(metricKey) {
  if (metricKey === "val_loss" || metricKey === "train_loss" || metricKey.endsWith("_loss")) return "min";
  return "max";
}

function sortByUpdatedDesc(items) {
  return [...(Array.isArray(items) ? items : [])].sort((a, b) => {
    const aTs = Date.parse(String(a?.updated_at ?? ""));
    const bTs = Date.parse(String(b?.updated_at ?? ""));
    if (!Number.isFinite(aTs) && !Number.isFinite(bTs)) return 0;
    if (!Number.isFinite(aTs)) return 1;
    if (!Number.isFinite(bTs)) return -1;
    return bTs - aTs;
  });
}

function defaultSelectedRunIds(items, count = 3) {
  const completed = sortByUpdatedDesc(items).filter((item) => item?.status === "completed");
  const picked = completed.slice(0, Math.max(1, count));
  return picked.map((item) => String(item.experiment_id));
}

function filterAnalyticsItems(items, options = {}) {
  const modelId = options.modelId ?? "";
  const showFailed = options.showFailed === true;
  return sortByUpdatedDesc(items).filter((item) => {
    if (!item || typeof item !== "object") return false;
    if (modelId && String(item.model_id) !== String(modelId)) return false;
    if (!showFailed && String(item.status) === "failed") return false;
    return true;
  });
}

function buildAnalyticsSummary(items) {
  const rows = Array.isArray(items) ? items : [];
  let bestAccuracy = null;
  let lowestValLoss = null;
  let failures = 0;
  for (const item of rows) {
    if (!item || typeof item !== "object") continue;
    if (item.status === "failed") failures += 1;

    const series = item.series && typeof item.series === "object" ? item.series : {};
    const valAcc = asNumber(item?.final?.val_accuracy ?? null);
    const valLoss = asNumber(item?.final?.val_loss ?? null);
    if (valAcc != null) bestAccuracy = bestAccuracy == null ? valAcc : Math.max(bestAccuracy, valAcc);
    if (valLoss != null) lowestValLoss = lowestValLoss == null ? valLoss : Math.min(lowestValLoss, valLoss);

    const seriesAcc = Array.isArray(series.val_accuracy) ? series.val_accuracy.map(asNumber).filter((v) => v != null) : [];
    const seriesLoss = Array.isArray(series.val_loss) ? series.val_loss.map(asNumber).filter((v) => v != null) : [];
    for (const value of seriesAcc) {
      bestAccuracy = bestAccuracy == null ? value : Math.max(bestAccuracy, value);
    }
    for (const value of seriesLoss) {
      lowestValLoss = lowestValLoss == null ? value : Math.min(lowestValLoss, value);
    }
  }
  return {
    totalRuns: rows.length,
    failures,
    bestAccuracy,
    lowestValLoss,
  };
}

function seriesPoints(item, metricKey) {
  const series = item?.series && typeof item.series === "object" ? item.series : {};
  const epochs = Array.isArray(series.epochs) ? series.epochs : [];
  const values = Array.isArray(series[metricKey]) ? series[metricKey] : [];
  const points = [];
  for (let index = 0; index < Math.min(epochs.length, values.length); index += 1) {
    const epoch = asNumber(epochs[index]);
    const value = asNumber(values[index]);
    if (epoch == null || value == null) continue;
    points.push({ epoch, value });
  }
  return points;
}

function bestRunByMetric(items, metricKey) {
  const direction = metricDirection(metricKey);
  let best = null;
  for (const item of Array.isArray(items) ? items : []) {
    const points = seriesPoints(item, metricKey);
    if (points.length === 0) continue;
    const candidate = direction === "min"
      ? points.reduce((min, row) => (row.value < min ? row.value : min), points[0].value)
      : points.reduce((max, row) => (row.value > max ? row.value : max), points[0].value);
    if (!best) {
      best = { id: String(item.experiment_id), value: candidate };
      continue;
    }
    if ((direction === "min" && candidate < best.value) || (direction === "max" && candidate > best.value)) {
      best = { id: String(item.experiment_id), value: candidate };
    }
  }
  return best?.id ?? null;
}

function hyperparamXValue(item, key) {
  const config = item?.config && typeof item.config === "object" ? item.config : {};
  if (key === "learning_rate") return asNumber(config?.optimizer?.lr);
  if (key === "batch_size") return asNumber(config?.batch_size);
  if (key === "epochs") return asNumber(config?.epochs);
  if (key === "augmentation") {
    const value = String(config?.augmentation_mode ?? config?.augmentation ?? "none");
    const map = { none: 0, light: 1, medium: 2, heavy: 3, custom: 4 };
    return map[value] ?? 0;
  }
  return null;
}

function scatterYValue(item, key) {
  if (key === "best_val_accuracy") {
    if (item?.best?.metric_name === "val_accuracy") return asNumber(item?.best?.metric_value);
    const points = seriesPoints(item, "val_accuracy");
    if (points.length === 0) return null;
    return points.reduce((max, row) => (row.value > max ? row.value : max), points[0].value);
  }
  if (key === "best_val_loss") {
    if (item?.best?.metric_name === "val_loss") return asNumber(item?.best?.metric_value);
    const points = seriesPoints(item, "val_loss");
    if (points.length === 0) return null;
    return points.reduce((min, row) => (row.value < min ? row.value : min), points[0].value);
  }
  if (key === "final_val_accuracy") return asNumber(item?.final?.val_accuracy);
  return null;
}

function scatterPoints(items, xKey, yKey) {
  const points = [];
  for (const item of Array.isArray(items) ? items : []) {
    const x = hyperparamXValue(item, xKey);
    const y = scatterYValue(item, yKey);
    if (x == null || y == null) continue;
    points.push({
      experimentId: String(item.experiment_id),
      name: String(item.name ?? item.experiment_id),
      x,
      y,
      status: String(item.status ?? ""),
      augmentationMode: String(item?.config?.augmentation_mode ?? item?.config?.augmentation ?? "none"),
      augmentationSummary: typeof item?.config?.augmentation_summary === "string" ? item.config.augmentation_summary : null,
    });
  }
  return points;
}

module.exports = {
  metricDirection,
  defaultSelectedRunIds,
  filterAnalyticsItems,
  buildAnalyticsSummary,
  seriesPoints,
  bestRunByMetric,
  hyperparamXValue,
  scatterYValue,
  scatterPoints,
};
